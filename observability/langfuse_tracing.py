"""Langfuse tracing helpers for the RAG request path.

Design (Langfuse SDK v4 - verified against langfuse 4.17.0):

* ONE logical trace per user request. FastAPI owns the ROOT trace; it mints
  the trace id and passes it (as ``langfuse_trace_id`` / the optional
  ``langfuse_parent_span_id``) to BentoML in the request body. BentoML then
  opens its operations as children of that SAME trace via ``trace_context``,
  so the whole request appears as a single correlated trace instead of two
  unrelated roots.
* The v4 client API used here (all verified to exist on ``Langfuse``):
      client.start_as_current_observation(trace_context=..., name=...,
                                          as_type=..., input=..., output=...,
                                          metadata=...)   # context manager
      client.create_trace_id()
      client.create_score(name=..., value=..., trace_id=..., comment=...)
      client.get_current_trace_id()
  We NEVER invent method names and never mix SDK major versions.

No-op safety
------------
When Langfuse is not configured (no public key, or the SDK is unavailable),
every helper degrades to a no-op so importing the module never fails and the
request path never blocks on tracing. Nothing here raises for a missing
trace backend.

Privacy
-------
Callers must pass ALREADY-REDACTED text. This module only forwards what it is
given to ``input``/``output``/``metadata``; it never injects raw PII itself,
but it also cannot clean what a caller hands it. Use the guardrail module
first and pass redacted text.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

from rag_project.config.settings import get_langfuse_config

_client = None
_client_ready = False


def _load_env_keys() -> dict:
    """Return the effective Langfuse config, allowing env-var overrides."""
    config = dict(get_langfuse_config())
    # Environment variables take precedence over config.yaml so secrets are
    # never committed to the repo; config.yaml holds only the host default.
    config["public_key"] = os.getenv(
        "LANGFUSE_PUBLIC_KEY", config.get("public_key", "")
    )
    config["secret_key"] = os.getenv(
        "LANGFUSE_SECRET_KEY", config.get("secret_key", "")
    )
    config["host"] = os.getenv("LANGFUSE_HOST", config.get("host", ""))
    config["release"] = os.getenv("LANGFUSE_RELEASE", config.get("release", ""))
    return config


def is_enabled() -> bool:
    """True only when Langfuse has both keys configured."""
    config = _load_env_keys()
    return bool(config.get("public_key") and config.get("secret_key"))


def get_client():
    """Return a shared Langfuse client, or None when not configured.

    The client is created lazily and cached. When disabled we return None so
    callers can skip tracing entirely (no network, no import side effects).
    """
    global _client, _client_ready
    if _client_ready:
        return _client
    _client_ready = True

    if not is_enabled():
        return None
    try:
        import langfuse

        config = _load_env_keys()
        kwargs = {
            "public_key": config["public_key"],
            "secret_key": config["secret_key"],
        }
        if config.get("host"):
            kwargs["host"] = config["host"]
        if config.get("release"):
            kwargs["release"] = config["release"]
        _client = langfuse.Langfuse(**kwargs)
    except Exception:  # noqa: BLE001 - tracing must never break a request
        _client = None
    return _client


def new_trace_id() -> str:
    """Mint a fresh trace id (used by the FastAPI gateway for the root)."""
    client = get_client()
    if client is not None:
        try:
            return client.create_trace_id()
        except Exception:  # noqa: BLE001
            pass
    # Deterministic-enough fallback id when tracing is disabled; callers use
    # it only for internal correlation, not for Langfuse.
    import uuid

    return str(uuid.uuid4())


def current_trace_id():
    """Return the active trace id from the SDK context, if any."""
    client = get_client()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def span(
    name: str,
    as_type: str = "span",
    trace_id: Optional[str] = None,
    input: Any = None,
    metadata: Optional[dict] = None,
):
    """Open a Langfuse observation as a context manager.

    * When ``trace_id`` is given, the observation is attached to that trace
      (used by BentoML to continue the FastAPI root trace). When omitted, the
      SDK attaches to the current trace in context.
    * ``as_type`` uses v4 observation types: span, generation, retriever,
      guardrail, embedding, evaluator, etc.
    * When tracing is disabled, this yields ``None`` and the body still runs,
      so calling code never needs a guard clause.

    Usage:
        with span("retrieval", as_type="retriever", trace_id=tid) as _:
            docs = retriever.invoke(question)
    """
    client = get_client()
    if client is None:
        yield None
        return

    trace_context = {"trace_id": trace_id} if trace_id else None

    # Open the SDK observation FIRST. If the SDK itself fails (bad key, network
    # glitch, version drift), degrade to a no-op span: yield None and let the
    # body still run. This guard only covers *entering* the observation.
    try:
        _cm = client.start_as_current_observation(
            trace_context=trace_context,
            name=name,
            as_type=as_type,
            input=input,
            metadata=metadata or {},
        )
        _observation = _cm.__enter__()
    except Exception:  # noqa: BLE001 - never fail a request for tracing
        yield None
        return

    # Now yield to the CALLER's body. We must NOT wrap this in a blanket
    # `except: yield None`: with @contextmanager an exception raised inside
    # the caller's `with` body is thrown back in at the yield, and swallowing
    # it here would make contextlib raise
    # "generator didn't stop after throw()", masking the real error.
    # Instead we close the observation and re-raise the original exception.
    try:
        yield _observation
    except BaseException as _body_exc:  # noqa: BLE001
        try:
            _cm.__exit__(type(_body_exc), _body_exc, _body_exc.__traceback__)
        except Exception:  # noqa: BLE001 - never let SDK close mask the error
            pass
        raise
    else:
        # Normal exit: close the observation, but a failure to close must not
        # fail the request either.
        try:
            _cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


def score_trace(trace_id: str, name: str, value, comment: Optional[str] = None):
    """Attach a score to an EXISTING trace by id (v4 create_score).

    Used to bind the RAGAS faithfulness score to the correct request's trace,
    including from a background/async evaluator that only has the trace id.
    Returns True when the score was sent, False otherwise (disabled/failed).
    """
    client = get_client()
    if client is None or not trace_id:
        return False
    try:
        client.create_score(
            name=name, value=value, trace_id=trace_id, comment=comment or ""
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def score_current_trace(name: str, value, comment: Optional[str] = None):
    """Attach a score to the trace currently open in the SDK context."""
    client = get_client()
    if client is None:
        return False
    try:
        client.score_current_trace(name=name, value=value, comment=comment or "")
        return True
    except Exception:  # noqa: BLE001
        return False


def flush():
    """Flush buffered events to Langfuse (safe no-op when disabled)."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # noqa: BLE001
        pass
