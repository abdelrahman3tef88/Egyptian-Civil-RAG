"""FastAPI gateway for the Egyptian Civil Code RAG service.

ROLE: this process is ONLY an HTTP gateway. It does NOT build or run the RAG
chain. It does not retrieve documents, build the RAG prompt, or generate
answers. All of that lives in the BentoML service (service/bentoml_service.py),
which owns the retriever, Chroma, the LangChain prompt, and the vLLM call.

Request flow (this file implements the gateway box):

    Client
      -> FastAPI /ask               (this file: validate + input guardrail)
      -> forward HTTP POST to BentoML /ask
      -> BentoML runs retrieval -> prompt -> vLLM -> output guardrail
      -> structured JSON comes back {"answer", "sources", ...}
      -> FastAPI returns the same contract to the client

Configuration is via environment variables (see .env.example):
    BENTOML_BASE_URL   BentoML service base (default http://127.0.0.1:3000)
    BENTOML_TIMEOUT    per-request timeout in seconds (default 300)
    GUARDRAIL_ENABLED  "1"/"true" to run the input PII guardrail (default on)

Security: HTTP errors never expose stack traces, API keys, or internal
secrets. Only a short, generic message is returned to the client; details are
logged server-side at debug level only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from guardrails import pii_detector
from monitoring import metrics as m
from observability import langfuse_tracing as lf

logger = logging.getLogger("rag_gateway")

DEFAULT_BENTOML_URL = "http://127.0.0.1:3000"
DEFAULT_TIMEOUT = 300.0

# BentoML routes. /ask_json returns the full non-streaming JSON contract
# {"answer", "sources", ...}; /ask is the streaming chunk endpoint that
# /ask/stream consumes.
_BENTOML_ASK_PATH = "/ask"
_BENTOML_ASK_JSON_PATH = "/ask_json"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def get_bentoml_base_url() -> str:
    return os.getenv("BENTOML_BASE_URL", DEFAULT_BENTOML_URL).strip().rstrip("/")


def get_bentoml_timeout() -> float:
    try:
        return float(os.getenv("BENTOML_TIMEOUT", str(DEFAULT_TIMEOUT)))
    except ValueError:
        return DEFAULT_TIMEOUT


app = FastAPI(title="Egyptian Civil Code RAG Gateway")


class AskRequest(BaseModel):
    """Request body for POST /ask (validated by Pydantic)."""

    question: str = Field(..., min_length=1)
    # Optional trace correlation id supplied by an upstream caller.
    langfuse_trace_id: Optional[str] = None


class AskResponse(BaseModel):
    """The externally-visible response contract.

    ``answer`` and ``sources`` are the required fields. ``question`` and
    ``langfuse_trace_id`` are preserved for existing clients/tests and never
    conflict with the required contract.
    """

    answer: str
    sources: List[str]
    question: Optional[str] = None
    langfuse_trace_id: Optional[str] = None


@app.get("/health")
def health():
    """Liveness check. Does NOT build the chain or contact the LLM.

    It only reports that the gateway is up and where it forwards requests. A
    true end-to-end readiness check would probe BentoML, which we deliberately
    avoid so /health stays cheap and never triggers model loading.
    """
    return {
        "status": "ok",
        "service": "fastapi-gateway",
        "bentoml_base_url": get_bentoml_base_url(),
        "guardrail_enabled": _env_flag("GUARDRAIL_ENABLED", True),
        "tracing_enabled": lf.is_enabled(),
    }


@app.get("/metrics")
def metrics_endpoint():
    """Expose Prometheus metrics for scraping (standard text format)."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask")
def ask(request: AskRequest):
    """Forward a validated question to BentoML and return its answer.

    This endpoint does NOT run the RAG chain. It (1) validates the body with
    Pydantic, (2) applies the input PII guardrail, (3) forwards to the BentoML
    /ask endpoint, (4) normalises the response into the required contract, and
    (5) records metrics + a Langfuse span. Any BentoML failure (connection,
    timeout, HTTP error, malformed body) becomes a safe, generic error.
    """
    start = time.perf_counter()

    # Mint/continue the request trace id (Langfuse no-ops when disabled).
    trace_id = request.langfuse_trace_id or lf.new_trace_id()
    question = request.question

    # --- Input guardrail (before forwarding) -----------------------------
    if _env_flag("GUARDRAIL_ENABLED", True):
        with lf.span(
            "input_guardrail",
            as_type="guardrail",
            trace_id=trace_id,
            metadata={"phase": "input"},
        ):
            scan = pii_detector.detect(question)
            action = pii_detector.decide(scan)
            m.record_guardrail(
                "fastapi", "input", action, ",".join(scan.kinds) or "none"
            )
            if action == pii_detector.BLOCK:
                m.record_request(
                    "fastapi", "/ask", "blocked", time.perf_counter() - start
                )
                raise HTTPException(
                    status_code=400,
                    detail="The request was blocked by the input guardrail.",
                )
            if action == pii_detector.MASK:
                question = pii_detector.mask(question)

    # --- Forward to BentoML ---------------------------------------------
    payload = {"question": question, "langfuse_trace_id": trace_id}
    url = get_bentoml_base_url() + _BENTOML_ASK_JSON_PATH
    timeout = get_bentoml_timeout()

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        m.record_error("fastapi", "/ask", "upstream_timeout")
        m.record_request("fastapi", "/ask", "error", time.perf_counter() - start)
        raise HTTPException(status_code=504, detail="The RAG service timed out.")
    except requests.exceptions.ConnectionError:
        m.record_error("fastapi", "/ask", "upstream_unavailable")
        m.record_request("fastapi", "/ask", "error", time.perf_counter() - start)
        raise HTTPException(status_code=503, detail="The RAG service is unavailable.")
    except requests.exceptions.RequestException:
        logger.debug("bentoml request failed", exc_info=True)
        m.record_error("fastapi", "/ask", "upstream_error")
        m.record_request("fastapi", "/ask", "error", time.perf_counter() - start)
        raise HTTPException(status_code=502, detail="The RAG service errored.")

    if response.status_code != 200:
        logger.debug("bentoml status %s", response.status_code)
        m.record_error("fastapi", "/ask", f"upstream_{response.status_code}")
        m.record_request("fastapi", "/ask", "error", time.perf_counter() - start)
        raise HTTPException(
            status_code=502, detail="The RAG service returned an error."
        )

    # --- Parse + normalise the contract ---------------------------------
    try:
        body = response.json()
        answer = body.get("answer")
        sources = body.get("sources", [])
        if not isinstance(answer, str):
            raise ValueError("answer is not a string")
        if not isinstance(sources, list):
            sources = []
    except (ValueError, AttributeError):
        logger.debug("bentoml malformed response", exc_info=True)
        m.record_error("fastapi", "/ask", "malformed_upstream")
        m.record_request("fastapi", "/ask", "error", time.perf_counter() - start)
        raise HTTPException(
            status_code=502,
            detail="The RAG service returned a malformed response.",
        )

    m.record_request("fastapi", "/ask", "success", time.perf_counter() - start)
    result = AskResponse(
        answer=answer,
        sources=[str(s) for s in sources],
        question=request.question,
        langfuse_trace_id=body.get("langfuse_trace_id", trace_id),
    )
    lf.flush()
    return result


# ==========================================================
# Streaming endpoint
# ----------------------------------------------------------
# POST /ask/stream forwards a question to the BentoML /ask endpoint and
# streams the generated answer back to the client token-by-token as
# Server-Sent Events (text/event-stream). Each SSE event is:
#
#     data: <token chunk>\n\n
#
# and a terminal event ``data: [DONE]`` marks the end of the stream.
#
# This is the ONLY endpoint that streams progressively. The plain JSON
# /ask endpoint returns the complete answer at once and must never be
# described as "streaming".
#
# We do NOT buffer the whole answer server-side before responding: each
# chunk BentoML emits is forwarded immediately, so the client sees the same
# progressive token flow BentoML produces from vLLM.
# ==========================================================

_SSE_DONE = "data: [DONE]\n\n"


def _sse_event(chunk: str) -> str:
    """Wrap a token chunk as a single SSE data event."""
    # Collapse any embedded newlines so one chunk stays one event.
    return f"data: {chunk.replace(chr(10), ' ')}\n\n"


def _stream_from_bentoml(question: str, trace_id: str):
    """Yield SSE events, forwarding each BentoML chunk as it arrives.

    Kept as a generator so the FastAPI StreamingResponse can flush chunks
    immediately instead of buffering the entire answer.
    """
    url = get_bentoml_base_url() + _BENTOML_ASK_PATH
    timeout = get_bentoml_timeout()
    payload = {"question": question, "langfuse_trace_id": trace_id}

    try:
        with requests.post(url, json=payload, timeout=timeout, stream=True) as response:
            if response.status_code != 200:
                # Upstream error surfaced as a terminal SSE event (no stack).
                yield _sse_event("Error: the RAG service returned an error.")
                return
            # iter_lines() yields each streamed line as it arrives; we forward
            # non-empty lines so tokens flow progressively, never buffered.
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # BentoML /ask yields raw string chunks; strip an optional
                # JSON string wrapper (e.g. "\"token\"") without over-parsing.
                chunk = line.strip()
                if chunk.startswith('"') and chunk.endswith('"'):
                    try:
                        chunk = json.loads(chunk)
                    except ValueError:
                        pass
                if chunk:
                    yield _sse_event(chunk)
    except requests.exceptions.RequestException:
        # Only a generic message reaches the client; details stay server-side.
        yield _sse_event("Error: the RAG service is unavailable.")
    finally:
        yield _SSE_DONE
        lf.flush()


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    """Stream the generated answer back to the client as SSE events.

    Applies the SAME input validation and input PII guardrail as /ask before
    forwarding. Returns ``text/event-stream`` so tokens arrive progressively.
    """
    start = time.perf_counter()
    trace_id = request.langfuse_trace_id or lf.new_trace_id()
    question = request.question

    # --- Input guardrail (before forwarding) -----------------------------
    if _env_flag("GUARDRAIL_ENABLED", True):
        with lf.span(
            "input_guardrail",
            as_type="guardrail",
            trace_id=trace_id,
            metadata={"phase": "input", "endpoint": "stream"},
        ):
            scan = pii_detector.detect(question)
            action = pii_detector.decide(scan)
            m.record_guardrail(
                "fastapi", "input", action, ",".join(scan.kinds) or "none"
            )
            if action == pii_detector.BLOCK:
                m.record_request(
                    "fastapi", "/ask/stream", "blocked", time.perf_counter() - start
                )
                raise HTTPException(
                    status_code=400,
                    detail="The request was blocked by the input guardrail.",
                )
            if action == pii_detector.MASK:
                question = pii_detector.mask(question)

    return StreamingResponse(
        _stream_from_bentoml(question, trace_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
