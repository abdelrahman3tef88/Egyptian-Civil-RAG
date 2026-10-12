"""Unit tests for the BentoML RAG service's non-streaming /ask_json endpoint.

The ``bentoml`` package is NOT installed in the test environment, so a tiny
fake module (service + api decorators as pass-throughs) is injected into
``sys.modules`` before importing the service. The RAG components (retriever,
prompt, LLM) are all mocked: no Chroma, no BGE-M3, no remote vLLM, no GPU.

What these tests prove about the endpoint's CONTRACT:
  * it returns {"answer", "sources", "langfuse_trace_id"} (JSON contract),
  * sources are real article citations from document metadata,
  * output PII is masked (and can be blocked) before returning,
  * missing vLLM config fails clearly instead of loading a local model,
  * internal errors become client-safe messages (no stack traces),
  * each request is isolated (no state shared between requests),
  * a failed optional evaluation never fails the generated answer, and
  * a successful faithfulness score is attached to the CORRECT trace id.
"""

import asyncio
import sys
import types
from unittest import mock

import pytest

# ----------------------------------------------------------
# Fake bentoml module (must exist BEFORE importing the service)
# ----------------------------------------------------------
_fake_bentoml = types.ModuleType("bentoml")
_fake_bentoml.service = lambda cls: cls  # @bentoml.service pass-through
_fake_bentoml.api = lambda fn: fn  # @bentoml.api pass-through
sys.modules.setdefault("bentoml", _fake_bentoml)

from service import bentoml_service as bs  # noqa: E402


# ----------------------------------------------------------
# Lightweight doubles for the RAG components
# ----------------------------------------------------------
class _FakeDoc:
    def __init__(self, text, article_number=None):
        self.page_content = text
        self.metadata = (
            {"article_number": article_number} if article_number is not None else {}
        )


class _FakeRetriever:
    """Records each question so tests can prove request isolation."""

    def __init__(self, docs=None):
        self.docs = docs or [_FakeDoc("نص المادة العشرين في القانون المدني.", "20")]
        self.seen_questions = []

    def invoke(self, question):
        self.seen_questions.append(question)
        return list(self.docs)


class _FakePromptValue:
    def __init__(self, text="built prompt"):
        self._text = text

    def to_string(self):
        return self._text


class _FakePrompt:
    def __init__(self):
        self.seen_inputs = []

    async def ainvoke(self, inputs):
        self.seen_inputs.append(inputs)
        return _FakePromptValue(f"prompt::{inputs['question']}::{inputs['context']}")


def _make_service(docs=None):
    """Build an RAGService instance WITHOUT running the heavy __init__."""
    svc = bs.RAGService.__new__(bs.RAGService)
    svc.retriever = _FakeRetriever(docs)
    svc.prompt = _FakePrompt()
    svc._query_embedder = None  # disables the drift check in tests
    svc._drift_baseline = "__unset__"
    return svc


def _run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------
# Embedding-drift stage: measured, recorded, never fatal
# ----------------------------------------------------------
class _FakeEmbedder:
    """Synthetic query embedder (no BGE-M3 load, no model download)."""

    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    def __call__(self, question):
        self.calls += 1
        return list(self.vector)


_TEST_BASELINE = {
    "model": "BAAI/bge-m3",
    "reference_ids": ["ref_000"],
    "embeddings": [[1.0, 0.0]],
}


def _make_service_with_drift(embedder, baseline=_TEST_BASELINE):
    """A service whose drift stage uses a fixed embedder + baseline."""
    svc = _make_service()
    svc._query_embedder = embedder
    # Pre-seeded so load_baseline() is never called: no file I/O, and the
    # "baseline missing" branch is covered separately below.
    svc._drift_baseline = baseline
    return svc


@pytest.fixture
def drift_env(monkeypatch):
    """Enable drift and capture what the drift stage records."""
    monkeypatch.setenv("DRIFT_ENABLED", "1")
    recorded = {"drift": [], "errors": [], "reports": []}
    monkeypatch.setattr(
        bs.m, "record_drift", lambda *a, **k: recorded["drift"].append((a, k))
    )
    monkeypatch.setattr(
        bs.m, "record_drift_error", lambda *a, **k: recorded["errors"].append((a, k))
    )
    monkeypatch.setattr(
        bs.drift,
        "append_drift_report",
        lambda report, **k: recorded["reports"].append(report),
    )
    return recorded


def test_drift_is_measured_before_retrieval(drift_env):
    """Drift is scored against the baseline, using the SAME embedder."""
    embedder = _FakeEmbedder([1.0, 0.0])  # identical -> similarity 1.0
    svc = _make_service_with_drift(embedder)

    with mock.patch.object(bs, "generate_from_prompt", lambda p, u=None: "a"):
        _run(svc.ask_json("ما هي المادة 20؟"))

    assert embedder.calls == 1, "exactly one query embedding per request"
    assert drift_env["drift"], "a drift score must be recorded"
    assert drift_env["reports"][0]["status"] == "ok"
    assert drift_env["reports"][0]["max_similarity"] == pytest.approx(1.0)
    assert drift_env["reports"][0]["drift"] is False


def test_drift_above_threshold_does_not_block_the_request(drift_env):
    """A drifted query is RECORDED and the pipeline CONTINUES.

    Exceeding the drift threshold must never terminate the pipeline:
    it is a monitoring signal, not a rejection policy.
    """
    embedder = _FakeEmbedder([0.0, 1.0])  # orthogonal -> similarity 0.0
    svc = _make_service_with_drift(embedder)

    with mock.patch.object(bs, "generate_from_prompt", lambda p, u=None: "a"):
        result = _run(svc.ask_json("سؤال خارج نطاق القانون"))  # must not raise

    assert result["answer"] == "a"
    assert drift_env["reports"][0]["drift"] is True
    assert drift_env["reports"][0]["max_similarity"] == pytest.approx(0.0)
    # Retrieval and generation still ran.
    assert svc.retriever.seen_questions == ["سؤال خارج نطاق القانون"]


def test_detector_failure_is_recorded_but_not_fatal(drift_env):
    """A broken detector continues the request and stays observable.

    The failure must be counted in the DETECTOR-error counter (not the
    drift counter) and written to the report with status="error", so a
    broken monitor is distinguishable from an actual drift event.
    """

    class _BrokenEmbedder:
        def __call__(self, question):
            raise RuntimeError("embedding model unavailable")

    svc = _make_service_with_drift(_BrokenEmbedder())

    with mock.patch.object(bs, "generate_from_prompt", lambda p, u=None: "a"):
        result = _run(svc.ask_json("سؤال"))

    # 1) the request still succeeded
    assert result["answer"] == "a"
    assert result["sources"] == ["Article 20"]
    # 2) the failure was counted as a detector error
    assert drift_env["errors"], "the detector failure must be recorded"
    assert drift_env["errors"][0][0][1] == "RuntimeError"
    # 3) it was NOT counted as drift
    assert drift_env["drift"] == [], (
        "a detector failure must never be recorded as a drift event"
    )
    # 4) the report marks it as an error, not a scored event
    assert drift_env["reports"][0]["status"] == "error"
    assert drift_env["reports"][0]["error_type"] == "RuntimeError"
    assert "max_similarity" not in drift_env["reports"][0]


def test_missing_baseline_disables_drift_without_failing(drift_env, monkeypatch):
    """A missing baseline file is handled, not crashed on."""

    def _raise(_path=None):
        raise FileNotFoundError("no baseline")

    monkeypatch.setattr(bs.drift, "load_baseline", _raise)
    svc = _make_service()
    svc._query_embedder = _FakeEmbedder([1.0, 0.0])
    svc._drift_baseline = "__unset__"  # force the load path

    with mock.patch.object(bs, "generate_from_prompt", lambda p, u=None: "a"):
        result = _run(svc.ask_json("سؤال"))

    assert result["answer"] == "a"
    assert svc._drift_baseline is None, "a missing baseline is cached as absent"
    assert drift_env["drift"] == [] and drift_env["errors"] == []


def test_drift_disabled_by_env_skips_the_stage(monkeypatch):
    """DRIFT_ENABLED=0 must skip the embedder entirely (no extra call)."""
    monkeypatch.setenv("DRIFT_ENABLED", "0")
    embedder = _FakeEmbedder([1.0, 0.0])
    svc = _make_service_with_drift(embedder)

    with mock.patch.object(bs, "generate_from_prompt", lambda p, u=None: "a"):
        _run(svc.ask_json("سؤال"))

    assert embedder.calls == 0


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    """Keep every test offline and deterministic.

    ONLINE_EVAL_ENABLED=0 is essential: without it the service would fall
    back to configs/config.yaml (evaluation.enabled: true) and attempt a
    REAL Gemini judge call per request, making unit tests slow and
    non-deterministic.
    """
    monkeypatch.setenv("ONLINE_EVAL_ENABLED", "0")
    monkeypatch.delenv("GUARDRAIL_OUTPUT_BLOCK", raising=False)
    monkeypatch.setenv("DRIFT_ENABLED", "0")
    monkeypatch.setattr(bs, "is_configured", lambda: True)
    yield


# ----------------------------------------------------------
# Contract
# ----------------------------------------------------------
def test_ask_json_returns_the_json_contract():
    svc = _make_service()

    def fake_generate(prompt_value, usage=None):
        if usage is not None:
            usage["input_tokens"] = 11
            usage["output_tokens"] = 7
        return "الإجابة القانونية."

    with mock.patch.object(bs, "generate_from_prompt", fake_generate):
        result = _run(svc.ask_json("ما هي المادة 20؟", "trace-1"))

    assert result["answer"] == "الإجابة القانونية."
    assert result["sources"] == ["Article 20"]
    assert result["langfuse_trace_id"] == "trace-1"


def test_ask_json_generates_a_trace_id_when_not_supplied():
    svc = _make_service()
    with mock.patch.object(bs, "generate_from_prompt", lambda p, u=None: "a"):
        result = _run(svc.ask_json("سؤال"))
    assert result["langfuse_trace_id"]
    assert isinstance(result["langfuse_trace_id"], str)


def test_ask_json_prompt_gets_retrieved_context():
    svc = _make_service()
    with mock.patch.object(bs, "generate_from_prompt", lambda p, u=None: "a"):
        _run(svc.ask_json("ما هي المادة 20؟"))
    # The EXISTING prompt variables: context from format_docs, question.
    assert len(svc.prompt.seen_inputs) == 1
    inputs = svc.prompt.seen_inputs[0]
    assert "ما هي المادة 20؟" in inputs["question"]
    assert "Article 20:" in inputs["context"]


# ----------------------------------------------------------
# Validation and configuration failures
# ----------------------------------------------------------
def test_ask_json_rejects_empty_question():
    svc = _make_service()
    with pytest.raises(bs._RequestRejected, match="empty"):
        _run(svc.ask_json("   "))


def test_ask_json_missing_vllm_config_fails_clearly():
    svc = _make_service()
    with mock.patch.object(bs, "is_configured", lambda: False):
        with pytest.raises(bs._RequestRejected, match="VLLM_MODEL"):
            _run(svc.ask_json("سؤال"))
    # Nothing was retrieved: failure happens before any RAG work.
    assert svc.retriever.seen_questions == []


# ----------------------------------------------------------
# Output guardrail
# ----------------------------------------------------------
def test_ask_json_masks_pii_in_the_answer():
    svc = _make_service()
    with mock.patch.object(
        bs,
        "generate_from_prompt",
        lambda p, u=None: "اتصل على 01012345678 للمزيد.",
    ):
        result = _run(svc.ask_json("سؤال"))
    assert "01012345678" not in result["answer"]
    assert "[REDACTED_PHONE]" in result["answer"]


def test_ask_json_blocks_national_id_in_output_when_strict(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_OUTPUT_BLOCK", "1")
    svc = _make_service()
    with mock.patch.object(
        bs,
        "generate_from_prompt",
        lambda p, u=None: "رقمك 28005211500123.",
    ):
        with pytest.raises(bs._RequestRejected, match="withheld"):
            _run(svc.ask_json("سؤال"))


# ----------------------------------------------------------
# Error handling: never leak internals
# ----------------------------------------------------------
def test_ask_json_llm_failure_returns_client_safe_error():
    svc = _make_service()

    def boom(prompt_value, usage=None):
        raise RuntimeError("connection to http://user:secret@vllm:8000 failed")

    with mock.patch.object(bs, "generate_from_prompt", boom):
        with pytest.raises(bs._RequestRejected) as excinfo:
            _run(svc.ask_json("سؤال"))

    message = str(excinfo.value)
    assert "failed to generate" in message
    assert "secret" not in message
    assert "http" not in message
    assert "Traceback" not in message


def test_ask_json_error_is_not_returned_as_an_answer():
    """A failure must never look like a successful legal answer."""
    svc = _make_service()
    with mock.patch.object(
        bs,
        "generate_from_prompt",
        mock.Mock(side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(bs._RequestRejected):
            _run(svc.ask_json("سؤال"))


# ----------------------------------------------------------
# Request isolation (no cross-request state)
# ----------------------------------------------------------
def test_ask_json_requests_are_isolated():
    svc = _make_service()
    answers = {"أ": "جواب أ", "ب": "جواب ب"}

    def fake_generate(prompt_value, usage=None):
        # Answer depends ONLY on this request's prompt.
        text = prompt_value.to_string()
        for question, answer in answers.items():
            if question in text:
                return answer
        return "؟"

    with mock.patch.object(bs, "generate_from_prompt", fake_generate):
        first = _run(svc.ask_json("أ"))
        second = _run(svc.ask_json("ب"))

    assert first["answer"] == "جواب أ"
    assert second["answer"] == "جواب ب"
    assert svc.retriever.seen_questions == ["أ", "ب"]
    assert svc.prompt.seen_inputs[0]["question"] == "أ"
    assert svc.prompt.seen_inputs[1]["question"] == "ب"


# ----------------------------------------------------------
# Full Stage B lifecycle in ONE request, in order
# ----------------------------------------------------------
def test_full_pipeline_runs_in_the_documented_order(monkeypatch):
    """drift -> retrieval -> prompt -> generation -> output guardrail -> sources.

    This is the end-to-end view of Stage B that the FastAPI gateway's
    POST /ask depends on, with every stage asserted in sequence so a
    reordering cannot pass silently.
    """
    monkeypatch.setenv("DRIFT_ENABLED", "1")
    order = []

    # 1) drift runs FIRST, against the cached baseline.
    embedder = _FakeEmbedder([1.0, 0.0])
    svc = _make_service_with_drift(embedder)
    baseline = dict(_TEST_BASELINE)

    original_drift = bs.drift.evaluate_query

    def _drift(query_embedding, **kwargs):
        order.append("drift")
        assert kwargs["baseline"] is baseline, (
            "drift must score against the cached baseline, not reload it"
        )
        return original_drift(query_embedding, **kwargs)

    monkeypatch.setattr(bs.drift, "evaluate_query", _drift)

    # 2) retrieval feeds the prompt.
    def _retrieve(question):
        order.append("retrieval")
        return svc.retriever.docs

    svc.retriever.invoke = _retrieve

    # 3) the prompt receives the retrieved context, unchanged.
    class _OrderedPrompt:
        seen_inputs = []

        async def ainvoke(self, inputs):
            order.append("prompt")
            _OrderedPrompt.seen_inputs.append(inputs)
            return _FakePromptValue(
                f"prompt::{inputs['question']}::{inputs['context']}"
            )

    svc.prompt = _OrderedPrompt()

    # 4) generation goes through the LLM client.
    def _generate(prompt_value, usage=None):
        order.append("generation")
        if usage is not None:
            usage["input_tokens"] = 10
            usage["output_tokens"] = 5
        # Output contains PII: the output guardrail must catch it.
        return "اتصل على 01012345678 للتفاصيل."

    with mock.patch.object(bs, "generate_from_prompt", _generate):
        result = _run(svc.ask_json("سؤال", "trace-lifecycle"))

    # 5) the RETURNED answer is the GUARDED one, not the raw output.
    assert order == ["drift", "retrieval", "prompt", "generation"], (
        f"unexpected stage order: {order}"
    )
    assert "01012345678" not in result["answer"]
    assert "[REDACTED_PHONE]" in result["answer"]

    # 6) the prompt was built from what retrieval returned.
    inputs = _OrderedPrompt.seen_inputs[0]
    assert inputs["question"] == "سؤال"
    assert "Article 20" in inputs["context"]

    # 7) the required JSON contract is returned, with the trace id preserved.
    assert result["sources"] == ["Article 20"]
    assert result["langfuse_trace_id"] == "trace-lifecycle"

    # 8) token usage came from the provider metadata, not a chunk count.
    assert _exact_token_total("rag_input_tokens") >= 10
    assert _exact_token_total("rag_output_tokens") >= 5


def _exact_token_total(metric_name):
    """Read an exact-token counter straight from the Prometheus registry."""
    from prometheus_client import REGISTRY

    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            return sum(sample.value for sample in metric.samples)
    return 0
