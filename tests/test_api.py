"""Tests for the FastAPI gateway (src/rag_project/api/app.py).

The gateway is a PURE HTTP gateway: it validates the request, applies the
input PII guardrail, and forwards to a BentoML endpoint. It does NOT build or
run the RAG chain, retrieve documents, or call the LLM. These tests therefore
never import the chain, Chroma, or the embedding model.

The BentoML endpoint is MOCKED at the ``requests.post`` boundary, so these
tests run offline with no remote GPU and no Chroma index.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_project.api import app as app_module
from rag_project.api.app import app

client = TestClient(app)


class _FakeResponse:
    """Minimal stand-in for a requests.Response returned by the mock."""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _ok_payload(answer="إجابة قانونية من الخدمة.", sources=None):
    return {
        "answer": answer,
        "sources": sources if sources is not None else ["المادة 20 من القانون المدني"],
        "langfuse_trace_id": "trace-abc",
    }


# ----------------------------------------------------------
# Health + validation (no BentoML contact)
# ----------------------------------------------------------
def test_health_endpoint():
    """GET /health reports the gateway healthy without touching BentoML."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "fastapi-gateway"


def test_ask_requires_a_question():
    """POST /ask rejects a body with no question (Pydantic 422)."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_gateway_does_not_build_the_chain():
    """The gateway module must NOT import/use the RAG chain builder."""
    source = open(app_module.__file__, encoding="utf-8").read()
    for forbidden in ("build_rag_chain", "build_vllm_llm", "generate_from_prompt"):
        assert forbidden not in source


# ----------------------------------------------------------
# /ask forwarding with a mocked BentoML endpoint
# ----------------------------------------------------------
@patch.object(app_module.requests, "post")
def test_ask_forwards_to_bentoml(mock_post):
    """POST /ask forwards the question and returns the required contract."""
    mock_post.return_value = _FakeResponse(200, _ok_payload())

    response = client.post("/ask", json={"question": "ما هي المادة 20؟"})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"] == "إجابة قانونية من الخدمة."
    assert isinstance(body["sources"], list) and body["sources"]
    # The question is echoed back for existing clients.
    assert body["question"] == "ما هي المادة 20؟"

    # Exactly one POST was made, to the BentoML /ask_json path.
    mock_post.assert_called_once()
    called_url = mock_post.call_args.args[0]
    assert called_url.endswith("/ask_json")


@patch.object(app_module.requests, "post")
def test_ask_handles_bentoml_unavailable(mock_post):
    """A BentoML connection error becomes a safe 503, not a stack trace."""
    import requests as _requests

    mock_post.side_effect = _requests.exceptions.ConnectionError("boom")
    response = client.post("/ask", json={"question": "سؤال"})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


@patch.object(app_module.requests, "post")
def test_ask_handles_malformed_upstream(mock_post):
    """A malformed BentoML body becomes a safe 502 error."""
    mock_post.return_value = _FakeResponse(200, {"answer": 123, "sources": None})
    response = client.post("/ask", json={"question": "سؤال"})
    assert response.status_code == 502
    assert "malformed" in response.json()["detail"].lower()


# ----------------------------------------------------------
# Input guardrail (blocking) before forwarding
# ----------------------------------------------------------
@patch.object(app_module.requests, "post")
def test_ask_blocks_national_id(mock_post):
    """A national ID in the question is blocked before forwarding."""
    response = client.post(
        "/ask", json={"question": "رقمي القومي 29001011234567، ما حقوقي؟"}
    )
    assert response.status_code == 400
    assert "blocked" in response.json()["detail"].lower()
    # Nothing was forwarded to BentoML when blocked.
    mock_post.assert_not_called()


# ----------------------------------------------------------
# /ask/stream forwarding (SSE) with a mocked BentoML endpoint
# ----------------------------------------------------------
class _FakeStreamResponse:
    """A streaming requests.Response whose iter_lines() yields chunks."""

    def __init__(self, status_code=200, lines=None):
        self.status_code = status_code
        self._lines = lines if lines is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


@patch.object(app_module.requests, "post")
def test_ask_stream_returns_sse(mock_post):
    """/ask/stream forwards chunks and returns text/event-stream."""
    mock_post.return_value = _FakeStreamResponse(200, ['"المادة"', '" "', '"20"'])
    response = client.post("/ask/stream", json={"question": "ما هي المادة 20؟"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    assert "data: المادة" in text
    assert "data: [DONE]" in text
