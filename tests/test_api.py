"""Tests for the FastAPI application.

Verifies GET /health and POST /ask. The /ask test runs the real RAG
chain (retriever + LLM), so it needs the generation key and a built
Chroma index; the request-shape test needs neither.
"""

import os

import pytest

from fastapi.testclient import TestClient

from rag_project.api.app import app
from rag_project.generation import llm
from rag_project.indexing import vector_store

# A test client bound to the FastAPI application.
client = TestClient(app)

requires_ready = pytest.mark.skipif(
    not (os.getenv(llm.GENERATION_API_KEY_VARIABLE) and vector_store.index_exists()),
    reason=(f"{llm.GENERATION_API_KEY_VARIABLE} and a built Chroma index are required"),
)


def test_health_endpoint():
    """GET /health reports the service as healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_requires_a_question():
    """POST /ask rejects a request without a question (validation only)."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


@requires_ready
def test_ask_endpoint_returns_answer():
    """POST /ask answers through the RAG chain."""
    question = "هل ينقضي عقد المقاولة باستحالة تنفيذ العمل المعقود عليه؟"
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 200
    body = response.json()
    # The endpoint echoes the question and returns the chain's answer.
    assert body["question"] == question
    assert body["answer"].strip() != ""
