"""FastAPI application exposing the RAG service.

Endpoints:

- GET  /health  -> service health/status check
- POST /ask     -> answer a question through the RAG chain

The API contains NO retrieval/generation logic of its own: it only
calls the integration layer (build_rag_chain) from
rag_project.generation.chain, which connects the config-driven
vector store, retriever, prompt, and LLM.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_project.generation.chain import build_rag_chain

# The FastAPI application object (uvicorn target: rag_project.api.app:app).
app = FastAPI(title="Egyptian Civil Code RAG")

# The chain is built once (embedding model + Chroma client are heavy)
# and reused for every request.
_rag_chain = None


def get_rag_chain():
    """Return the cached RAG chain, building it on first use."""
    global _rag_chain
    if _rag_chain is None:
        # One call wires retrieval + prompt + LLM from the config.
        _rag_chain = build_rag_chain()
    return _rag_chain


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str


@app.get("/health")
def health():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    """Answer a question using the RAG chain."""
    try:
        # All retrieval + generation work happens inside the chain.
        answer = get_rag_chain().invoke(request.question)
    except Exception as error:
        # Surface configuration/setup problems clearly to the caller.
        raise HTTPException(status_code=500, detail=str(error))

    return {"question": request.question, "answer": answer}
