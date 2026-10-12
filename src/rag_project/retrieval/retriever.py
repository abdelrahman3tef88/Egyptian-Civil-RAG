"""Retriever, driven by configs/config.yaml.

The search type and top-k value come from the central configuration
file; the retrieval behavior itself (similarity search over the
vector store) is unchanged.

Language isolation (strict):
--------------------------------------
Every retrieval result is filtered to the language of the *question* before
it can reach the prompt, the generation chain, or the RAGAS metrics.

* The language is detected reliably from the question text (``detect_language``),
  not by guessing from the question shape.
* Each chunk is checked two ways:
    1. its ``language`` metadata (written by the indexing stage), and
    2. the script of its text (fallback when metadata is missing/ambiguous).
* A chunk whose language differs from the question language is dropped.
  No cross-language chunk is ever promoted to a fallback or a filler.
  Only the remaining chunks keep their original retrieval order.
* The same language-filtered retriever is used by the chain and by the
  evaluation pipeline so generation and scoring always see the identical set.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.documents import Document

from rag_project.config import settings

# Configuration section for retrieval (configs/config.yaml).
CONFIG = settings.CONFIG["retrieval"]

# Regular expressions used for the text-script fallback check.
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_ENGLISH_LETTERS = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Return ``"ar"``, ``"en"`` or ``"other"`` for *text*.

    Arabic script wins when Arabic characters are present; otherwise the
    script with Latin letters; otherwise a neutral ``"other"``.  The
    decision is based on the actual script of the text, never on the
    surrounding shape of the question.
    """
    if not text:
        return "other"
    has_ar = bool(_ARABIC.search(text))
    has_en = bool(_ENGLISH_LETTERS.search(text))
    if has_ar and not has_en:
        return "ar"
    if has_en and not has_ar:
        return "en"
    if has_ar and has_en:
        # Both scripts present.  Pick the one that starts earlier.
        return (
            "ar"
            if _ARABIC.search(text).start() < _ENGLISH_LETTERS.search(text).start()
            else "en"
        )
    return "other"


def _chunk_language(chunk: Document) -> Optional[str]:
    """Return the language of *chunk* according to metadata, with a text fallback."""
    meta = getattr(chunk, "metadata", None) or {}
    lang = meta.get("language")
    if lang is not None and str(lang) in {"ar", "en"}:
        return str(lang)
    text = chunk.page_content or ""
    return detect_language(text)


class LanguageFilterRetriever:
    """Duck-typed wrapper: retrieve as the inner retriever does, then filter by language.

    This plain class keeps the retriever usable by the LCEL chain
    (``retriever | format_docs``), the evaluation pipeline
    (``retriever.invoke(question)``), and any async caller.

    ``__or__`` support is needed because the plain class does not implement
    LangChain's pipe operator; the fallback wraps ``invoke`` in a
    ``RunnableLambda`` so ``retriever | format_docs`` keeps working.
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def __or__(self, other: Any) -> Any:
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self.invoke) | other

    def invoke(self, query: str, **kwargs: Any) -> List[Document]:
        docs = self.retriever.invoke(query, **kwargs)
        query_language = detect_language(query)
        if query_language == "other":
            # No usable script in the question - keep the raw results and let
            # the caller decide how to handle them.
            return docs
        return [doc for doc in docs if _chunk_language(doc) == query_language]

    async def ainvoke(self, query: str, **kwargs: Any) -> List[Document]:
        docs = await self.retriever.ainvoke(query, **kwargs)
        query_language = detect_language(query)
        if query_language == "other":
            return docs
        return [doc for doc in docs if _chunk_language(doc) == query_language]

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        # Classic LangChain retriever interface (abstract in older versions).
        return self.invoke(query, **kwargs)

    def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return self.invoke(query, **kwargs)

    async def aget_relevant_documents(
        self, query: str, **kwargs: Any
    ) -> List[Document]:
        return await self.ainvoke(query, **kwargs)


# ==========================================================
# Create Retriever
# ----------------------------------------------------------
# Responsible for retrieving the most relevant chunks from the
# vector database built by the indexing stage.
# ==========================================================
def create_retriever(vector_database):
    """Turn the vector database into a retriever using the configured settings."""
    retriever = vector_database.as_retriever(
        # Search type from the config (e.g. "similarity")
        search_type=CONFIG["search_type"],
        # Search configuration
        search_kwargs={
            # Number of chunks to return (top_k from the config)
            "k": CONFIG["top_k"],
        },
    )
    # Return a wrapper that enforces language isolation on every search.
    return LanguageFilterRetriever(retriever)
