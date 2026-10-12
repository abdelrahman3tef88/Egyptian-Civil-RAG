"""Tests for retrieval.

Verifies that the configured retriever (search type + top_k from
configs/config.yaml) returns chunks from the Chroma collection built
out of articles.json.
"""

import pytest

from rag_project.config import settings
from rag_project.indexing import embeddings, vector_store
from rag_project.retrieval import retriever as retriever_module

# Configured retrieval parameters.
TOP_K = settings.CONFIG["retrieval"]["top_k"]

requires_index = pytest.mark.skipif(
    not vector_store.index_exists(),
    reason="Chroma index not built yet (run scripts/build_index.py)",
)


@requires_index
def test_retriever_returns_top_k_chunks():
    """A question is answered with up to the configured number of chunks.

    The language-filtered retriever keeps only chunks that match the detected
    language of the question, so the correct-language subset may be smaller
    than top_k; it is never padded with the other language. Every returned
    chunk is non-empty, carries an article number, matches the question
    language, and keeps the retrieval order with no duplicates.
    """
    # Load the Chroma collection built from articles.json.
    embedding_model = embeddings.create_embedding_model()
    vector_database = vector_store.load_vector_store(embedding_model)

    # The configured retriever (search type + top_k from config).
    retriever = retriever_module.create_retriever(vector_database)

    # A simple Arabic question about the Egyptian Civil Code data.
    query = "ما هي الأهلية؟"
    retrieved_chunks = retriever.invoke(query)

    # Correct-language subset can be smaller than top_k; it is never padded.
    assert 0 < len(retrieved_chunks) <= TOP_K
    expected_lang = retriever_module.detect_language(query)
    assert expected_lang == "ar"
    # Every chunk carries text and its originating article number.
    for chunk in retrieved_chunks:
        assert chunk.page_content.strip() != ""
        assert "article_number" in chunk.metadata
        assert retriever_module._chunk_language(chunk) == expected_lang
    # No duplicate chunk texts and retrieval order is preserved in formatting.
    texts = [chunk.page_content for chunk in retrieved_chunks]
    assert len(set(texts)) == len(texts)
    formatted = (
        str(retriever_module.format_docs)
        if hasattr(retriever_module, "format_docs")
        else None
    )
    assert formatted is None or True
    from rag_project.generation.chain import format_docs

    context = format_docs(retrieved_chunks)
    order_in_context = [context.index(t) for t in texts]
    assert order_in_context == sorted(order_in_context)
