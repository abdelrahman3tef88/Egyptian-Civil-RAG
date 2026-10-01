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
    """A question is answered with the configured number of chunks."""
    # Load the Chroma collection built from articles.json.
    embedding_model = embeddings.create_embedding_model()
    vector_database = vector_store.load_vector_store(embedding_model)

    # The configured retriever (search type + top_k from config).
    retriever = retriever_module.create_retriever(vector_database)

    # A simple question about the Egyptian Civil Code data.
    query = "ما هي الأهلية؟"
    retrieved_chunks = retriever.invoke(query)

    # Configured top_k: exactly that many chunks are returned.
    assert len(retrieved_chunks) == TOP_K
    # Every chunk carries text and its originating article number.
    for chunk in retrieved_chunks:
        assert chunk.page_content.strip() != ""
        assert "article_number" in chunk.metadata
