"""Tests for the indexing pipeline.

Verifies, with the real data/processed/articles.json and the values
from configs/config.yaml, that indexing works: loading -> chunking
-> embeddings -> Chroma creation/persistence.
"""

import pytest

from rag_project.config import settings
from rag_project.indexing import chunking, embeddings, vector_store

# Paths and parameters come from the central config (not hard-coded).
ARTICLES_JSON_PATH = settings.resolve_path(settings.CONFIG["data"]["articles_json"])
CHUNK_SIZE = settings.CONFIG["chunking"]["chunk_size"]

requires_articles = pytest.mark.skipif(
    not ARTICLES_JSON_PATH.exists(),
    reason="data/processed/articles.json not built yet (run extract_data.py)",
)


@requires_articles
def test_load_articles_from_json():
    """articles.json records become LangChain Documents with metadata."""
    documents = chunking.load_articles()
    # One document per article record.
    assert len(documents) > 0
    # article_number is preserved in metadata (first record = article 1).
    assert documents[0].metadata["article_number"] == 1
    # The article text is the chunk source.
    assert documents[0].page_content.strip() != ""


@requires_articles
def test_chunking_works_on_articles_json():
    """The configured splitter chunks the articles and keeps metadata."""
    documents = chunking.load_articles()
    chunks = chunking.split_documents(documents)
    # Chunks were produced from the real dataset.
    assert len(chunks) > 0
    # Configured chunk_size is respected.
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)
    # The splitter carries the article metadata into the chunks.
    assert "article_number" in chunks[0].metadata


@requires_articles
def test_embedding_and_vector_store_can_be_created(tmp_path):
    """Embeddings + Chroma creation/persistence work on real chunks."""
    # Use a small slice of the real dataset to keep the test fast.
    documents = chunking.load_articles()
    chunks = chunking.split_documents(documents[:5])
    assert len(chunks) > 0

    # The configured embedding model.
    embedding_model = embeddings.create_embedding_model()

    # Chroma creation into a temporary directory (config drives production).
    persist_directory = tmp_path / "chroma"
    vector_database = vector_store.create_vector_store(
        chunks, embedding_model, persist_directory=persist_directory
    )
    assert vector_database is not None
    vector_store.save_vector_store(vector_database)

    # A similarity query proves the stored chunks are searchable.
    results = vector_database.similarity_search("مادة", k=2)
    assert len(results) == 2
    assert "article_number" in results[0].metadata

    # Reopening the persisted collection works (the retriever needs this).
    reloaded = vector_store.load_vector_store(
        embedding_model, persist_directory=persist_directory
    )
    assert reloaded is not None
