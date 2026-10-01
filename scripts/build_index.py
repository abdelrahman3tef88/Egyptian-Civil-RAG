"""build_index.py — articles.json → chunks → embeddings → Chroma.

Small orchestration script: it only CALLS the components in
src/rag_project/indexing/. All parameters and paths come from
configs/config.yaml, so nothing is hard-coded here.

Run with:  uv run python scripts/build_index.py
"""

from rag_project.indexing import chunking, embeddings, vector_store


def main():
    """Load articles.json, chunk, embed, and persist the Chroma collection."""
    # 1) Load articles.json as Documents (path from config).
    documents = chunking.load_articles()

    # 2) Chunking (chunk_size / chunk_overlap from config).
    chunks = chunking.split_documents(documents)

    # 3) Embedding model (model name / device from config).
    embedding_model = embeddings.create_embedding_model()

    # 4) Build the Chroma collection (type / directory / collection
    #    name from config) and make sure it is persisted.
    vector_database = vector_store.create_vector_store(chunks, embedding_model)
    vector_store.save_vector_store(vector_database)

    # Summary of what was written where.
    print("=" * 60)
    print("Vector Database Created Successfully")
    print("=" * 60)
    print(f"Database            : {vector_store.CONFIG['type']}")
    print(f"Persist directory   : {vector_store.PERSIST_DIRECTORY}")
    print(f"Collection          : {vector_store.COLLECTION_NAME}")
    print(f"Total Chunks Stored : {len(chunks)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
