"""reindex.py — rebuild the vector index when indexing configuration changes.

Intended responsibility: rebuild the vector store from scratch whenever
indexing configuration (chunking, embedding model, ...) changes, ensuring
artifacts/vector_store/ stays consistent with the current settings.

Workflow intentionally not implemented yet (Day 1 structure only).
"""
