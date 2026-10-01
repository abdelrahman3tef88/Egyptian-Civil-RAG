"""Retriever, driven by configs/config.yaml.

The search type and top-k value come from the central configuration
file; the retrieval behavior itself (similarity search over the
vector store) is unchanged.
"""

from rag_project.config import settings

# Configuration section for retrieval (configs/config.yaml).
CONFIG = settings.CONFIG["retrieval"]


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
            "k": CONFIG["top_k"]
        },
    )
    return retriever
