"""Chroma vector store, driven by configs/config.yaml.

The database type, persistence directory, and collection name come
from the central configuration file. The indexing/retrieval behavior
(chunk documents in, similarity search out, metadata preserved) is
the same as before.
"""

from langchain_chroma import Chroma

from rag_project.config import settings

# Configuration section for the vector database (configs/config.yaml).
CONFIG = settings.CONFIG["vector_db"]

# Persisted index directory (configured, relative to the project root).
PERSIST_DIRECTORY = settings.resolve_path(CONFIG["persist_directory"])

# Collection that holds the Civil Code chunks.
COLLECTION_NAME = CONFIG["collection_name"]


# ==========================================================
# Create Chroma Vector Database
# ==========================================================
def create_vector_store(chunks, embedding_model, persist_directory=None):
    """Build the Chroma collection from the article chunks + embeddings."""
    # Tests may point at a temporary directory; production uses config.
    if persist_directory is None:
        persist_directory = PERSIST_DIRECTORY

    vector_database = Chroma.from_documents(
        # List of document chunks (metadata is stored alongside them)
        documents=chunks,
        # Embedding model
        embedding=embedding_model,
        # Collection name from the config
        collection_name=COLLECTION_NAME,
        # Where the collection is persisted
        persist_directory=str(persist_directory),
    )
    return vector_database


# ==========================================================
# Save Vector Database
# ----------------------------------------------------------
# Chroma (>= 1.x) writes documents to its persist_directory as soon
# as they are added, and no longer exposes a persist() call. This
# function is kept so the indexing flow (create -> save) stays
# explicit and testable.
# ==========================================================
def save_vector_store(vector_database):
    """Confirm the collection is persisted (nothing to do for Chroma)."""
    # Returning the store makes the step usable in a single expression.
    return vector_database


# ==========================================================
# Load Vector Database
# ==========================================================
def load_vector_store(embedding_model, persist_directory=None):
    """Open the persisted Chroma collection for retrieval."""
    if persist_directory is None:
        persist_directory = PERSIST_DIRECTORY

    vector_database = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(persist_directory),
    )
    return vector_database


def index_exists(persist_directory=None):
    """True when a persisted Chroma collection is present on disk."""
    if persist_directory is None:
        persist_directory = PERSIST_DIRECTORY
    # Chroma stores its data in a SQLite file inside the directory.
    return (persist_directory / "chroma.sqlite3").exists()
