"""Embedding model, driven by configs/config.yaml.

The model name, device policy, and normalization flag all come from
the central configuration file.
"""

import torch

from langchain_huggingface import HuggingFaceEmbeddings

from rag_project.config import settings

# Configuration section for embeddings (configs/config.yaml).
CONFIG = settings.CONFIG["embedding"]


def _resolve_device(value):
    """Turn the configured device value into a concrete torch device name."""
    # "auto" keeps the code portable: GPU when present, CPU otherwise.
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # Any explicit value from the config is used as-is.
    return value


# ==========================================================
# Load Embedding Model (config-driven)
# ==========================================================
def create_embedding_model():
    """Create the configured HuggingFace embedding model."""
    embedding_model = HuggingFaceEmbeddings(
        # Name of the embedding model from Hugging Face
        model_name=CONFIG["embedding_model"],
        # Model configuration (device resolved from the config policy)
        model_kwargs={
            "device": _resolve_device(CONFIG.get("device", "auto"))
        },
        # Encoding configuration
        encode_kwargs={
            "normalize_embeddings": CONFIG.get("normalize_embeddings", True)
        },
    )
    return embedding_model
