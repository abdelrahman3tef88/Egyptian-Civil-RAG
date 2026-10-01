"""Minimal MLflow model wrapper for the selected RAG configuration.

This module defines the ONLY MLflow model class used by the experiment
phase. It deliberately does NOT package the RAG pipeline: its single job
is to make the selected configuration (chunking + embedding + the fixed
retrieval settings, with the faithfulness score and the source run that
produced it) versioned, registered, traceable and retrievable from the
MLflow Model Registry.

The configuration travels in two ways so that a registered version is
self-describing:

  1. as the model's in-memory configuration (``PythonModel`` instance
     attribute, which MLflow serialises with the model), and
  2. as a ``best_config.json`` artifact attached to the model, which a
     human (or any MLflow client) can read directly.
"""

import json
import os

try:
    import mlflow
    from mlflow.pyfunc import PythonModel
except ImportError as error:  # pragma: no cover - dependency guard.
    raise ImportError(
        "MLflow is required for the model registry layer. "
        "Install the project dependencies (mlflow is declared in "
        "pyproject.toml) before importing registry.config_model."
    ) from error

# Name of the configuration file attached to every registered version.
CONFIG_FILENAME = "best_config.json"

# Keys copied into the model's own metadata for traceability.
TRACEABLE_KEYS = (
    "run_id",
    "source_run_id",
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "search_type",
    "top_k",
    "faithfulness",
    "golden_dataset_version",
)


class ChunkingConfigModel(PythonModel):
    """A registered MLflow model that carries the selected configuration.

    The model is intentionally inert: ``predict`` returns the
    configuration it was registered with, so a consumer (or a reviewer)
    can ask the Registry which chunking/embedding setup is the current
    ``champion`` without re-running any experiment.
    """

    def __init__(self, config=None):
        # The selected configuration as a plain dictionary.
        self.config = dict(config or {})

    # ------------------------------------------------------
    # Load the configuration attached to the logged model
    # ------------------------------------------------------
    def load_context(self, context):
        """Read best_config.json from the model's artifacts when present."""
        # MLflow passes the artifact paths declared at log time.
        config_path = (context.artifacts or {}).get("config")
        if not config_path:
            return

        candidate = os.path.join(config_path, CONFIG_FILENAME)
        if not os.path.exists(candidate):
            return

        with open(candidate, "r", encoding="utf-8") as config_file:
            # The artifact is authoritative once loaded.
            self.config = json.load(config_file)

    # ------------------------------------------------------
    # Expose the configuration
    # ------------------------------------------------------
    def predict(self, context, model_input=None, params=None) -> dict:
        """Return the registered configuration (no RAG work is done here).

        Type hints are supplied so MLflow can infer a signature when the
        model is logged.
        """
        return self.config


    def summary(self):
        """Human-readable one-line summary of the registered setup."""
        return (
            "embedding_model={embedding_model} | chunk_size={chunk_size} | "
            "chunk_overlap={chunk_overlap} | search_type={search_type} | "
            "top_k={top_k} | faithfulness={faithfulness}".format(**{
                key: self.config.get(key) for key in (
                    "embedding_model",
                    "chunk_size",
                    "chunk_overlap",
                    "search_type",
                    "top_k",
                    "faithfulness",
                )
            })
        )


def build_config_model(best_config):
    """Create the model instance for a best_config.json payload."""
    return ChunkingConfigModel(config=best_config)


def config_tags(best_config):
    """Tags that link a registered version back to its source run."""
    tags = {}
    for key in TRACEABLE_KEYS:
        value = best_config.get(key)
        # MLflow tags are strings; skip what the payload does not carry.
        if value is not None:
            tags[key] = str(value)

    # MLflow reserves the "mlflow." tag prefix for itself.
    return {key: value for key, value in tags.items() if not key.startswith("mlflow.")}


# Re-exported so callers do not need to import mlflow directly.
__all__ = [
    "CONFIG_FILENAME",
    "TRACEABLE_KEYS",
    "ChunkingConfigModel",
    "build_config_model",
    "config_tags",
    "mlflow",
]
