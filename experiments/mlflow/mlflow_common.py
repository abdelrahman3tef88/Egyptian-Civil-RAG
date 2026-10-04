"""Shared helpers for the MLflow workflow scripts.

Kept dependency-free (no MLflow, no RAG imports) so that every script in
experiments/mlflow/ can use it without pulling in the heavy pipeline.

Also performs the src/ bootstrap: the project's package is importable
when it is installed, and this makes it importable when it is not (the
same thing pytest does with `pythonpath = ["src"]`).
"""

import json
import re
import sys
from pathlib import Path

# ----------------------------------------------------------
# Project locations
# ----------------------------------------------------------
# PROJECT_ROOT = <repo>/ (this file lives in <repo>/experiments/mlflow/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "experiments" / "mlflow" / "experiment_config.yaml"

# Local tracking store used when neither experiment_config.yaml nor the
# MLFLOW_TRACKING_URI environment variable names one.
#
# Why a SQLite store instead of the classic ./mlruns filesystem store:
# MLflow 3.x put the filesystem tracking backend into maintenance mode
# and raises on it unless MLFLOW_ALLOW_FILE_STORE=true is set (verified
# with mlflow 3.16.1). SQLite is the supported local backend, needs no
# server, and is the same store the documented MLflow server serves.
DEFAULT_LOCAL_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


def load_experiment_config(config_path=None):
    """Read experiments/mlflow/experiment_config.yaml."""
    import yaml

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise SystemExit(f"Experiment config not found: {path}")
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def slugify(value):
    """Turn a model name / label into a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def write_json(path, payload):
    """Write JSON with UTF-8 (Arabic) preserved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


def read_json(path):
    """Read a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def configure_stdout():
    """Keep Arabic readable when the scripts print to a console."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def ensure_experiment(mlflow, name, artifact_location=None):
    """Create the MLflow experiment on first use, then select it."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    existing = client.get_experiment_by_name(name)
    if existing is None and artifact_location:
        client.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)


def configure_tracking(mlflow, config):
    """Point MLflow at the experiment's tracking store.

    Resolution order:

    1. ``experiment.tracking_uri`` in experiment_config.yaml (for example
       ``http://localhost:5000`` when an MLflow server is running),
    2. the ``MLFLOW_TRACKING_URI`` environment variable - MLflow's own
       resolution is left untouched in that case,
    3. the local SQLite store (see DEFAULT_LOCAL_TRACKING_URI), so the
       scripts also work with no server at all.
    """
    import os

    tracking_uri = config["experiment"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(DEFAULT_LOCAL_TRACKING_URI)
    return mlflow.get_tracking_uri()
