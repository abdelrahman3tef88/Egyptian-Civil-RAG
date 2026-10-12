"""
Configuration Management for the Egyptian Civil Code RAG project.

Every stage (indexing, retrieval, generation, API) loads its values from
this file instead of hard-coding hyperparameters or paths.
"""

from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load secrets (API keys, ...) from the project .env file into the
# environment once, at the configuration layer, so every module
# (scripts, tests, API) sees the same environment variables.
load_dotenv()

# Project root, derived from this file's location.
# src/rag_project/config/settings.py -> parents[3] = the repository root.
# This is deliberately NOT Path.cwd(): scripts and the MLflow experiment
# runner are executed from several directories (the repo root,
# experiments/mlflow, ...), and a cwd-based root makes importing this
# module fail with "Configuration file not found" whenever the working
# directory is not the project root. It is also correct inside Docker,
# where the source is copied to /app/src, giving /app as the root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Central configuration file.
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=None):
    """Read the YAML configuration file and return it as a dictionary."""
    config_path = Path(path) if path is not None else CONFIG_PATH

    # Fail clearly when the configuration file is missing.
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # UTF-8 keeps the file readable with any language in it.
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


# Loaded once at import; every module reads its section from here.
CONFIG = load_config()


# config.yaml -> load_config() -> CONFIG -> indexing / retrieval / generation / API


def resolve_path(relative_path):
    """Turn a project-relative config path into an absolute path."""
    path = Path(relative_path)
    # Absolute paths are used as they are.
    if path.is_absolute():
        return path
    # Relative paths are anchored at the project root.
    return PROJECT_ROOT / path


# ----------------------------------------------------------
# Safe nested config access helpers for the new sections
# ----------------------------------------------------------
def get_config(section, key, default=None):
    """Read a nested config value without raising KeyError."""
    return CONFIG.get(section, {}).get(key, default)


def get_vllm_config():
    """Return the configured vLLM values (URL, model, auth)."""
    vllm = CONFIG.get("vllm") or {}
    return {
        "base_url": vllm.get("base_url", ""),
        "model": vllm.get("model", ""),
        "api_key": vllm.get("api_key", ""),
        "timeout_seconds": vllm.get("timeout_seconds", 300),
        "max_tokens": vllm.get("max_tokens", 4096),
    }


def get_evaluation_config():
    """Return the configured evaluation values."""
    evaluation = CONFIG.get("evaluation") or {}
    return {
        "backend": evaluation.get("backend", "ragas"),
        "enabled": evaluation.get("enabled", True),
        "gemini_evaluator_model": evaluation.get("gemini_evaluator_model", ""),
    }


def get_drift_config():
    """Return the configured drift values."""
    drift = CONFIG.get("drift") or {}
    return {
        "threshold": drift.get("threshold", 0.45),
        "reference_questions_path": drift.get(
            "reference_questions_path", "data/reference_questions.json"
        ),
        "baseline_cache_path": drift.get(
            "baseline_cache_path", "monitoring/baseline/reference_embeddings.json"
        ),
        "model": drift.get("model", "BAAI/bge-m3"),
        "device": drift.get("device", "auto"),
        "reference_count": drift.get("reference_count", 50),
    }


def get_langfuse_config():
    """Return the configured Langfuse values."""
    langfuse = CONFIG.get("langfuse") or {}
    return {
        "host": langfuse.get("host", "https://cloud.langfuse.com"),
        "public_key": langfuse.get("public_key", ""),
        "secret_key": langfuse.get("secret_key", ""),
        "release": langfuse.get("release", ""),
        "sample_rate": langfuse.get("sample_rate", 1.0),
    }
