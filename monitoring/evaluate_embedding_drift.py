"""Query embedding drift detection for the Egyptian Civil Code RAG.

Goal
----
For each incoming query we compute ONE number: the maximum cosine similarity
between the query embedding and a fixed set of 50 reference embeddings that
were precomputed from the actual civil-law PDF. That number is the initial
*novelty score*:

    max_similarity = max(cosine(query_embedding, reference_embedding_i))

If ``max_similarity`` falls BELOW a configurable threshold, we record a
*potential drift event*.

What drift is NOT
-----------------
* A low similarity is NOT proof the question is invalid or out of domain.
* We do NOT automatically reject low-similarity questions. Processing
  continues unless a separate guardrail/policy blocks the request.
* There is NO universal threshold for BGE-M3. The default here is PROVISIONAL
  and must be calibrated offline with in-domain and out-of-domain examples
  (see ``calibrate_threshold`` and the README).

Baseline
--------
The 50 reference embeddings are computed ONCE when the baseline is prepared
and persisted to ``monitoring/baseline/reference_embeddings.json`` together
with the model identifier and metadata. They are NOT recomputed per request,
and the legal PDF is NOT re-embedded.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from rag_project.config.settings import get_drift_config, resolve_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ==========================================================
# Cosine similarity
# ==========================================================
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def max_reference_similarity(
    query_embedding: List[float], reference_embeddings: List[List[float]]
) -> tuple:
    """Return ``(max_similarity, closest_index)`` against the reference set."""
    best = -1.0
    best_index = -1
    for index, reference in enumerate(reference_embeddings):
        score = cosine_similarity(query_embedding, reference)
        if score > best:
            best = score
            best_index = index
    return best, best_index


# ==========================================================
# Baseline: load precomputed reference embeddings (no recompute per request)
# ==========================================================
def load_baseline(path: Optional[str] = None) -> dict:
    """Load the persisted reference baseline (embeddings + metadata).

    Returns a dict with keys: ``model``, ``reference_count``,
    ``reference_ids``, and ``embeddings``. Raises FileNotFoundError with a
    clear message if the baseline has not been prepared yet.
    """
    config = get_drift_config()
    baseline_path = Path(path) if path else resolve_path(config["baseline_cache_path"])
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Embedding baseline not found at {baseline_path}. "
            f"Run prepare_baseline() first."
        )
    with open(baseline_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_baseline(
    embeddings: List[List[float]],
    reference_ids: List[str],
    model: str,
    path: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Persist reference embeddings + metadata to the baseline artifact."""
    config = get_drift_config()
    baseline_path = Path(path) if path else resolve_path(config["baseline_cache_path"])
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "reference_count": len(reference_ids),
        "reference_ids": reference_ids,
        "created_at": _utc_now(),
        "metadata": extra_metadata or {},
        "embeddings": embeddings,
    }
    with open(baseline_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return baseline_path


def prepare_baseline(reference_questions_path: Optional[str] = None) -> dict:
    """Compute and persist the 50 reference embeddings from the reference set.

    This is an OFFLINE / setup step. It loads BGE-M3 once, embeds the 50
    reference questions, and writes the baseline artifact. It is NOT called
    on the request path.
    """
    # Imported lazily so importing this module does not load BGE-M3.
    from rag_project.indexing.embeddings import create_embedding_model

    config = get_drift_config()
    questions_path = (
        Path(reference_questions_path)
        if reference_questions_path
        else resolve_path(config["reference_questions_path"])
    )
    with open(questions_path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    items = dataset["questions"] if isinstance(dataset, dict) else dataset
    reference_ids = [item["id"] for item in items]
    texts = [item["question"] for item in items]

    embedding_model = create_embedding_model()
    embeddings = embedding_model.embed_documents(texts)

    save_baseline(
        embeddings=[list(map(float, vector)) for vector in embeddings],
        reference_ids=reference_ids,
        model=config["model"],
        extra_metadata={"source": str(questions_path)},
    )
    return {
        "model": config["model"],
        "reference_count": len(reference_ids),
        "baseline_path": str(resolve_path(config["baseline_cache_path"])),
    }


# ==========================================================
# Per-request evaluation
# ==========================================================
def evaluate_query(
    query_embedding: List[float],
    baseline: Optional[dict] = None,
    threshold: Optional[float] = None,
) -> dict:
    """Score a query embedding against the baseline reference set.

    Returns a privacy-safe report dict (no raw embedding, no raw question):
    ``max_similarity``, ``threshold``, ``drift`` (bool), ``closest_reference_id``,
    ``model``, and a ``timestamp``.
    """
    baseline = baseline if baseline is not None else load_baseline()
    threshold = threshold if threshold is not None else get_drift_config()["threshold"]

    embeddings = baseline["embeddings"]
    reference_ids = baseline.get("reference_ids", [])
    max_similarity, closest_index = max_reference_similarity(
        query_embedding, embeddings
    )

    closest_id = (
        reference_ids[closest_index]
        if 0 <= closest_index < len(reference_ids)
        else None
    )
    return {
        "max_similarity": max_similarity,
        "threshold": threshold,
        "drift": max_similarity < threshold,
        "closest_reference_id": closest_id,
        "model": baseline.get("model"),
        "timestamp": _utc_now(),
    }


def append_drift_report(report: dict, path: Optional[str] = None) -> Path:
    """Append one drift evaluation to the JSONL drift report (privacy-safe).

    The caller must pass a report whose fields contain NO raw PII: only the
    correlation id, a privacy-safe question representation, and the numbers
    produced by ``evaluate_query``.
    """
    reports_dir = Path(path) if path else resolve_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "embedding_drift_report.jsonl"
    with open(report_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    return report_path


# ==========================================================
# Offline threshold calibration (does NOT pick a universal value)
# ==========================================================
def calibrate_threshold(
    in_domain_scores: List[float], out_domain_scores: List[float]
) -> dict:
    """Suggest a threshold from representative in/out-domain similarity lists.

    This is a heuristic ONLY. It returns the midpoint between the mean
    in-domain and mean out-of-domain max-similarity, plus the raw means so a
    human can judge. It does NOT claim the value is correct for BGE-M3: the
    selected threshold and its rationale must be recorded in config/docs
    before use.
    """
    if not in_domain_scores or not out_domain_scores:
        raise ValueError("both in_domain_scores and out_domain_scores are required")
    in_mean = sum(in_domain_scores) / len(in_domain_scores)
    out_mean = sum(out_domain_scores) / len(out_domain_scores)
    suggested = (in_mean + out_mean) / 2.0
    return {
        "suggested_threshold": suggested,
        "in_domain_mean": in_mean,
        "out_domain_mean": out_mean,
        "note": "PROVISIONAL until calibrated with real in/out-domain examples.",
    }
