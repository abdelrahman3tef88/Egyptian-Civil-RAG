"""evaluation/run_ragas.py - final RAGAS evaluation with MLflow tracking.

What this script does, end to end:

  1. Load the 50 final questions from data/final_questions/final_ragas_questions.json.
  2. For each question, run the EXISTING RAG pipeline exactly as the app does
     (vector store -> retriever -> prompt -> Gemini -> answer), passing ONLY
     the question text. The ground-truth answer is NEVER sent to generation.
  3. Collect the generated answer and the retrieved contexts.
  4. Keep the reference/ground-truth answer only for the RAGAS stage.
  5. Run the four RAGAS metrics (Context Precision, Context Recall,
     Faithfulness, Answer Relevancy) per question with the Gemini judge.
  6. Aggregate each metric across all questions (mean of per-question scores).
  7. Open ONE MLflow run, log fixed parameters + the four metric values,
     and store the per-question results as an artifact.
  8. If overall Faithfulness < 0.80, raise a clearly visible ALERT (terminal
     + an MLflow tag), implemented independently of CI/CD and Langfuse.

Run:  uv run python evaluation/run_ragas.py
Then view results in the normal MLflow UI (see printed tracking URI).
"""

import asyncio
import json
import sys
from pathlib import Path

# Project layout: this file = evaluation/run_ragas.py -> parents[1] = root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make the project package importable (same as tests' pythonpath=["src"]).
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Reuse the existing MLflow helper (local sqlite tracking store + helpers).
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "mlflow"))
# This script's own directory, so the sibling evaluation/ragas_metrics.py
# can be imported as a plain top-level module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlflow_common import (  # noqa: E402
    configure_stdout,
    configure_tracking,
    ensure_experiment,
    write_json,
)

import ragas_metrics  # noqa: E402
from rag_project.generation import llm as llm_module  # noqa: E402
from rag_project.generation.chain import format_docs  # noqa: E402
from rag_project.generation.prompts import prompt as rag_prompt  # noqa: E402
from rag_project.indexing import embeddings as embeddings_module, vector_store  # noqa: E402
from rag_project.retrieval import retriever as retriever_module  # noqa: E402

# ---- Evaluation settings ---------------------------------------------------
QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "final_questions" / "final_ragas_questions.json"
)
EXPECTED_QUESTION_COUNT = 50
# One MLflow experiment for the whole final evaluation (not per metric).
EXPERIMENT_NAME = "egyptian-civil-law-final-ragas"
# Faithfulness alert threshold (terminal + MLflow tag).
FAITHFULNESS_ALERT_THRESHOLD = 0.80
# Generated answers/contexts cached on disk right AFTER the generation stage
# and BEFORE RAGAS scoring, so they need not be regenerated on a re-run.
# The legacy final_ragas cache is intentionally untouched; v2 uses an isolated path.
LEGACY_GENERATED_RECORDS_PATH = (
    PROJECT_ROOT / "reports" / "mlflow" / "final_ragas" / "generated_records.json"
)
GENERATED_RECORDS_PATH = (
    PROJECT_ROOT / "reports" / "mlflow" / "final_ragas_v2" / "generated_records.json"
)
ARTICLES_PATH = PROJECT_ROOT / "data" / "processed" / "articles.json"
CACHE_SCHEMA_VERSION = "final-ragas-v2.3"


def load_questions():
    """Load and validate the 50-question final dataset."""
    if not QUESTIONS_PATH.exists():
        raise SystemExit(f"Final questions file not found: {QUESTIONS_PATH}")
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = data["questions"]
    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_QUESTION_COUNT} questions, found {len(questions)}."
        )
    return data, questions


def _retrieval_fingerprint_inputs():
    """Return the retrieval settings covered by the v2 cache fingerprint."""
    from rag_project.config import settings as config_settings

    retrieval = config_settings.CONFIG.get("retrieval", {})
    embedding = config_settings.CONFIG.get("embedding", {})
    chunking = config_settings.CONFIG.get("chunking", {})
    return {
        "search_type": retrieval.get("search_type"),
        "top_k": retrieval.get("top_k"),
        "embedding_model": embedding.get("embedding_model"),
        "chunk_size": chunking.get("chunk_size"),
        "chunk_overlap": chunking.get("chunk_overlap"),
        # Code-behavior flags: any change here must invalidate old caches.
        # v2.2: strict question-language filtering on every retrieval, and
        # single-retrieval generation contexts shared with scoring/reporting.
        "language_filter": "strict_question_language",
        "generation_context": "shared_single_retrieval",
    }


def compute_cache_fingerprint(questions):
    """Fingerprint the dataset identity plus retrieval configuration.

    No model calls are made; this hashes only local dataset/config values.
    """
    import hashlib

    identity = [
        {
            "id": item.get("id"),
            "question": item.get("question"),
            "ground_truth": item.get("ground_truth"),
            "relevant_articles": [str(a) for a in item.get("relevant_articles", [])],
        }
        for item in questions
    ]
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "question_count": len(identity),
        "questions": identity,
        "retrieval": _retrieval_fingerprint_inputs(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest


def load_article_numbers():
    """Return verified article numbers from articles.json, or an empty set."""
    try:
        data = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    verified = set()
    for article in data.get("articles", []):
        number = article.get("article_number")
        if number is None:
            continue
        verified.add(str(number))
    return verified


def verified_ground_truth_article_numbers(relevant_articles, verified_articles):
    """Keep only verifiable ground-truth article numbers; never invent any."""
    numbers = []
    missing = []
    for raw in relevant_articles or []:
        value = str(raw).strip()
        if value and value in verified_articles:
            numbers.append(value)
        else:
            missing.append(value or None)
    return numbers, missing


def build_retrieved_chunks(documents, verified_articles):
    """Capture retrieved-chunk provenance from actual document metadata only."""
    chunks = []
    missing = []
    for rank, document in enumerate(documents):
        metadata = dict(getattr(document, "metadata", None) or {})
        raw_number = metadata.get("article_number")
        article_number = None
        if raw_number is not None and str(raw_number).strip() != "":
            candidate = str(raw_number).strip()
            if candidate in verified_articles:
                article_number = candidate
            else:
                missing.append({"rank": rank, "unverifiable_article_number": candidate})
        else:
            missing.append({"rank": rank, "unverifiable_article_number": None})
        chunks.append(
            {
                "rank": rank,
                "article_number": article_number,
                "chunk_id": metadata.get("chunk_id"),
                "language": metadata.get("language"),
                "metadata": metadata,
                "text": document.page_content,
            }
        )
    return chunks, missing


def build_rag_chain():
    """Build the existing RAG pipeline from the persisted index (not rebuilt).

    Mirrors run_ci_test.py: load the vector store, create the retriever and
    the Gemini generation model, and compose the chain. Only the question
    ever reaches the chain.
    """

    generation_llm = llm_module.create_llm()
    embedding_model = embeddings_module.create_embedding_model()
    store = vector_store.load_vector_store(embedding_model)
    retriever = retriever_module.create_retriever(store)
    return retriever, rag_prompt, generation_llm


def label_evaluation_context(text, metadata):
    """Return the RAGAS evaluation context for one chunk.

    Prefixes the ORIGINAL text with the article number taken strictly from
    the chunk's own ``metadata``:

        Arabic chunk  -> "المادة 51:\\n<text>"
        English chunk -> "Article 216:\\n<text>"
        no article number -> "<text>" unchanged (never invent a number)

    ``text`` itself is never modified; metadata is only read.
    """
    meta = metadata or {}
    number = meta.get("article_number")
    if number is None or str(number).strip() == "":
        return text
    language = meta.get("language")
    if language not in ("ar", "en"):
        # Metadata has no usable language: decide the label LANGUAGE from the
        # text script (the number itself still comes from metadata).
        import re as _re

        language = "ar" if _re.search(r"[\u0600-\u06FF]", text) else "en"
    label = f"المادة {number}" if language == "ar" else f"Article {number}"
    return f"{label}:\n{text}"


def build_evaluation_contexts(documents):
    """Build RAGAS evaluation contexts with article labels from METADATA only.

    Each element preserves the chunk's original text byte-for-byte and is
    prefixed with a label derived strictly from the chunk's own metadata
    (see :func:`label_evaluation_context`).

    WHY: ``ContextRecall`` / ``ContextPrecision`` / ``Faithfulness`` judge the
    reference and statements (which name the article) against the raw chunk
    text, but ``article_number`` lives only in Chroma metadata - so a correct
    retrieval scored 0 for "the context does not mention Article 216".
    The generation context (``format_docs``) already carries these labels;
    this brings the EVALUATION contexts to parity without touching retrieval,
    chunk texts, metadata, ground truth, or the generation prompt.
    """
    return [
        label_evaluation_context(doc.page_content, getattr(doc, "metadata", None) or {})
        for doc in documents
    ]


def run_pipeline(retriever, prompt, generation_llm, questions, verified_articles=None):
    """Run every question through the RAG pipeline and collect per-question data.

    Returns a list of dicts with: id, difficulty, language, question, answer,
    contexts, retrieved_contexts, retrieved_chunks, ground_truth_article_numbers,
    ground_truth, and a RAGAS SingleTurnSample. The ground_truth is
    carried along ONLY for the later RAGAS stage - it never reaches generation.
    The scored contexts are exactly the retrieved documents used for generation,
    in retrieval order; no second retrieval pass is performed.
    """
    from ragas.dataset_schema import SingleTurnSample  # ragas sample container
    from rag_project.generation.chain import (
        create_rag_chain_with_context,
    )  # no-retrieval answer step

    checked_articles = (
        verified_articles if verified_articles is not None else load_article_numbers()
    )
    records = []
    for index, item in enumerate(questions, start=1):
        question = item["question"]

        # Single retrieval, performed once with the language-filtered retriever.
        # The returned documents are the single source of truth for all three
        # downstream stages: generation, RAGAS scoring, and reporting.
        documents = retriever.invoke(question)
        contexts = format_docs(documents)
        # Evaluation contexts only: same documents, same order, original text
        # unchanged - just prefixed with the article number from metadata so
        # the RAGAS judges can match reference/statement labels.
        retrieved_contexts = build_evaluation_contexts(documents)
        retrieved_chunks, _missing_chunks = build_retrieved_chunks(
            documents, checked_articles
        )
        ground_truth_numbers, _missing_ground_truth = (
            verified_ground_truth_article_numbers(
                item.get("relevant_articles", []), checked_articles
            )
        )

        # Generate the answer against the SAME pre-fetched context that was
        # scored above, via a retrieval-free final step (no second retrieval).
        # Only the question + this context reach generation; neither ground
        # truth nor metadata is sent to the LLM.
        answer_chain = create_rag_chain_with_context(contexts, prompt, generation_llm)
        answer = answer_chain({"question": question})

        print(f"  [{index:02d}/{len(questions)}] {item['id']} done", flush=True)

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            # RAGAS 0.4.3 expects list[str]; contexts are the labeled
            # evaluation contexts (article number from metadata + original text).
            retrieved_contexts=retrieved_contexts,
            reference=item["ground_truth"],  # used only by RAGAS metrics
        )
        records.append(
            {
                "id": item["id"],
                "difficulty": item["difficulty"],
                "language": item["language"],
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "retrieved_contexts": retrieved_contexts,
                "retrieved_chunks": retrieved_chunks,
                "ground_truth_article_numbers": ground_truth_numbers,
                "ground_truth": item["ground_truth"],
                "sample": sample,
            }
        )
    return records


def save_generated_records(records, path, questions=None, fingerprint=None):
    """Persist the generated answers + contexts to disk BEFORE RAGAS scoring.

    Caches the expensive generation step so a re-run can skip regeneration.
    Stores only the JSON-serializable fields (the RAGAS sample object is not
    serializable and is rebuilt from these fields on load). The v2 payload is
    a dict carrying the cache fingerprint plus one record per question.
    """
    if fingerprint is None:
        if questions is None:
            questions = [
                {
                    "id": r["id"],
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "relevant_articles": r.get("ground_truth_article_numbers", []),
                }
                for r in records
            ]
        fingerprint = compute_cache_fingerprint(questions)
    if questions is None:
        questions = [
            {
                "id": r["id"],
                "question": r["question"],
                "ground_truth": r["ground_truth"],
            }
            for r in records
        ]
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "question_count": len(records),
        "records": [
            {
                "id": r["id"],
                "difficulty": r["difficulty"],
                "language": r["language"],
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                # Persist the list[str] RAGAS actually scores against, so the
                # sample can be rebuilt on cache-load exactly as freshly built.
                "retrieved_contexts": list(r["sample"].retrieved_contexts),
                "retrieved_chunks": r.get("retrieved_chunks", []),
                "ground_truth_article_numbers": r.get(
                    "ground_truth_article_numbers", []
                ),
                "ground_truth": r["ground_truth"],
            }
            for r in records
        ],
    }
    write_json(path, payload)
    print(f"Saved {len(payload['records'])} generated records -> {path}")
    return path


def load_generated_records(path, questions=None, fingerprint=None):
    """Load cached generated records so RAG generation can be skipped.

    Returns a list of record dicts (each with a rebuilt RAGAS SingleTurnSample)
    ONLY if the v2 cache file exists, carries the expected fingerprint, holds
    exactly EXPECTED_QUESTION_COUNT valid records, and matches the current
    question IDs/questions/ground truths. Otherwise returns None so the caller
    regenerates fresh. The legacy final_ragas cache is never consulted here.
    """
    from ragas.dataset_schema import SingleTurnSample  # ragas sample container

    # No cache file -> caller must generate.
    if not path.exists():
        return None
    if questions is None or fingerprint is None:
        # Without the current dataset identity there is nothing safe to validate.
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    # v2 payload shape only; legacy list-shaped caches are always rejected.
    if not isinstance(data, dict):
        return None
    if data.get("schema") != CACHE_SCHEMA_VERSION:
        return None
    if data.get("fingerprint") != fingerprint:
        return None
    cached = data.get("records")
    # Wrong count or shape -> do not use this cache.
    if not isinstance(cached, list) or len(cached) != EXPECTED_QUESTION_COUNT:
        return None
    if len(cached) != len(questions):
        return None

    records = []
    for cached_record, current in zip(cached, questions):
        rc = cached_record.get("retrieved_contexts")
        chunks = cached_record.get("retrieved_chunks")
        # Require the fields RAGAS scoring depends on, with correct types.
        if cached_record.get("id") != current.get("id"):
            return None
        if cached_record.get("question") != current.get("question"):
            return None
        if cached_record.get("ground_truth") != current.get("ground_truth"):
            return None
        if not isinstance(cached_record.get("answer"), str) or not isinstance(
            cached_record.get("question"), str
        ):
            return None
        if not (isinstance(rc, list) and all(isinstance(x, str) for x in rc)):
            return None
        if not isinstance(chunks, list):
            return None
        for rank, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                return None
            if chunk.get("rank") != rank:
                return None
            if not isinstance(chunk.get("text"), str):
                return None
            if rank >= len(rc):
                return None
            # Contexts are article-labeled (schema v2.3); accept exactly the
            # form derived from this chunk's own metadata/text. Nothing else.
            if rc[rank] != label_evaluation_context(
                chunk["text"], chunk.get("metadata") or {}
            ):
                return None
            if not isinstance(chunk.get("metadata"), dict):
                return None
        ground_truth_numbers = cached_record.get("ground_truth_article_numbers", [])
        if not isinstance(ground_truth_numbers, list) or not all(
            isinstance(x, str) for x in ground_truth_numbers
        ):
            return None
        sample = SingleTurnSample(
            user_input=cached_record["question"],
            response=cached_record["answer"],
            retrieved_contexts=rc,
            reference=cached_record["ground_truth"],  # used only by RAGAS metrics
        )
        records.append(
            {
                "id": cached_record.get("id"),
                "difficulty": cached_record.get("difficulty"),
                "language": cached_record.get("language"),
                "question": cached_record["question"],
                "answer": cached_record["answer"],
                "contexts": cached_record.get("contexts"),
                "retrieved_contexts": rc,
                "retrieved_chunks": chunks,
                "ground_truth_article_numbers": ground_truth_numbers,
                "ground_truth": cached_record.get("ground_truth"),
                "sample": sample,
            }
        )
    return records


def score_metric(metric, records):
    """Score one RAGAS metric across all records and return per-question scores.

    RAGAS computes each value through its Gemini judge; no custom logic here.
    """
    scores = []
    for record in records:
        value = asyncio.run(metric.single_turn_ascore(record["sample"]))
        scores.append(value)
    return scores


def aggregate(scores):
    """Mean of the per-question scores (RAGAS-style aggregate)."""
    usable = [s for s in scores if s is not None]
    return sum(usable) / len(usable) if usable else 0.0


def emit_faithfulness_alert(overall_faithfulness):
    """Print a clearly visible ALERT when Faithfulness drops below threshold."""
    if overall_faithfulness >= FAITHFULNESS_ALERT_THRESHOLD:
        print(
            f"Faithfulness {overall_faithfulness:.4f} >= {FAITHFULNESS_ALERT_THRESHOLD} - OK"
        )
        return "OK"

    # Independent of CI/CD and Langfuse: a plain, loud terminal + MLflow tag.
    banner = "=" * 78
    print("\n" + banner)
    print("  ***************************** ALERT *****************************")
    print("  Faithfulness < 0.80")
    print(
        f"  Overall Faithfulness = {overall_faithfulness:.4f} "
        f"(threshold {FAITHFULNESS_ALERT_THRESHOLD})"
    )
    print("  The RAG answers are not sufficiently grounded in the retrieved context.")
    print("  *****************************************************************")
    print(banner + "\n")
    return "ALERT"


def main():
    """Load data, run the pipeline + RAGAS metrics, log everything to MLflow."""
    configure_stdout()

    # Reuse the existing Gemini judge + embedding model for the RAGAS metrics.
    judge_llm = llm_module.create_judge_llm()
    embedding_model = embeddings_module.create_embedding_model()
    metrics = ragas_metrics.build_metrics(judge_llm, embedding_model)

    dataset, questions = load_questions()
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")
    fingerprint = compute_cache_fingerprint(questions)

    # Reuse a valid v2 cached generation if present; otherwise generate + cache.
    # The legacy final_ragas cache is never read here.
    records = load_generated_records(GENERATED_RECORDS_PATH, questions, fingerprint)
    if records is not None:
        print(
            f"Loaded {len(records)} cached generated records -> {GENERATED_RECORDS_PATH}\n"
            "Skipping RAG generation; scoring directly."
        )
    else:
        # Build retrieval + prompt + LLM once; per-question generation reuses
        # each question's pre-fetched documents (single retrieval per question).
        retriever, prompt, generation_llm = build_rag_chain()
        print("Running the RAG pipeline over all questions...\n")
        records = run_pipeline(retriever, prompt, generation_llm, questions)

        # Generate 50 -> Save records -> Score: cache the generated answers and
        # their contexts to disk immediately after generation, before RAGAS runs.
        save_generated_records(records, GENERATED_RECORDS_PATH, questions, fingerprint)

    # Run the four RAGAS metrics over the collected answers/contexts.
    print("\nScoring with RAGAS (Gemini judge)...")
    per_metric_scores = {
        name: score_metric(metric, records) for name, metric in metrics.items()
    }
    overall = {name: aggregate(scores) for name, scores in per_metric_scores.items()}

    print("\n" + "=" * 78)
    for name, value in overall.items():
        print(f"  {name:<20}: {value:.4f}")
    print("=" * 78)

    # ---- MLflow: one run, fixed params + four metrics + artifact -----------
    import mlflow

    # Use a dedicated SQLite tracking store for the final RAGAS evaluation,
    # completely separate from the nine chunking/embedding experiments' mlflow.db.
    RAGAS_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow_ragas.db').as_posix()}"
    config = {"experiment": {"tracking_uri": RAGAS_TRACKING_URI}}
    configure_tracking(mlflow, config)
    ensure_experiment(mlflow, EXPERIMENT_NAME)

    # Detailed per-question results saved as an MLflow artifact.
    detail_path = (
        PROJECT_ROOT
        / "reports"
        / "mlflow"
        / "final_ragas_v2"
        / "final_ragas_results.json"
    )
    per_question = []
    for i, record in enumerate(records):
        per_question.append(
            {
                "id": record["id"],
                "difficulty": record["difficulty"],
                "language": record["language"],
                "question": record["question"],
                "answer": record["answer"],
                "ground_truth": record["ground_truth"],
                "scores": {name: per_metric_scores[name][i] for name in metrics},
            }
        )
    write_json(detail_path, {"overall": overall, "per_question": per_question})

    alert_status = emit_faithfulness_alert(overall["faithfulness"])

    with mlflow.start_run(run_name="final-ragas"):
        # Fixed parameters (dataset + judge model + retrieval settings).
        mlflow.log_params(
            {
                "dataset": "data/final_questions/final_ragas_questions.json",
                "dataset_version": dataset.get("version", "final-ragas-v1"),
                "question_count": len(questions),
                "judge_model": "gemini-3.1-flash-lite (project Gemini judge)",
                "ragas_version": "0.4.3",
                "faithfulness_alert_threshold": FAITHFULNESS_ALERT_THRESHOLD,
            }
        )
        # The four final metric values.
        mlflow.log_metrics(overall)
        # Faithfulness alert state, visible in the MLflow UI run tags.
        mlflow.set_tag("faithfulness_alert", alert_status)
        if alert_status == "ALERT":
            mlflow.set_tag("warning", "Faithfulness < 0.80")
        # Detailed per-question results as an artifact.
        mlflow.log_artifact(str(detail_path))

    print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment   : {EXPERIMENT_NAME}")
    print(f"Detailed artifact   : {detail_path}")


if __name__ == "__main__":
    main()
