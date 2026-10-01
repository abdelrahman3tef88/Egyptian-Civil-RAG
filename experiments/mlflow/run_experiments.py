"""run_experiments.py - the MLflow chunking x embedding experiment.

For every (chunk_size, chunk_overlap, embedding model) combination this
script:

  1. builds (or reuses) an ISOLATED Chroma index for that exact
     configuration,
  2. runs all golden-dataset questions through the project's existing RAG
     pipeline (chunking -> embeddings -> Chroma -> retriever -> prompt ->
     LLM), sending ONLY the question - never the ground truth,
  3. scores the generated answers with Faithfulness (the single metric of
     this phase),
  4. logs parameters, the faithfulness metric, and the run artifacts to
     MLflow.

4 chunking configurations x 3 embedding models = 12 MLflow runs, all
generated programmatically from experiments/mlflow/experiment_config.yaml.

The production configuration (configs/config.yaml) is NEVER modified: the
retrieval settings are read from it and only logged.

Run with:  uv run python experiments/mlflow/run_experiments.py
See also:  --dry-run (print the matrix without running anything),
           --max-questions N (small smoke test).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Shared MLflow helpers. Importing this also puts src/ on sys.path (the
# project package must be importable whether or not it is installed),
# so it comes BEFORE the rag_project imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlflow_common import (  # noqa: E402
    configure_stdout,
    configure_tracking,
    ensure_experiment,
    load_experiment_config,
    slugify,
    write_json,
)

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from rag_project.config import settings  # noqa: E402
from rag_project.evaluation import faithfulness as faithfulness_module  # noqa: E402
from rag_project.generation import chain as chain_module  # noqa: E402
from rag_project.generation import llm as llm_module  # noqa: E402
from rag_project.generation.prompts import prompt as rag_prompt  # noqa: E402
from rag_project.indexing import chunking, embeddings, vector_store  # noqa: E402
from rag_project.retrieval import retriever as retriever_module  # noqa: E402

# File written next to each cached index describing what built it.
INDEX_MARKER = "index_config.json"

# ----------------------------------------------------------
# Experiment-space constraints
# ----------------------------------------------------------
# The experiment compares chunking x embedding and nothing else.
EXPECTED_EMBEDDING_MODELS = 3
EXPECTED_CHUNKING_CONFIGS = 4

# Languages every compared embedding model must handle: the corpus is a
# bilingual Egyptian Civil Code and the golden questions are Arabic/English.
REQUIRED_LANGUAGES = {"ar", "en"}

# Models that cannot satisfy the Arabic + English requirement. This is a
# guard rail so an English-only model cannot quietly re-enter the matrix
# (sentence-transformers/all-MiniLM-L6-v2 was removed from it for exactly
# this reason; the model still belongs to the production config, which the
# experiment never touches).
ENGLISH_ONLY_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2",
    "intfloat/e5-small-v2",
    "intfloat/e5-large-v2",
    "thenlper/gte-base",
    "thenlper/gte-large",
}

# Faithfulness is the only metric of this phase; nothing else is allowed.
ALLOWED_METRICS = {"faithfulness"}





# ==========================================================
# Configuration + golden dataset loading
# ==========================================================
def load_golden_dataset(relative_path):
    """Read the golden dataset (a list of questions, or a wrapper object)."""
    path = settings.resolve_path(relative_path)
    if not path.exists():
        raise SystemExit(
            f"Golden dataset not found: {path}\n"
            "Build it first with: uv run python scripts/build_golden_dataset.py"
        )

    with open(path, "r", encoding="utf-8") as dataset_file:
        data = json.load(dataset_file)

    # Accept both a bare list and the versioned wrapper object.
    if isinstance(data, dict):
        questions = data.get("questions")
        version = data.get("version", "unknown")
    else:
        questions = data
        version = "unknown"

    if not questions:
        raise SystemExit(f"Golden dataset contains no questions: {path}")

    required = ("id", "question", "ground_truth", "relevant_articles")
    for record in questions:
        missing = [key for key in required if key not in record]
        if missing:
            raise SystemExit(
                f"Golden dataset record {record.get('id')} is missing {missing}"
            )

    return questions, version, path


def validate_fixed_retrieval(config):
    """Ensure the experiment keeps the project's retrieval configuration.

    search_type and top_k are NOT experiment dimensions: they must equal
    the production values in configs/config.yaml.
    """
    experimental = config["retrieval"]
    production = settings.CONFIG["retrieval"]

    if experimental["search_type"] != production["search_type"]:
        raise SystemExit(
            f"search_type mismatch: experiment_config.yaml has "
            f"{experimental['search_type']!r} but configs/config.yaml has "
            f"{production['search_type']!r}"
        )
    if int(experimental["top_k"]) != int(production["top_k"]):
        raise SystemExit(
            f"top_k mismatch: experiment_config.yaml has "
            f"{experimental['top_k']} but configs/config.yaml has "
            f"{production['top_k']}"
        )
    if experimental["search_type"] != "similarity":
        raise SystemExit(
            "this experiment phase must use similarity search only "
            "(MMR is deliberately out of scope)"
        )
    return experimental["search_type"], int(experimental["top_k"])


def validate_experiment_space(config):
    """Validate the experiment space BEFORE anything is downloaded or called.

    Checks, all fail-fast:

    * exactly 3 embedding models and 4 chunking configurations (=> 12 runs),
    * no English-only model is present,
    * every model is declared multilingual AND covers ar + en,
    * every model declares its embedding dimension,
    * faithfulness is the only evaluation metric.
    """
    problems = []

    models = config["embedding"]["models"]
    chunkings = config["chunking"]["configurations"]

    if len(models) != EXPECTED_EMBEDDING_MODELS:
        problems.append(
            f"expected exactly {EXPECTED_EMBEDDING_MODELS} embedding models, "
            f"found {len(models)}"
        )
    if len(chunkings) != EXPECTED_CHUNKING_CONFIGS:
        problems.append(
            f"expected exactly {EXPECTED_CHUNKING_CONFIGS} chunking "
            f"configurations, found {len(chunkings)}"
        )

    for model in models:
        name = model.get("name")
        if not name:
            problems.append("an embedding model has no name")
            continue

        # English-only models cannot serve this bilingual corpus.
        if name in ENGLISH_ONLY_MODELS:
            problems.append(
                f"{name} is an English-only embedding model and does not "
                f"support the required Arabic + English retrieval"
            )

        # The multilingual claim must be declared, not inferred from a name.
        if not model.get("multilingual"):
            problems.append(f"{name} is not marked multilingual: true")

        languages = {str(code).lower() for code in (model.get("languages") or [])}
        missing = REQUIRED_LANGUAGES - languages
        if missing:
            problems.append(
                f"{name} is missing required language(s) "
                f"{sorted(missing)} (declared: {sorted(languages)})"
            )

        # The dimension is needed for the runtime dimension check.
        if model.get("dimension") is None:
            problems.append(f"{name} does not declare an embedding dimension")

    metric = str(config["evaluation"].get("metric", "")).strip().lower()
    if metric not in ALLOWED_METRICS:
        problems.append(
            f"evaluation.metric must be one of {sorted(ALLOWED_METRICS)}, "
            f"found {metric!r} (this phase evaluates faithfulness only)"
        )

    if problems:
        raise SystemExit(
            "Experiment configuration is invalid - nothing was run:\n  - "
            + "\n  - ".join(problems)
        )

    return models, chunkings


def build_combinations(config):
    """Generate the full experiment matrix programmatically."""
    combinations = []
    for chunk in config["chunking"]["configurations"]:
        for model in config["embedding"]["models"]:
            combinations.append(
                {
                    "embedding_model": model["name"],
                    "embedding_dimension": model.get("dimension"),
                    "chunk_size": int(chunk["chunk_size"]),
                    "chunk_overlap": int(chunk["chunk_overlap"]),
                }
            )
    return combinations


def filter_combinations(combinations, models=None, chunk_sizes=None):
    """Apply the optional CLI filters (used for smoke tests)."""
    selected = combinations
    if models:
        wanted = {name.strip() for name in models}
        selected = [c for c in selected if c["embedding_model"] in wanted]
    if chunk_sizes:
        wanted = {int(size) for size in chunk_sizes}
        selected = [c for c in selected if c["chunk_size"] in wanted]
    return selected


def combination_slug(combination):
    """Deterministic identity of one configuration.

    Includes the embedding model, chunk_size and chunk_overlap, so an
    index can never be confused with an index built by another
    configuration.
    """
    return (
        f"{slugify(combination['embedding_model'])}"
        f"__cs{combination['chunk_size']}"
        f"__co{combination['chunk_overlap']}"
    )


def prompt_version():
    """Reproducible version label for the project's RAG prompt.

    Derived from a hash of the real prompt template, so the label always
    matches the prompt that was used (nothing is hard-coded).
    """
    template = str(rag_prompt.messages[0].prompt.template)
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


# ==========================================================
# Per-configuration components
# ----------------------------------------------------------
# The project modules expose ONE configured instance each (bound to
# configs/config.yaml). An experiment needs a different instance per
# arm, so the same classes/functions are used with the arm's values -
# the project modules themselves are never modified.
# ==========================================================
def project_separators():
    """The separator list used by the project's own splitter.

    langchain-text-splitters 1.x keeps the list in a private attribute;
    older versions expose it publicly, so both are checked.
    """
    splitter = chunking.text_splitter
    separators = getattr(splitter, "separators", None) or getattr(
        splitter, "_separators", None
    )
    if not separators:
        raise SystemExit(
            "Could not read the separator list from the project's splitter "
            "(rag_project.indexing.chunking.text_splitter)."
        )
    return list(separators)


def create_experiment_splitter(chunk_size, chunk_overlap):
    """Chunker for one arm: same splitter and separators, new sizes."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Reuse the separator list defined by the project's chunking module.
        separators=project_separators(),
    )


# ==========================================================
# Query / passage prefixing (embedding adapter layer)
# ----------------------------------------------------------
# The intfloat E5 family is trained with asymmetric instruction prefixes:
# a question must be embedded as "query: <text>" and a corpus passage as
# "passage: <text>". Without them the vectors are not comparable across
# the two sides and retrieval quality drops.
#
# This is handled HERE, in the embedding adapter only, and it is driven
# by the per-model flag already present in experiment_config.yaml
# (`expects_query_passage_prefixes`) - no model name is hard-coded, so a
# model without the flag (BAAI/bge-m3) receives raw text unchanged.
# ==========================================================
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class PrefixedEmbeddings(Embeddings):
    """Wrap an embedding model to add the E5 query/passage prefixes.

    Everything except `embed_query` and `embed_documents` is delegated to
    the wrapped model, so the object still behaves like the original
    HuggingFaceEmbeddings for every other caller (Chroma, the retriever,
    the dimension check).
    """

    def __init__(
        self,
        base_embeddings,
        query_prefix=QUERY_PREFIX,
        passage_prefix=PASSAGE_PREFIX,
    ):
        self.base_embeddings = base_embeddings
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    # Any other attribute/method comes from the wrapped model.
    def __getattr__(self, name):
        # __getattr__ only runs for attributes not found normally, so
        # base_embeddings/query_prefix/passage_prefix are not affected.
        return getattr(self.base_embeddings, name)

    # --- the two prefixed entry points -------------------------------
    def embed_query(self, text: str):
        """Embed a user question as `query: <text>`."""
        return self.base_embeddings.embed_query(f"{self.query_prefix}{text}")

    def embed_documents(self, texts):
        """Embed corpus passages as `passage: <text>`."""
        prefixed = [f"{self.passage_prefix}{text}" for text in texts]
        return self.base_embeddings.embed_documents(prefixed)

    def describe(self):
        """Human-readable description of the prefixing in effect."""
        return {
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "prefixed": True,
        }


class RawEmbeddings(Embeddings):
    """Wrap an embedding model and pass text through completely unchanged.

    Used for models whose flag is false (e.g. BAAI/bge-m3), so the
    experiment code path is identical for every arm and only the wrapping
    decision differs.
    """

    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings

    def __getattr__(self, name):
        return getattr(self.base_embeddings, name)

    def embed_query(self, text: str):
        return self.base_embeddings.embed_query(text)

    def embed_documents(self, texts):
        return self.base_embeddings.embed_documents(texts)

    def describe(self):
        return {
            "query_prefix": None,
            "passage_prefix": None,
            "prefixed": False,
        }


def model_config_entry(config, model_name):
    """Return the configured metadata block for one embedding model."""
    for entry in config["embedding"]["models"]:
        if entry.get("name") == model_name:
            return entry
    raise SystemExit(
        f"Embedding model {model_name!r} is not described in "
        f"experiment_config.yaml; add its metadata (dimension, "
        f"multilingual, languages) before running the experiment."
    )


def uses_query_passage_prefixes(config, model_name):
    """True when the configured model needs the E5 query/passage prefixes.

    Read from the per-model `expects_query_passage_prefixes` flag, so no
    model name is ever hard-coded in the code.
    """
    return bool(
        model_config_entry(config, model_name).get(
            "expects_query_passage_prefixes", False
        )
    )



def create_experiment_embeddings(model_name, config):
    """Embedding model for one arm.

    Mirrors rag_project.indexing.embeddings.create_embedding_model()
    (same class, same model_kwargs/encode_kwargs, same device policy)
    with the arm's model name, and reuses the module's device resolver.

    The returned object is always the adapter layer, so query/passage
    prefixing is decided per model from `expects_query_passage_prefixes`
    and never from the model name.
    """
    embedding_config = config["embedding"]
    base_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": embeddings._resolve_device(  # noqa: SLF001 - shared policy
                embedding_config.get("device", "auto")
            )
        },
        encode_kwargs={
            "normalize_embeddings": embedding_config.get(
                "normalize_embeddings", True
            )
        },
    )

    # Model-level flag decides prefixing (no model names hard-coded).
    if uses_query_passage_prefixes(config, model_name):
        return PrefixedEmbeddings(base_embeddings)

    return RawEmbeddings(base_embeddings)



def index_directory(config, combination):
    """Cache directory of one arm's isolated index."""
    base = settings.resolve_path(config["execution"]["index_cache_dir"])
    return base / combination_slug(combination)


def marker_matches(directory, combination, config=None):
    """True when a cached index was built by exactly this configuration.

    The embedding prefixing is part of the index identity: an index built
    WITHOUT the E5 query/passage prefixes stores raw-passage vectors, so
    reusing it while sending prefixed queries would silently compare
    mismatched vector spaces. Such an index must be rebuilt instead.
    """
    marker_path = directory / INDEX_MARKER
    if not marker_path.exists():
        return False
    try:
        with open(marker_path, "r", encoding="utf-8") as marker_file:
            marker = json.load(marker_file)
    except (json.JSONDecodeError, OSError):
        return False

    if not (
        marker.get("embedding_model") == combination["embedding_model"]
        and int(marker.get("chunk_size", -1)) == combination["chunk_size"]
        and int(marker.get("chunk_overlap", -1)) == combination["chunk_overlap"]
    ):
        return False

    # Marker written before prefixing existed has no key -> treat as
    # unprefixed, so it is only reused by an unprefixed model.
    if config is not None:
        expected_prefixed = uses_query_passage_prefixes(
            config, combination["embedding_model"]
        )
        if bool(marker.get("query_passage_prefixes", False)) != expected_prefixed:
            return False

    return True



def verify_embedding_dimension(embedding_model, expected_dimension, model_name):
    """Confirm the loaded model really produces `expected_dimension` values.

    Implementation-agnostic: it embeds a probe string and measures the
    resulting vector, so the check reflects what the runtime actually
    produces rather than what a model card claims. A mismatch means the
    configuration is wrong, so the caller aborts instead of comparing
    mis-configured models.
    """
    probe = embedding_model.embed_query("embedding dimension probe")
    actual_dimension = len(probe)

    if actual_dimension != int(expected_dimension):
        raise SystemExit(
            "Embedding dimension mismatch - the experiment was not run:\n"
            f"  model            : {model_name}\n"
            f"  declared in yaml : {expected_dimension}\n"
            f"  actual output    : {actual_dimension}\n"
            "Fix experiment_config.yaml before running the experiments."
        )
    return actual_dimension


def load_or_build_index(config, combination, rebuild=False):
    """Return (vector_store, embedding_model, reused) for one arm.

    A cached index is reused only when its marker proves it was built by
    the same embedding model and chunking configuration.
    """
    directory = index_directory(config, combination)
    embedding_model = create_experiment_embeddings(
        combination["embedding_model"], config
    )

    # Verify the loaded model matches its declared dimension BEFORE the
    # index is built or reused, so no run can silently compare a
    # mis-configured model.
    actual_dimension = verify_embedding_dimension(
        embedding_model,
        combination["embedding_dimension"],
        combination["embedding_model"],
    )

    if (
        not rebuild
        and vector_store.index_exists(directory)
        and marker_matches(directory, combination, config)
    ):
        store = vector_store.load_vector_store(
            embedding_model, persist_directory=directory
        )
        return store, embedding_model, True, actual_dimension

    # Build the index for this exact configuration.
    directory.mkdir(parents=True, exist_ok=True)

    # 1) Articles -> Documents (same loader the project uses; the path
    #    comes from configs/config.yaml).
    documents = chunking.load_articles()
    # 2) Chunk with this arm's chunk_size / chunk_overlap.
    splitter = create_experiment_splitter(
        combination["chunk_size"], combination["chunk_overlap"]
    )
    chunks = splitter.split_documents(documents)
    # 3) Embed + persist into this arm's own directory.
    store = vector_store.create_vector_store(
        chunks, embedding_model, persist_directory=directory
    )
    vector_store.save_vector_store(store)

    # Record what this index is, so it can never be reused by accident.
    marker = {
        "embedding_model": combination["embedding_model"],
        "embedding_dimension": combination["embedding_dimension"],
        "embedding_dimension_verified": actual_dimension,
        # Part of the index identity: an index built with these prefixes
        # stores prefixed-passage vectors.
        "query_passage_prefixes": uses_query_passage_prefixes(
            config, combination["embedding_model"]
        ),
        "chunk_size": combination["chunk_size"],

        "chunk_overlap": combination["chunk_overlap"],
        "chunk_count": len(chunks),
        "document_count": len(documents),
        "collection_name": vector_store.COLLECTION_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(directory / INDEX_MARKER, "w", encoding="utf-8") as marker_file:
        json.dump(marker, marker_file, ensure_ascii=False, indent=2)

    return store, embedding_model, False, actual_dimension



# ==========================================================
# Evaluation loop
# ==========================================================
def evaluate_questions(rag_chain, retriever, questions, llm, config, limit=None):
    """Answer every question with the pipeline, then score faithfulness.

    IMPORTANT: only ``question`` is passed to the RAG pipeline. The
    ground truth, the relevant article numbers and the ground-truth
    contexts never leave this function - they are evaluator-side data.
    """
    backend = config["evaluation"].get("backend", "builtin")
    max_claims = int(config["evaluation"].get("max_claims", 25))
    selected = questions[:limit] if limit else questions

    results = []
    contexts_by_question = {}
    for record in selected:
        question = record["question"]

        started = time.perf_counter()
        # 1) + 2) the project's chain retrieves the context and generates
        #    the answer from the question alone.
        answer = rag_chain.invoke(question)
        # The project's chain returns only the answer text, so the
        # retrieved documents are fetched again here - the alternative
        # would be modifying the project's chain, which is out of scope.
        documents = retriever.invoke(question)
        elapsed = time.perf_counter() - started

        contexts = [document.page_content for document in documents]
        # Kept for the retrieved_contexts.json artifact.
        contexts_by_question[record["id"]] = contexts

        # 3) Faithfulness is computed from the answer and the retrieved
        #    contexts ONLY.
        score = faithfulness_module.evaluate_faithfulness(
            question,
            answer,
            contexts,
            llm=llm,
            backend=backend,
            max_claims=max_claims,
        )

        results.append(
            {
                "id": record["id"],
                "question": question,
                "language": record.get("language"),
                "question_type": record.get("question_type"),
                "difficulty": record.get("difficulty"),
                # Evaluator-side reference (never sent to the pipeline).
                "ground_truth_articles": record.get("relevant_articles"),
                "answer": answer,
                "faithfulness": score.score,
                "faithfulness_backend": score.backend,
                "n_claims": score.n_claims,
                "n_supported": score.n_supported,
                "claim_verdicts": score.claims,
                # Diagnostics kept in the artifacts (not logged as metrics).
                "retrieved_article_numbers": [
                    document.metadata.get("article_number")
                    for document in documents
                ],
                "answer_seconds": round(elapsed, 3),
            }
        )

    return results, contexts_by_question


def summarise(results, combination, dataset_version, reused_index, chunk_count):
    """Aggregate one run into the numbers and the artifacts it publishes."""
    scores = [result["faithfulness"] for result in results]
    faithfulness = faithfulness_module.aggregate_faithfulness(scores)

    backends = sorted({result["faithfulness_backend"] for result in results})
    # The honest, human-readable name of the metric that was computed.
    evaluator = faithfulness_module.faithfulness_label(backends[0])
    return {
        "faithfulness": faithfulness,
        "faithfulness_metric_name": evaluator,
        "questions_evaluated": len(results),
        "faithfulness_backend": ", ".join(backends),
        "embedding_model": combination["embedding_model"],
        "embedding_dimension": combination["embedding_dimension"],
        "chunk_size": combination["chunk_size"],
        "chunk_overlap": combination["chunk_overlap"],
        "golden_dataset_version": dataset_version,
        "index_reused": reused_index,
        "indexed_chunks": chunk_count,
    }



# ==========================================================
# One complete MLflow run
# ==========================================================
def run_single_experiment(
    config,
    combination,
    questions,
    dataset_version,
    runtime,
    max_questions=None,
    rebuild=False,
    reports_root=None,
):
    """Run one configuration end to end and log it as one MLflow run."""
    import mlflow

    slug = combination_slug(combination)
    if reports_root is None:
        reports_root = config["execution"]["reports_dir"]
    reports_root = settings.resolve_path(reports_root)
    evaluation_dir = reports_root / "evaluation_results"
    artifacts_dir = reports_root / "artifacts" / slug


    # 1) Isolated index for this exact configuration.
    store, _embedding_model, reused, actual_dimension = load_or_build_index(
        config, combination, rebuild=rebuild
    )
    marker_path = index_directory(config, combination) / INDEX_MARKER
    chunk_count = 0
    if marker_path.exists():
        with open(marker_path, "r", encoding="utf-8") as marker_file:
            chunk_count = int(json.load(marker_file).get("chunk_count", 0))

    # 2) Retriever + chain: the project's own components with the
    #    project's fixed retrieval settings.
    retriever = retriever_module.create_retriever(store)
    rag_chain = chain_module.create_rag_chain(retriever, rag_prompt, runtime["llm"])

    # 3) Answer + evaluate every golden question. The answer comes from
    #    the generation model, the score from the judge model.
    results, contexts_by_question = evaluate_questions(
        rag_chain,
        retriever,
        questions,
        runtime["judge_llm"],
        config,
        limit=max_questions,
    )

    summary = summarise(
        results, combination, dataset_version, reused, chunk_count
    )

    # 4) Local artifacts (kept inspectable even without an MLflow server).
    evaluation_path = evaluation_dir / f"{slug}.json"
    contexts_path = artifacts_dir / "retrieved_contexts.json"
    summary_path = artifacts_dir / "run_summary.json"

    write_json(
        evaluation_path,
        {
            "configuration": {
                "embedding_model": combination["embedding_model"],
                "embedding_dimension": combination["embedding_dimension"],
                "chunk_size": combination["chunk_size"],
                "chunk_overlap": combination["chunk_overlap"],
                "search_type": runtime["search_type"],
                "top_k": runtime["top_k"],
            },
            "golden_dataset_version": dataset_version,
            "faithfulness": summary["faithfulness"],
            "questions_evaluated": summary["questions_evaluated"],
            "results": results,
        },
    )
    write_json(contexts_path, contexts_by_question)
    write_json(summary_path, summary)

    # 5) MLflow logging.
    params = {
        "embedding_model": combination["embedding_model"],
        "embedding_dimension": str(actual_dimension),
        # Reproducibility: whether this arm embedded with the E5
        # query/passage prefixes.
        "query_passage_prefixes": str(
            uses_query_passage_prefixes(config, combination["embedding_model"])
        ).lower(),
        "chunk_size": str(combination["chunk_size"]),
        "chunk_overlap": str(combination["chunk_overlap"]),
        "search_type": runtime["search_type"],
        "top_k": str(runtime["top_k"]),
        "golden_dataset_version": dataset_version,
        "prompt_version": runtime["prompt_version"],
        # The generation model (answers) and the Faithfulness judge (scoring)
        # are logged separately so both roles stay traceable in MLflow.
        "generation_model": runtime["generation_model"],
        "faithfulness_judge_model": runtime["faithfulness_judge_model"],
        "experiment_name": runtime["experiment_name"],

        # Which Faithfulness implementation actually produced the score,
        # so a score is never mislabelled (e.g. as RAGAS when it is not).
        "faithfulness_evaluator": summary["faithfulness_metric_name"],
        "faithfulness_backend": summary["faithfulness_backend"],
        "evaluation_metric": config["evaluation"].get("metric", "faithfulness"),
    }


    with mlflow.start_run(run_name=slug) as active_run:
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "configuration_slug": slug,
                "experiment_phase": "chunking_x_embedding",
                "index_reused": str(reused),
                "faithfulness_backend": summary["faithfulness_backend"],
            }
        )
        # The single evaluation metric of this phase.
        mlflow.log_metric("faithfulness", summary["faithfulness"])
        # Naturally available execution information (not an evaluation metric).
        mlflow.log_metric("questions_evaluated", summary["questions_evaluated"])

        mlflow.log_artifact(str(evaluation_path), artifact_path="evaluation_results")
        mlflow.log_artifact(str(summary_path), artifact_path="artifacts")
        mlflow.log_artifact(str(contexts_path), artifact_path="artifacts")

        run_id = active_run.info.run_id

    summary["run_id"] = run_id
    summary["run_name"] = slug
    write_json(summary_path, summary)
    return summary


# ==========================================================
# CLI
# ==========================================================
def parse_args(argv=None):
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Run the MLflow chunking x embedding experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment-config", default=None, help="path to experiment_config.yaml"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="use only the first N golden questions (smoke test)",
    )
    parser.add_argument(
        "--models", nargs="*", default=None, help="restrict to these embedding models"
    )
    parser.add_argument(
        "--chunk-sizes", nargs="*", default=None, help="restrict to these chunk sizes"
    )
    parser.add_argument(
        "--limit-runs", type=int, default=None, help="run at most N configurations"
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="override the reports directory (default: execution.reports_dir)",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="rebuild the indexes even if cached"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the experiment matrix and exit"
    )
    return parser.parse_args(argv)


def split_values(values):
    """Accept both "--models a b" and "--models a,b"."""
    if not values:
        return None
    flattened = []
    for value in values:
        flattened.extend(part for part in str(value).split(",") if part.strip())
    return flattened


def print_matrix(combinations, questions, dataset_version, search_type, top_k, models):
    """Show the experiment matrix that will be executed."""
    print("=" * 92)
    print("EXPERIMENT MATRIX")
    print("=" * 92)
    print(f"Golden questions      : {len(questions)} (version {dataset_version})")
    print(f"Fixed retrieval       : search_type={search_type}, top_k={top_k}")
    print(f"Embedding models      : {len(models)} (all bilingual: ar + en)")
    for model in models:
        print(
            f"  - {model['name']:<40} dim={model['dimension']:<6} "
            f"multilingual={str(model.get('multilingual')).lower()}"
        )
    print(f"Chunking configs      : 4 (chunk_size + chunk_overlap always paired)")
    print(f"Total runs            : {len(combinations)}")
    print("-" * 92)
    print(
        f"{'#':<3} {'embedding_model':<40} {'dim':>5} {'chunk_size':>11} "
        f"{'overlap':>8} {'search':>10} {'top_k':>6}"
    )
    print("-" * 92)
    for position, combination in enumerate(combinations, start=1):
        print(
            f"{position:<3} {combination['embedding_model']:<40} "
            f"{combination['embedding_dimension']:>5} "
            f"{combination['chunk_size']:>11} {combination['chunk_overlap']:>8} "
            f"{search_type:>10} {top_k:>6}"
        )
    print("=" * 92)



def main(argv=None):
    """Load the config/dataset, generate the combinations, run, and log."""
    args = parse_args(argv)
    configure_stdout()

    config = load_experiment_config(args.experiment_config)
    search_type, top_k = validate_fixed_retrieval(config)
    # Fail fast on an invalid experiment space (wrong model count, an
    # English-only model, a non-multilingual model, a bad metric, ...)
    # before any model download or LLM call happens.
    configured_models, _ = validate_experiment_space(config)

    combinations = filter_combinations(
        build_combinations(config),
        split_values(args.models),
        split_values(args.chunk_sizes),
    )
    if args.limit_runs:
        combinations = combinations[: args.limit_runs]

    questions, dataset_version, dataset_path = load_golden_dataset(
        config["golden_dataset"]["path"]
    )

    print_matrix(
        combinations,
        questions,
        dataset_version,
        search_type,
        top_k,
        configured_models,
    )

    print(f"Golden dataset file   : {dataset_path}")
    print(
        "Index cache directory : "
        f"{settings.resolve_path(config['execution']['index_cache_dir'])}"
    )
    print(
        "Reports directory     : "
        f"{settings.resolve_path(config['execution']['reports_dir'])}"
    )

    if args.dry_run:
        print("\n--dry-run: nothing was executed and MLflow was not contacted.")
        return 0

    # Fail fast: the generation key is required for the answers and the
    # judge key for the Faithfulness scoring.

    missing_keys = [
        variable
        for variable in (
            llm_module.GENERATION_API_KEY_VARIABLE,
            llm_module.JUDGE_API_KEY_VARIABLE,
        )
        if not os.getenv(variable)
    ]
    if missing_keys:
        raise SystemExit(
            "Missing API key(s) in the environment: "
            + ", ".join(missing_keys)
            + ". Put them in the project .env file (see .env.example) or "
            "export them before running the experiment."
        )

    import mlflow

    configure_tracking(mlflow, config)
    experiment_name = config["experiment"]["name"]
    ensure_experiment(
        mlflow, experiment_name, config["experiment"].get("artifact_location")
    )

    # Two separate models: the generator answers, the judge scores the
    # answer against the retrieved contexts.
    runtime = {
        "llm": llm_module.create_llm(),
        "judge_llm": llm_module.create_judge_llm(),
        "search_type": search_type,
        "top_k": top_k,
        "generation_model": settings.CONFIG["generation"]["model"],
        "faithfulness_judge_model": settings.CONFIG["generation"]["judge_model"],
        "prompt_version": prompt_version(),
        "experiment_name": experiment_name,
    }


    print(f"MLflow tracking URI   : {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment     : {experiment_name}")

    summaries = []
    for position, combination in enumerate(combinations, start=1):
        print("\n" + "-" * 78)
        print(
            f"[{position}/{len(combinations)}] {combination['embedding_model']} | "
            f"chunk_size={combination['chunk_size']} | "
            f"chunk_overlap={combination['chunk_overlap']}"
        )
        print("-" * 78, flush=True)
        try:
            summary = run_single_experiment(
                config,
                combination,
                questions,
                dataset_version,
                runtime,
                max_questions=args.max_questions,
                rebuild=args.rebuild,
                reports_root=args.reports_dir,
            )
        except Exception as error:  # noqa: BLE001 - continue with the next arm.
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            summaries.append(
                {
                    "embedding_model": combination["embedding_model"],
                    "chunk_size": combination["chunk_size"],
                    "chunk_overlap": combination["chunk_overlap"],
                    "faithfulness": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        print(
            f"  faithfulness={summary['faithfulness']:.4f} | "
            f"questions={summary['questions_evaluated']} | "
            f"chunks={summary['indexed_chunks']} | "
            f"index_reused={summary['index_reused']} | "
            f"run_id={summary['run_id']}",
            flush=True,
        )
        summaries.append(summary)

    print("\n" + "=" * 78)
    print("EXPERIMENT RESULTS (sorted by faithfulness)")
    print("=" * 78)
    print(f"{'embedding_model':<44} {'size':>6} {'overlap':>8} {'faithfulness':>13}")
    print("-" * 78)
    ranked = sorted(
        summaries,
        key=lambda item: (
            item["faithfulness"] is not None,
            item["faithfulness"] or 0.0,
        ),
        reverse=True,
    )
    for summary in ranked:
        score = summary["faithfulness"]
        score_text = f"{score:.4f}" if isinstance(score, float) else "FAILED"
        print(
            f"{summary['embedding_model']:<44} {summary['chunk_size']:>6} "
            f"{summary['chunk_overlap']:>8} {score_text:>13}"
        )
    print("=" * 78)
    print("Next step: uv run python experiments/mlflow/promote_best_config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






