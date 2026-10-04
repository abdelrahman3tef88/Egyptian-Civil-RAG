"""run_ci_test.py - CI gate for the RAG pipeline.

ARCHITECTURE (this script is the CONSUMER, never the BUILDER):

    CI workflow:  dvc pull -> dvc repro -> [this script]
                                                     |
    DVC "index" stage (dvc.yaml) builds              v
    artifacts/vector_store/  ------->  vector_store.load_vector_store()
                                             -> retriever.create_retriever()
                                                 -> prompt -> llm -> answer
                                                     -> faithfulness.evaluate_faithfulness()
                                                     -> aggregate_faithfulness()
                                                     -> threshold -> exit 0 / 1

WHAT THIS SCRIPT MUST NEVER DO
------------------------------
It must not ingest, load documents for indexing, chunk, embed, create a
vector store, create a Chroma collection, or build/rebuild any index.
The ONLY thing it may do with the index is LOAD and USE the one DVC
already produced. That is why every component below is called from
src/rag_project/ with no indexing function in this file:

    vector_store.load_vector_store()   <- opens the existing collection
    (vector_store.create_vector_store() is NEVER called here)

It also never runs dvc pull / dvc repro / any MLflow experiment. Those
belong to the CI workflow, which owns the order of operations.

Exit codes:
    0  overall RAGAS Faithfulness >= FAITHFULNESS_THRESHOLD
    1  overall RAGAS Faithfulness <  FAITHFULNESS_THRESHOLD
    2  the evaluation could not be performed (missing index, missing
       dataset, RAGAS failure, ...). An error is NEVER reported as a
       passing CI run.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_project.evaluation import faithfulness as faithfulness_module  # noqa: E402
from rag_project.generation import llm as llm_module  # noqa: E402
from rag_project.generation.chain import create_rag_chain, format_docs  # noqa: E402
from rag_project.generation.prompts import prompt as rag_prompt  # noqa: E402
from rag_project.indexing import embeddings, vector_store  # noqa: E402
from rag_project.retrieval import retriever as retriever_module  # noqa: E402

# ---- CI policy ----------------------------------------------------------
# The PASS/FAIL decision lives HERE, never in faithfulness.py: that module
# computes and aggregates a metric, it does not decide CI policy.
FAITHFULNESS_THRESHOLD = 0.75

# The CI dataset. Deliberately NOT the 25-question MLflow golden dataset
# and not any experiment run.
CI_DATASET_PATH = REPO_ROOT / "data" / "ci_test_questions.json"
EXPECTED_QUESTION_COUNT = 20

# CI outcome / could-not-run.
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

REFUSAL_SENTINEL = faithfulness_module.REFUSAL_SENTINEL
BACKEND = faithfulness_module.DEFAULT_BACKEND  # "ragas"


class SetupError(RuntimeError):
    """The evaluation could not run (missing index / bad CI dataset).

    Kept distinct from a regression failure on purpose: this exits 2, so a
    broken environment is never reported as "FAIL (exit 1)" and can never
    be mistaken for a genuine metric regression - nor for a PASS.
    """


def load_ci_questions():
    """Load and validate the CI dataset (exactly 20 questions)."""
    if not CI_DATASET_PATH.exists():
        raise SetupError(
            f"CI dataset not found: {CI_DATASET_PATH}\n"
            "The CI evaluator consumes data/ci_test_questions.json and "
            "cannot generate questions itself."
        )

    payload = json.loads(CI_DATASET_PATH.read_text(encoding="utf-8"))
    questions = payload["questions"] if isinstance(payload, dict) else payload

    # Validated BEFORE any evaluation runs, so the count is never adjusted
    # later and no question is silently dropped or added.
    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise SetupError(
            f"CI dataset must hold exactly {EXPECTED_QUESTION_COUNT} "
            f"questions, found {len(questions)}"
        )

    for record in questions:
        # The question text is the ONLY field the pipeline may ever see.
        if not str(record.get("question", "")).strip():
            raise SetupError(f"{record.get('id')}: empty question")

    return questions


def load_production_index():
    """LOAD the DVC-produced production index. Never builds one.

    embeddings.create_embedding_model() is required because a Chroma
    collection cannot be opened without an embedding function to encode
    incoming queries; it embeds the QUESTIONS only. It does not embed
    or write any document, and no index is created or rebuilt here.
    """
    persist_directory = vector_store.PERSIST_DIRECTORY

    if not vector_store.index_exists():
        raise SetupError(
            f"Production vector index not found at {persist_directory}.\n"
            "Restore it FIRST, then run this script:\n"
            "  dvc pull\n"
            "  dvc repro\n"
            "This script never builds the index itself."
        )

    embedding_model = embeddings.create_embedding_model()
    # load (not create): opens the collection DVC already produced.
    return vector_store.load_vector_store(embedding_model), persist_directory


def main():
    questions = load_ci_questions()

    print("=" * 78)
    print("RAG CI EVALUATION")
    print("=" * 78)
    print(f"CI dataset        : {CI_DATASET_PATH.relative_to(REPO_ROOT)}")
    print(f"Questions         : {len(questions)}")
    print(f"Faithfulness backend: {BACKEND} (actual RAGAS metric)")
    print(f"Threshold         : {FAITHFULNESS_THRESHOLD}")

    store, persist_directory = load_production_index()
    print(f"Vector index      : {persist_directory} (loaded, NOT built)")
    print(f"Collection        : {vector_store.COLLECTION_NAME}")

    # Existing RAG components - no duplicated pipeline.
    retriever = retriever_module.create_retriever(store)
    llm = llm_module.create_llm()
    judge_llm = llm_module.create_judge_llm()
    rag_chain = create_rag_chain(retriever, rag_prompt, llm)

    scores = []
    refusal_count = 0
    print("\n" + "-" * 78)
    for record in questions:
        question = record["question"]

        # ONLY the question reaches generation. No ground truth, no
        # difficulty, no language, no metadata, no expected answer.
        answer = rag_chain.invoke(question)

        # Same retrieval the answer was generated from, formatted the same
        # way, so the judge scores the answer against the real context.
        documents = retriever.invoke(question)
        contexts = [format_docs(documents)]

        # RAGAS Faithfulness: question + answer + retrieved contexts only.
        result = faithfulness_module.evaluate_faithfulness(
            question, answer, contexts, llm=judge_llm, backend=BACKEND
        )
        scores.append(result.score)

        refused = REFUSAL_SENTINEL.strip().lower() in str(answer).strip().lower()
        refusal_count += 1 if refused else 0

        print(
            f"  {record['id']}  faithfulness={result.score:.4f}"
            f"{'  [REFUSAL]' if refused else ''}",
            flush=True,
        )

    # No question may be silently skipped: a missing score must fail the
    # run loudly instead of quietly shrinking the denominator and
    # inflating the mean.
    if len(scores) != EXPECTED_QUESTION_COUNT:
        raise SetupError(
            f"expected {EXPECTED_QUESTION_COUNT} faithfulness scores, "
            f"collected {len(scores)} - a question was skipped"
        )
    for question_id, score in zip((record["id"] for record in questions), scores):
        if score is None or (isinstance(score, float) and score != score):
            raise SetupError(f"{question_id}: undefined faithfulness score")

    overall = faithfulness_module.aggregate_faithfulness(scores)
    answer_rate = (len(questions) - refusal_count) / len(questions)

    print("\n" + "=" * 78)
    print(f"Overall RAGAS Faithfulness : {overall:.4f}")
    print(f"Questions evaluated        : {len(scores)}")
    print(f"Refusals ('I don't know')  : {refusal_count}")
    print(f"Answer rate                : {answer_rate:.4f}")
    print(f"Threshold                  : {FAITHFULNESS_THRESHOLD}")
    print("=" * 78)

    if overall >= FAITHFULNESS_THRESHOLD:
        print(f"RESULT: PASS ({overall:.4f} >= {FAITHFULNESS_THRESHOLD})")
        return EXIT_PASS

    print(f"RESULT: FAIL ({overall:.4f} < {FAITHFULNESS_THRESHOLD})")
    return EXIT_FAIL


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        # A RAGAS / evaluator / setup failure is an ERROR (exit 2), never a
        # silent PASS and never a regression FAIL. CI must not report
        # success - or a metric drop - on a broken run.
        print(f"\nRESULT: ERROR - {type(error).__name__}: {error}")
        raise SystemExit(EXIT_ERROR)
