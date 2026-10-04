"""Faithfulness - the only evaluation metric of the experiment phase.

The RAG pipeline must NOT receive the golden ground truth: the evaluator
is given the question, the answer the pipeline produced, and the contexts
the pipeline retrieved, and nothing else.

Faithfulness is defined (as in RAGAS) as:

    number of claims in the answer that are supported by the retrieved
    contexts  /  total number of claims in the answer

ACTIVE BACKEND: ``ragas``
--------------------------------
The experiment evaluates faithfulness with the **real RAGAS
``Faithfulness`` metric** (``ragas.metrics.collections.Faithfulness``),
using the project's existing Gemini chat model as the judge through
RAGAS' ``LangchainLLMWrapper``. That is the whole evaluation path:

    Question -> RAG pipeline -> Answer + Retrieved Contexts
              -> RAGAS Faithfulness -> Gemini judge -> score

``backend = "ragas"`` is the default and the value used by
``experiments/mlflow/experiment_config.yaml``.

There is NO silent fallback. If RAGAS cannot be imported, or if the
metric fails or returns an undefined score, the error is raised so the
run fails loudly. A score produced by anything other than RAGAS is never
reported as a RAGAS score.

``builtin`` remains available ONLY as an explicit, opt-in backend (it is
the project's own claim-decomposition judge). It is never selected
automatically and is not used by the experiment.

Naming: the MLflow metric key is always ``faithfulness``. The backend
that actually produced each score is recorded per run as the parameter
``faithfulness_evaluator`` (see ``faithfulness_label()``).

CONTRACT - what this evaluator sees:

    it receives : the generated ANSWER and the RETRIEVED CONTEXTS
    it asks     : "are the claims made by the answer supported by the
                  retrieved contexts?"
    it never    : receives ground_truth, relevant_articles or
                  ground_truth_contexts - they are not parameters of any
                  function in this module, and they are not needed,
                  because Faithfulness is reference-free.

Both backends are used only by the evaluator; neither is part of the RAG
pipeline itself.

HOW A SCORE IS PRODUCED (two levels)
------------------------------------
    level 1 (per question)
        evaluate_faithfulness(question, answer, contexts, llm)
        -> one RAGAS Faithfulness score for that single question, judged
           ONLY against the contexts retrieved for that question. Each
           question is scored independently.

    level 2 (the whole batch)
        aggregate_faithfulness([per-question scores])
        -> the single OVERALL Faithfulness score reported by the
           experiment, taken as the mean of the per-question scores.

The overall score is therefore always a mean of real per-question RAGAS
scores; it is never a separate judgement, never a re-judging of all
answers at once, and never a threshold/PASS-FAIL verdict.
"""

import asyncio
import json
import re
import warnings
from dataclasses import dataclass, field

from rag_project.evaluation import ragas_compat
from rag_project.generation import llm as llm_module

# The exact refusal sentence produced by the project's RAG prompt
# (generation/prompts.py, instruction 3). An answer that is an honest
# refusal asserts nothing, so it cannot be unfaithful.
REFUSAL_SENTINEL = "I don't know based on the provided documents."

# The backend used when the caller does not choose one.
DEFAULT_BACKEND = "ragas"

# Human-readable name of each backend's metric. The MLflow metric key is
# always "faithfulness"; this is the label describing HOW it was computed.
FAITHFULNESS_LABELS = {
    "builtin": "LLM-based Faithfulness (claim decomposition + claim verification)",
    "ragas": "RAGAS Faithfulness",
}


def faithfulness_label(backend):
    """Return the honest label for a backend name."""
    return FAITHFULNESS_LABELS.get(
        str(backend).split(" ")[0], FAITHFULNESS_LABELS[DEFAULT_BACKEND]
    )


class RagasEvaluationError(RuntimeError):
    """Raised when the RAGAS Faithfulness metric cannot produce a score.

    This is deliberately fatal for the run: the experiment never quietly
    substitutes the built-in judge for a RAGAS score.
    """


# ==========================================================
# Result of one faithfulness evaluation
# ==========================================================
@dataclass
class FaithfulnessResult:
    """Numeric faithfulness score plus the evidence behind it."""

    score: float
    # Per-question detail (kept so the run artifacts are inspectable).
    n_claims: int = 0
    n_supported: int = 0
    backend: str = "builtin"
    claims: list = field(default_factory=list)


# ==========================================================
# Chat model message -> plain text
# ==========================================================
def _message_text(message):
    """Flatten an LLM response object into plain text.

    Chat providers (and other chat models) may return the text as a plain
    string, as a list of content blocks, or - for reasoning models -
    alongside a separate reasoning trace, so all shapes are handled.


    """
    content = getattr(message, "content", message)

    # Plain string response.
    if isinstance(content, str):
        return content

    # List-of-blocks response: keep the text parts only.
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts)

    return str(content)


# ==========================================================
# Backend 1: minimal built-in judge (no extra dependency)
# ==========================================================
CLAIM_EXTRACTION_INSTRUCTIONS = """You are a strict fact-checking assistant.

Task: break the ANSWER below into its atomic factual claims.

Rules:
- One claim per entry: a single, self-contained statement of fact.
- Split compound sentences into separate claims.
- Ignore greetings, filler, and pure formatting words.
- If the ANSWER makes no factual claim at all (for example it only
  refuses to answer), return an empty list.
- Reply with ONLY a JSON array of strings. No prose, no code fences.

ANSWER:
{answer}
"""

CLAIM_VERIFICATION_INSTRUCTIONS = """You are a strict fact-checking assistant.

Task: decide whether the CLAIM is fully supported by the CONTEXT.

Rules:
- Use ONLY the CONTEXT. Do not use outside knowledge.
- Answer "yes" only if the CONTEXT itself states or directly entails
  every part of the CLAIM.
- Otherwise answer "no".

CONTEXT:
{context}

CLAIM:
{claim}

Reply with exactly one word: yes or no.
"""


def _parse_claim_list(raw_text):
    """Extract the JSON array of claims from the judge's reply."""
    # Prefer a real JSON array anywhere in the reply.
    match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip() != ""]

    # Fallback: treat bullet / numbered lines as claims.
    claims = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        cleaned = cleaned.strip('",')
        if cleaned and not cleaned.startswith("[") and not cleaned.startswith("]"):
            claims.append(cleaned)
    return claims


def _builtin_faithfulness(question, answer, contexts, llm, max_claims):
    """Score faithfulness with claim decomposition + verification."""
    # An honest refusal asserts nothing unsupported.
    if REFUSAL_SENTINEL.split(".")[0].lower() in answer.lower():
        return FaithfulnessResult(
            score=1.0, n_claims=0, n_supported=0, backend="builtin"
        )

    # Step 1: decompose the answer into atomic claims.
    extraction = llm.invoke(CLAIM_EXTRACTION_INSTRUCTIONS.format(answer=answer))
    claims = _parse_claim_list(_message_text(extraction))

    # Cap the number of judge calls per question.
    claims = claims[:max_claims]

    # No claims -> nothing that could contradict the contexts.
    if not claims:
        return FaithfulnessResult(
            score=1.0, n_claims=0, n_supported=0, backend="builtin"
        )

    # Step 2: verify every claim against the retrieved contexts.
    context_block = "\n\n".join(contexts)
    supported = 0
    verdicts = []
    for claim in claims:
        verdict_raw = (
            _message_text(
                llm.invoke(
                    CLAIM_VERIFICATION_INSTRUCTIONS.format(
                        context=context_block, claim=claim
                    )
                )
            )
            .strip()
            .lower()
        )
        # "yes" at the start of the reply counts as supported.
        is_supported = verdict_raw.startswith("yes") or verdict_raw.startswith("نعم")
        supported += 1 if is_supported else 0
        verdicts.append({"claim": claim, "supported": bool(is_supported)})

    score = supported / len(claims)
    return FaithfulnessResult(
        score=score,
        n_claims=len(claims),
        n_supported=supported,
        backend="builtin",
        claims=verdicts,
    )


# ==========================================================
# Backend 2: RAGAS Faithfulness (used when ragas is installed)
# ==========================================================
# Cache of built RAGAS scorers, keyed by the judge model identity. Building
# a scorer re-creates its prompts, and the experiment scores 25 questions x
# N configurations with the same judge, so the scorer is built once.
_RAGAS_SCORERS = {}


def _ragas_scorer(llm, faithfulness_cls, wrapper_cls):
    """Return a RAGAS Faithfulness scorer wrapping the project's judge LLM."""
    key = (type(llm).__name__, getattr(llm, "model", None))
    if key not in _RAGAS_SCORERS:
        _RAGAS_SCORERS[key] = faithfulness_cls(llm=wrapper_cls(llm))
    return _RAGAS_SCORERS[key]


def _ragas_faithfulness(question, answer, contexts, llm):
    """Score faithfulness with the real RAGAS ``Faithfulness`` metric.

    Verified empirically against the installed ragas 0.4.3:

    * ``ragas.metrics.Faithfulness`` + ``LangchainLLMWrapper`` WORKS with
      the project's Gemini chat model and returns a numeric score.
    * ``ragas.metrics.collections.Faithfulness`` (the newer, non-deprecated
      module) REJECTS ``LangchainLLMWrapper``: it raises
      "Collections metrics only support modern InstructorLLM", and its
      ``llm_factory(provider="google")`` route additionally requires
      LiteLLM, which the project does not depend on.

    So the deprecated module path is used deliberately: it is the only
    RAGAS Faithfulness entry point in 0.4.3 that can drive a non-OpenAI
    provider (Gemini) without adding another LLM gateway dependency. It
    is the same RAGAS metric implementation (same prompts, same
    supported / total-claims arithmetic), and the deprecation warning is
    silenced on purpose so it does not pollute the experiment output.

    Any failure is raised as RagasEvaluationError. There is no fallback
    to the built-in judge: a RAGAS score must always be a RAGAS score.
    """
    # 1) Make RAGAS importable (no LangChain downgrade; see ragas_compat).
    ragas_compat.ensure_ragas_importable()

    try:
        with warnings.catch_warnings():
            # Intentional: this module path is deprecated but is the only
            # one that supports a non-OpenAI provider in ragas 0.4.3.
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.dataset_schema import SingleTurnSample
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import Faithfulness

        scorer = _ragas_scorer(llm, Faithfulness, LangchainLLMWrapper)
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=list(contexts),
        )
        score = asyncio.run(scorer.single_turn_ascore(sample))
    except RagasEvaluationError:
        raise
    except Exception as error:  # noqa: BLE001 - re-raised as a clear error.
        raise RagasEvaluationError(
            f"RAGAS Faithfulness failed ({type(error).__name__}: {error}). "
            "The experiment requires the real RAGAS metric and will not "
            "fall back to the built-in judge."
        ) from error

    # An undefined (NaN/None) score is a failure, never a silent 0.
    if score is None or score != score:
        raise RagasEvaluationError(
            "RAGAS returned an undefined faithfulness score for this sample."
        )

    return FaithfulnessResult(score=float(score), backend="ragas")


# ==========================================================
# Public entry point
# ==========================================================
def evaluate_faithfulness(
    question,
    answer,
    contexts,
    llm=None,
    backend=DEFAULT_BACKEND,
    max_claims=25,
):
    """Return the faithfulness score of one answer against its contexts.

    ``question`` is accepted for completeness and logging only - the
    metric itself is computed from the answer and the retrieved
    contexts, and the golden ground truth is never passed in.

    ``llm`` is the JUDGE model. When it is not supplied, the project's
    Faithfulness judge is created (the configured Gemini model) - the
    generation role is never used as the judge.

    ``backend`` defaults to ``DEFAULT_BACKEND`` ("ragas"), the real
    RAGAS Faithfulness metric. The ``"builtin"`` judge is only used when
    it is requested explicitly, and a RAGAS failure is NEVER turned into
    a built-in score.
    """
    # The judge is a separate model from the answer generator.
    if llm is None:
        llm = llm_module.create_judge_llm()

    backend = backend or DEFAULT_BACKEND

    # Normalise the contexts (a single string is accepted too).
    if isinstance(contexts, str):
        contexts = [contexts]
    contexts = [context for context in contexts if context and context.strip()]

    # An empty answer cannot be faithful.
    if not answer or not answer.strip():
        return FaithfulnessResult(score=0.0, backend=backend)

    # No retrieved context at all: any claim would be unsupported.
    if not contexts:
        return FaithfulnessResult(score=0.0, backend=backend)

    # Active evaluator: the real RAGAS metric. Errors propagate.
    if backend == "ragas":
        return _ragas_faithfulness(question, answer, contexts, llm)

    # Explicit opt-in only (NOT used by the experiment).
    if backend == "builtin":
        return _builtin_faithfulness(question, answer, contexts, llm, max_claims)

    raise ValueError(
        f"unknown faithfulness backend {backend!r}; "
        f"expected one of {sorted(FAITHFULNESS_LABELS)}"
    )


def aggregate_faithfulness(scores):
    """Return the OVERALL Faithfulness score for the whole question set.

    This is the second of two levels, and the only one reported as the
    run's metric:

        level 1 - evaluate_faithfulness() returns one RAGAS Faithfulness
                  score per question, computed independently against
                  that question's own retrieved contexts;
        level 2 - this function combines those per-question scores into
                  the single overall score for the whole batch.

    Aggregation rule: the arithmetic MEAN of the per-question scores,
    i.e. every question contributes equally regardless of how many
    claims it produced. A question with many claims does not get more
    weight than a one-sentence answer, and no configuration can raise
    its overall score by making some single question very long.

    Samples that carry no usable score (None, or a NaN that some metric
    backends can return) are EXCLUDED from the mean rather than being
    silently counted as 0.0, because a 0.0 would mean "the answer was
    unfaithful" while a missing score means "nothing was measured" -
    two very different claims. An empty batch scores 0.0.

    This function only aggregates. It deliberately applies no
    threshold, no PASS/FAIL verdict and no confidence interval: those
    are interpretation decisions and do not belong to the metric.
    """
    usable = []
    for score in scores:
        # Drop only the samples that were never measured.
        if score is None:
            continue
        # NaN != NaN: a NaN score is undefined, not a number to average.
        if isinstance(score, float) and score != score:
            continue
        usable.append(float(score))

    if not usable:
        return 0.0
    return sum(usable) / len(usable)


def format_contexts(retrieved_documents):
    """Join retrieved documents exactly the way the project's chain does.

    Reuses rag_project.generation.chain.format_docs, so the evaluator
    sees precisely the text (and order) the RAG prompt saw.
    """
    from rag_project.generation.chain import format_docs

    return format_docs(retrieved_documents)
