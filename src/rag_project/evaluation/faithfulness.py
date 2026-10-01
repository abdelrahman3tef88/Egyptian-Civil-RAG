"""Faithfulness - the only evaluation metric of the experiment phase.

The RAG pipeline must NOT receive the golden ground truth: the evaluator
is given the question, the answer the pipeline produced, and the contexts
the pipeline retrieved, and nothing else.

Faithfulness is defined (as in RAGAS) as:

    number of claims in the answer that are supported by the retrieved
    contexts  /  total number of claims in the answer

Two interchangeable backends produce that number:

  * ``builtin`` (ACTIVE) - the project's own **LLM-based Faithfulness**:
    the generated answer is decomposed into atomic claims, and every
    claim is verified against the retrieved contexts. The judge model is
    the configured Gemini model (see generation.judge_model) and it
    needs no extra dependency.
  * ``ragas`` (OPTIONAL) - the RAGAS ``Faithfulness`` metric, wrapped
    around the same project LLM. Used only when the ``ragas`` package is
    importable.

Naming: because the active implementation is the project's own judge,
its scores are reported as **"LLM-based Faithfulness"** and are never
described as RAGAS scores. ``faithfulness_label()`` returns the exact
name for a backend, and the runner records it per run as the MLflow
parameter ``faithfulness_evaluator``.

Why ``builtin`` is the default: RAGAS is not currently usable in this
environment. ragas 0.4.3 imports
``langchain_community.chat_models.vertexai``, which no longer exists in
the project's langchain-community 0.4.2, so ``import ragas`` fails; making
it import would mean downgrading the project's LangChain stack. RAGAS is
therefore declared as an optional extra in pyproject.toml, and this module
falls back to the built-in judge. The backend that actually produced each
score is always reported in ``FaithfulnessResult.backend`` and recorded in
the run artifacts.

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
"""

import json
import re
from dataclasses import dataclass, field

from rag_project.generation import llm as llm_module

# The exact refusal sentence produced by the project's RAG prompt
# (generation/prompts.py, instruction 3). An answer that is an honest
# refusal asserts nothing, so it cannot be unfaithful.
REFUSAL_SENTINEL = "I don't know based on the provided documents."

# Human-readable name of each backend's metric. The MLflow metric key is
# always "faithfulness"; this is the label describing HOW it was computed.
FAITHFULNESS_LABELS = {
    "builtin": "LLM-based Faithfulness (claim decomposition + claim verification)",
    "ragas": "RAGAS Faithfulness",
}


def faithfulness_label(backend):
    """Return the honest label for a backend name.

    Anything that is not a pure built-in run (for example the recorded
    fallback "builtin (ragas unavailable: ImportError)") is reported as
    the built-in judge, because that is what actually produced the score.
    """
    return FAITHFULNESS_LABELS.get(
        str(backend).split(" ")[0], FAITHFULNESS_LABELS["builtin"]
    )



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
def _ragas_faithfulness(question, answer, contexts, llm):
    """Score faithfulness with the RAGAS Faithfulness metric."""
    # Imported lazily so this module stays importable without RAGAS.
    import asyncio

    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    scorer = Faithfulness(llm=LangchainLLMWrapper(llm))
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=list(contexts),
    )
    score = asyncio.run(scorer.single_turn_ascore(sample))

    # A NaN score means RAGAS could not judge the sample; treat it as a
    # hard failure so the caller falls back instead of logging NaN.
    if score is None or score != score:
        raise ValueError("RAGAS returned an undefined faithfulness score")

    return FaithfulnessResult(score=float(score), backend="ragas")


# ==========================================================
# Public entry point
# ==========================================================
def evaluate_faithfulness(
    question,
    answer,
    contexts,
    llm=None,
    backend="builtin",
    max_claims=25,
):
    """Return the faithfulness score of one answer against its contexts.

    ``question`` is accepted for completeness and logging only - the
    metric itself is computed from the answer and the retrieved
    contexts, and the golden ground truth is never passed in.

    ``llm`` is the JUDGE model. When it is not supplied, the project's
    Faithfulness judge is created (the configured Gemini model) - the
    generation role is never used as the judge.
    """
    # The judge is a separate model from the answer generator.
    if llm is None:
        llm = llm_module.create_judge_llm()


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

    # Requested RAGAS backend.
    if backend == "ragas":
        try:
            return _ragas_faithfulness(question, answer, contexts, llm)
        except Exception as error:  # noqa: BLE001 - keep the run alive.
            # Fall back to the built-in judge, but record the fallback in
            # the result so the logged score stays transparent.
            result = _builtin_faithfulness(
                question, answer, contexts, llm, max_claims
            )
            result.backend = f"builtin (ragas unavailable: {type(error).__name__})"
            return result

    # Default: minimal built-in judge.
    return _builtin_faithfulness(question, answer, contexts, llm, max_claims)


def aggregate_faithfulness(scores):
    """Average the per-question scores, ignoring unanswered samples."""
    usable = [score for score in scores if score is not None]
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

