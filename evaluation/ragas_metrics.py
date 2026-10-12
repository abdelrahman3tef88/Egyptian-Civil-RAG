"""evaluation/ragas_metrics.py - configure the four RAGAS metrics.

This module ONLY wires up the four RAGAS metrics used by the final
evaluation, pointing every LLM-based metric at the project's existing
Gemini judge model:

  * Context Precision   - is the retrieved context relevant to the question?
  * Context Recall      - can the reference answer be attributed to the context?
  * Faithfulness        - is the answer supported by the retrieved context?
  * Answer Relevancy    - is the answer relevant to the question?

There is NO custom scoring logic here. Each metric is the real RAGAS
metric object; RAGAS computes the actual value via its Gemini-backed LLM
judge (and, for Answer Relevancy, its embeddings). This file just builds
and returns them so the runner stays small and readable.

RAGAS version: 0.4.3 (installed). Its top-level ``import ragas`` fails on
the project's LangChain version because RAGAS eagerly imports a Vertex AI
symbol that no longer exists; ``ragas_compat.ensure_ragas_importable()``
registers a tiny shim first (see src/rag_project/evaluation/ragas_compat.py).
"""

import warnings

# Make ``import ragas`` succeed on this LangChain stack before any ragas
# import below runs (registers the Vertex AI shim; see the module docstring).
from rag_project.evaluation import ragas_compat

ragas_compat.ensure_ragas_importable()

# ragas 0.4.3 marks ragas.metrics.* as deprecated in favour of a collections
# API that only supports an OpenAI/LiteLLM judge. The deprecated module path
# is the ONLY one that can drive the project's Gemini judge through a plain
# LangChain wrapper, so we silence the (harmless) deprecation warning.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )


def build_metrics(judge_llm, embedding_model):
    """Return the four configured RAGAS metrics as a name -> metric dict.

    ``judge_llm``       : the project's Gemini judge (llm.create_judge_llm()).
    ``embedding_model`` : the project's embedding model, used only by
                          Answer Relevancy to score generated-question
                          similarity against the original question.
    """
    # Wrap the Gemini chat model so RAGAS uses it as its LLM judge.
    # bypass_n=True prevents the wrapper from setting n=strictness on the
    # underlying Gemini model, which rejects "Multiple candidates" requests.
    ragas_llm = LangchainLLMWrapper(judge_llm, bypass_n=True)
    # Wrap the embedding model for the Answer Relevancy metric.
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    return {
        # Context Precision: ranks retrieved contexts by relevance (LLM judge).
        "context_precision": ContextPrecision(llm=ragas_llm),
        # Context Recall: can the reference answer be attributed to the context?
        "context_recall": ContextRecall(llm=ragas_llm),
        # Faithfulness: is every claim in the answer supported by the context?
        "faithfulness": Faithfulness(llm=ragas_llm),
        # Answer Relevancy: is the answer on-topic for the question?
        # strictness=1 -> RAGAS generates ONE candidate question per answer
        # (n=strictness); Gemini Flash rejects multi-candidate generation with
        # "Multiple candidates is not enabled", so we cap it at a single one.
        "answer_relevancy": AnswerRelevancy(
            llm=ragas_llm, embeddings=ragas_embeddings, strictness=1
        ),
    }
