"""LLM, driven by configs/config.yaml.

Provider and chat model come from the central configuration file;
the API key is read from the environment/.env - never hardcoded.

TWO roles are served by the project, using the same Gemini model:

  * the GENERATION model produces the RAG answer from the question +
    retrieved context, and
  * the FAITHFULNESS JUDGE model scores whether that answer is supported
    by the retrieved context.

The judge keeps its own model entry (``generation.judge_model``) and its
own factory so both roles stay visible in MLflow, but it is configured
with the same provider, model and API key as the generator.
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI

from rag_project.config import settings

# Configuration section for generation (configs/config.yaml).
# (The project's .env file is loaded by the config module, so the
# API key is already in the environment here.)
CONFIG = settings.CONFIG["generation"]

# Environment variable holding the API key (never hardcoded). Both the
# generation model and the Faithfulness judge use this one key.
GENERATION_API_KEY_VARIABLE = "GEMINAI_API_KEY"
JUDGE_API_KEY_VARIABLE = "GEMINAI_API_KEY"


# ==========================================================
# Create the generation model (config-driven)
# ==========================================================
def google_model_id(model_name):
    """Return the model id in the form the Google Gemini API expects.

    The configuration may carry a vendor-prefixed identifier such as
    "gemini/gemini-3.1-flash-lite", while Google's client wants the bare
    model id ("gemini-3.1-flash-lite"). Only a leading vendor prefix is
    removed; the model name itself is never rewritten or invented.
    """
    model_id = str(model_name).strip()
    for prefix in ("gemini/", "models/"):
        if model_id.lower().startswith(prefix):
            model_id = model_id[len(prefix) :]
            break
    if not model_id:
        raise SystemExit("configs/config.yaml: generation.model is empty.")
    return model_id


def create_llm():
    """Create the configured Gemini chat model used to answer questions."""
    return ChatGoogleGenerativeAI(
        # Generation model from the config (vendor prefix removed)
        model=google_model_id(CONFIG["model"]),
        # API key from environment/.env - never hardcoded
        google_api_key=os.getenv(GENERATION_API_KEY_VARIABLE),
        # Lower temperature = more factual answers (from the config)
        temperature=CONFIG["temperature"],
    )


# ==========================================================
# Create the Faithfulness judge model (config-driven)
# ==========================================================
def create_judge_llm():
    """Create the configured Gemini chat model used to judge faithfulness.

    Same job as always: it scores the generated answer against the
    retrieved contexts. It is configured from the same Gemini model and
    the same API key as the generator, and stays deterministic.
    """
    judge_model = CONFIG.get("judge_model")
    if not judge_model:
        raise SystemExit(
            "configs/config.yaml: generation.judge_model is missing. It must "
            "name the Faithfulness judge model (e.g. "
            "gemini/gemini-3.1-flash-lite)."
        )

    return ChatGoogleGenerativeAI(
        # Faithfulness judge model from the config (vendor prefix removed)
        model=google_model_id(judge_model),
        # API key from environment/.env - never hardcoded
        google_api_key=os.getenv(JUDGE_API_KEY_VARIABLE),
        # The judge must be deterministic, like the generation model.
        temperature=CONFIG.get("judge_temperature", CONFIG["temperature"]),
    )
