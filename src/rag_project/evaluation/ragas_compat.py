"""Make ``import ragas`` work without changing any LangChain version.

Why this file exists
--------------------
RAGAS 0.4.3 (the latest release) eagerly imports Vertex AI support at
module load time::

    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI

The project's ``langchain-community`` is 0.4.2, where the
``chat_models.vertexai`` module no longer exists (it moved to the
standalone ``langchain-google-vertexai`` package), so the import raises
``ModuleNotFoundError`` and ``import ragas`` fails.

RAGAS uses ``ChatVertexAI`` for exactly one thing: an ``isinstance``
check in ``MULTIPLE_COMPLETION_SUPPORTED``. This project never uses
Vertex AI - the judge is Gemini - so the symbol is irrelevant here.

The only ways to import RAGAS are therefore:

  a) downgrade langchain-community / langchain-core so the old Vertex AI
     module comes back (forbidden: it destabilises the whole LangChain
     stack), or
  b) register a placeholder for the one missing module before RAGAS is
     imported (this file).

This module implements (b). It is deliberately minimal and explicit:

  * it never modifies an installed package on disk,
  * it only runs if the genuine module is missing,
  * it is called from exactly one place (the RAGAS backend of
    rag_project/evaluation/faithfulness.py), before ``import ragas``,
  * if anything about the environment differs from what is expected, it
    raises instead of pretending RAGAS is available.
"""

import sys
import types


def ensure_ragas_importable():
    """Register the missing Vertex AI shim when langchain-community dropped it.

    Returns the ``ChatVertexAI`` placeholder class that RAGAS will see.
    """
    # Already fine (older langchain-community that still ships Vertex AI)?
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI

        return ChatVertexAI
    except ModuleNotFoundError:
        pass

    # The module is genuinely missing: build a placeholder for it.
    import langchain_community.chat_models as chat_models

    if hasattr(chat_models, "vertexai"):
        # A partially present package - do not guess.
        raise RuntimeError(
            "langchain_community.chat_models.vertexai is not importable but "
            "is registered as an attribute; refusing to guess its contents."
        )

    module_name = "langchain_community.chat_models.vertexai"

    class ChatVertexAI:  # pragma: no cover - placeholder, never instantiated
        """Placeholder for RAGAS' Vertex AI isinstance check.

        RAGAS only checks ``isinstance(llm, ChatVertexAI)``; the project
        judges with Gemini, so this class is never constructed.
        """

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "langchain_community ChatVertexAI is not available in this "
                "environment; RAGAS' Vertex AI backend cannot be used."
            )

    shim = types.ModuleType(module_name)
    shim.ChatVertexAI = ChatVertexAI
    shim.__doc__ = "Compatibility shim: see rag_project/evaluation/ragas_compat.py"

    sys.modules[module_name] = shim
    setattr(chat_models, "vertexai", shim)

    # Verify RAGAS can now actually be imported; if not, fail loudly rather
    # than letting the caller believe RAGAS is available.
    try:
        import ragas  # noqa: F401
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "The Vertex AI shim was registered but 'import ragas' still "
            f"fails: {type(error).__name__}: {error}. RAGAS is required for "
            "the experiment's Faithfulness metric and must not be replaced "
            "by a fallback evaluator."
        ) from error

    return ChatVertexAI
