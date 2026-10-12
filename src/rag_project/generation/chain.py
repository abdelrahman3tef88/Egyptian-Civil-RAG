"""RAG chain - the main integration layer of the pipeline.

Connects the retrieval stage with the generation stage:

    Question -> Retriever -> Retrieved Chunks -> format_docs()
             -> Prompt Template -> LLM -> Answer

Two entry points:
  - create_rag_chain(retriever, prompt, llm): compose given components
  - build_rag_chain(): build everything from configs/config.yaml
    (vector store -> retriever -> prompt -> LLM -> chain) so callers
    such as the FastAPI app only need one call.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional

from rag_project.generation.prompts import prompt


# ==========================================================
# Convert Retrieved Documents into One Context String
# ==========================================================
def format_docs(docs):
    """Join the retrieved chunks into the single context block.

    Each chunk is prefixed with its article number, taken from the
    document metadata, in the form::

        Article 51:
        <page_content>

    WHY the label is needed: ``article_number`` lives ONLY in the
    document metadata and never in ``page_content``. Sending the raw
    text alone therefore gives the LLM no way to answer questions such
    as "which article contains this provision?", and it refuses with
    "I don't know based on the provided documents." even when the
    correct article is the very first retrieved chunk. Prefixing each
    chunk with its number lets the model both identify the article and
    quote its text.

    Guarantees:
      * the retrieval order is preserved exactly as retrieved,
      * ``page_content`` is passed through verbatim - never edited,
        re-ordered, truncated or stripped,
      * the metadata itself is only read, never modified,
      * no article number is hard-coded; a document whose metadata has
        no ``article_number`` (or a None value) falls back to its plain
        ``page_content``, so the pipeline keeps working on documents
        that carry no article number at all.
    """
    # The retriever returns a list of Document objects.
    parts = []
    for doc in docs:
        # Read the article number from the metadata; tolerate a missing
        # key, a None value, or metadata that is absent entirely.
        number = (getattr(doc, "metadata", None) or {}).get("article_number")
        page_content = doc.page_content

        if number is None:
            # No usable article number: send the text exactly as it is.
            parts.append(page_content)
        else:
            parts.append(f"Article {number}:\n{page_content}")

    # Join the labelled chunks into one context block.
    return "\n\n".join(parts)


# ==========================================================
# Extract Meaningful Legal Citations from Retrieved Documents
# ==========================================================
def format_sources(documents):
    """Return the meaningful legal article citations from *documents*.

    The API contract returns the `sources` list as legal article
    citations (e.g. ``["Article 51", "Article 119"]``), NOT Chroma
    chunk IDs. Each value is taken from the retrieved document's
    ``metadata["article_number"]`` - the same field that
    :func:`format_docs` uses - so the citation is grounded in the
    actual document, never invented.

    Guarantees:
      * Order follows retrieval order.
      * Duplicates are removed while preserving order.
      * A document without a usable ``article_number`` is skipped.
      * Nothing is fabricated: if no document carries an article number,
        an empty list is returned.
    """
    sources = []
    for doc in documents:
        metadata = getattr(doc, "metadata", None) or {}
        article_number = metadata.get("article_number")
        if article_number is None:
            continue
        # Normalise int/float article numbers to ``Article N``.
        if isinstance(article_number, (int, float)):
            sources.append(f"Article {int(article_number)}")
        else:
            sources.append(f"Article {article_number}")

    seen = set()
    unique = []
    for source in sources:
        if source not in seen:
            seen.add(source)
            unique.append(source)
    return unique


# ==========================================================
# Build the Complete RAG Chain
# ==========================================================
def create_rag_chain(retriever, prompt, llm, context: Optional[str] = None):
    """Compose retrieval + prompt + LLM into one callable chain.

    Args:
        retriever: the (language-filtered) retriever that supplies context.
        prompt: prompt template.
        llm: model that produces the answer.
        context: optional fixed context string. When provided, the chain uses
            this exact context for generation without retrieving again. This is
            how the evaluation pipeline guarantees that the generated answer,
            the RAGAS ``retrieved_contexts`` and the saved contexts are the same
            documents in the same order.
    """
    if context is not None:
        # No retrieval: reuse the already-filtered context delivered by the
        # caller so generation and scoring see the identical documents.
        return lambda inputs: (prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": inputs["question"]}
        )
    rag_chain = (
        {
            # Retrieved chunks become the prompt's context
            "context": retriever | format_docs,
            # The user's question passes straight through
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def create_rag_chain_with_context(context, prompt, llm):
    """Compose the final answer step with a pre-fetched context (no retrieval).

    This is the correct way to avoid duplicate retrieval: ``documents`` are
    fetched once with the language-filtered retriever, ``context`` is built
    from them once with :func:`format_docs`, and the same values are handed
    both to generation (through this chain) and to scoring/reporting, so all
    three stages see the identical documents in the identical order.
    """

    def answer_with_context(inputs):
        return (prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": inputs["question"]}
        )

    return answer_with_context


# ==========================================================
# Build the Whole Pipeline From Configuration
# ----------------------------------------------------------
# This is the integration layer used by the API: it wires the
# configured Chroma collection, the retriever, the prompt, and
# the LLM together, so app.py never touches indexing/retrieval
# internals. Imports are done inside the function so importing
# this module (e.g. for tests of format_docs) stays lightweight.
# ==========================================================
def build_rag_components():
    """ابني الـretriever الموجود ونرجّعه مع الـprompt.

    دي النسخة "المكوّنة" من build_rag_chain(): نفس الـwiring بالظبط
    (embedding model -> Chroma -> retriever) بس من غير LLM.

    محتاجينها للـstreaming endpoint في bentoml_service: هناك الـprompt
    بيتبني عادي، بس الرد لازم يرجع token-by-token من vLLM مش كنص واحدة.

    الخطوة الوحيدة الناقصة عندها هي الخطوة اللي قبل الـLLM،
    يعني مفيش تكرار لـretrieval ولا إعادة بناء لـindex.
    """
    # Imported here to keep module import cheap and side-effect free.
    from rag_project.indexing import embeddings, vector_store
    from rag_project.retrieval import retriever as retriever_module

    # Embedding model (same one that built the index).
    embedding_model = embeddings.create_embedding_model()
    # Persisted Chroma collection built by the indexing stage.
    vector_database = vector_store.load_vector_store(embedding_model)
    # Retriever with the configured search type and top_k.
    retriever = retriever_module.create_retriever(vector_database)

    return retriever, prompt


# ==========================================================
# vLLM adapter (PromptValue -> str, used by the serving path)
# ==========================================================
def build_vllm_llm():
    """Return the RunnableLambda that turns the prompt's PromptValue
    into the generated answer string via the remote vLLM server.

    This is the vLLM analogue of :func:`create_llm` in
    ``rag_project/generation/llm.py``: it is the ONLY step that
    changes between the Gemini path and the vLLM path. The retriever,
    ``format_docs`` and the existing prompt are all unchanged.
    """
    # Imported inside the function so importing this module stays cheap.
    from langchain_core.runnables import RunnableLambda

    from service.llm_client import generate_from_prompt

    return RunnableLambda(generate_from_prompt)


def build_vllm_chain(retriever, prompt, vllm_llm):
    """Compose retrieval + prompt + vLLM LLM into one callable chain.

    This is the vLLM counterpart of :func:`create_rag_chain`, used by
    the BentoML streaming endpoint. The retriever, prompt and
    ``format_docs`` are exactly the same as the Gemini path.
    """
    return create_rag_chain(retriever, prompt, vllm_llm)


def build_rag_chain():
    """Build the full RAG chain from configs/config.yaml."""
    # Same retrieval wiring as the streaming path (single source of truth).
    retriever, prompt_template = build_rag_components()

    # ----------------------------------------------------------
    # LLM = configured Gemini generation model (generation/llm.py).
    # Only this step changed:
    #   * retriever unchanged,
    #   * format_docs() unchanged,
    #   * existing prompt (generation/prompts.py) unchanged,
    #   * only the answering model is the configured Gemini LLM.
    # No vLLM server is required for this path.
    # ----------------------------------------------------------
    from rag_project.generation import llm as llm_module

    llm = llm_module.create_llm()

    # The chain itself (context + question -> prompt -> LLM -> string).
    return create_rag_chain(retriever, prompt_template, llm)


def _create_vllm_llm():
    """نجيب الـvLLM adapter من service/llm_client.

    الاستيراد متعمّد جوّا الدالة (زي باقي استيرادات build_rag_chain)
    عشان استيراد chain.py يفضل خفيف، وعشان ندي error واضح لو مجلد
    service/ مش موجود.
    """
    try:
        # service/ موجود في الـproject root مش جوّا src/، فهو مش داخل
        # packages بتاعة pyproject. ده شغّال عادي طول ما اشتغلت من جذر
        # المشروع (bentoml serve / python من الـproject root).
        from service.llm_client import build_vllm_llm
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "مقدرناش نلاقي حزمة service. شغّل الأمر من جذر المشروع "
            "(اللي فيه ملف service/) وتأكد إن المجلد موجود."
        ) from error

    # RunnableLambda: (PromptValue -> str) جوّا الـLCEL chain.
    return build_vllm_llm()
