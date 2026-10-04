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
# Build the Complete RAG Chain
# ==========================================================
def create_rag_chain(retriever, prompt, llm):
    """Compose retrieval + prompt + LLM into one callable chain."""
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


# ==========================================================
# Build the Whole Pipeline From Configuration
# ----------------------------------------------------------
# This is the integration layer used by the API: it wires the
# configured Chroma collection, the retriever, the prompt, and
# the LLM together, so app.py never touches indexing/retrieval
# internals. Imports are done inside the function so importing
# this module (e.g. for tests of format_docs) stays lightweight.
# ==========================================================
def build_rag_chain():
    """Build the full RAG chain from configs/config.yaml."""
    # Imported here to keep module import cheap and side-effect free.
    from rag_project.indexing import embeddings, vector_store
    from rag_project.generation import llm as llm_module
    from rag_project.retrieval import retriever as retriever_module

    # Embedding model (same one that built the index).
    embedding_model = embeddings.create_embedding_model()
    # Persisted Chroma collection built by the indexing stage.
    vector_database = vector_store.load_vector_store(embedding_model)
    # Retriever with the configured search type and top_k.
    retriever = retriever_module.create_retriever(vector_database)
    # LLM with the configured model and temperature.
    llm = llm_module.create_llm()

    # The chain itself (context + question -> prompt -> LLM -> string).
    return create_rag_chain(retriever, prompt, llm)
