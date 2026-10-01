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
    """Join the retrieved chunks into the single context block."""
    # The retriever returns a list of Document objects.
    # We join their page_content into one text block.
    return "\n\n".join(doc.page_content for doc in docs)


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
