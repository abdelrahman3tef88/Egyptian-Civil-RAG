"""Tests for generation.

Local tests verify the prompt and context formatting. The real
end-to-end test (question -> retriever -> prompt -> LLM -> answer)
runs only when the generation key is set and the Chroma index exists.
"""

import os

import pytest

from langchain_core.documents import Document

from rag_project.generation import chain, llm, prompts
from rag_project.indexing import embeddings, vector_store
from rag_project.retrieval import retriever as retriever_module

requires_index = pytest.mark.skipif(
    not vector_store.index_exists(),
    reason="Chroma index not built yet (run scripts/build_index.py)",
)
requires_api_key = pytest.mark.skipif(
    not os.getenv(llm.GENERATION_API_KEY_VARIABLE),
    reason=(
        f"{llm.GENERATION_API_KEY_VARIABLE} not set "
        "(add it to .env to run the LLM test)"
    ),
)



def test_prompt_has_context_and_question():
    """The prompt template asks for context + question."""
    assert "context" in prompts.prompt.input_variables
    assert "question" in prompts.prompt.input_variables


def test_format_docs_joins_chunks():
    """format_docs joins retrieved chunks into one context block."""
    documents = [
        Document(page_content="مادة الأولى"),
        Document(page_content="مادة ثانية"),
    ]
    assert chain.format_docs(documents) == "مادة الأولى\n\nمادة ثانية"


@requires_api_key
@requires_index
def test_end_to_end_rag_answer():
    """question -> retriever -> context -> prompt -> LLM -> answer."""
    # Build the chain from the configured components.
    embedding_model = embeddings.create_embedding_model()
    vector_database = vector_store.load_vector_store(embedding_model)
    retriever = retriever_module.create_retriever(vector_database)
    rag_chain = chain.create_rag_chain(retriever, prompts.prompt, llm.create_llm())

    # One real question through the whole chain.
    answer = rag_chain.invoke("ما هي الأهلية؟")

    # The chain returns a plain non-empty string (StrOutputParser).
    assert isinstance(answer, str)
    assert answer.strip() != ""


@requires_api_key
@requires_index
def test_build_rag_chain_from_config():
    """The integration layer builds a working chain from the config alone."""
    rag_chain = chain.build_rag_chain()
    answer = rag_chain.invoke("ما هي الأهلية؟")
    assert isinstance(answer, str)
    assert answer.strip() != ""
