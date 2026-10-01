"""Chunking for articles.json, driven by configs/config.yaml.

Splits the structured articles into retrieval chunks. The splitter
parameters (chunk_size / chunk_overlap) are read from the central
configuration file instead of being hard-coded.
"""

import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_project.config import settings

# Configuration section for chunking (configs/config.yaml).
CONFIG = settings.CONFIG["chunking"]


# ==========================================================
# Load articles.json as LangChain Documents (BILINGUAL)
# ----------------------------------------------------------
# The ingestion output provides one record per Egyptian Civil
# Code article, and every record now carries BOTH languages:
#     article["text"]["ar"]  -> the Arabic article text
#     article["text"]["en"]  -> the English translation
#
# One record therefore becomes up to TWO Documents: one per
# language. Each Document keeps the SAME metadata
# (article_number, is_repealed, page_start, page_end), plus a
# "language" field so a retrieved chunk still shows which
# language it came from.
#
# Nothing else changes: the splitter, chunk_size, chunk_overlap
# and libraries are exactly the same as before, and a record with
# an empty language side is skipped for that language only
# (the Arabic of Article 1022 is genuinely absent in the PDF).
# ==========================================================
def load_articles(json_path=None):
    """Read articles.json and return one Document per article language."""
    # Default to the path configured in configs/config.yaml.
    if json_path is None:
        json_path = settings.resolve_path(settings.CONFIG["data"]["articles_json"])

    # Open the ingestion output (UTF-8 keeps the Arabic text intact).
    with open(json_path, "r", encoding="utf-8") as articles_file:
        data = json.load(articles_file)

    documents = []
    # One document per language per structured article record.
    for article in data["articles"]:
        # The same metadata for both languages, built once per article.
        metadata = {
            "article_number": article["article_number"],
            "is_repealed": article["is_repealed"],
            "page_start": article["page_start"],
            "page_end": article["page_end"],
        }

        # The bilingual ingestion output stores the text as a dict with
        # the "ar" and "en" keys; both are indexed so the retriever can
        # match an Arabic question against Arabic text AND an English
        # question against English text.
        for language in ("ar", "en"):
            # Skip this language when the source has no text for it.
            article_text = article["text"].get(language, "")
            if article_text.strip() == "":
                continue

            language_metadata = dict(metadata)  # Copy, keep records independent.
            language_metadata["language"] = language  # Remember the language.

            document = Document(
                # The article text plays the role of page_content.
                page_content=article_text,
                # Metadata preserved so retrieved chunks keep their origin.
                metadata=language_metadata,
            )
            documents.append(document)
    return documents


# ==========================================================
# Create Recursive Text Splitter (config-driven)
# ----------------------------------------------------------
# chunk_size / chunk_overlap come from configs/config.yaml, so
# experiments only change the YAML file - never this code.
# ==========================================================
text_splitter = RecursiveCharacterTextSplitter(
    # Maximum number of characters in each chunk
    chunk_size=CONFIG["chunk_size"],
    # Number of overlapping characters between chunks
    chunk_overlap=CONFIG["chunk_overlap"],
    # Function used to count characters
    length_function=len,
    # Try to split using larger separators first
    separators=[
        "\n\n",   # Paragraph
        "\n",     # New line
        ".",      # Sentence
        " ",      # Space
        ""        # Character
    ],
)


# ==========================================================
# Split All Documents into Chunks
# ==========================================================
def split_documents(documents):
    """Split the article Documents into chunks."""
    chunks = text_splitter.split_documents(documents)
    return chunks
