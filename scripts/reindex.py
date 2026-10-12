"""reindex.py — INCREMENTAL update: add a new document's articles to the
existing Chroma index WITHOUT rebuilding or replacing it.

    data/test/new_test_document.pdf
        -> loaders.open_pdf + extraction.extract_articles  (existing ingestion)
    data/processed/new_article.json  (NEW output, articles.json untouched)
        -> chunking.load_articles + split_documents        (existing chunking)
    chunks -> embeddings.create_embedding_model            (existing BGE-M3)
        -> vector_store.load_vector_store                  (opens EXISTING collection)
    db.add_documents(...)  =>  ADD new vectors, KEEP existing vectors

Does NOT delete the collection, rebuild, remove vectors, or overwrite
data/processed/articles.json.  Run:  uv run python scripts/reindex.py
"""

import json
import sys
import time
from collections import defaultdict

from rag_project.config import settings
from rag_project.indexing import chunking, embeddings, vector_store
from rag_project.ingestion import extraction, loaders

TEST_PDF = settings.PROJECT_ROOT / "data" / "test" / "new_test_document.pdf"
NEW_JSON = settings.PROJECT_ROOT / "data" / "processed" / "new_article.json"


def main():
    started_at = time.perf_counter()

    # 1) Open the NEW test PDF with the existing loader.
    if not TEST_PDF.exists():
        print(f"ERROR: test PDF not found: {TEST_PDF}")
        return 1
    document = loaders.open_pdf(TEST_PDF)
    page_count = document.page_count

    # 2) Extract bilingual article records (existing extraction).
    articles, preamble, _rejected = extraction.extract_articles(document)
    document.close()
    print(f"Extracted {len(articles)} article records from {TEST_PDF.name}")

    # 3) Write the NEW processed output (articles.json is untouched).
    payload = {
        "source_pdf": str(TEST_PDF),
        "page_count": page_count,
        "preamble": preamble,
        "articles": articles,
    }
    NEW_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(NEW_JSON, "w", encoding="utf-8") as out_file:
        json.dump(payload, out_file, ensure_ascii=False, indent=2)
    print(f"Wrote {len(articles)} articles to {NEW_JSON}")

    # 4) Chunk with the existing chunking logic.
    documents = chunking.load_articles(json_path=NEW_JSON)
    chunks = chunking.split_documents(documents)
    print(f"Produced {len(chunks)} chunks")
    # 5) Embedding model (model name / device from config).
    embedding_model = embeddings.create_embedding_model()

    # 6) Open the EXISTING Chroma collection (never create/replace).
    if not vector_store.index_exists():
        print("ERROR: no existing index. Run scripts/build_index.py first.")
        return 1
    db = vector_store.load_vector_store(embedding_model)

    # 7) Record the BEFORE count (authoritative read).
    before_ids = db.get()["ids"]
    before_count = len(before_ids)
    print(f"Vectors BEFORE: {before_count}")

    # 8) New IDs get a "newdoc_" prefix so they can never collide with
    #    the existing UUID ids already stored in the collection.
    per_key = defaultdict(int)
    new_ids = []
    for chunk in chunks:
        key = (chunk.metadata["article_number"], chunk.metadata["language"])
        idx = per_key[key]
        per_key[key] += 1
        new_ids.append(f"newdoc_{key[0]}_{key[1]}_{idx}")

    # 9) ADD the new vectors (no delete, no rebuild).
    db.add_documents(documents=chunks, ids=new_ids)
    vector_store.save_vector_store(db)

    # 10) Re-open the persisted collection and verify final counts.
    reloaded = vector_store.load_vector_store(embedding_model)
    after_ids = reloaded.get()["ids"]
    after_count = len(after_ids)
    added_count = after_count - before_count
    old_preserved = set(before_ids).issubset(set(after_ids))

    elapsed = time.perf_counter() - started_at

    # 11) Summary (actual numbers only).
    print("=" * 60)
    print("Incremental Reindex Complete")
    print("=" * 60)
    print(f"Test PDF             : {TEST_PDF}")
    print(f"Processed output     : {NEW_JSON}")
    print(f"Articles extracted   : {len(articles)}")
    print(f"Chunks added         : {len(chunks)}")
    print(f"Vectors BEFORE       : {before_count}")
    print(f"Vectors AFTER        : {after_count}")
    print(f"Added (after-before) : {added_count}")
    print(f"Existing preserved   : {'PASS' if old_preserved else 'FAIL'}")
    print(f"Collection           : {vector_store.COLLECTION_NAME}")
    print(f"Persist directory    : {vector_store.PERSIST_DIRECTORY}")
    print(f"Time                 : {elapsed:.2f}s")
    print("=" * 60)
    return 0 if old_preserved else 1


if __name__ == "__main__":
    sys.exit(main())
