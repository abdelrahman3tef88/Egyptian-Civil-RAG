# Incremental Re-indexing Test

## 1. Test Document

| Item | Result |
|------|--------|
| Source PDF | `data/raw/القانون المصري.pdf` (170 pages, 2,599,703 bytes) |
| Generated test PDF | `data/test/new_test_document.pdf` (10 pages, 286,334 bytes) |
| Articles | ~100 real articles (pages containing articles 1–104 copied verbatim from the source; existing extraction emitted 133 records = 104 on-page + 389–417 for repealed articles via `cleaning.REPEALED_RANGES`) |

Method: `scripts/make_test_pdf.py` copies the original PDF pages that hold articles 1–100 (0-indexed pages 0–9) with `pymupdf.Document.insert_pdf`, so the two-column layout, Arabic reading order, and codepoints are byte-identical to the source. The existing ingestion (`loaders.open_pdf` + `extraction.extract_articles`) parses it unchanged. Article numbers are preserved exactly (1…104). Verified readable: 10 pages, Arabic headers `مادة N` and English headers `Article N` present.

## 2. Re-indexing

| Item | Result |
|------|--------|
| Script | `scripts/reindex.py` |
| Processed output | `data/processed/new_article.json` (78,684 bytes, 133 records) |
| Chroma collection | `egyptian_civil_code` at `artifacts/vector_store` (opened existing, not rebuilt) |
| Status | PASS |

Flow executed: `new_test_document.pdf` → `loaders.open_pdf` + `extraction.extract_articles` (existing ingestion) → `data/processed/new_article.json` → `chunking.load_articles` + `split_documents` (235 chunks) → `embeddings.create_embedding_model` (BGE-M3) → `vector_store.load_vector_store` (existing collection) → `db.add_documents` with `newdoc_`-prefixed IDs. No collection delete, no rebuild, no vector removal. `data/processed/articles.json` SHA256 unchanged before/after: `5E670F5D45A4D49481A0CCA3316A92EF2F9F1ECD4B3C3DE0D4020F68DF9C879E`.

## 3. Chroma Verification

| Metric | Result |
|--------|--------|
| Vectors before | 6348 |
| New vectors added | 235 |
| Vectors after | 6583 |
| Existing vectors preserved | PASS |

`added = after - before = 6583 - 6348 = 235`, equal to the chunk count produced from the test PDF. Existing-ID set verified as a strict subset of the after-ID set (`set(before).issubset(set(after))` → PASS). 235 new IDs carry the `newdoc_` prefix (e.g. `newdoc_100_en_0`), so they cannot collide with the existing UUID IDs.

## 4. Retrieval Verification

### Query

```
Acceptance in the case of a contract of adhesion is confined to adhesion to standard conditions which are drawn up by the offeror and which are not subject to d...
```

(Article 100 English text taken from `data/processed/new_article.json`, run through the project's `retriever.create_retriever(db)` with `search_type=similarity, k=7`.)

### Retrieved Result

| Rank | article_number | language | id |
|------|----------------|----------|-----|
| #0 | 100 | en | `96ce3b79-42b1-45d8-943e-1142b119fdcd` (existing) |
| #1 | 100 | en | `3f7c0a16-c802-4932-8548-2cddb3073003` (existing) |
| #2 | 100 | en | **`newdoc_100_en_0` (newly indexed)** |
| #3 | 100 | ar | `22b4d1f8-cc63-427f-aef0-4fd7398aef8c` (existing) |
| #4 | 100 | ar | `a8813b2e-9921-475b-bc2d-b90a21de4bfd` (existing) |
| #5 | 100 | ar | **`newdoc_100_ar_0` (newly indexed)** |
| #6 | 94 | en | `f515b644-5778-4c61-bf50-ab0310871cf9` (existing) |

`ARTICLE 100 IN RESULTS: True`; newly indexed IDs present: `['newdoc_100_ar_0', 'newdoc_100_en_0']`.

### Verification

PASS

## 5. Final Result

PASS
