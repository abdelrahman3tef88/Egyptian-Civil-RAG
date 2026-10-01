# Day 1 — RAG Foundation (Complete Technical Build Log)

> Scope: this document is the complete Day 1 build record — environment,
> configuration, implementation, validation, and commands. Everything below
> describes the **actual project state** (verified from the files on disk);
> nothing is aspirational and no command, file, or configuration is invented.

## 1. Day 1 Objective

Day 1 built the working foundation of a production-oriented RAG application
over the Egyptian Civil Code (القانون المصري): the Python environment and
packaging, central YAML configuration, the full offline pipeline from the
source PDF to a persisted vector index, and the runtime path from a user
question to a generated answer served over HTTP — all tested end to end.

### Offline / build-time stages (run when data or the index changes)

```
PDF (data/raw/القانون المصري.pdf)
 ↓
Ingestion (loaders → cleaning → extraction)
 ↓
articles.json (data/processed/)
 ↓
Indexing (chunking → embeddings → vector store)
 ↓
Vector Database (artifacts/vector_store/, Chroma collection)
```

### Runtime stages (run per user request)

```
User Question
 ↓
FastAPI (src/rag_project/api/app.py)
 ↓
RAG Chain (generation/chain.py: build_rag_chain())
 ↓
Retriever → Chroma similarity search → top-3 chunks
 ↓
Prompt (retrieved context + question) → LLM (Gemini gemini-3.5-flash)
 ↓
Answer
```

## 2. Day 1 Pipeline Overview

```
PDF
↓
Ingestion
↓
Structured JSON
↓
Chunking
↓
Embeddings
↓
Vector Store
↓
Retriever
↓
Prompt
↓
LLM
↓
RAG Chain
↓
FastAPI
↓
Tests
```

## 3. Development Environment

Setup was done in a Windows PowerShell terminal, in the project folder
`D:\Egyptian Civil RAG Project` (referred to below as the project root).

### 3.1 Check Python (verified: Python 3.11.9, requires-python `>=3.11`)

```powershell
python --version
```

### 3.2 Create the virtual environment

The `.venv/` directory was created with the standard library tool (this is
exactly what `.venv/pyvenv.cfg` records: `command = ... python.exe -m venv
...\.venv`):

```powershell
python -m venv .venv
```

- **What `.venv` is:** an isolated Python installation for this project
  (its own interpreter + installed packages), living inside the project
  folder so the setup is reproducible.
- **Why it exists:** project packages never conflict with the system
  Python or with other projects.

### 3.3 Use the environment

The project uses the environment's interpreter directly (this is why the
validation commands below call `.venv\Scripts\python.exe`):

```powershell
.venv\Scripts\python.exe --version   # -> Python 3.11.9
```

(`uv run ...` selects the same project environment automatically; both
invocation styles appear in the command reference in §30.)

### 3.4 The `uv` package manager (verified: uv 0.12.18)

`uv` is the project's package/dependency manager. In this project `uv` is
responsible for: adding dependencies (`uv add`), resolving the full
dependency graph into `uv.lock`, installing/synchronizing packages into
`.venv` (`uv sync`), and running commands inside the project environment
(`uv run`).

## 4. Python Project Initialization

`pyproject.toml` was **created manually** as a text file (no `uv init`
command was used — do not document one here). It declares a
setuptools-based package with an `src/` layout
(`[tool.setuptools.packages.find] where = ["src"]`), distribution name
`egyptian-civil-rag-project`, version `0.1.0`. The `src/` tree and every
`__init__.py` were then created manually to match that layout.

## 5. `pyproject.toml` (actual content, verified)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "egyptian-civil-rag-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [ ...see section 7... ]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

What each part means:

- **`[build-system]`** — how the package itself is built (setuptools).
- **`[project]`** — metadata: `name` (distribution name),
  `version`, `requires-python` (minimum interpreter), `dependencies`
  (the direct runtime requirements every install gets).
- **`[dependency-groups] dev`** — development-only tools (`pytest`, the
  test runner), not runtime dependencies.
- **`[tool.setuptools.packages.find] where = ["src"]`** — setuptools
  discovers importable packages under `src/` (the `src/` layout).
- **`[tool.pytest.ini_options]`** — `testpaths = ["tests"]` (test
  location) and `pythonpath = ["src"]` (so `import rag_project`
  resolves).

There are **no** `[project.optional-dependencies]`, coverage, lint, or
mypy sections — only what is shown above.

Why the file matters: it is the single source of truth for what the
project is and what it needs.

### The four related files, precisely distinguished

```text
pyproject.toml → project metadata + DIRECT dependencies (edited by hand / `uv add`)
uv.lock        → EXACT resolved dependency versions (1,036,454 bytes; generated by uv, never hand-edited)
.venv/         → the installed Python environment (interpreter + packages)
.env           → local secrets/configuration (GOOGLE_API_KEY); never committed
```

## 6. `uv.lock` (1,036,454 bytes)

`uv.lock` is the exact resolved dependency graph (every transitive package
pinned to one version). Relationship:

```text
pyproject.toml
      ↓ dependency resolution (uv add / uv lock / uv sync)
uv.lock              ← exact versions recorded
      ↓ installation
.venv                ← those exact versions installed
```

Generated/updated by `uv` commands in this project: `uv lock` (initial
resolve), `uv sync` (environment sync), and every `uv add ...` in §7. It
must not be edited by hand.

## 7. Installing Dependencies (actual, from `pyproject.toml`)

Runtime dependencies and why each is here:

| Package | Purpose in this project |
|---|---|
| `pymupdf>=1.24` | Coordinate-level PDF extraction (dynamic Arabic/English column split) |
| `pypdf>=6.19` | General PDF utilities |
| `sentence-transformers>=6.1` | Embedding-model backend (MiniLM) |
| `langchain`, `langchain-core` | RAG primitives: Documents, runnables, output parsers |
| `langchain-text-splitters>=1.1.2` | `RecursiveCharacterTextSplitter` for chunking |
| `langchain-huggingface>=1.2.2` | `HuggingFaceEmbeddings` wrapper |
| `langchain-community>=0.4.2` | Document-loader utilities in the stack |
| `langchain-chroma>=1.1.0` | Maintained Chroma vector-store integration |
| `chromadb>=1.5.9` | The Chroma vector database engine itself |
| `faiss-cpu>=1.15.1` | Declared but **superseded**: the store was first built on FAISS, then rebuilt on Chroma (§20) |
| `langchain-google-genai>=4.4.0`, `google-genai>=2.25` | Gemini LLM (`gemini-3.5-flash`) via LangChain |
| `fastapi>=0.141`, `uvicorn[standard]>=0.53` | HTTP API layer |
| `pydantic>=2.13`, `pydantic-settings>=2.15` | Request models / settings support |
| `python-dotenv>=1.2.3` | Load `.env` secrets into the environment |
| `pyyaml>=6.0.3` | Parse `configs/config.yaml` |
| `pytest>=8.0` (dev group) | Test runner |

### Actual `uv add` sequence used (each modified `pyproject.toml` + `uv.lock` + `.venv`)

```powershell
uv add langchain langchain-community langchain-core langchain-huggingface `
       langchain-text-splitters sentence-transformers faiss-cpu `
       langchain-google-genai pymupdf pypdf
uv add langchain-chroma
uv add pyyaml
uv add python-dotenv
```

What `uv add <package>` does: adds the requirement to `pyproject.toml`,
re-resolves the graph into `uv.lock`, then installs/synchronizes the result
into `.venv`. (`uv sync` alone installs from an existing lockfile without
adding anything.)

## 8. Environment Variables

- **`.env.example`** — committed template documenting `GOOGLE_API_KEY=`
  (used by `generation/llm.py`). It carries no executable code; only the
  commented-out placeholders (`LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`,
  `EMBEDDING_MODEL`, `VECTOR_STORE_PATH`).
- **`.env`** — local, untracked file holding the real Gemini key.
  `config/settings.py` loads it once via `load_dotenv()` so every module
  (scripts, tests, API) sees the same environment variables.
- Secrets are never committed: `.gitignore` contains `.env`.

> ⚠️ **Security note (real finding):** during documentation inspection the
> current `.env.example` was found to contain a **real-looking
> `GOOGLE_API_KEY` value** (not an empty placeholder). Treat that key as
> compromised: rotate it in the Google console, and restore `.env.example`
> to an empty `GOOGLE_API_KEY=` placeholder. Never commit real secrets.

## 9. Git / `.gitignore`

**Status: this project is not a git repository** (no `.git/` directory
exists), but `.gitignore` already defines the commit policy for when it
becomes one. Actual entries and what they protect:

```text
__pycache__/  *.py[cod]  *.egg-info/  .eggs/  build/  dist/   → build caches
.venv/  venv/  env/                                     → local environments
.env                                                   → secrets
.vscode/  .idea/  .ipynb_checkpoints/                  → editor/notebook noise
.pytest_cache/  .coverage  htmlcov/                    → test/tooling caches
*.pdf                                                  → user PDFs are not committed
data/raw/*  data/processed/* (except .gitkeep files)   → data contents not committed
artifacts/vector_store/* (except .gitkeep)             → generated index not committed
```

**Committed (tracked) project files:** `pyproject.toml`, `uv.lock`,
`README.md`, `README_D1.md`, `.gitignore`, `.env.example`, `configs/`,
`src/`, `scripts/`, `tests/`, `notebooks/`, the reference notebook
`Standard_RAG(_Q&A_Courses)(2).ipynb`, and directory `.gitkeep` files.
**Never committed:** `.env`, the PDF, `articles.json`, the Chroma index
(`chroma.sqlite3` + HNSW shard files).

## 10. Project Structure (actual, verified from disk)

```text
rag-project/
│
├── pyproject.toml
├── uv.lock
├── README.md
├── README_D1.md
├── .gitignore
├── .env
├── .env.example
├── Standard_RAG(_Q&A_Courses)(2).ipynb   (reference implementation, root)
│
├── configs/
│   └── config.yaml                        (central hyperparameters/paths)
│
├── src/
│   └── rag_project/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py            (loads configs/config.yaml + .env)
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py             (PDF path/open/check_pdf_type → 1/0)
│       │   ├── cleaning.py            (conservative cleaning, repealed ranges)
│       │   └── extraction.py          (columns, order, headers, state tracker)
│       ├── indexing/
│       │   ├── __init__.py
│       │   ├── chunking.py            (config chunk_size/overlap → chunks)
│       │   ├── embeddings.py          (config model/device → vectors)
│       │   └── vector_store.py        (Chroma create/load, collection)
│       ├── retrieval/
│       │   ├── __init__.py
│       │   └── retriever.py           (config search_type/top_k → top-k)
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── prompts.py             (RAG prompt template)
│       │   ├── llm.py                 (Gemini gemini-3.5-flash, config model/temp)
│       │   └── chain.py               (integration layer: build_rag_chain())
│       └── api/
│           ├── __init__.py
│           └── app.py                 (FastAPI GET /health, POST /ask via chain)
│   ├── scripts/
│   ├── extract_data.py          (ingestion orchestration — implemented)
│   ├── build_index.py           (indexing orchestration — implemented)
│   └── reindex.py               (Day-1 docstring/placeholder — NOT implemented)
│
│   ├── data/
│   │   ├── raw/
│   │   │   ├── .gitkeep
│   │   │   └── القانون المصري.pdf   (2,599,703 bytes, user-provided, untracked)
│   │   └── processed/
│   │       ├── .gitkeep
│   │       └── articles.json          (626,908 bytes, 1091 articles, generated, untracked)
│   │
│   ├── artifacts/
│   │   └── vector_store/
│   │       ├── .gitkeep
│   │       ├── chroma.sqlite3         (7,745,536 bytes, generated, untracked)
│   │       └── 66639adc-.../         (Chroma HNSW shard: header.bin,
│   │                                  length.bin, link_lists.bin,
│   │                                  data_level0.bin, index_metadata.pickle)
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_ingestion.py          (10 tests — implemented)
│   │   ├── test_indexing.py           (3 tests — implemented)
│   │   ├── test_retrieval.py          (1 test — implemented)
│   │   ├── test_generation.py         (4 tests — implemented)
│   │   └── test_api.py                (3 tests — implemented)
│   │
│   └── notebooks/
│       └── exploration.ipynb          (skeleton, unused by the pipeline)
```

### Directory and file responsibilities

- **`pyproject.toml`** — package metadata + direct dependencies (setuptools `src/` layout, distribution name `egyptian-civil-rag-project 0.1.0`).
- **`uv.lock`** — resolved dependency graph (1,036,454 bytes), generated by `uv`, never hand-edited.
- **`README.md`** — minimal main project README (3 lines), expanded on future days.
- **`README_D1.md`** — this document: the complete Day 1 build log.
- **`.gitignore`** — keeps caches, environments, secrets, raw PDFs, generated data, and the index out of version control.
- **`.env` / `.env.example`** — local Gemini API key (untracked) / its committed key template.
- **`Standard_RAG(_Q&A_Courses)(2).ipynb`** — reference implementation (root): chunking, embeddings, vector DB, retriever, prompt, LLM, RAG chain. Its course dataset was never used.
- **`configs/config.yaml`** — the central hyperparameters/paths file every module reads (see §13).
- **`src/`** — the installable Python package (`rag_project`) with all reusable application components.
- **`config/`** — `settings.py` loads `configs/config.yaml` once, exposes `CONFIG`/`PROJECT_ROOT`/`resolve_path()`, and loads `.env` secrets via `load_dotenv()`.
- **`ingestion/`** — PDF loading (`loaders.py`), conservative cleaning (`cleaning.py`), Arabic column extraction + article reconstruction (`extraction.py`).
- **`indexing/`** — `chunking.py` (splitter from config values), `embeddings.py` (HuggingFace model from config), `vector_store.py` (Chroma create/load against the configured directory/collection).
- **`retrieval/`** — `retriever.py` builds the similarity retriever with `search_type`/`top_k` from config.
- **`generation/`** — `prompts.py` (prompt template), `llm.py` (Gemini client from config model/temperature, key from env), `chain.py` (`create_rag_chain()` + config-built `build_rag_chain()` integration layer).
- **`api/`** — `app.py` (FastAPI `GET /health`, `POST /ask` via the chain; no RAG logic inside).
- **`scripts/`** — executable orchestrators (`extract_data.py`, `build_index.py`); `reindex.py` remains the original Day-1 docstring-only file and is not part of the pipeline.
- **`data/raw/`** — input PDF (2,599,703 bytes, user-provided, untracked). **`data/processed/`** — generated `articles.json` (626,908 bytes, untracked).
- **`artifacts/vector_store/`** — generated Chroma index (`chroma.sqlite3` 7,745,536 bytes + HNSW shard dir, untracked).
- **`tests/`** — 21 implemented tests across the five stage files.
- **`notebooks/`** — `exploration.ipynb` skeleton, unused by the pipeline.

## 11. Separation of Responsibilities

- **`src/`** contains reusable application components — the library code
  (config, ingestion, indexing, retrieval, generation, API) imported by
  scripts, tests, and the service.
- **`configs/`** holds the one YAML file every stage reads for values;
  only secrets live in `.env`.
- **`scripts/`** contains executable workflows that orchestrate those
  components (running ingestion, building the index).
- **`tests/`** verifies individual components and the API behavior
  end-to-end (live Gemini calls where an API key is needed).
- **`data/`** contains input and processed data (PDF in, structured JSON out).
- **`artifacts/`** contains generated runtime/indexing artifacts (the Chroma
  store), regenerated by `build_index.py` as needed and never hand-edited.

## 12. Day 1 Components (implemented, not placeholders)

- **`loaders.py`** — `PDF_PATH` constant; `open_pdf()` (exists/open/pages checks); `check_pdf_type()` → 1 (digital) / 0 (scan) from sampled character volume (~4,240 chars/page measured).
- **`cleaning.py`** — conservative legal-text cleaning: whitespace collapse, invisible/bidi artifact removal, Arabic-Indic digit normalization, clause-marker parenthesis repair; `REPEALED_RANGES = [(54, 80), (389, 417)]` + `is_repealed_article()`; paragraph assembly.
- **`extraction.py`** — dynamic `mid_x = page.rect.width / 2` clipping of the right Arabic column with Y-order; validated reading-order restoration (word-order reversal + digit-run reversal — naive char reversal was tested and rejected); anchored `مادة` header regex incl. fragmented `ما دة` forms; `current_open_article` state tracker with monotonic guard; English column extraction for validation only.
- **`chunking.py`** — builds LangChain Documents from `articles.json` and splits them with the config splitter (size 1000, overlap 200); metadata flows into chunks.
- **`embeddings.py`** — wraps the config HuggingFace model (all-MiniLM-L6-v2, normalized; device auto → CPU here).
- **`vector_store.py`** — creates/opens the configured Chroma collection (`egyptian_civil_code` at `artifacts/vector_store`); `index_exists()` gate for dependent stages.
- **`retriever.py`** — builds the similarity retriever with config `search_type` and `top_k: 3`.
- **`prompts.py`** — the reference prompt (context + question variables, exact-refusal instruction).
- **`llm.py`** — Gemini client from config (`gemini-3.5-flash`, `temperature: 0`); key from `GOOGLE_API_KEY`.
- **`chain.py`** — `format_docs()` + `create_rag_chain()` composition, plus `build_rag_chain()` (one-call config wiring used by the API).
- **`app.py`** — FastAPI application with `GET /health` and `POST /ask`, both calling the RAG chain.

## 13. `configs/config.yaml` (central configuration, verified)

```yaml
chunking:    {chunk_size: 1000, chunk_overlap: 200}
embedding:   {embedding_model: sentence-transformers/all-MiniLM-L6-v2,
              device: auto, normalize_embeddings: true}
vector_db:   {type: chroma, persist_directory: artifacts/vector_store,
              collection_name: egyptian_civil_code}
retrieval:   {search_type: similarity, top_k: 3}
generation:  {model: gemini-3.5-flash, temperature: 0}
data:        {articles_json: data/processed/articles.json}
```

`config/settings.py` loads it once (`CONFIG`, `CONFIG_PATH`), resolves
relative paths against the project root, and loads `.env` secrets. Every
stage module reads its section — verified by grep that no hyperparameter
is hard-coded in `chunking`/`embeddings`/`vector_store`/`retriever`/`llm`.

## 14. Ingestion Stage (actual: 170 pages → 1091 articles)

Entry point `scripts/extract_data.py` (loads PDF → type check → `extract_articles()`
→ English cross-check → prints calculated summary + 5 validation checks →
writes `articles.json` **only if all checks pass**).

Measured results (printed by the script, not assumed): DIGITAL; 170 pages;
**1091 articles (first 1, last 1149)**; 98 multi-page (p12→13 confirmed);
**repealed range 54–80 and 389–417 are absent from this PDF edition**
(verified: jumps 53→81, 388→418 + the PDF's own repeal notes on pp. 7, 53),
so `is_repealed` flags are consistent and `repealed_count = 0`;
Arabic ratio 97.16%, 0 English-contamination tokens; English cross-check
94.32% match, 105.61% coverage; 44 chapter headings preserved, not deleted;
3 rejected backwards headers kept as body text (logged as diagnostics).

## 15. Ingestion Data Contract (`data/processed/articles.json`, 626,908 bytes)

Top level: `source_pdf`, `page_count`, `preamble` (decree text before
article 1), `articles` (one record per article, strictly increasing):

```json
{
  "article_number": 664,
  "text": "ينقضي عقد المقاولة باستحالة تنفيذ العمل المعقود عليه.",
  "page_start": 91,
  "page_end": 91,
  "is_repealed": false
}
```

(Multi-page example: `{"article_number": 120, ..., "page_start": 12,
"page_end": 13, "is_repealed": false}`.) Two articles the PDF itself
corrupts stay evidence-visible: `601` (digits stored scrambled, text kept
inside article 600) and `1022` (absent from both columns).

## 16. Ingestion Validation (`tests/test_ingestion.py`, 10 tests)

Verifies: Arabic→Western digit normalization; all real header forms incl.
fragmented `ما دة`/`م ادة`/`ماد ة` (and rejects mid-sentence mentions);
repealed-range boundaries (54/80/389/417 in, 53/81/388/418 out);
word+digit reading-order restoration; clause-marker repair (dash forms and
years like `1949 (` untouched); paragraph assembly; missing-PDF error;
`check_pdf_type == 1`; and a full-PDF integration run (monotonic numbering,
valid page spans, non-empty text, multi-page ≥ 1, repealed gaps exactly at
53→81 and 388→418). Command: `uv run pytest tests/test_ingestion.py -v`.

## 17. Indexing Stage (actual: 1091 articles → 1092 chunks → Chroma)

Entry point `scripts/build_index.py`:
`load_articles()` → `split_documents()` → `create_embedding_model()` →
`create_vector_store()` → `save_vector_store()` + summary print. Verified
run output: `Database: chroma / Persist directory: …/artifacts/vector_store /
Collection: egyptian_civil_code / Total Chunks Stored: 1092`.

Adapted from `Standard_RAG(_Q&A_Courses)(2).ipynb` cells 18/20 (splitter),
29 (MiniLM + normalize), 34/37 (store build/save), keeping the same
parameters with `data/processed/articles.json` as the dataset. Initial
build used FAISS; the store was then rebuilt on **Chroma
(`langchain-chroma`, collection `egyptian_civil_code`; `faiss-cpu` remains
declared but unused)** — see §20.

## 18. Notebook Adaptation

`Standard_RAG(_Q&A_Courses)(2).ipynb` (project root, 65 cells) is the
structural/implementation reference for chunking → embeddings → vector DB
→ retriever → prompt → LLM → RAG chain. Its course files (DOCX/PDF/TXT/CSV
loaders), course-specific cleaning (TOC-cutting, skip-first-pages), Colab
paths, and display cells were explicitly **not** reused; verification
runs proved those rules would delete or corrupt legal text. Only the
prompt's refusal phrase changed (`course materials` → `documents`),
because the data source changed; all index/retrieval/LLM values are
unchanged notebook parameters, now driven from `configs/config.yaml`.

## 19. Indexing Script (actual command used)

`scripts/build_index.py` — orchestration only (no implementation logic).
Actual command (verified output shown in §17):

```powershell
uv run python scripts/build_index.py
```

`uv run` executes the script inside the project `.venv`.

## 20. Vector Store Artifact

`artifacts/vector_store/` holds `chroma.sqlite3` (7,745,536 bytes) plus one
HNSW shard directory (`66639adc-.../`: `header.bin`, `length.bin`,
`link_lists.bin`, `data_level0.bin`, `index_metadata.pickle`), collection
`egyptian_civil_code`. Generated by `scripts/build_index.py`; never
hand-edited; never committed (per `.gitignore`). Deleted superseded FAISS
files (`index.faiss`, `index.pkl`) when the store moved to Chroma.

## 21. Retriever Stage (actual)

Flow: user question → query embedding → Chroma similarity search →
configured `top_k: 3` chunks with metadata. `retrieval/retriever.py` reads
`search_type`/`top_k` from config and returns the retriever. Verified by
`tests/test_retrieval.py` (1 test): on the real index, `ما هي الأهلية؟`
returns exactly 3 non-empty chunks, each carrying `article_number`.

## 22. Retriever Testing

Command: `uv run pytest tests/test_retrieval.py -v` → **1 passed** (skips
cleanly with a reason when the Chroma index is absent).

## 23. Generation Stage (actual files)

- **`generation/prompts.py`** — the reference prompt (context + question
  variables, exact-refusal instruction).
- **`generation/llm.py`** — Gemini client from config (`gemini-3.5-flash`,
  `temperature: 0`); key from `GOOGLE_API_KEY` in `.env` (never hard-coded).
- **`generation/chain.py`** — `format_docs()` (chunks → context block),
  `create_rag_chain()` (composition), and **`build_rag_chain()`** — the
  one-call integration layer (Chroma → retriever → prompt → LLM → parser)
  that the API uses.

## 24. RAG Chain (runtime orchestration, not build-time)

`build_rag_chain()` runs **only at request time**: it loads the persisted
Chroma collection (never rebuilds it), retrieves, formats, prompts, calls
Gemini, and returns a plain string. Ingestion and indexing never run on a
user request — they are the offline pipeline in §1.

## 25. Build/Offline vs Runtime (must-read distinction)

### Build / offline pipeline (when data or the index changes)

```text
PDF
 ↓
Ingestion (extract_data.py)
 ↓
articles.json
 ↓
Indexing (build_index.py)
 ↓
Vector Store (artifacts/vector_store, chroma.sqlite3 ...)
```

### Runtime pipeline (per user request)

```text
User Question
 ↓
FastAPI (app.py: /ask)
 ↓
RAG Chain (chain.build_rag_chain())
 ↓
Retriever → Chroma similarity → 3 chunks → context
 ↓
Prompt (context + question) → Gemini gemini-3.5-flash
 ↓
Answer
```

## 26. FastAPI Layer (actual: `src/rag_project/api/app.py`)

`app = FastAPI(title="Egyptian Civil Code RAG")` with a module-level cached
chain (`get_rag_chain()` builds once — embedding model + Chroma client are
heavy — and reuses it). **No RAG logic lives in the API.** The served path
below was verified with FastAPI's `TestClient` rather than a live server
process (no `uvicorn` serve command was needed or run).

## 27. FastAPI and the Other Components

```text
                    FASTAPI (GET /health, POST /ask)
                       │
                       ▼
                    app.py (routes + cached build_rag_chain())
                       │
                       ▼
                  RAG Chain (chain.py)
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Retriever             LLM (Gemini)
              │                 ▲
              ▼                 │
        Chroma Vector Store     │
              │                 │
              ▼                 │
        Retrieved Context       │
              └──────→ Prompt ──┘
                       │
                       ▼
                    Answer
```

Responsibilities: ingestion prepared `articles.json`; indexing prepared the
Chroma store; the retriever searches it; generation uses the context;
`chain.py` orchestrates runtime RAG; `app.py` exposes that flow over HTTP.
Verified end-to-end example (live Gemini, real index):
Q «هل ينقضي عقد المقاولة باستحالة تنفيذ العمل المعقود عليه؟» →
retrieved articles [664, 564, 313] → A: «نعم، ينقضي عقد المقاولة باستحالة
تنفيذ العمل المعقود عليه.» (matches article 664). Refusal path also
verified live: out-of-context questions return exactly
"I don't know based on the provided documents."

## 28. FastAPI Test (`tests/test_api.py`, 3 tests, all passing)

Verifies: `GET /health` → 200 `{"status": "ok"}`; `POST /ask` with `{}` →
422 (pydantic validation, no key/index needed); `POST /ask` with a real
question → 200 with echoed question + non-empty chain answer (needs
`GOOGLE_API_KEY` + built index, otherwise skipped with a reason).
Command: `uv run pytest tests/test_api.py -v`.

## 29. Complete Day 1 Pipeline (actual project)

## 30. Command Reference (all actually used, grouped, each explained)

### Environment

```powershell
python -m venv .venv
```

Creates the isolated project interpreter in `.venv/` (recorded in
`.venv/pyvenv.cfg`). Next step: run everything through it (`uv run` or
`.venv\Scripts\python.exe`) so installs never touch system Python.

```powershell
python --version
.venv\Scripts\python.exe --version   # -> Python 3.11.9 (verifies the env)
```

Checks the interpreter before building on it.

### Dependencies (each: edits `pyproject.toml` → re-resolves `uv.lock` → syncs `.venv`)

```powershell
uv add langchain langchain-community langchain-core langchain-huggingface `
       langchain-text-splitters sentence-transformers faiss-cpu `
       langchain-google-genai pymupdf pypdf
uv add langchain-chroma
uv add pyyaml
uv add python-dotenv
```

`uv add` is how every package above entered the project; `uv sync` alone
only installs from the existing lockfile (it was used for environment
syncs). `uv lock` produces the initial `uv.lock`.

### Ingestion (builds `data/processed/articles.json`)

```powershell
uv run python scripts/extract_data.py
```

Runs the PDF → articles pipeline; prints the calculated summary + 5
validation checks; writes the JSON **only if all checks pass**. Next step
after success: `build_index.py`.

### Ingestion tests (10 tests: units + full-PDF integration)

```powershell
uv run pytest tests/test_ingestion.py -v
```

Pass/fail proves the extraction rules on the real PDF.

### Build index (builds `artifacts/vector_store/`)

```powershell
uv run python scripts/build_index.py
```

Chunks articles.json (1092 chunks), embeds them, and persists the Chroma
collection. Verified output: `Total Chunks Stored : 1092`.

### Indexing tests (3 tests: load/chunk/metadata, embeddings, Chroma round-trip)

```powershell
uv run pytest tests/test_indexing.py -v
```

### Retrieval tests (verifies top-k chunks from the real index)

```powershell
uv run pytest tests/test_retrieval.py -v
```

### Generation tests (prompt/format units + live Gemini chain)

```powershell
uv run pytest tests/test_generation.py -v
```

### API tests (health, validation, live `/ask`)

```powershell
uv run pytest tests/test_api.py -v
```

`uv run` always executes inside the project `.venv`.

### What was NOT run

No `uvicorn` serve command was executed: the served path was verified with
FastAPI's `TestClient` (see §28). No `uv init` command was used (the
`pyproject.toml` was written manually). No docker/CI/CD commands exist.

## 31. Explaining the key commands (why each was needed)

- **`python -m venv .venv`** — creates the isolated interpreter; without
  it, the 20+ RAG packages would pollute or conflict with system Python.
- **`uv add <package>`** — the only way dependencies entered the project:
  it declares the requirement, re-resolves the full graph, and installs it.
  Each call in §7 changed `pyproject.toml`, `uv.lock`, and `.venv` together.
- **`uv run python scripts/extract_data.py`** — executes the ingestion
  pipeline inside `.venv`; it created `data/processed/articles.json`.
- **`uv run pytest ... -v`** — runs the test files with verbose output;
  `-v` names each passing test so the report above can list evidence per
  test, not just totals.
- **`uv run python scripts/build_index.py`** — executes indexing inside
  `.venv`; it created `chroma.sqlite3` and the HNSW shard directory.

## 32. Day 1 File Inventory (actual files only)

| File | Purpose | Stage |
|---|---|---|
| `pyproject.toml` | Metadata + direct deps, setuptools `src/` layout | Environment |
| `uv.lock` | Exact resolved versions (1,036,454 bytes) | Environment |
| `.gitignore` | Commit policy (env, secrets, PDF, data, index) | Environment |
| `.env` / `.env.example` | Real Gemini key (untracked) / its template | Environment (⚠️ key leaked into example — rotate) |
| `configs/config.yaml` | Central hyperparameters + paths | Configuration |
| `src/rag_project/config/settings.py` | Loads YAML once + `.env`; `CONFIG`, paths | Configuration |
| `src/rag_project/ingestion/loaders.py` | PDF path/open/`check_pdf_type` | Ingestion |
| `src/rag_project/ingestion/cleaning.py` | Conservative cleaning, repealed ranges | Ingestion |
| `src/rag_project/ingestion/extraction.py` | Columns, reading order, headers, state tracker | Ingestion |
| `scripts/extract_data.py` | Ingestion orchestration + validation report | Ingestion |
| `data/processed/articles.json` | 1091 article records (generated) | Ingestion output |
| `src/rag_project/indexing/chunking.py` | 1092 chunks from config values | Indexing |
| `src/rag_project/indexing/embeddings.py` | MiniLM vectors (CPU) | Indexing |
| `src/rag_project/indexing/vector_store.py` | Chroma collection create/load | Indexing |
| `scripts/build_index.py` | Indexing orchestration | Indexing |
| `artifacts/vector_store/chroma.sqlite3` + HNSW dir | Persisted index (generated) | Indexing output |
| `src/rag_project/retrieval/retriever.py` | similarity top-3 retriever | Retrieval |
| `src/rag_project/generation/prompts.py` | RAG prompt template | Generation |
| `src/rag_project/generation/llm.py` | Gemini client (config model/temp) | Generation |
| `src/rag_project/generation/chain.py` | `build_rag_chain()` integration layer | Generation |
| `src/rag_project/api/app.py` | FastAPI `/health` + `/ask` via the chain | API |
| `tests/test_ingestion.py` | 10 unit + integration tests | Validation |
| `tests/test_indexing.py` | 3 indexing tests | Validation |
| `tests/test_retrieval.py` | 1 retrieval test | Validation |
| `tests/test_generation.py` | 4 generation tests (incl. live E2E ×2) | Validation |
| `tests/test_api.py` | 3 API tests (incl. live `/ask`) | Validation |
| `Standard_RAG(_Q&A_Courses)(2).ipynb` | Reference implementation (root) | Reference |
| `scripts/reindex.py` | **Placeholder** — not implemented | (future) |

## 33. Test / Validation Summary (actual runs, evidence-based)

| Stage | Status | Evidence |
|---|---|---|
| Environment | Completed | Python 3.11.9, uv 0.12.18, `.venv` from `python -m venv` |
| Configuration | Completed | YAML loads; grep proves no hard-coded hyperparameters |
| Ingestion | Tested | 10/10 pass; 1091 articles, 5/5 validation checks, JSON written |
| Indexing | Tested | 3/3 pass; 1092 chunks in Chroma |
| Retriever | Tested | 1/1 pass; top-3 with article metadata |
| Generation | Tested | 4/4 pass; live Gemini E2E (also via `build_rag_chain`) |
| FastAPI | Tested | 3/3 pass; `/health` 200, `/ask` 422 + live 200 |
| **Total** | **21/21 passed** | full suite, 363 s, exit code 0 |

## 34. Not in Day 1 (no future work claimed as done)

None of these exist in the project: DVC, MLflow, RAGAS, Langfuse,
Prometheus, Grafana, Evidently, Locust, BentoML, vLLM, Docker, CI/CD,
production deployment. `faiss-cpu` is declared but superseded by Chroma;
`scripts/reindex.py` is an unimplemented placeholder. Future work belongs
to later days.

## 35. Day 1 Errors Encountered (honest reproducibility record)

1. **`re.sub` callback bug** — the digit-run reverser received a `Match`,
   not a string (`IndexError: no such group`). Fixed by reversing
   `match.group(0)`. Found on the first real PDF run.
2. **cp1252 `UnicodeEncodeError`** — printing Arabic samples crashed when
   stdout was a pipe. Fixed with `sys.stdout.reconfigure(encoding="utf-8")`
   in the scripts (same class of fix verified in tests via file output).
3. **Wrong validation assumption** — first runs required `repealed_count > 0`
   and failed; raw-PDF scanning proved ranges 54–80/389–417 are physically
   absent from this edition (gaps 53→81, 388→418 + the PDF's own repeal
   notes). Rule corrected to consistency of every present flag.
4. **Broken marker leftovers** — probe runs exposed `ما دة`-split headers
   (fixed: spaced-letter regex, +13 articles) and `1( ( `-style clause
   markers (fixed: line-start rule, 303 occurrences, 0 conflicts).
5. **Unrecoverable source corruptions (reported, not hidden)** — article
   `601`'s digits are scrambled in the PDF itself (its text is preserved
   inside article 600, flagged in rejected diagnostics); article `1022`
   is absent from both columns.
6. **`.env` loading order** — API tests skipped because `llm` was only
   lazily imported; fixed by loading `.env` in `config/settings.py` so all
   modules share the environment.
7. **SDK stderr notice** — the google-genai SDK prints an "automatic
   function calling" warning into the error stream on every LLM call;
   verified it is **not** part of the parsed answer (1 text block, refusal
   string intact).
8. **Slow full-suite** — 21 tests take ~6 minutes (embedding model loads +
   live Gemini calls); suites are also runnable per-file.

## 36. Day 1 Completion Record

**Done:** environment (venv, pyproject, uv.lock, deps) → configuration
(YAML + settings + secrets) → ingestion (170 pp → 1091 articles, JSON) →
indexing (1092 chunks → Chroma) → retriever (top-3) → generation
(prompt + Gemini + chain) → FastAPI (`/health`, `/ask` via chain) →
21/21 tests green → verified live Q&A grounded in article 664.

```text
What did we build?     A working PDF → articles → Chroma → retriever →
                       Gemini → FastAPI pipeline over the Egyptian Civil Code.
How did we build it?   §§3–9 (environment/config/deps) then §§14,17 (pipelines).
What commands?         `python -m venv .venv`; `uv add …` (§7); `uv run python
                       scripts/extract_data.py`; `uv run python scripts/build_index.py`;
                       `uv run pytest … -v` per file (§30–31).
What files?            §10 tree + §32 inventory (actual sizes included).
Why each file?         §12 responsibilities + §§13–17, 21, 23–27 stage docs.
How stages connect?    §1, §11, §25, §27, §29 diagrams.
How tested?            §§16, 22, 28 + §33 evidence table (21/21 passed).
```
