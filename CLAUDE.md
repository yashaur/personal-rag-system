# CLAUDE.md

Project memory for Claude Code. **Read this first.** Fuller detail in `ARCHITECTURE.md`.

---

## What this is

A **first** RAG system — deliberately minimal. A fully local "chat with my documents"
tool over a personal knowledge base (PDFs, lecture slides, technical/legal docs). All
inference runs on local **Ollama** models; nothing leaves the machine. Target hardware:
M1 MacBook Air.

Two surfaces: a minimal **FastAPI** backend + a **Streamlit** frontend.

This is v1. Favor the simplest thing that works. Do not add infrastructure or abstraction
that isn't needed yet.

---

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain (**LCEL only**) |
| Backend | FastAPI (minimal, **synchronous**) |
| Frontend | Streamlit |
| LLM + embeddings | Local Ollama |
| Vector store | **Chroma** (persistent local folder — no Docker, no SQL) |
| Keyword search | in-memory `BM25Retriever` (`rank_bm25`) |
| Fusion | `EnsembleRetriever` (RRF) |
| Reranker | LangChain `CrossEncoderReranker` + `BAAI/bge-reranker-base` (local) |
| Conversation | single + multi-turn (one endpoint, optional history) |
| Logging | plain `logging` |
| Document formats | **PDF (PyMuPDF) + TXT** to start |

---

## Golden rules (do not violate)

1. **LCEL only.** No `LLMChain`, `ConversationalRetrievalChain`, or other legacy chain
   classes. Compose everything as LCEL runnables.
2. **Chroma is the store.** Persistent local directory. **No Postgres, no SQLAlchemy, no
   Alembic, no Docker.** Chroma holds both vectors and chunk metadata.
3. **Synchronous v1.** No async background tasks, no job tracking, no polling. Ingest runs
   in-request and returns when done.
4. **Config is the single source of truth.** Every tunable lives in `config.py`
   (Pydantic `BaseSettings`, read from `.env`). Reference via `settings.<field>`; no
   hardcoded values elsewhere.
5. **Concrete over abstract.** Write direct functions/modules. **No abstract base classes
   or factories** until a second real implementation exists.
6. **Plain logging.** Standard library `logging`. No `structlog`, no contextvars.
7. **Always cite sources.** Every answer returns the source filename (and page where
   available) for each chunk used.

---

## Retrieval pipeline (order)

```
hybrid retrieve (Chroma dense + BM25, fused with RRF)
  → CrossEncoder rerank (keep top_n)
  → LCEL generation chain (Ollama)
```

For multi-turn: condense (history + follow-up) into a standalone question first, then run
the pipeline above.

**Build it incrementally:** get pure semantic retrieval working end to end first, then add
the BM25 leg + RRF, then add reranking. See what each layer contributes.

---

## In scope (v1)

Hybrid search, reranking, single + multi-turn chat, source citations, a minimal document
list. PDF + TXT ingestion.

## Out of scope (deferred — do NOT build yet)

Semantic caching, async/background ingestion + job tracking, HyDE, Multi-Query,
contextual compression, feature-flag toggles, structured logging, Postgres/pgvector,
Docker/compose, migrations, document-delete API, formal test suite, extra formats
(docx/pptx/csv/html), Parent Document Retrieval, semantic chunking.

---

## Build order

1. `config.py` + `llm.py` — settings + Ollama LLM/embeddings
2. `loaders.py` → `vectorstore.py` → `ingestion.py` — get docs into Chroma
3. `retrieval.py` — semantic-only first, then BM25 + RRF, then rerank
4. `prompts.py` + `chains.py` — single-turn first, then conversational
5. `schemas.py` + `api.py` + `main.py`
6. `frontend/`

---

## Run commands

```bash
# Ollama (native, separate terminal)
ollama serve
ollama pull <llm-model>           # e.g. llama3.1
ollama pull <embedding-model>     # e.g. nomic-embed-text

# Backend (from backend/)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (from frontend/)
streamlit run app.py
```

---

## Notes

- Pick the embedding model up front. Changing it later means re-embedding the whole corpus
  (a Chroma collection should not mix models).
- Each new document format is just one small loader function added to `loaders.py` and
  wired into the extension dispatch.
- Reranking downloads `BAAI/bge-reranker-base` (~270 MB) once to the HuggingFace cache.
