# Personal RAG System

A fully local **"chat with my documents"** system over a personal knowledge base
(PDFs, lecture slides, technical/legal docs). All inference runs on local
[Ollama](https://ollama.com) models — **nothing leaves the machine**. Built to run on an
M1 MacBook Air.

Two surfaces: a minimal **FastAPI** backend and a **Streamlit** frontend.

---

## Features

- **Retrieval-augmented generation** over a personal PDF/TXT corpus, with **inline source
  citations** (filename + page) on every answer.
- **Hybrid retrieval pipeline** — dense vector search (Chroma) fused with **BM25 keyword
  search** via **Reciprocal Rank Fusion** (`EnsembleRetriever`), then **cross-encoder
  reranking** (`BAAI/bge-reranker-base`) to keep the most relevant chunks. Composed entirely
  with **LangChain LCEL** runnables.
- **Multi-turn conversation** — follow-up questions are condensed into a standalone query
  (using a dedicated, more capable condenser model) before retrieval, so context carries
  across turns.
- **Token streaming** — answers stream to the UI token-by-token; sources are sent first, then
  the generated text.
- **Full document lifecycle** — upload/ingest, list, and delete (single or all) documents. The
  dense (Chroma) and keyword (in-memory BM25) indexes are kept in sync on every change.
- **LLMOps observability** — [Langfuse](https://langfuse.com) tracing plus a custom callback
  that instruments each generation with **TTFT** (time-to-first-token), **TPOT**
  (time-per-output-token), **TPS** (tokens/sec), and token counts, tagging each trace with the
  git release.

---

## Architecture

```
question (+ optional chat_history) ──▶ POST /query
  └─ if multi-turn: condense (history + follow-up) ──▶ standalone question
       └─ hybrid retrieve: Chroma dense + BM25, fused with RRF (EnsembleRetriever)
            └─ rerank: CrossEncoder (BAAI/bge-reranker-base), keep top_n
                 └─ LCEL generation chain (Ollama) ──▶ answer + source citations
```

**Ingestion** (synchronous, in-request): upload → load (PyMuPDF for PDF, native TXT) → chunk
(`RecursiveCharacterTextSplitter`) → embed (Ollama) → add to Chroma with metadata → rebuild the
BM25 index. Chroma is the single store — it holds both the vectors and the chunk metadata, so
the document list is derived from Chroma metadata (no separate database).

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design rationale.

### Stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain (LCEL) |
| Backend | FastAPI (synchronous) |
| Frontend | Streamlit |
| LLM + embeddings | Local Ollama |
| Vector store | Chroma (persistent local folder) |
| Keyword search | in-memory `BM25Retriever` (`rank_bm25`) |
| Fusion | `EnsembleRetriever` (Reciprocal Rank Fusion) |
| Reranker | `CrossEncoderReranker` + `BAAI/bge-reranker-base` (local) |
| Observability | Langfuse |
| Document formats | PDF (PyMuPDF) + TXT |

---

## Project structure

```
app/                     # FastAPI backend
  config.py              # Pydantic settings (single source of truth)
  llm.py                 # Ollama LLM + embeddings
  loaders.py             # PDF/TXT loaders
  vectorstore.py         # Chroma wrapper + document-list helper
  ingestion_deletion.py  # ingest / delete, keeping indexes in sync
  retrieval.py           # hybrid retrieve + RRF + rerank
  prompts.py             # prompt templates
  chains.py              # LCEL generation + condense chains (+ streaming)
  observability.py       # Langfuse tracing + custom latency callback
  schemas.py             # Pydantic request/response models
  api.py                 # routes
  main.py                # FastAPI app
frontend/                # Streamlit app
  app.py                 # landing page
  api_client.py          # httpx wrapper (the only backend doorway)
  pages/1_chat.py        # chat UI (multi-turn, streaming, citations)
  pages/2_documents.py   # upload + document management
ARCHITECTURE.md          # detailed design reference
```

---

## Setup

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com), and a
[Langfuse](https://langfuse.com) instance (self-hosted or cloud).

### 1. Ollama (native, in a separate terminal)

```bash
ollama serve
ollama pull llama3.2:3b        # generation
ollama pull ministral-3:8b     # multi-turn condenser
ollama pull nomic-embed-text   # embeddings
```

### 2. Environment

Create a `.env` file in the project root. Langfuse keys are required (the app initializes
tracing on startup); everything else has sensible defaults in [`app/config.py`](app/config.py).

```dotenv
# Langfuse (required)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000

# Optional overrides (defaults shown)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:3b
OLLAMA_CONDENSER_MODEL=ministral-3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
TOP_K=5                     # candidates per retrieval leg (dense + BM25)
RERANK_TOP_N=3              # kept after cross-encoder rerank
HYBRID_WEIGHTS=[0.6, 0.4]   # EnsembleRetriever (RRF) weights: [semantic, BM25]
```

### 3. Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> First boot is slow: the cross-encoder reranker downloads `BAAI/bge-reranker-base` (~270 MB)
> to the HuggingFace cache once, and loads at import.

### 4. Frontend

```bash
streamlit run frontend/app.py
```

---

## API

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/ingest` | Upload + ingest a PDF/TXT file |
| `POST` | `/query` | Ask a question (single or multi-turn) |
| `POST` | `/query/stream` | Same, streamed as NDJSON (sources first, then tokens) |
| `GET`  | `/documents` | List ingested documents (filename, pages, chunks) |
| `DELETE` | `/delete_single_file` | Delete one document and its chunks |
| `DELETE` | `/delete_all_files` | Delete the entire corpus |

Interactive docs are available at `http://localhost:8000/docs` once the backend is running.

---

## Observability

Every query is traced in Langfuse via a LangChain callback handler. A custom
`OllamaLatencyCallback` ([`app/observability.py`](app/observability.py)) parses Ollama's
`response_metadata` and attaches per-generation performance scores — TTFT, TPOT, TPS,
load/total duration, and input/output token counts — to each trace, which is itself tagged with
the current git short-SHA as the `release`.

---

## Scope

**In scope (v1):** hybrid search, reranking, single + multi-turn chat, token streaming, source
citations, document upload/list/delete, PDF + TXT.

**Deferred:** semantic caching, async/background ingestion, query transforms (HyDE, multi-query),
a formal evaluation suite, additional formats (docx/pptx/csv), and containerization. This is a
deliberately minimal v1 — the goal was a working core, then adding layers one at a time to see
what each contributes.
