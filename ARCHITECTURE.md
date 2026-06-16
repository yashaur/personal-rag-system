# Architecture — Local RAG System (v1)

Detailed design reference. Conventions and the quick map are in `CLAUDE.md`.

A first, deliberately minimal Retrieval-Augmented Generation system over a personal
knowledge base. Fully local: Ollama models, LangChain (LCEL) orchestration, a minimal
FastAPI backend, a Streamlit frontend, and **Chroma** as the vector store (persistent
local folder — no database server, no Docker).

The guiding principle for v1: the simplest thing that works, end to end. Three real
features are in scope — hybrid search, reranking, and multi-turn chat — but everything
else (caching, async jobs, query transforms, abstractions, containers) is deferred.

---

## 1. Data flow

**Ingestion (synchronous)**

```
Upload (Streamlit) → POST /ingest
  → load (PDF via PyMuPDF, or TXT)
  → chunk (RecursiveCharacterTextSplitter)
  → embed (Ollama)
  → add to Chroma (with metadata)
  → (re)build BM25 index from all chunks, persist to data/bm25/
  → return a summary (filename, chunk count)
```

No job IDs, no polling. The request returns when ingestion is done (a Streamlit spinner
covers the wait).

**Query**

```
question + optional chat_history → POST /query
  → if multi-turn: condense (history + follow-up) → standalone question
  → hybrid retrieve: Chroma dense + BM25, fused with RRF (EnsembleRetriever)
  → rerank: CrossEncoder (BAAI/bge-reranker-base), keep top_n
  → LCEL generation chain (Ollama)
  → return answer + source citations
```

---

## 2. Configuration (`config.py`)

A single Pydantic `BaseSettings` read from `.env`. Everything tunable lives here and is
referenced as `settings.<field>`; nothing hardcoded elsewhere. Fields:

- `ollama_base_url`, `ollama_llm_model`, `ollama_embed_model`, `llm_temperature`
- `chroma_dir` (e.g. `./data/chroma`), `collection_name`
- `chunk_size` (default 1000), `chunk_overlap` (default 200)
- `top_k` (candidates retrieved per leg, default 5), `rerank_top_n` (kept after rerank,
  default 3)
- `hybrid_weights` (default `[0.6, 0.4]` = semantic, BM25)
- `reranker_model` (default `BAAI/bge-reranker-base`)
- `bm25_dir` (e.g. `./data/bm25`), `uploads_dir` (e.g. `./data/uploads`)

---

## 3. Loaders (`loaders.py`)

A small dispatch on file extension returns `List[langchain.Document]`:

| Format | Library | Notes |
|---|---|---|
| `.pdf` | PyMuPDF (`fitz`) | One Document per page; `page_number` + `source` in metadata |
| `.txt` | native | Whole file as one Document; `source` in metadata |

Adding a format later = one function plus a line in the dispatch. No factory/ABC.

---

## 4. Chunking

`RecursiveCharacterTextSplitter`, `chunk_size`/`chunk_overlap` from config. Each chunk
carries `source`, `page_number` (when available), and `chunk_index` in its metadata. This
metadata is what produces source citations in answers.

---

## 5. Vector store (`vectorstore.py`)

Chroma with a persistent client pointed at `settings.chroma_dir`, wrapped by LangChain's
`langchain_chroma.Chroma`. Embeddings come from the shared `OllamaEmbeddings` instance.
Chroma persists to disk automatically, so the collection survives restarts. On backend
startup the existing collection is loaded; if empty, the app simply waits for ingestion.

Chroma stores chunk text + metadata, so it doubles as the source for the "list documents"
view — no separate registry table is needed (derive the document list from chunk metadata,
or keep a tiny JSON manifest alongside if a cleaner list is wanted).

---

## 6. Ingestion (`ingestion.py`)

A single synchronous function: `load → chunk → embed → Chroma.add_documents → rebuild
BM25`. Because BM25 is in-memory over the full corpus, it is rebuilt from all current
chunks after each ingest and re-serialized to `settings.bm25_dir`. At this scale (tens to
low hundreds of docs) a full rebuild is trivially fast and far simpler than incremental
updates.

---

## 7. Retrieval (`retrieval.py`)

Three stages, assembled with LangChain components:

1. **Semantic** — Chroma's retriever (`as_retriever`, `k = top_k`) over Ollama embeddings.
2. **Keyword** — `BM25Retriever` loaded from the serialized index (`k = top_k`).
3. **Fusion** — `EnsembleRetriever([semantic, bm25], weights=hybrid_weights)` applies
   Reciprocal Rank Fusion. RRF is used (not score averaging) because the two methods
   produce incomparable score scales.
4. **Rerank** — wrap the ensemble in a `ContextualCompressionRetriever` whose compressor
   is a `CrossEncoderReranker` backed by `HuggingFaceCrossEncoder(reranker_model)`. It
   reorders the fused candidates by true query–document relevance and keeps `rerank_top_n`.

Build this incrementally: get the semantic retriever returning sensible chunks first, then
add the BM25 leg and the ensemble, then add the reranker. Each step is independently
verifiable.

```python
# sketch
semantic = chroma.as_retriever(search_kwargs={"k": settings.top_k})
bm25 = BM25Retriever.from_documents(all_docs); bm25.k = settings.top_k
ensemble = EnsembleRetriever(retrievers=[semantic, bm25],
                             weights=settings.hybrid_weights)
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name=settings.reranker_model),
    top_n=settings.rerank_top_n)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=ensemble)
```

---

## 8. Prompts & chains (`prompts.py`, `chains.py`)

**Prompts**

- `rag_prompt` — answer using only the provided context; cite the source filename when
  referencing specific information; say so when the context doesn't contain the answer.
- `condense_prompt` — rewrite a follow-up question into a standalone question given the
  chat history (multi-turn only).

**Chains (LCEL)**

Single-turn:

```python
chain = (
    RunnableParallel({"context": retriever | format_docs,
                      "question": RunnablePassthrough()})
    | rag_prompt | llm | StrOutputParser()
)
```

Multi-turn: condense `(history, follow_up)` into a standalone question, then feed it
through the single-turn chain. Chat history is passed in the request body and held in
Streamlit `session_state` — the backend stays stateless.

---

## 9. API (`api.py`, `schemas.py`, `main.py`)

Routes:

- `POST /ingest` — multipart upload; runs ingestion synchronously; returns
  `{filename, chunk_count}`.
- `POST /query` — body below; returns the answer with sources. Handles both single-turn
  (no/empty history) and multi-turn (history present).
- `GET /documents` — list ingested documents (filename, chunk count).

Schemas:

```python
class QueryRequest(BaseModel):
    question: str
    chat_history: list[dict] = []          # [{"role": "user"|"assistant", "content": ...}]
    mode: Literal["single", "multi"] = "single"

class SourceDocument(BaseModel):
    content: str
    source: str
    page_number: int | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
```

`main.py` builds the singletons once at startup (Ollama LLM + embeddings, Chroma
collection, BM25 index, the assembled retriever and chains) and exposes them to the routes.

---

## 10. Frontend (`frontend/`)

- `app.py` — Streamlit home / landing.
- `pages/1_chat.py` — a single/multi-turn toggle, chat history via `st.chat_message`,
  `st.chat_input`, and a collapsible "Sources" expander showing filename + page per chunk.
  History lives in `st.session_state` and is sent with each request.
- `pages/2_documents.py` — `st.file_uploader` (PDF/TXT) with a spinner during ingest, and a
  table of currently ingested documents.
- `api_client.py` — a thin `httpx` wrapper; base URL from env, one method per endpoint.

---

## 11. Folder structure

```
rag-system/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app + startup (build/load singletons)
│   │   ├── config.py          # Pydantic settings — single source of truth
│   │   ├── llm.py             # ChatOllama + OllamaEmbeddings singletons
│   │   ├── loaders.py         # pdf (PyMuPDF) + txt → List[Document], dispatch by ext
│   │   ├── ingestion.py       # load → chunk → embed → store (synchronous)
│   │   ├── vectorstore.py     # Chroma persistent client
│   │   ├── retrieval.py       # hybrid (Chroma + BM25, RRF) → CrossEncoder rerank
│   │   ├── prompts.py         # rag prompt + condense prompt
│   │   ├── chains.py          # LCEL: single-turn + conversational
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── api.py             # routes: /ingest, /query, /documents
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app.py
│   ├── pages/
│   │   ├── 1_chat.py
│   │   └── 2_documents.py
│   └── api_client.py
├── data/                      # gitignored
│   ├── chroma/                # Chroma persistence
│   ├── bm25/                  # serialized BM25 index
│   └── uploads/               # uploaded files
├── README.md
└── .gitignore
```

---

## 12. Dependencies

```
langchain
langchain-community
langchain-ollama
langchain-chroma
chromadb
fastapi
uvicorn
streamlit
pymupdf
rank-bm25
sentence-transformers      # backs HuggingFaceCrossEncoder (reranker)
pydantic-settings
httpx                       # frontend api_client
```

---

## 13. Build order

1. `config.py` + `llm.py`
2. `loaders.py` → `vectorstore.py` → `ingestion.py`
3. `retrieval.py` (semantic-only → + BM25/RRF → + rerank)
4. `prompts.py` + `chains.py` (single-turn → conversational)
5. `schemas.py` + `api.py` + `main.py`
6. `frontend/`

---

## 14. Later (separate efforts)

The features cut from v1 — caching, async ingestion, HyDE / Multi-Query, contextual
compression, the enterprise pgvector tier, Docker, and evaluation/LLMOps (RAGAS, LangFuse,
CI/CD, monitoring) — are intentionally out of scope. Add them one at a time onto a working
core so you can measure what each contributes.
