# Build Guide — `app/api.py` (+ the minimal `main.py` to run it)

The HTTP surface. This is where everything you've built stops being importable functions
and becomes a running server the frontend can talk to.

The single most important mindset for this file: **api.py is a thin translation layer.**
It does almost no logic of its own. Each route does three things — (1) receive a request
FastAPI has already validated against your schemas, (2) call a function you already wrote
(`answer_question`, `ingest_file`, `refresh`), (3) shape the result into a response schema.
If you find yourself writing retrieval or generation logic *here*, something has leaked out
of the layer it belongs in.

Three routes (ARCHITECTURE §9): `POST /ingest`, `POST /query`, `GET /documents`. No delete
endpoint — it's explicitly out of scope (CLAUDE.md). You'll also write a ~5-line `main.py`
at the end so you can actually run and test it.

---

## 0. Two decisions made for you, with reasons

**(a) Routes are `def`, not `async def`.** This looks wrong to people who've seen FastAPI
tutorials full of `async def`, so here's the why — as a shop.

Your server is **a shop with one cashier**; each request is a customer. One task is slow:
asking the local AI for an answer takes ~10 seconds.

- **`async def`** = "Cashier, do it yourself, but help other customers during any waiting
  gaps." This only helps if the slow task has a **pause button** (the keyword `await`) the
  cashier can press to step away. Your AI call is **blocking** — 10 seconds of solid work,
  no `await`, no pause button. So the cashier stands frozen for 10 seconds and **everyone
  else in line waits.** The shop jams.
- **`def`** (what you use) = "Cashier, hand this to a helper in the back room." FastAPI runs
  a plain `def` route on a separate threadpool worker, so the slow AI work happens in back
  and the **cashier stays free to serve everyone else.**

Your work (Ollama calls, embedding, the reranker) is blocking with no `await`, so `def` is
both correct *and* what "synchronous v1" (golden rule #3) asks for. Use `def`.
>
> **`def` is *not* "one user at a time."** The back room isn't one helper — it's a pool of
> ~40 (FastAPI's threadpool). Each in-flight request gets its own helper thread, returned to
> the pool when the response is sent. So while User 1's 10-second query runs on one thread,
> User 2's query is handed to another and runs **concurrently** — a single server process
> already serves many users at once. With this blocking stack, `def` is what *gives* you that
> concurrency; `async def` over blocking calls would instead run them on the cashier and
> freeze everyone. (To make `async def` actually help, you'd first rewrite the LLM/retrieval
> calls onto async clients — e.g. `ainvoke` — and `await` them throughout.)
> **This has nothing to do with *who* calls the API.** The shop story never cared whether
> customers walked in, phoned, or used an app — only that one *task* was slow and couldn't
> pause. So it's not about Streamlit, browsers, or CORS (§4). `def` vs `async def` is decided
> solely by what the route does *inside*; the answer is the same whether the caller is
> Streamlit, a browser, curl, or another server.

**(b) An `APIRouter` in api.py, the `FastAPI()` app in main.py.** ARCHITECTURE §9/§11 split
it this way: `api.py` owns the routes, `main.py` assembles the app and includes the router.
Keeps "what the endpoints do" separate from "how the app is wired."

```python
# top of api.py
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.schemas import QueryRequest, QueryResponse, SourceDocument, IngestResponse, DocumentInfo
from app.ingestion import ingest_file
from app.chains import answer_question
from app.vectorstore import vectorstore
import app.retrieval as retrieval          # module import — same lazy-lookup reason as chains.py

router = APIRouter()
```

> Note the `import app.retrieval as retrieval` again, not `from app.retrieval import refresh`.
> You need to call `retrieval.refresh()` and have it rebind the module-global `retriever`
> that the chain reads. Same names-vs-objects lesson from chains.py.

---

## 1. `POST /ingest` — upload, store, **refresh**

The flow: receive an uploaded file → save it to `uploads_dir` → `ingest_file(path)` →
**`retrieval.refresh()`** → return the summary.

That `refresh()` is the whole reason this endpoint isn't trivial — it's the lazy-retriever
gotcha you already met, now in production. `ingest_file` adds chunks to Chroma, but the
live `retrieval.retriever` (with its in-memory BM25 index) was built from the *old* corpus.
Without `refresh()`, the very next `/query` would search stale data — or, if this was the
first ever ingest, `retriever` is still `None` and `answer_question` returns "No documents
have been ingested yet." **Ingest that doesn't refresh is a silent bug.**

Sketch (most of this is new, so it's fairly complete — understand each line, don't just
paste):

```python
ALLOWED = {".pdf", ".txt"}

@router.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    name = Path(file.filename).name              # strip any path components (traversal guard)
    if Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED}.")

    dest = Path(settings.uploads_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file.file.read())           # save the upload to disk

    try:
        result = ingest_file(str(dest))          # -> {"filename": ..., "chunk_count": ...}
    except ValueError as e:                       # ingest_file raises this on an empty file
        raise HTTPException(status_code=400, detail=str(e))

    retrieval.refresh()                          # <-- the step you must not forget
    return IngestResponse(**result)              # keys already match the schema
```

Things to notice:
- `UploadFile = File(...)` needs `python-multipart` (already installed).
- `IngestResponse(**result)` works *only because* `ingest_file` returns exactly
  `{"filename", "chunk_count"}` — the alignment you verified in the schemas review pays off.
- Translating `ValueError → HTTPException(400)` is the api layer's job: turn an internal
  exception into an honest HTTP status. Don't let it bubble up as an opaque 500.

---

## 2. `POST /query` — and where `mode` finally does its job

FastAPI parses and validates the body into a `QueryRequest` for you. Your route then has to
do the two conversions we discussed when designing the schema:

1. **`mode` → whether to pass history.** This is the payoff of putting `mode` in the
   request and *not* in chains.py. `answer_question` decides single vs. multi by whether
   history is present; the api layer decides *whether to hand over the history at all*
   based on `mode`. So `"single"` ⇒ pass `None` (ignore any history the client sent);
   `"multi"` ⇒ convert and pass it.
2. **`ChatMessage` objects → dicts.** `answer_question` → `to_messages` reads
   `message["role"]` (dict access). Your request carries `ChatMessage` *objects*, so
   `model_dump()` each one back to a dict at the boundary. (chains.py stays untouched.)
3. **`Document` → `SourceDocument`.** `answer_question` returns
   `{"answer": str, "sources": list[Document]}` (raw langchain Documents). Map each one
   into your response schema — and use `.get("page_number")`, not `["page_number"]`, because
   TXT chunks don't have it (the exact reason the field is optional).

```python
@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    history = [m.model_dump() for m in req.chat_history] if req.mode == "multi" else None

    result = answer_question(req.question, history)   # {"answer": str, "sources": [Document]}

    sources = [
        SourceDocument(
            content=doc.page_content,
            source=doc.metadata["source"],
            page_number=doc.metadata.get("page_number"),
        )
        for doc in result["sources"]
    ]
    return QueryResponse(answer=result["answer"], sources=sources)
```

The "no documents yet" case needs **no special handling here** — `answer_question`'s own
guard returns `{"answer": "No documents...", "sources": []}`, which maps cleanly to a
`QueryResponse` with an empty `sources` list. The endpoint just works.

---

## 3. `GET /documents` — derive the list from Chroma

There's no separate registry of uploaded files (golden rule: Chroma is the store). So you
*derive* the document list by reading all chunk metadata and counting chunks per `source`.
This is the one route you'll assemble mostly yourself — it's short.

The pieces:
- `vectorstore.get(include=["metadatas"])` returns `{"ids": [...], "metadatas": [ {...}, ... ]}`.
  Each metadata dict has a `source` key (the filename).
- Count occurrences of each `source` — `collections.Counter` is built for exactly this:
  `Counter(m["source"] for m in data["metadatas"])` gives `{filename: chunk_count}`.
- Turn that into `list[DocumentInfo]`. Declare the route's `response_model=list[DocumentInfo]`.
- Empty store ⇒ empty `metadatas` ⇒ you return `[]`. No special-casing needed.

Write this route from that description. It's ~4 lines in the body.

---

## 4. The minimal `main.py` (so you can run it)

This is plumbing, not a learning target — here it is in full:

```python
from fastapi import FastAPI
from app.api import router

app = FastAPI(title="Personal RAG System")
app.include_router(router)
```

Run it (Ollama up first, models pulled):

```bash
uvicorn app.main:app --reload --port 8000
```

Two things worth knowing:

- **No explicit "build singletons at startup" step is needed.** Your llm, vectorstore,
  retriever, and chains are already module-level globals constructed when their modules are
  imported — and importing `app.api` pulls all of them in. On boot, `retrieval.retriever` is
  built from whatever is already persisted in Chroma, so a server started with existing data
  is immediately query-ready. (ARCHITECTURE §9 phrases this as a startup builder; in your
  code it's just import-time construction.)
- **First boot is slow.** The CrossEncoder reranker (`bge-reranker-base`, ~270 MB) loads
  when `retrieval.py` is imported — the first time it also downloads. The server isn't hung;
  it's loading the model. Subsequent boots are fast (cached).

**You do *not* need CORS middleware.** This trips people up: Streamlit isn't a browser app
calling your API from JavaScript. The Streamlit *server* runs `api_client.py` (httpx) and
calls FastAPI **server-to-server**, on localhost. CORS is a browser mechanism for
cross-origin *browser* requests — it simply doesn't apply here. Adding `CORSMiddleware`
would be cargo-culting. (If you ever called the API from browser JS directly, then you'd
add it.)

---

## 5. Gotchas checklist

- [ ] `retrieval.refresh()` is called at the end of `/ingest` (the silent-bug trap).
- [ ] Routes are `def`, not `async def` (blocking work → threadpool; golden rule #3).
- [ ] `page_number=doc.metadata.get("page_number")` — `.get`, not `[ ]`.
- [ ] `mode == "multi"` gates whether history is forwarded; `ChatMessage`s are
      `model_dump()`-ed to dicts.
- [ ] `ValueError` from `ingest_file` → `HTTPException(400, ...)`, not a leaked 500.
- [ ] Upload filename sanitized with `Path(file.filename).name`.
- [ ] `import app.retrieval as retrieval` (module form) so `refresh()` rebinds the global
      the chain reads.
- [ ] Ignore `bm25_dir` / `data/bm25/` from ARCHITECTURE — your retriever rebuilds BM25
      in-memory via `refresh()`. There's nothing to serialize.

---

## 6. Verification

Start Ollama (`ollama serve`, models pulled) and the server
(`uvicorn app.main:app --reload --port 8000`). Then:

1. **Swagger UI** — open `http://localhost:8000/docs`. All three routes should appear with
   your schema field descriptions. This alone confirms the app wired up.
2. **Ingest** a file:
   ```bash
   curl -F "file=@data/uploads/test.pdf" http://localhost:8000/ingest
   # -> {"filename":"test.pdf","chunk_count": N}
   ```
3. **List documents:**
   ```bash
   curl http://localhost:8000/documents
   # -> [{"filename":"test.pdf","chunk_count": N}]
   ```
4. **Query (single-turn):**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question":"What is multicollinearity?"}'
   # -> {"answer":"...","sources":[{"content":"...","source":"test.pdf","page_number":14}, ...]}
   ```
5. **Query (multi-turn):** repeat with `"mode":"multi"` and a `chat_history` of one
   user/assistant pair, and confirm a follow-up like `"why does it matter?"` gets answered
   as if it were standalone (the condense step firing through the API).
6. **Bad inputs:** a `.docx` upload and an empty file should each return **400**, not 500.
   A query with `chat_history` containing `{"role":"system",...}` should return **422**
   (your `ChatMessage` Literal doing its job).

When `/docs` lists everything, an ingest→documents→query round-trip works, and the 400/422
cases behave — api.py is done.

## What's next

`main.py` is essentially finished above, so after I review your api.py the only thing left
on the backend is a quick pass over it — then we're onto the **frontend** (`api_client.py`
first, since the pages depend on it).
