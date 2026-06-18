# Build Guide — `frontend/api_client.py`

The frontend's **only** doorway to the backend. Every HTTP call lives here; the Streamlit
pages call plain Python functions (`query(...)`, `ingest(...)`, `list_documents()`) and
never touch `httpx` themselves. That separation is the whole point — the pages worry about
*what to show*, this file worries about *how to talk to the API*. One function per endpoint.

It's a thin wrapper, so it's small. But there's one non-obvious thing that will bite you
immediately if you miss it (the timeout — see §1), precisely *because* your queries are slow.

Stack: `httpx` (already installed). This file imports **nothing from `app/`** — the
frontend is a separate process and stays decoupled from the backend package. It needs only
`httpx` and `os`.

---

## 0. Setup — base URL + a shared client

The frontend and backend are **separate processes**, so the client needs to know where the
backend lives. Read it from the environment with a sensible default (don't import
`app.config` — that's backend-side):

```python
import os
import httpx

BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
```

Then make **one module-level client** the functions share (connection reuse; imported once
per Streamlit process):

```python
client = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(120.0))
```

`base_url` means your functions can use relative paths (`client.post("/query", ...)`).

---

## 1. The timeout — the one thing you must not miss

**`httpx`'s default timeout is 5 seconds.** Your `/query` does retrieval + cross-encoder
rerank + local LLM generation — easily 10–60s on an M1 (you already felt the "little
slowly"). With the default, the client would raise `httpx.ReadTimeout` *long before* the
answer comes back, and every real query would appear to "fail" even though the backend is
working fine.

So set a **generous** timeout, as above (`httpx.Timeout(120.0)`). Bump it higher if your
machine is slow; `timeout=None` disables it entirely (waits forever — simplest, fine for a
local single-user tool, but a large finite value is tidier). This applies to `/ingest` too —
embedding many chunks isn't instant.

This is the #1 frontend gotcha for LLM apps. Burn it in.

---

## 2. `ingest(uploaded_file)` — multipart upload

Streamlit's `st.file_uploader` hands you an `UploadedFile` object with `.name`, `.type`, and
`.getvalue()` (the bytes). Your `/ingest` route expects a multipart file under the form
field named **`file`** (because the route parameter is `file: UploadFile`). So the field key
here **must be `"file"`** — mismatch it and FastAPI returns 422.

```python
def ingest(uploaded_file) -> dict:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = client.post("/ingest", files=files)
    response.raise_for_status()
    return response.json()        # -> {"filename": ..., "chunk_count": ...}
```

The tuple is `(filename, content_bytes, content_type)`. That `filename` is what arrives as
`file.filename` on the server — the name it saves under and cites. `raise_for_status()`
turns a 4xx/5xx into an exception the page can catch (see §5).

---

## 3. `query(question, chat_history, mode)` — JSON body

This one sends JSON matching your `QueryRequest`. `chat_history` is the list of
`{"role": ..., "content": ...}` dicts the chat page keeps in `st.session_state` — you pass
it straight through (the backend validates it into `ChatMessage`s; the frontend never needs
the schema classes). Write this one yourself from the pattern:

- `client.post("/query", json={...})` with keys `question`, `chat_history`, `mode`.
- `raise_for_status()`, then `return response.json()`.
- The returned dict is your `QueryResponse`: `{"answer": str, "sources": [{"content","source","page_number"}, ...]}`. Return it as-is; the chat page renders `answer` plus a "Sources" expander over `sources`.

This is the concrete payoff of "the backend is stateless": the *frontend* owns the history
in `session_state` and ships it on every call; `api_client` is a pure pass-through.

## 4. `list_documents()` — simple GET

```python
def list_documents() -> list[dict]:
    response = client.get("/documents")
    response.raise_for_status()
    return response.json()        # -> [{"filename": ..., "chunk_count": ...}, ...]
```

The documents page renders this as a table.

---

## 5. Error handling — where it lives

Keep `api_client` honest and simple: call `raise_for_status()` and let exceptions propagate.
The **pages** decide how to show them (e.g. `try: ... except httpx.HTTPStatusError as e: st.error(...)`). Two reasons it belongs in the page, not here: (1) only the UI knows how to
display an error to the user, and (2) it keeps each client function a one-liner. Don't
swallow errors here and return `None` — that just hides failures.

(If you want friendlier messages, you *can* read `e.response.json()["detail"]` in the page —
that's the `detail` string from your backend's `HTTPException`s, e.g. "Unsupported file
type…".)

---

## 6. Checklist before review

- [ ] `BASE_URL` from `os.getenv` with a localhost default; **no `app.` imports**.
- [ ] A shared `httpx.Client` with a **large timeout** (not the 5s default).
- [ ] `ingest` posts multipart with the field key **`"file"`** and `(name, bytes, type)`.
- [ ] `query` posts JSON with `question` / `chat_history` / `mode`; returns the parsed dict.
- [ ] `list_documents` GETs and returns the parsed list.
- [ ] Every function calls `raise_for_status()` and returns `response.json()`.

## 7. Verification

With the backend running (`uvicorn app.main:app --port 8000`, Ollama up), add a temporary
`__main__` block or use a scratch REPL:

```python
if __name__ == "__main__":
    print(list_documents())                                  # [] or your ingested files
    print(query("What is multicollinearity?", [], "single")) # answer + sources
```

Confirm: `list_documents()` returns your corpus; `query(...)` returns an answer **without
timing out** (this is what validates §1); and a multi-turn call with a real `chat_history`
gets a condensed follow-up. You don't need Streamlit for this — it's plain Python over HTTP.

## What's next

After review: the two pages. `pages/2_documents.py` (uploader → `ingest`, table →
`list_documents`) is the simpler one and a good warm-up; then `pages/1_chat.py` (the
`st.chat_input` / `st.session_state` / sources-expander page that uses `query`), and finally
the small `app.py` landing page.
