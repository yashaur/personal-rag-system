# Build Guide — `app/schemas.py`

The API's **contract**. These Pydantic models are the typed boundary between the outside
world (JSON over HTTP) and the Python objects your app already produces. `api.py` (next
file) will declare them as its request/response types, and FastAPI does the rest.

This is a small file — you'll write all of it. The value here isn't volume, it's getting
the field types and defaults to line up *exactly* with the shapes `chains.py` and
`ingestion.py` already return, so `api.py` is a thin, honest translation layer with no
surprises.

Stack: Pydantic **v2** (2.13.x), FastAPI 0.137, Python 3.14 — so modern type syntax
(`int | None`) and v2 method names (`model_dump`, not the v1 `.dict`).

---

## 0. Mental model: what a Pydantic model does in FastAPI

A `BaseModel` subclass is a typed data shape that FastAPI uses for three jobs:

1. **Validation / parsing (inbound).** When a request body is declared as `QueryRequest`,
   FastAPI parses the incoming JSON into a `QueryRequest` instance and validates every
   field. Wrong type or missing required field → FastAPI returns a **422** with a precise
   error, and your route function never even runs. You get clean, typed Python inside.
2. **Serialization (outbound).** When a route is declared to return `QueryResponse`,
   FastAPI turns your model into JSON on the way out.
3. **Docs.** Every model feeds the auto-generated OpenAPI page at `/docs` — free,
   accurate API documentation.

So think of `schemas.py` as the **edge** of your system: the place where untyped JSON
becomes trustworthy typed objects, and vice-versa. Everything inside the edge
(`chains.py`, `retrieval.py`, …) can then assume the data is well-formed.

---

## 1. The shapes you're mirroring

The models aren't invented from scratch — each mirrors something that already exists:

- `answer_question()` ([chains.py:92](app/chains.py#L92)) returns
  `{'answer': str, 'sources': list[Document]}`.
- `ingest_file()` ([ingestion.py:12](app/ingestion.py#L12)) returns
  `{'filename': str, 'chunk_count': int}`.
- Each chunk's `.metadata` carries `source`, and (PDF only) `page_number`
  ([loaders.py](app/loaders.py)).

Your job is to give those shapes a typed, validated face for the API.

---

## 2. The models to write

Six small models. Write each one yourself from the spec; I show snippets only for the new
concepts.

| Model | Fields | Used by | Mirrors |
|---|---|---|---|
| `ChatMessage` | `role`, `content` | nested in `QueryRequest` | the dicts `to_messages` reads |
| `QueryRequest` | `question`, `chat_history`, `mode` | `POST /query` body | ARCHITECTURE §9 |
| `SourceDocument` | `content`, `source`, `page_number` | inside `QueryResponse` | a langchain `Document` |
| `QueryResponse` | `answer`, `sources` | `POST /query` response | `answer_question()` output |
| `IngestResponse` | `filename`, `chunk_count` | `POST /ingest` response | `ingest_file()` output |
| `DocumentInfo` | `filename`, `chunk_count` | `GET /documents` (`list[DocumentInfo]`) | grouped Chroma metadata |

### `ChatMessage` — your first `Literal` field

`role` should only ever be `"user"` or `"assistant"`. `Literal` makes that a validation
rule, not a hope:

```python
from typing import Literal
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
```

Now `ChatMessage(role="system", content="…")` raises a `ValidationError` instead of
quietly sailing through. That matters because `to_messages` only handles `user`/`assistant`
([chains.py:62-68](app/chains.py#L62-L68)) and silently drops anything else.

### `QueryRequest` — nested model + defaults

```python
class QueryRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] = []
    mode: Literal["single", "multi"] = "single"
```

- `question` is required (no default).
- `chat_history` defaults to empty — a single-turn request just omits it. (See §3 for why
  it's `list[ChatMessage]` and not `list[dict]`.)
- `mode` defaults to `"single"`.

> **Mutable default `= []`?** In plain Python that's the classic shared-mutable-default
> bug. In **Pydantic it's safe** — Pydantic gives each instance its own copy. So `= []` is
> idiomatic here; you don't need `Field(default_factory=list)` (though it's also fine).

### `SourceDocument` — the optional field

```python
class SourceDocument(BaseModel):
    content: str
    source: str
    page_number: int | None = None
```

`page_number` **must** be optional: TXT chunks have no page (your `_load_txt` sets only
`source`), so this field is `None` for them. `int | None = None` is the modern v2 way to
say "an int or nothing, defaulting to nothing."

### `QueryResponse`, `IngestResponse`, `DocumentInfo` — your turn

Write these three from the table. They're all plain `str`/`int`/`list[...]` fields, no new
concepts:

- `QueryResponse`: `answer: str`, `sources: list[SourceDocument]`.
- `IngestResponse`: `filename: str`, `chunk_count: int` — must match `ingest_file()`'s
  return keys exactly.
- `DocumentInfo`: `filename: str`, `chunk_count: int`. Note the naming choice: Chroma
  stores the file name under metadata key `source`, but for a **consistent public API** we
  expose it as `filename` in both `IngestResponse` and `DocumentInfo`. `api.py` will map
  `metadata["source"] → filename`. Internal storage key vs. public field name is a
  distinction worth being deliberate about.

---

## 3. The one real decision: `list[ChatMessage]` vs `list[dict]`

ARCHITECTURE.md sketched `chat_history: list[dict]`. I'm recommending you go one better
with `list[ChatMessage]`, and here's the honest trade-off so you choose knowingly.

**`list[dict]`** — minimal, zero extra models. But it throws away Pydantic's entire reason
for existing on this field: a client could POST `{"chat_history": [{"foo": "bar"}]}` and
FastAPI would happily accept it, because *any* dict satisfies `dict`. The failure then
happens later and uglier — a `KeyError: 'role'` deep inside `to_messages`, surfacing as an
opaque **500**.

**`list[ChatMessage]`** — FastAPI validates every message at the edge. A malformed history
is rejected with a precise **422** ("role: field required") before your code runs. It also
self-documents the wire format in `/docs`. This is exactly what Pydantic is *for*.

The one catch, and how to handle it: `chains.py`'s `to_messages` reads messages as **dicts**
(`message['role']`). If the request carries `ChatMessage` *objects*, you must convert them
back to dicts at the boundary — in `api.py`, not here:

```python
# in api.py later (preview), when calling the chain:
answer_question(
    req.question,
    [m.model_dump() for m in req.chat_history],   # ChatMessage -> {"role":..., "content":...}
)
```

`model_dump()` turns each `ChatMessage` back into the plain dict `to_messages` expects. Net
result: **the schema validates the edge, and `chains.py` keeps its dict contract
unchanged** — no edits to the file you just finished. That's the clean seam, and it's why
I recommend this path. (If you'd rather keep it dead-simple and match ARCHITECTURE
literally, `list[dict]` works and needs no conversion — your call, just know what you're
giving up.)

---

## 4. Gotchas to keep in mind

1. **No langchain imports in this file.** `schemas.py` is pure Pydantic — data shapes
   only. The `Document → SourceDocument` mapping belongs in `api.py`:
   ```python
   # api.py later, NOT here:
   SourceDocument(
       content=doc.page_content,
       source=doc.metadata["source"],
       page_number=doc.metadata.get("page_number"),   # .get → None-safe for TXT
   )
   ```
   This keeps the dependency direction clean: `chains.py` doesn't import schemas, and
   `schemas.py` doesn't import langchain. Each layer stays unaware of the other's
   internals.
2. **`page_number` optional** — covered above, but it's the easiest field to get wrong.
3. **`mode` is `api.py`'s concern, not `chains.py`'s.** Recall `chains.py` branches purely
   on whether history is present; it never looks at `mode`. `api.py` will use `mode` to
   decide *whether to forward `chat_history`* to `answer_question`. The field lives in the
   request model regardless.
4. **Pydantic v2 names:** `model_dump()` / `model_dump_json()` (the v1 `.dict()` /
   `.json()` are deprecated). `ValidationError` is imported from `pydantic`.

---

## 5. Checklist before you ask me to review

- [ ] Six models: `ChatMessage`, `QueryRequest`, `SourceDocument`, `QueryResponse`,
      `IngestResponse`, `DocumentInfo`.
- [ ] `role` and `mode` use `Literal`; `page_number` is `int | None = None`.
- [ ] `IngestResponse` field names match `ingest_file()`'s return keys (`filename`,
      `chunk_count`) exactly.
- [ ] No `langchain` import anywhere in the file.
- [ ] `chat_history` decision made consciously (`list[ChatMessage]` recommended).
- [ ] A `__main__` block that proves validation works (see below).

## 6. Verification (`__main__` block — your usual convention)

No server needed; this is pure model construction. `python -m app.schemas` should:

1. **Happy path:** build a `QueryRequest` with a couple of `ChatMessage`s and print
   `req.model_dump()` — confirm you get a clean nested dict ready for JSON.
2. **Validation payoff:** wrap a bad payload in `try/except ValidationError` and confirm it
   raises:
   ```python
   from pydantic import ValidationError
   try:
       ChatMessage(role="system", content="nope")   # 'system' not in Literal
   except ValidationError as e:
       print("rejected as expected:\n", e)
   ```
3. **Response side:** build a `SourceDocument` from a hand-made
   `{content, source, page_number}`, wrap it in a `QueryResponse(answer="…", sources=[…])`,
   and print `model_dump()`.
4. **Ingest side:** construct `IngestResponse(filename="test.pdf", chunk_count=12)` — and
   sanity-check it accepts exactly what `ingest_file()` returns.

If the bad payload in step 2 raises and the good ones don't, your contract is sound.

## What's next

After review, **`api.py`** — where these models finally get wired to routes (`POST /query`,
`POST /ingest`, `GET /documents`), the `ChatMessage → dict` and `Document → SourceDocument`
conversions happen, and `answer_question` / `ingest_file` get called for real. Then
`main.py` assembles the app, and we're onto the frontend.
