# Build Guide — token streaming (the project's last feature)

The finishing touch. Right now every answer arrives in one lump: the user stares at a
spinner for the whole retrieve → rerank → generate cycle, then the full text appears at
once. Streaming changes the *felt* latency — the first words show up the moment the model
produces them, and the rest flows in as it's written. The total time is unchanged; the
**wait** is transformed. On a local `llama3.2:3b`, that's the single biggest UX win left.

This guide adds streaming **end to end** — a new backend route, a new client function, and
a change to the chat page — while leaving the existing non-streaming path completely intact
as a fallback. No code here (per how we've been working); you implement, I review.

> **Note on the v1 "no streaming" line.** `chains.md` §0 said "we won't use streaming in
> v1." We're consciously revisiting that now as the closing feature. Worth updating that
> sentence in `chains.md`/`ARCHITECTURE.md` when you're done so the docs don't contradict
> the code.

---

## 0. The mental model — what can and cannot stream

This is the whole game, so internalise it before touching a file.

Your pipeline has two phases with very different timing:

```
  ── BLOCKING PRELUDE ─────────────────────────  ── STREAMABLE ──
  (multi-turn) condense → retrieve → rerank        generate
  └──────────── must FINISH first ──────────┘      └─ emits token by token ─┘
        (no partial output is meaningful here)
```

**Only the final generation can stream.** A half-finished retrieval or a half-reranked list
isn't a partial answer — it's nothing the user can see. Condensing a follow-up into a
standalone question is an internal reformulation that's never shown. So those steps stay
exactly as they are: blocking `.invoke()` calls that run to completion. Streaming touches
**only** the last leg, where the LLM produces text one chunk at a time.

The most useful consequence falls straight out of that diagram: **by the time the first
token exists, retrieval has already finished — so the sources are fully known before the
answer starts.** That's why the design sends **sources first**, then streams the answer.
You never have to "wait until the end" to know what to cite.

**One more framing decision:** we add a *parallel* streaming path rather than retrofitting
the existing `answer_question` chain. Two reasons. (1) The existing chain is built around
`RunnablePassthrough.assign(...)` accumulating a dict and a final `RunnableLambda` that
reshapes it to `{answer, sources}` — that shape is designed to be `.invoke()`d and returns
one dict, not a stream. (2) As §2 shows (and as I verified in your venv), one of those
pieces *silently breaks streaming*. It's cleaner and safer to write a small, purpose-built
generator for the streaming case and keep the battle-tested blocking path as a fallback.

---

## 1. Design the contract first — the event stream

Before either end, pin down **what travels over the wire**, because both `api.py` and
`api_client.py` have to agree on it. This is the streaming equivalent of the `QueryResponse`
schema — the spine everything hangs off.

The stream is a sequence of small JSON objects, **one per line** (newline-delimited JSON,
"NDJSON"). Three event types, in this order:

```
{"type": "sources", "sources": [ {"content": "...", "source": "stats.pdf", "page_number": 4}, ... ]}
{"type": "token",   "text": "Multi"}
{"type": "token",   "text": "collinearity occurs"}
{"type": "token",   "text": " when ..."}
...
{"type": "done"}
```

- **`sources`** — emitted **once, first**. Its payload is *exactly the same shape* as
  today's `QueryResponse.sources` (a list of `{content, source, page_number}`). That's
  deliberate: the frontend's existing `render_sources` ([1_chat.py:29](../frontend/pages/1_chat.py#L29))
  already consumes that exact shape, so it'll work unchanged.
- **`token`** — emitted many times, one per chunk the model produces. (A "token" here is
  really whatever chunk LangChain hands you; could be a word-piece or a few characters.
  Doesn't matter — concatenated, they're the answer.)
- **`done`** — emitted once, last. A clean end-of-stream sentinel so the client knows the
  answer is complete (vs. the connection dropping). You could also carry timing here later.

**Why NDJSON and not SSE?** Server-Sent Events (`text/event-stream`, `data: …\n\n` frames)
is the web-standard for token streaming — *when the consumer is a browser's `EventSource`*.
Ours isn't: the consumer is `httpx` running inside the Streamlit **server** (recall the
"server-to-server, no CORS" point from `api.md` §4). For an httpx client, SSE just means
stripping `data:` prefixes and handling blank-line framing for zero benefit. One JSON object
per line is the simpler thing that works — `iter_lines()` on the client gives you exactly
one event per line to `json.loads`. (If you'd rather do real SSE, it's fine — just keep the
two ends consistent. The rest of this guide assumes NDJSON.)

Lock this table in your head; §4 produces these frames, §6 consumes them.

---

## 2. `app/chains.py` — the streaming generator (the core)

### 2a. The streamable core chain

Define a second small chain alongside `generation_chain`, but stopping one step earlier:
the prompt, the LLM, and the parser — and **not** `format_output`. Conceptually it's
`rag_prompt | llm | parser`. When you call `.stream(...)` on that, `ChatOllama` emits
message chunks, `StrOutputParser` passes each chunk's text straight through, and you get an
iterator of answer fragments.

> **Gotcha — `format_output` silently kills streaming. (Verified in your venv.)**
> `format_output` is `RunnableLambda(remove_asterisk)` ([chains.py:43-46](../app/chains.py#L43-L46)),
> a plain `str → str` function. A `RunnableLambda` only streams if the wrapped function is a
> *generator* that transforms an input iterator; over an ordinary function it does the only
> thing it can — **buffers the entire upstream into one string, calls the function once, and
> yields a single result.** I reproduced this offline while planning: `fake_stream |
> StrOutputParser()` yielded **3** chunks; appending `| RunnableLambda(strip "**")` collapsed
> it to **one** chunk (`"Hello world"`). So if you leave `format_output` on the streaming
> chain, you've technically "streamed" — exactly one giant token at the very end. Useless.
>
> The fix: drop it from the streaming path. Your system prompt already instructs "plaintext,
> not markdown" ([prompts.py:4](../app/prompts.py#L4)), so `**` should be rare. If a stray
> pair still slips through, strip it **per-token in the frontend** (cheap string replace on
> each chunk) or just accept it. Don't try to be clever stripping across chunk boundaries —
> a `**` split across two chunks can't be caught per-chunk anyway, which is the deeper reason
> the buffering lambda existed. For streaming, lean on the prompt.

### 2b. The new generator function

Write a generator — call it something like `stream_answer_question`, taking the question and
an optional chat history, mirroring `answer_question`'s signature
([chains.py:113](../app/chains.py#L113)). It reuses everything you already built:

1. **Resolve the standalone question.** Reuse `standalone_question(...)`
   ([chains.py:95](../app/chains.py#L95)) — pass it the question + history dict and it
   returns either the question as-is (no history) or the condensed standalone version. This
   is a **blocking** `.invoke()` inside, and that's correct: the condense step is internal,
   never streamed.
2. **Retrieve, blocking.** `retrieval.retriever.invoke(question)`
   ([chains.py:49](../app/chains.py#L49)) → the list of `Document`s. (Module-level
   `retrieval.retriever` access, same lazy-lookup reason as everywhere else.)
3. **Build context.** `format_chunks(...)` ([chains.py:27](../app/chains.py#L27)) → the
   context string.
4. **Hand back sources first, then a token iterator.** Create the token iterator by calling
   `.stream({"question": ..., "context": ...})` on your §2a core chain. Note this is *lazy*
   — calling `.stream()` returns an iterator but doesn't hit Ollama until something starts
   consuming it. That's exactly what we want: the docs are already in hand, and generation
   only fires once the route begins iterating tokens (after it's sent the sources).

**Recommended return shape: `(sources, token_iterator)`** — a tuple of the raw `Document`
list and the lazy string iterator. Here's why that shape and not, say, yielding ready-made
event dicts: it preserves the layer boundary `chains.md` §1 fought for — *chains.py returns
raw LangChain `Document`s and knows nothing about HTTP/JSON shapes; api.py maps `Document →
SourceDocument` and owns the wire format*. Keep that boundary; api.py (§4) turns this tuple
into the NDJSON frames from §1.

> **Gotcha — `add_timer` can't wrap a stream.** `add_timer`
> ([chains.py:18-25](../app/chains.py#L18-L25)) calls `.invoke()`, which fully materialises
> the result — wrap the stream in it and you've un-streamed it, same failure mode as
> `format_output`. If you want timing (and you should — **time-to-first-token is the exact
> metric streaming improves**), do it stream-aware: write a tiny generator that wraps the
> token iterator, records `perf_counter()` when it's created, logs TTFT at the **first**
> yield, and logs total elapsed when the iterator is **exhausted**, then keep yielding tokens
> through untouched. It's the streaming cousin of `add_timer`, and it keeps timing in
> chains.py next to your other `logger.info` timers instead of leaking into api.py. (Optional,
> but it's the satisfying part — you'll *see* TTFT drop from ~full-answer-time to a second
> or two.)

> **Gotcha — mirror the empty-store guard.** `answer_question` returns the "No documents
> have been ingested yet!" payload when `retrieval.retriever is None`
> ([chains.py:114-117](../app/chains.py#L114-L117)). Your streaming function must do the same
> *before* step 2, or `None.invoke(...)` throws. In the tuple shape, that's simply: return an
> **empty sources list** and a **one-item iterator** yielding that message string. The route
> then handles it identically to a real answer — no special-casing downstream.

Leave `answer_question` and the whole existing chain **untouched** — `/query` and the
`__main__` self-test still depend on them.

---

## 3. `app/llm.py` — nothing to do

`ChatOllama` streams natively the instant you call `.stream()`; modern `langchain_ollama`
(yours is 1.1) needs no `streaming=True` flag or callback wiring. Leave
[llm.py:9-13](../app/llm.py#L9-L13) exactly as it is. (Listed here only so you don't go
hunting for a switch that doesn't need flipping.)

---

## 4. `app/api.py` — the `POST /query/stream` route

A new route that returns a `StreamingResponse` (from `fastapi.responses`) wrapping a
**generator** that emits the NDJSON frames from §1.

**Keep it a plain `def` route with a plain `def` generator — not `async`.** This is the
"shop with one cashier" rule from `api.md` §0 again: your work (Ollama, retrieval, the
reranker) is blocking with no `await`, so a synchronous generator handed to
`StreamingResponse` is both correct and what golden rule #3 ("synchronous v1") demands.
Starlette iterates a sync generator on the threadpool and flushes each yielded piece to the
client as it's produced — true incremental delivery, no async machinery, no background jobs.

What the route does:

1. Parse the body into `QueryRequest` exactly like `/query` does, and reuse the same
   `mode → history` logic ([api.py:51-55](../app/api.py#L51-L55)): `"multi"` forwards the
   converted history, `"single"` passes `None`.
2. Call your §2 `stream_answer_question(...)` to get `(sources, token_iterator)`.
3. Inside the generator you hand to `StreamingResponse`, yield in this order:
   - **the `sources` frame** — map each `Document → SourceDocument` with the *exact same
     code you already have* at [api.py:59-64](../app/api.py#L59-L64) (`content`, `source`,
     `page_number=.get(...)`), then JSON-encode the list into the `{"type":"sources", ...}`
     shape and yield it plus a newline;
   - **a `token` frame per token** — for each string from `token_iterator`, yield
     `{"type":"token","text": ...}` + newline;
   - **the `done` frame** — once, after the loop.
   Each frame is one JSON object followed by `"\n"` (that newline is the NDJSON delimiter the
   client splits on).
4. Set `media_type="application/x-ndjson"` on the `StreamingResponse`.

Two specifics:

- **No `response_model`.** It only applies to a single returned object; a stream has none.
  Omit it (this route won't appear in `/docs` with a nice schema — that's expected, and the
  protocol is documented here instead).
- **Keep `/query`** ([api.py:47](../app/api.py#L47)) exactly as-is. It's the non-streaming
  fallback and several things still lean on it (the `__main__` checks, anything that wants a
  single JSON blob). Adding `/query/stream` is purely additive.

---

## 5. `app/schemas.py` — optional, almost certainly nothing

A stream isn't one Pydantic body, so `QueryResponse` doesn't model it. You already have
`SourceDocument` for the sources frame, and the token/done frames are trivial. Per "concrete
over abstract" + "simplest thing that works," just emit plain dicts in §4 and skip new event
models. (You *may* add a small `StreamEvent`-ish model purely for documentation, but it buys
nothing functional — I'd leave it.)

---

## 6. `frontend/api_client.py` — a `query_stream` generator

Add a streaming sibling to `query()`. It uses httpx's streaming API and is itself a
**generator** that yields the parsed event dicts (`{"type": ...}`) as they arrive.

The shape (in prose, since the *gotcha* is structural):

- Open the request as a context manager: `with client.stream("POST", "/query/stream",
  json={...}) as response:` — same JSON body as `query()`.
- Iterate `response.iter_lines()`, and for each non-empty line `json.loads` it and **yield**
  the resulting dict.
- The shared module-level `client` ([api_client.py:7](../frontend/api_client.py#L7)) is fine
  as-is; its 300s timeout governs the gap *between* chunks, which stays tiny while tokens
  flow.

> **Gotcha #1 — yield from *inside* the `with` block.** The connection lives only as long as
> the `with client.stream(...)` context is open. If you collect lines, exit the `with`, and
> *then* try to yield — or if the generator is abandoned early — the stream is already closed
> and you'll get nothing or an error. So every `yield` must happen *inside* the `with` and
> inside the `iter_lines()` loop. The whole point is that the caller pulls items while the
> socket is still open.

> **Gotcha #2 — no `.json()`, and `raise_for_status()` timing.** In streaming mode the body
> isn't read yet, so you cannot call `response.json()` (that's the non-streaming path). For
> error handling: the status line + headers *are* available right after the `with` opens, so
> call `response.raise_for_status()` there, before the iteration loop, to catch a 4xx/5xx up
> front. One subtlety — to read the error *detail* body on a streaming response you must
> `response.read()` first (httpx hasn't pulled it). For v1 you can keep it simple: let
> `raise_for_status()` raise and let the page show a generic message.

Keep `query()` ([api_client.py:15](../frontend/api_client.py#L15)) for the fallback path.

---

## 7. `frontend/pages/1_chat.py` — render with `st.write_stream`

This is where the payoff becomes visible. Streamlit's `st.write_stream(gen)` consumes a
generator, renders each string chunk live (with a typing cursor), **and returns the full
concatenated string** when the generator is exhausted — which is perfect, because you store
that return value in `session_state` exactly like `answer` is stored today.

Replace the buffering block ([1_chat.py:60-70](../frontend/pages/1_chat.py#L60-L70)) — the
`api_client.query(...)` → `st.markdown(answer)` part — with this flow:

1. Call `api_client.query_stream(prompt, history, mode)` to get the event generator.
2. **Pull the sources event off first.** It's guaranteed to arrive before any token (§1), so
   advance the generator once (e.g. `next(...)`) and read its `"sources"` list. Stash it.
3. **Feed the *rest* to `st.write_stream`, adapted to strings.** `st.write_stream` wants
   strings, but your generator yields event *dicts*. So wrap the remaining generator in a
   tiny adapter that yields `event["text"]` for each `"token"` event and stops at `"done"`.
   Pass that adapter to `st.write_stream`; capture its return value as the full answer text.
   (Because a generator is a single forward iterator, pulling the sources event in step 2
   leaves it positioned right at the first token — exactly what you feed in here.)
4. After the stream finishes, call the existing `render_sources(sources)`
   ([1_chat.py:29](../frontend/pages/1_chat.py#L29)) to draw the expander — unchanged,
   because the sources payload is the same shape it always was (§1).
5. Append `{"role": "assistant", "content": <full text>, "sources": sources}` to
   `st.session_state.messages`, exactly as today ([1_chat.py:72](../frontend/pages/1_chat.py#L72)).

A few notes:

- **Spinner.** Keep a short `st.spinner("Thinking…")` around steps 1–2 only — the
  blocking prelude (condense + retrieve + rerank) is the part with no visible output. Once
  tokens start flowing into `st.write_stream`, the live text *is* the progress indicator, so
  the spinner's job is done.
- **History replay is unchanged.** The loop that redraws past turns
  ([1_chat.py:41-45](../frontend/pages/1_chat.py#L41-L45)) still uses `st.markdown` — you do
  **not** re-stream old messages, you only stream the new one. Leave that loop alone.
- **Errors.** Wrap the streaming consume in the same `try/except httpx.HTTPError` spirit as
  today ([1_chat.py:62-68](../frontend/pages/1_chat.py#L62-L68)). If it fails before any
  token, show the error message in the bubble; if it dies mid-stream, the partial text is
  already on screen — append what you have (or a short "(interrupted)" note) so
  `session_state` stays consistent.

---

## 8. Checklist before you ask me to review

- [ ] Streaming core chain is `rag_prompt | llm | parser` — **`format_output` removed** from
      it (the buffering trap).
- [ ] `stream_answer_question` reuses `standalone_question`, `retrieval.retriever`, and
      `format_chunks`; condense + retrieve stay blocking; returns `(sources, token_iterator)`.
- [ ] Empty-store guard mirrored (empty sources + one-item message iterator), no `NoneType`.
- [ ] `answer_question` and the existing chain untouched; `/query` still works.
- [ ] `/query/stream` is a `def` route returning `StreamingResponse` over a `def` generator,
      `media_type="application/x-ndjson"`, no `response_model`.
- [ ] Frames are exactly `sources` (once, first) → `token` (many) → `done` (once), one JSON
      object per line; sources frame reuses the `Document → SourceDocument` mapping.
- [ ] `query_stream` yields parsed events from **inside** the `with client.stream(...)` block;
      `query()` kept for fallback.
- [ ] Chat page pulls sources first, feeds a token-only adapter to `st.write_stream`, stores
      the returned full text, then `render_sources`; history replay loop unchanged.
- [ ] (Optional) TTFT + total time logged via a stream-aware timer in chains.py.

## 9. Verification

1. **Backend alone (the real test of streaming).** Ollama up, a doc ingested, then:
   ```bash
   curl -N -X POST http://localhost:8000/query/stream \
     -H "Content-Type: application/json" \
     -d '{"question":"What is multicollinearity?"}'
   ```
   `-N` disables curl's buffering so you see frames live. Confirm: the `{"type":"sources",...}`
   line appears **first and immediately**, then `{"type":"token",...}` lines tick out
   **one at a time** (not all at once at the end — if they dump together, suspect a leftover
   buffering step like `format_output`, or a proxy), then a final `{"type":"done"}`.
2. **Frontend.** `streamlit run frontend/app.py`, ask a question, and watch the answer *type
   itself out* while the Sources expander appears after. Compare the felt latency against the
   old path (flip back to `query()` mentally) — first words in ~a second or two vs. waiting
   for the whole answer.
3. **Multi-turn.** Toggle multi-turn on, ask a follow-up ("why does it matter?"), and confirm
   it's answered as if standalone — i.e. the condense still fires (blocking) and only the
   final answer streams; the sources reflect the *condensed* question.
4. **Regression.** `curl` the old `POST /query` and confirm it still returns a single full
   `QueryResponse` JSON. Nothing about the blocking path changed.

## What's next

This is the last feature on the list — once your streaming implementation passes review, the
project is effectively done. Loose ends to consider closing out: update the "no streaming in
v1" notes in `chains.md`/`ARCHITECTURE.md`, and do a final commit. Tell me when the six edits
are in and I'll review them the usual way.
