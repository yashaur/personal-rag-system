# Build Guide — `app/chains.py`

The LCEL layer. This is the file where everything you've built so far finally clicks
together: the `retriever` (retrieval.py), the prompts (prompts.py), and the `llm`
(llm.py) get composed into a single callable that takes a question and returns a **cited
answer**.

By the end you'll have two behaviours behind one entry point:

- **Single-turn** — `question → {answer, sources}`
- **Multi-turn** — `(question + chat_history) →` rewrite the follow-up into a standalone
  question → run the single-turn pipeline → `{answer, sources}`

We build it in five small steps, each runnable and checkable on its own (same incremental
philosophy you used in retrieval.py).

---

## 0. First, the mental model: what is LCEL actually?

You've used `|` already in retrieval.py conceptually, but here it's the whole game, so
let's be precise.

**LCEL = LangChain Expression Language.** Everything is a `Runnable` — an object with an
`.invoke(input)` method. The pipe operator `|` connects them: `a | b` means "run `a`,
feed its output into `b`." The result is itself a Runnable, so you can keep chaining.

Three primitives do 95% of the work, and you'll use all three here:

| Primitive | What it does | Mental picture |
|---|---|---|
| `a \| b \| c` | sequential pipe | output of one is input of the next |
| `RunnableParallel` (or a **plain dict** `{"x": ra, "y": rb}`) | run several runnables on the **same** input, collect results into a dict | a fork: one input, many labelled outputs |
| `RunnableLambda(fn)` | wrap any plain Python function as a Runnable | escape hatch to ordinary code |
| `RunnablePassthrough()` | pass the input straight through, unchanged | a wire |
| `RunnablePassthrough.assign(k=fn)` | run `fn` on the input dict, **add** its result under key `k`, pass the **whole dict** onward | a fork that keeps the original *and* adds a new field |

Two conveniences worth burning into memory, because the sketches below rely on them:

1. **A bare dict in a chain position is auto-promoted to `RunnableParallel`.** So
   `{"context": foo, "question": bar} | prompt` runs `foo` and `bar` on the same input
   and hands `prompt` a dict with `context` and `question` keys.
2. **A bare function where a Runnable is expected is auto-wrapped in `RunnableLambda`.**
   So inside `.assign(docs=lambda x: ...)`, that lambda becomes a Runnable for you.

Why LCEL at all (and why the golden rule bans `LLMChain`/`ConversationalRetrievalChain`)?
Because composing runnables gives you streaming, batching, and async for free, and the
data flow is explicit and inspectable — no hidden chain magic. We won't use streaming in
v1, but the composition style is the foundation everything else sits on.

---

## 1. The contract this file must satisfy

Look ahead at what `api.py` will need (ARCHITECTURE.md §9). The response is:

```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]   # SourceDocument = {content, source, page_number}
```

So `chains.py` must return **both the answer string and the source `Document`s** that
produced it. Hold onto this — it's the single design constraint that shapes the whole
file, and it's exactly where the textbook RAG sketch falls short (see Step 2).

`chains.py` does **not** import the Pydantic schemas and does **not** build
`SourceDocument` objects. It returns raw LangChain `Document`s under a `sources` key;
`api.py` later maps those into `SourceDocument`. Keep that boundary clean — chains.py
knows about retrieval and the LLM, not about HTTP shapes.

---

## 2. Imports (and one import that will bite you if you get it wrong)

These paths are correct for the LangChain 1.x in your venv (I verified them):

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document          # for type hints only

from app.llm import llm
from app.prompts import rag_prompt, condenser_prompt    # your names from prompts.py
import app.retrieval as retrieval                        # <-- module import, on purpose
```

**Read that last line twice.** Do **not** write `from app.retrieval import retriever`.

Here's why. In retrieval.py, `retriever` is a module-level variable that is `None` until
documents exist, and `refresh()` *reassigns* it after each ingest:

```python
retriever = _build_retriever()      # None when the store is empty
def refresh() -> None:
    global retriever
    retriever = _build_retriever()   # rebinds the name to a NEW object
```

`from app.retrieval import retriever` copies the *current value* (likely `None` at import
time) into your module and never sees the rebind. Your chain would be frozen pointing at
`None` forever. By importing the **module** (`import app.retrieval as retrieval`) and
reading `retrieval.retriever` **at call time** (inside a lambda), you always get whatever
`refresh()` last installed. This is a real Python gotcha about names vs. objects, and
it's the kind of thing I'll ask you to explain back in review.

`StrOutputParser` is here because `llm` is a **chat** model (`ChatOllama`) — it returns an
`AIMessage`, not a string. `StrOutputParser()` pulls out the `.content`. Whenever you
pipe `llm` and want plain text, end with `| StrOutputParser()`.

---

## 3. Step 0 — `format_docs(docs) -> str`  *(you implement this)*

The retriever hands back a list of `Document`s. The prompt template wants a single
`{context}` string. `format_docs` bridges them.

But it does one more critical job. Re-read your system prompt in prompts.py: *"When you
state specific information, cite the source filename it came from, as shown in the
context."* The model can only cite what it can **see**. So `format_docs` must stamp each
chunk with its own citation header. If you just `"\n\n".join(d.page_content ...)`, you've
thrown the filenames away and the citation instruction is dead on arrival.

**Your task.** Write `format_docs(docs: list[Document]) -> str` that turns the list into
one string where each chunk is preceded by a citation line. Target output shape:

```
[Source: stats_notes.pdf, page 4]
Multicollinearity occurs when two or more predictors ...

[Source: stats_notes.pdf, page 5]
... inflates the variance of the coefficient estimates ...
```

Hints / requirements:

- The metadata keys come from loaders.py: PDFs have `source` **and** `page_number`; TXT
  files have only `source`. So read the page with `d.metadata.get("page_number")` and
  build the header *without* the page part when it's `None`. Never use `d.metadata["page_number"]`
  directly — it'll `KeyError` on TXT chunks.
- Separate chunks with a blank line (`"\n\n".join(...)`) so they're visually distinct to
  the model.
- Keep it a plain function (no Runnable here) — it's just string work. You'll wrap it
  where needed in the chain.

Write it, then eyeball the output against a couple of real retrieved chunks before moving
on. Garbage context = garbage answer, and this is the easiest place to introduce garbage.

---

## 4. Step 1 — single-turn, answer-only (warm-up)

This mirrors the sketch in ARCHITECTURE.md §8 and exists to get the core pipe working in
your head. Input is a bare question **string**; output is the answer **string**. (We
throw sources away for now and fix that in Step 2 — don't skip ahead.)

```python
single_turn_v1 = (
    {
        "context": RunnableLambda(lambda q: format_docs(retrieval.retriever.invoke(q))),
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)
# in:  "What is multicollinearity?"     out:  "Multicollinearity is ... [Source: ...]"
```

Walk through it slowly, because every later step is a variation on this shape:

1. The input is the question string `q`.
2. The **dict** is a `RunnableParallel`: both values receive the same `q`.
   - `"context"`: retrieve with the *current* retriever (`retrieval.retriever.invoke(q)`
     — note the call-time lookup we set up in Step 2), then `format_docs` the result.
   - `"question"`: `RunnablePassthrough()` just forwards `q` untouched.
3. The dict's output `{"context": "...", "question": "..."}` feeds `rag_prompt`, whose
   template placeholders are exactly `{context}` and `{question}` — they must match these
   keys, which is why naming matters.
4. `rag_prompt` produces a list of messages → `llm` produces an `AIMessage` →
   `StrOutputParser()` extracts the text.

If you can run this and get a grounded, source-citing answer, the spine of your RAG
system works. Verify it before continuing (see Step 8).

---

## 5. Step 2 — return the sources too (the real version)

Here's the problem Step 1 leaves us with. Once you pipe `... | llm | StrOutputParser()`,
the only thing coming out is a string. The `Document`s were created inside the `context`
lambda, consumed by `format_docs`, and dropped on the floor. The API needs them. We can't
fish them back out after the fact — we have to **keep them as we go**.

The fix is `RunnablePassthrough.assign`, which is the single most important pattern in
this file. `.assign(k=fn)` takes a **dict** flowing through the chain, runs `fn` on that
whole dict, and adds the result under key `k` **without discarding the existing keys**.
You chain `.assign` calls to accumulate fields, then reshape at the end.

```python
generate = rag_prompt | llm | StrOutputParser()          # the {context, question} -> answer sub-chain

single_turn = (
    # input: {"question": "..."}
    RunnablePassthrough.assign(
        docs=lambda x: retrieval.retriever.invoke(x["question"])
    )
    # now: {"question": "...", "docs": [Document, ...]}
    | RunnablePassthrough.assign(
        answer=(
            {
                "context": lambda x: format_docs(x["docs"]),
                "question": lambda x: x["question"],
            }
            | generate
        )
    )
    # now: {"question": "...", "docs": [...], "answer": "..."}
    | RunnableLambda(lambda x: {"answer": x["answer"], "sources": x["docs"]})
    # out: {"answer": "...", "sources": [Document, ...]}
)
```

Trace the dict as it grows — this is the whole trick:

- **Input** is now a dict `{"question": ...}`, not a bare string. (That's why we needed a
  string-only warm-up first: `.assign` only works on dicts, because it adds *keys*.)
- **First `.assign(docs=...)`** retrieves once and stashes the `Document`s under `docs`.
  The original `question` is still there. We retrieve exactly once and reuse the result
  for both the answer and the sources — no double retrieval.
- **Second `.assign(answer=...)`** builds the answer *from the docs we already have*. The
  inner dict-runnable maps the current dict into `{context, question}` (note `context`
  reads `x["docs"]`, not a fresh retrieval) and pipes it through `generate`.
- **Final `RunnableLambda`** reshapes the accumulated dict into the clean
  `{answer, sources}` contract, dropping the intermediate `question`/`docs` plumbing keys
  (well — `docs` becomes `sources`).

That last rename is deliberate: the *outside world* (api.py) speaks `sources`; the
*inside* of the chain speaks `docs`. The lambda is the translation point.

> Why not `RunnableParallel({"answer": generate, "sources": retrieve})` at the top? Because
> that would run retrieval **twice** (once for each branch) and the two retrievals could
> even return different docs under load. `.assign` retrieves once and shares. Keep that in
> mind — it's a classic RAG-with-sources mistake.

---

## 6. Step 3 — multi-turn: condense the follow-up first

Multi-turn chat has one job before retrieval: turn a context-dependent follow-up ("why
does it matter?") into a self-contained question ("why does multicollinearity matter in
regression?") so the retriever has something searchable. That's what your
`condenser_prompt` is for.

The condense chain itself is trivial — same pattern as `generate`:

```python
condense = condenser_prompt | llm | StrOutputParser()
```

Two wiring details to handle:

**(a) `chat_history` arrives as dicts, but `MessagesPlaceholder` wants message objects.**
The API/frontend send history as `[{"role": "user"|"assistant", "content": "..."}]`
(ARCHITECTURE.md §9). Your `condenser_prompt` uses `MessagesPlaceholder('chat_history')`,
which expects a list of LangChain messages (`HumanMessage`/`AIMessage`). So you need a
small converter.

**Your task** — write:

```python
def to_messages(history: list[dict]) -> list:
    """[{'role','content'}, ...] -> [HumanMessage | AIMessage, ...]"""
    ...
```

Map `role == "user"` to `HumanMessage(content=...)` and `role == "assistant"` to
`AIMessage(content=...)`. Decide what to do with any other role (skipping it is fine for
v1). Keep it boring and explicit — a `for` loop is perfectly good here.

**(b) Only condense when there's history.** On the first turn there's no history, and
asking the LLM to "rephrase given the conversation" with an empty conversation just wastes
a call and can mangle a perfectly good question. So branch in plain Python:

```python
def standalone_question(x: dict) -> str:
    history = x.get("chat_history") or []
    if not history:
        return x["question"]                 # first turn: use as-is
    return condense.invoke({
        "question": x["question"],
        "chat_history": to_messages(history),
    })
```

A plain function with an `if` is intentional here. You *could* express this with
`RunnableBranch`, but for one binary condition it's harder to read than the function and
buys nothing — and the golden rules tell us to favour the simplest thing that works. We'll
wrap this function as a Runnable in the next step.

---

## 7. Step 4 — unify, guard, expose

Now stitch the condense step in front of `single_turn`, and give `api.py` one clean
function to call.

```python
rag_chain = (
    RunnablePassthrough.assign(question=RunnableLambda(standalone_question))
    | single_turn
)
```

`.assign(question=...)` runs `standalone_question` on the input dict and **overwrites**
the `question` key with the standalone version (assign overwrites an existing key). Then
`single_turn` runs exactly as before — it retrieves and answers using the rewritten
question, and never needs to know whether condensing happened. The `chat_history` key is
still in the dict; `single_turn` simply ignores it. Notice how the multi-turn feature
slots in as one extra step in front of a pipeline that didn't change — that's the payoff
of the explicit-dataflow style.

Finally, the entry point with the empty-store guard:

```python
def answer_question(question: str, chat_history: list[dict] | None = None) -> dict:
    if retrieval.retriever is None:
        return {"answer": "No documents have been ingested yet.", "sources": []}
    return rag_chain.invoke({
        "question": question,
        "chat_history": chat_history or [],
    })
```

The guard matters: before the first ingest, `retrieval.retriever` is `None`, and any
`.invoke` inside the chain would blow up with `AttributeError: 'NoneType' ...`. Catch it
here and return a clean, honest payload instead. This function is what api.py imports and
calls — single-turn is just `answer_question(q)`, multi-turn is
`answer_question(q, history)`.

> **On the request's `mode` flag.** The `QueryRequest` schema has
> `mode: Literal["single","multi"]`. Notice `chains.py` doesn't look at it at all — it
> branches purely on whether `chat_history` is non-empty. That's deliberate: let `api.py`
> own the `mode` flag (it decides whether to forward history to `answer_question`), and
> keep this file reacting only to the data it's given. One concern per layer.

---

## 8. Step 8 — verification  *(you implement this `__main__` block)*

Use the same self-test convention as your other modules (the `if __name__ == '__main__':`
block in retrieval.py is your template — copy its reset/ingest/refresh dance).

Prereqs: `ollama serve` running, with your configured LLM and `nomic-embed-text` pulled,
and a `test.pdf` in `data/uploads/`.

Write a block that does, in order:

1. `from app.ingestion import ingest_file` and reset: `vectorstore.reset_collection()`.
   (Import `vectorstore` for this — or call `retrieval.vectorstore`; either works.)
2. Ingest `settings.uploads_dir + '/test.pdf'`, then **`retrieval.refresh()`** — this is
   the step beginners forget. Without it, `retrieval.retriever` is still `None`/stale and
   your guard returns the empty message. (This is the lazy-lookup gotcha showing up in
   practice — a good thing to feel once.)
3. **Single-turn:** `print(answer_question("What is multicollinearity?"))`. Confirm:
   - `answer` is grounded in the PDF and names a source filename.
   - `sources` is a list of `Document`s whose `.metadata` carries `source` (and
     `page_number` for the PDF).
4. **Multi-turn:** build a tiny history, e.g.
   ```python
   history = [
       {"role": "user", "content": "What is multicollinearity?"},
       {"role": "assistant", "content": "<paste the answer you just got>"},
   ]
   print(answer_question("why does it matter?", history))
   ```
   Confirm the follow-up gets answered as if it asked the full standalone question. If you
   want to *see* the rewrite, temporarily `print(standalone_question({...}))` — watching
   "why does it matter?" become "why does multicollinearity matter?" is the satisfying
   moment of this whole file.
5. **(Optional) guard:** reset the collection, call `refresh()`, then
   `answer_question("anything")` and confirm you get the "No documents..." payload rather
   than a crash.
6. Clean up with `vectorstore.reset_collection()`.

> **What `reset_collection()` does (and doesn't do).** It's a *logical* reset — it deletes
> the `personal_rag` collection from Chroma's sqlite DB and recreates an empty one. It does
> **not** wipe `data/chroma/`. The UUID-named folders in there are per-segment HNSW indexes;
> a delete orphans the old folder and a recreate makes a new one, so the folders **pile up**
> across runs and `chroma.sqlite3` persists. That's expected and harmless — the orphaned
> folders are unreferenced, so retrieval is unaffected and your next ingest starts from a
> genuinely empty collection. If you ever want a truly pristine directory, delete it from
> the shell (`rm -rf data/chroma`) **before** the process starts — not with `shutil.rmtree`
> mid-run, because the `vectorstore` singleton (vectorstore.py) holds the sqlite file open
> from import time.

---

## Checklist before you ask me to review

- [ ] `import app.retrieval as retrieval` (module form), and every retriever access is
      `retrieval.retriever.invoke(...)` inside a lambda — no top-level
      `from app.retrieval import retriever`.
- [ ] `format_docs` emits a `[Source: ...]` header per chunk and survives TXT chunks (no
      `page_number`) without `KeyError`.
- [ ] Retrieval happens **once** per query (via `.assign(docs=...)`), reused for both the
      answer and the returned sources.
- [ ] `answer_question` returns `{"answer": str, "sources": list[Document]}` in **both**
      single- and multi-turn paths.
- [ ] Empty-store guard returns the clean payload, no exception.
- [ ] No Pydantic / `SourceDocument` imports here — that mapping is api.py's job.
- [ ] A `__main__` block that proves single-turn, multi-turn, and (optionally) the guard.

## What's next

After I review chains.py, the build order takes us to **`schemas.py`** (the Pydantic
request/response models — small and quick), then **`api.py`** (where `answer_question`
finally gets wired to `POST /query`, and the `Document → SourceDocument` mapping lives),
then **`main.py`**, then the Streamlit frontend.
