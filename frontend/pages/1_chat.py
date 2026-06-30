import streamlit as st
import httpx

import api_client
import json

from uuid import uuid4

st.set_page_config(page_title="Chat", page_icon="💬")

st.title("💬 Chat")

# --- Sidebar controls -------------------------------------------------------
multi_turn = st.sidebar.toggle(
    "Multi-turn",
    value=True,
    help="When turned on, follow-up questions are interpreted in the context of the conversation "
         "(the backend condenses them into a standalone question before retrieving).",
)

mode = "multi" if multi_turn else "single"

if mode == 'multi':
    if not st.session_state.get('session_id'):
        st.session_state['session_id'] = str(uuid4())
else:
    st.session_state['session_id'] = None

session_id = st.session_state['session_id']

stream_answer = st.sidebar.toggle(
    "Stream LLM response",
    value = True,
    help = "When turned on, the interface streams each word (token) as it is produced instead of "
           "outputting the entire answer at once."
)

stream_mode = "stream" if stream_answer else "once"

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    del st.session_state['session_id']
    st.rerun()

# --- Conversation state -----------------------------------------------------
# Each message: {"role": "user"|"assistant", "content": str, "sources": list (assistant only)}
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            page = f", p. {s['page_number']}" if s.get("page_number") is not None else ""
            st.markdown(f"**{i}. {s['source']}{page}**")
            snippet = s["content"].strip()
            st.caption(snippet[:500] + ("…" if len(snippet) > 500 else ""))


# --- Replay the conversation so far -----------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources"))

# --- Handle a new question --------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prior turns only (exclude the question we just appended), as {role, content}.
    # Always a list — never None — so the backend never sees `chat_history: null`.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        try:    
            if stream_mode == 'stream':
                with st.spinner('Thinking...'):
                    token_generator = api_client.query_stream(prompt, history, mode, session_id)
                    sources = next(token_generator)['sources']

                token_adaptor = (frame['token'] for frame in token_generator if frame['type'] == 'token')
                answer = st.write_stream(token_adaptor)

                render_sources(sources)

            else:
                with st.spinner('Thinking...'):
                    response = api_client.query(prompt, history, mode, session_id)
                    answer = response["answer"]
                    sources = response.get("sources", [])
                    st.markdown(answer)
                    render_sources(sources)

        except httpx.HTTPError as e:
            answer = f"⚠️ Could not get an answer: {e}"
            sources = []

        except Exception as e:
            answer = f"⚠️ Could not get an answer: {e}"
            sources = []
    
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
        )
    