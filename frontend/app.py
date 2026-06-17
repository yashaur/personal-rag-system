import streamlit as st

st.set_page_config(
    page_title="Personal RAG System",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Personal RAG System")

st.markdown(
    """
    A fully local *chat with your documents* tool. Everything runs on your machine —
    documents are embedded and searched locally, and answers come from a local Ollama model.
    Nothing leaves your computer.

    **Use the pages in the sidebar:**

    - **📄 Documents** — upload PDFs or text files and see what's currently in your knowledge base.
    - **💬 Chat** — ask questions and get answers grounded in your documents, with source citations.
    """
)

st.info(
    "Make sure the backend is running first:  `uvicorn app.main:app --port 8000`  "
    "(with `ollama serve` up and your models pulled).",
    icon="ℹ️",
)
