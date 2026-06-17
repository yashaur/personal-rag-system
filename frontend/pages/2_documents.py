import streamlit as st
import httpx

import api_client

st.set_page_config(page_title="Documents", page_icon="📄")

st.title("📄 Documents")

# --- Upload -----------------------------------------------------------------
st.subheader("Upload")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

# Gate ingestion behind the button: Streamlit re-runs the whole script on every
# interaction, and `uploaded_files` stays populated across re-runs. Without the
# button, the files would be re-ingested (and re-duplicated in Chroma) on every
# re-run. The button is True only on the run triggered by the click.
if uploaded_files and st.button("Ingest", type="primary"):
    for f in uploaded_files:
        with st.spinner(f"Ingesting {f.name}…"):
            try:
                result = api_client.ingest(f)
            except httpx.HTTPStatusError as e:
                detail = e.response.json().get("detail", str(e))
                st.error(f"Failed to ingest **{f.name}**: {detail}")
            except httpx.HTTPError as e:
                st.error(f"Could not reach the backend while ingesting **{f.name}**: {e}")
            else:
                st.success(f"Ingested **{result['filename']}** — {result['chunk_count']} chunks.")

st.divider()

# --- Ingested documents -----------------------------------------------------
st.subheader("Ingested documents")

try:
    docs = api_client.list_documents()
except httpx.HTTPError as e:
    st.error(f"Could not reach the backend: {e}")
    docs = []

if docs:
    st.dataframe(docs, hide_index=True)
else:
    st.info("No documents yet — upload a PDF or TXT above.")
