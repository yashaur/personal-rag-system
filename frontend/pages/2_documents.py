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

# A delete triggers a rerun (below) so the list re-fetches without the removed file.
# The API's response is stashed in session_state just before that rerun and surfaced
# here on the next run — this is how we "relay the JSON response" after the list resets.
_status = st.session_state.pop("doc_action_status", None)
if _status is not None:
    (st.success if _status["ok"] else st.error)(_status["msg"])

try:
    docs = api_client.list_documents()
except httpx.HTTPError as e:
    st.error(f"Could not reach the backend: {e}")
    docs = []

if not docs:
    st.info("No documents yet — upload a PDF or TXT above.")
else:
    # Delete-all, gated behind a confirmation so a stray click can't wipe everything.
    confirm_all = st.checkbox("I understand this permanently removes every document")
    if st.button("🗑️  Delete all files", disabled=not confirm_all):
        try:
            resp = api_client.delete_all_files()
            _result = {"ok": True, "msg": f"{resp['message']} ({resp['chunk_count']} chunks removed)", "json": resp}
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            _result = {"ok": False, "msg": f"Delete-all failed: {detail}", "json": None}
        except httpx.HTTPError as e:
            _result = {"ok": False, "msg": f"Could not reach the backend: {e}", "json": None}
        st.session_state["doc_action_status"] = _result
        st.rerun()

    st.divider()

    # One row per document, each with its own 🗑 button. st.dataframe can't host
    # interactive buttons in its cells, so the "table" is laid out with st.columns.
    head = st.columns([6, 2, 2, 1])
    head[0].markdown("**Filename**")
    head[1].markdown("**Pages**")
    head[2].markdown("**Chunks**")
    head[3].markdown("**Delete**")

    for doc in docs:
        fname = doc["filename"]
        c_name, c_pages, c_chunks, c_del = st.columns([6, 2, 2, 1])
        c_name.write(fname)
        c_pages.markdown(doc['page_count'])
        c_chunks.markdown(doc["chunk_count"])
        if c_del.button("🗑️", key=f"del::{fname}", help=f"Delete {fname}"):
            try:
                resp = api_client.delete_single_file(fname)
                _result = {"ok": True, "msg": f"{resp['message']} ({resp['chunk_count']} chunks removed)", "json": resp}
            except httpx.HTTPStatusError as e:
                detail = e.response.json().get("detail", str(e))
                _result = {"ok": False, "msg": f"Failed to delete '{fname}': {detail}", "json": None}
            except httpx.HTTPError as e:
                _result = {"ok": False, "msg": f"Could not reach the backend: {e}", "json": None}
            st.session_state["doc_action_status"] = _result
            st.rerun()
