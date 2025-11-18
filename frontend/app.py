import os
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="ASK MY PDF", layout="wide")
st.title("ASK MY PDF – AI PDF Assistant")
st.caption("Upload a PDF, extract context, and chat with it securely.")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "extracted" not in st.session_state:
    st.session_state.extracted = None


with st.sidebar:
    st.header("System Status")
    backend_url = st.text_input("Backend URL", BACKEND_URL)
    try:
        status = requests.get(f"{backend_url}/").json()
        st.success(f"{status.get('status', 'unknown')} – {status.get('documents', 0)} docs indexed")
    except Exception as err:
        st.error(f"Backend unreachable: {err}")
        st.stop()

st.subheader("1. Upload PDF")
uploaded_file = st.file_uploader("Select a PDF under 20MB", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Uploading..."):
        response = requests.post(
            f"{backend_url}/upload",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
        )
    if response.status_code == 200:
        payload = response.json()
        st.session_state.doc_id = payload["doc_id"]
        st.success(f"Uploaded {payload['filename']} ({payload['size_mb']} MB)")
    else:
        st.error(response.json().get("detail", "Upload failed"))

if st.session_state.doc_id:
    st.subheader("2. Extract Text & Build Vector Store")
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk size", min_value=200, max_value=1500, value=500, step=50)
    with col2:
        overlap = st.slider("Chunk overlap", min_value=50, max_value=400, value=100, step=10)

    if st.button("Extract & Index", use_container_width=True):
        with st.spinner("Extracting..."):
            response = requests.post(
                f"{backend_url}/extract-text",
                json={"doc_id": st.session_state.doc_id, "chunk_size": chunk_size, "overlap": overlap},
            )
        if response.status_code == 200:
            st.session_state.extracted = response.json()
            st.success("PDF processed successfully.")
        else:
            st.error(response.json().get("detail", "Extraction failed"))

if st.session_state.get("extracted"):
    st.subheader("Extracted Preview")
    st.write(st.session_state.extracted["text_preview"])
    st.info(
        f"Chunks indexed: {st.session_state.extracted['chunks_indexed']} "
        f"| Tables detected: {st.session_state.extracted['tables_detected']}"
    )

if st.session_state.doc_id and st.session_state.get("extracted"):
    st.subheader("3. Ask a Question")
    question = st.text_area("Your question", placeholder="What does the executive summary say?")
    relevance = st.slider("Manual relevance feedback (optional)", 0.0, 1.0, 0.5, 0.1)
    top_k = st.number_input("Top-K chunks", min_value=1, max_value=10, value=5, step=1)

    if st.button("Submit Question", use_container_width=True):
        if question.strip():
            with st.spinner("Querying..."):
                response = requests.post(
                    f"{backend_url}/query",
                    json={
                        "doc_id": st.session_state.doc_id,
                        "question": question.strip(),
                        "top_k": top_k,
                        "relevance_score": relevance,
                    },
                )
            if response.status_code == 200:
                data = response.json()
                st.success(data["answer"])
                st.metric("Latency (ms)", round(data["metrics"]["latency_ms"], 2))
                st.metric("Chunks retrieved", data["metrics"]["retrieved"])
                st.metric("Retrieval accuracy", round(data["metrics"]["retrieval_accuracy"], 2))
                st.metric("Relevance score", data["metrics"].get("relevance_score", "N/A"))
                with st.expander("Context Chunks"):
                    for chunk in data["relevant_chunks"]:
                        st.write(chunk)
            else:
                st.error(response.json().get("detail", "Query failed"))

st.subheader("4. Recent History")
history_response = requests.get(f"{backend_url}/history")
if history_response.status_code == 200 and history_response.json().get("history"):
    for entry in history_response.json()["history"]:
        with st.container(border=True):
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(f"**A:** {entry['answer']}")
            st.caption(
                f"Chunks: {entry['chunks_returned']} · "
                f"Latency: {round(entry['metrics']['latency_ms'], 2)} ms · "
                f"Relevance: {entry['metrics'].get('relevance_score') or 'N/A'}"
            )
else:
    st.info("No history yet. Upload and ask a question to populate this section.")

