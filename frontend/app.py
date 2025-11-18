import streamlit as st

st.set_page_config(page_title="ASK MY PDF", layout="wide", initial_sidebar_state="expanded")

import base64
import os
from datetime import datetime
from io import BytesIO

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Session state initialization ---
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "extracted" not in st.session_state:
    st.session_state.extracted = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {"latency_ms": 0, "retrieved": 0, "retrieval_accuracy": 0, "relevance_score": 0}
if "current_references" not in st.session_state:
    st.session_state.current_references = {}

# --- Custom CSS ---
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Metrics bar styling */
    .metrics-container {
        background: #2C3E50;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        gap: 1.5rem;
        align-items: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-item {
        color: white;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        margin-left: 0.3rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .user-message {
        background: #25D366;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .assistant-message {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        color: #333;
        margin-right: auto;
        border-bottom-left-radius: 5px;
    }
    
    .references {
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #E0E0E0;
        font-size: 0.8rem;
    }
    
    .reference-link {
        display: inline-block;
        background: #F0F0F0;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem 0.2rem 0.2rem 0;
        border-radius: 5px;
        color: #2C3E50;
        text-decoration: none;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .reference-link:hover {
        background: #2C3E50;
        color: white;
    }
    
    .message-role {
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        opacity: 0.7;
    }
    
    .message-content {
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* PDF viewer styling */
    .pdf-container {
        border: 2px solid #E0E0E0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #34495E !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton>button {
        background: rgba(255, 255, 255, 0.2);
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background: rgba(255, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 20px;
        border: 2px solid #E0E0E0;
        padding: 0.5rem 1rem;
    }
    
    /* Chat container */
    .chat-container {
        height: 500px;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def display_pdf(pdf_bytes):
    """Display PDF using base64 encoding in iframe"""
    if pdf_bytes:
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(f'<div class="pdf-container">{pdf_display}</div>', unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='color: white; text-align: center;'>📄 ASK MY PDF</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Backend status
    try:
        status = requests.get(f"{BACKEND_URL}/", timeout=5).json()
        st.success(f"✅ Backend Connected")
        st.markdown(f"<p style='color: white; font-size: 0.85rem;'>Documents indexed: {status.get('documents', 0)}</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error("❌ Backend Offline")
        st.markdown(f"<p style='color: white; font-size: 0.85rem;'>Make sure backend is running on port 8000</p>", unsafe_allow_html=True)
        st.stop()
    
    st.markdown("---")
    
    # File upload section
    st.markdown("<h3 style='color: white;'>📤 Upload PDF</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file (max 20MB)", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        st.session_state.pdf_bytes = file_bytes
        st.session_state.filename = uploaded_file.name
        
        # Upload to backend
        with st.spinner("Uploading..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": (uploaded_file.name, file_bytes, "application/pdf")},
                )
                if response.status_code == 200:
                    payload = response.json()
                    st.session_state.doc_id = payload["doc_id"]
                    st.success(f"✅ {payload['filename']} ({payload['size_mb']} MB)")
                else:
                    st.error(response.json().get("detail", "Upload failed"))
            except Exception as e:
                st.error(f"Upload error: {str(e)}")
    
    # Extract & Build Index button - shown right after upload
    if st.session_state.doc_id:
        chunk_size = 500
        overlap = 100
        if st.button("🔨 Extract & Build Index", use_container_width=True, type="primary"):
            with st.spinner("Extracting and indexing..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/extract-text",
                        json={
                            "doc_id": st.session_state.doc_id,
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                        },
                    )
                    if resp.status_code == 200:
                        st.session_state.extracted = True
                        data = resp.json()
                        st.success(f"✅ Indexed {data.get('chunks_indexed', 0)} chunks")
                    else:
                        st.error(resp.json().get("detail", "Extraction failed"))
                except Exception as e:
                    st.error(f"Extraction error: {str(e)}")
    
    st.markdown("---")
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        chunk_size = st.slider("Chunk size", 200, 1500, 500, 50)
        overlap = st.slider("Chunk overlap", 50, 400, 100, 10)
        top_k = st.number_input("Top-K chunks", 1, 20, 5, 1)
        manual_relevance = st.slider("Manual relevance feedback", 0.0, 1.0, 0.5, 0.1)
    
    st.markdown("---")
    
    # Chat history management
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.doc_id = None
        st.session_state.extracted = False
        st.session_state.messages = []
        st.session_state.pdf_bytes = None
        st.session_state.filename = None
        st.rerun()

# --- Main Content Area ---
# Metrics bar at the top
metrics = st.session_state.last_metrics
st.markdown(f"""
<div class="metrics-container">
    <div class="metric-item">⚡ Latency: <span class="metric-value">{metrics.get('latency_ms', 0):.0f} ms</span></div>
    <div class="metric-item">📦 Chunks: <span class="metric-value">{metrics.get('retrieved', 0)}</span></div>
    <div class="metric-item">🎯 Accuracy: <span class="metric-value">{metrics.get('retrieval_accuracy', 0):.2f}</span></div>
    <div class="metric-item">⭐ Relevance: <span class="metric-value">{metrics.get('relevance_score', 0):.2f}</span></div>
</div>
""", unsafe_allow_html=True)

# Two column layout: Chat on left, PDF preview on right
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("💬 Chat with Your PDF")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Upload a PDF, extract it, and start asking questions!")
        else:
            for idx, message in enumerate(st.session_state.messages):
                role = message["role"]
                content = message["content"]
                references = message.get("references", [])
                
                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-role" style="color: white;">You</div>
                        <div class="message-content" style="color: white;">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Build references HTML
                    refs_html = ""
                    if references:
                        refs_html = '<div class="references"><strong>📎 References:</strong> '
                        for ref in references:
                            page = ref.get("page", 1)
                            refs_html += f'<a class="reference-link" href="#page{page}" onclick="return false;">Page {page}</a> '
                        refs_html += '</div>'
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="message-role">AI Assistant</div>
                        <div class="message-content">{content}</div>
                        {refs_html}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat input at the bottom
    st.markdown("---")
    user_question = st.text_input("Ask a question about your PDF:", key="user_input", placeholder="What is this document about?")
    
    col_send, col_clear = st.columns([4, 1])
    
    with col_send:
        send_button = st.button("📤 Send", use_container_width=True, type="primary")
    
    with col_clear:
        if st.button("🗑️", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    if send_button and user_question.strip():
        if not st.session_state.doc_id:
            st.error("⚠️ Please upload a PDF first")
        elif not st.session_state.extracted:
            st.error("⚠️ Please extract and index the PDF first (check Advanced Settings)")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_question.strip()})
            
            # Query backend
            with st.spinner("🤔 Thinking..."):
                try:
                    top_k = st.session_state.get("top_k", 5) if "top_k" in st.session_state else 5
                    manual_relevance = st.session_state.get("manual_relevance", 0.5) if "manual_relevance" in st.session_state else 0.5
                    
                    resp = requests.post(
                        f"{BACKEND_URL}/query",
                        json={
                            "doc_id": st.session_state.doc_id,
                            "question": user_question.strip(),
                            "top_k": top_k,
                            "relevance_score": manual_relevance,
                        },
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "No answer generated")
                        references = data.get("references", [])
                        
                        # Add assistant message with references
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "references": references
                        })
                        
                        # Update metrics
                        st.session_state.last_metrics = data.get("metrics", {})
                        
                        st.rerun()
                    else:
                        error_msg = resp.json().get("detail", "Query failed")
                        st.error(f"❌ {error_msg}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with col2:
    st.subheader("📄 PDF Preview")
    
    if st.session_state.pdf_bytes:
        st.caption(f"**{st.session_state.filename}**")
        display_pdf(st.session_state.pdf_bytes)
    else:
        st.info("📂 No PDF uploaded yet. Upload a file from the sidebar to preview it here.")
        st.image("https://via.placeholder.com/400x600/EFEFEF/666666?text=PDF+Preview", width=400)

