# Context.md: Ask My PDF – LangChain RAG Assistant

## 1. Project Overview

- **Goal:** Deliver a secure, production-ready Retrieval-Augmented Generation experience for chatting with PDFs. 
- **Current Architecture:** 
  - **Backend:** FastAPI (`backend/main.py`) with LangChain for chunking, FAISS retrieval, and LLM orchestration.
  - **LLMs:** Swappable providers (OpenAI, Gemini, Hugging Face router). Hugging Face calls are routed through the OpenAI-compatible API.
  - **Vector Store:** Local LangChain FAISS index persisted in `models/langchain_index`.
  - **Frontend:** Streamlit (`frontend/app.py`) for upload, extraction preview, chat, and history.
  - **Docs:** Architecture diagram, sprint plan, threat model, troubleshooting playbook, sprint tracking.

The original “cloud-only” constraint has been relaxed; the project now optimises for reproducible local LangChain workflows with optional cloud LLMs.

---

## 2. Evolution by Sprint

| Sprint | Focus | Delivered Artifacts |
|--------|-------|---------------------|
| **Sprint 1** | Minimum viable RAG (PDF → chunks → embeddings → QA) | FastAPI skeleton, PDF extraction utilities, initial `/upload` + `/query` endpoints. |
| **Sprint 2** | API + UI polish | Streamlit front-end, `/extract-text`, `/history`, input validation, document registry. |
| **Sprint 3** | Security & robustness | STRIDE threat model, validation for file type/size, auto-index safeguards, router troubleshooting. |
| **Sprint 4** | Packaging | LangChain migration, Hugging Face router support, documentation appendix, requirements refresh. |

---

## 3. Technical Building Blocks

### Backend (`backend/`)
- `main.py`: FastAPI app, environment config, document registry, LangChain LLM factory, auto-index helpers.
- `extract.py`: Multi-layer PDF extraction (pdfplumber, PyPDF, PyMuPDF) with sanitisation and validation.
- `rag.py`: LangChain pipeline encapsulating chunking, embeddings, FAISS persistence, LangChain prompt → LLM chain, and query metrics logging.

### Frontend (`frontend/`)
- `app.py`: Streamlit workflow (upload → extract → chat → history). Communicates via the FastAPI endpoints; includes manual relevance slider and status indicators.

### Models & Data
- `models/`: Serialized FAISS index (`langchain_index/index.faiss` + `index.pkl`) plus document metadata JSON.
- `data/uploaded_files/`: Temporary PDF storage (auto-cleaned in production deployments).

### Docs & Scripts
- `docs/`: Architecture diagram, sprint plan PDF, threat model, appendix.
- `SPRINT_TRACKING.md`, `TROUBLESHOOTING.md`: Operations references.
- `scripts/generate_docs.py`: Rebuilds architecture image + PDFs.

---

## 4. Key Decisions
1. **LangChain-first pipeline:** Chosen for traceability, easier chaining, and a growing ecosystem of retrievers and LLM wrappers.
2. **Local FAISS persistence:** Keeps the repo self-contained and avoids Pinecone/Weaviate dependencies; acceptable for single-tenant use cases.
3. **Hugging Face router compatibility:** Instead of the deprecated `api-inference` host, the OpenAI-compatible router is used when `LLM_PROVIDER=huggingface`.
4. **Auto-index safety net:** `/query` will re-trigger chunking/indexing if the document isn’t processed yet, avoiding “document not found” loops.

---

## 5. Future Enhancements
- Multi-document namespace management + semantic reranking.
- Auth + persistent chat histories per user.
- Optional cloud vector databases for horizontal scaling.
- CI/CD template (GitHub Actions) and HF Spaces deployment script.