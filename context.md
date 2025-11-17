# Context.md: AskMyPDF (Cloud-Native RAG Agent)

## 1. Project Overview & Goal

**Project Name:** AskMyPDF: Cloud-Native RAG Agent
**Core Value:** Develop a scalable, production-ready RAG application that chat with PDFs, prioritizing cloud deployment, security, and the use of open-source Hugging Face models.

**Architecture Constraint:** This project must follow a **Cloud-Native** architecture. This means the LLM and Vector Database must be externally hosted/managed APIs, and the application itself must be a lightweight **FastAPI** service, ready for deployment to platforms like Huging Face Spaces or Cloud Run.

**NO local inference or local FAISS is allowed.**

---

## 2. Sprint 1 Goal (Core Functional Loop)

The goal for Sprint 1 (3 weeks) is to establish the **Minimum Viable RAG Pipeline**. This means successfully performing data ingestion, retrieval, and generation using the required **external/cloud components**, accessible via a local FastAPI server.

**Deliverable:** A functional `api_server.py` that, upon startup, successfully connects to all external services and can execute a full RAG cycle.

---

## 3. Technical Context & Reusability

We will reuse the proven core RAG logic from the existing files, but integrate new wrappers.

### A. Reusable Core Functions (`rag_utils.py`):

The following functions are **essential** and are reused by the FastAPI service:

* `open_and_read_pdf`: PDF text extraction.
* `create_chunks_from_pages`: Text chunking logic.
* `text_formatter`: Text cleaning.
* `create_embeddings`: Embedding generation (still runs locally on the client's embedding model).

Vector search and LLM orchestration now live directly inside `api_server.py` so the utilities stay lightweight.

### B. New Required Libraries (`requirements.txt` updates):

The existing `requirements.txt` must be updated to replace local tools with cloud/API tools:

| Local Tool (Removed) | Cloud-Native Tool (Added) | Purpose |
| :--- | :--- | :--- |
| `streamlit` | **`fastapi` & `uvicorn`** | API infrastructure. |
| `faiss-cpu` | **`pinecone-client`** | Cloud-managed Vector Database. |
| (Not present) | **Provider SDKs (google-generativeai / openai) + httpx** | Hosted LLM access. |

---

## 4. Sprint 1 Tasks

These tasks correspond to the highest priority stories (R-01, R-02, D-01, S-01, M-01) and must be completed in order.

### Task 1: Setup & Environment

* **Action:** Create a new Python project structure, including `api_server.py` and a new `requirements.txt`.
* **Update:** Modify `requirements.txt` to include `fastapi`, `uvicorn`, the hosted LLM SDKs (`google-generativeai`, `openai`), and the chosen vector DB client.
* **Constraint:** Set up environment variables for the Hugging Face API key (`HUGGINGFACEHUB_API_TOKEN`) and the chosen Vector DB API key/endpoint.

### Task 2: Implement Hosted LLM Inference (R-01)

* **Action:** In `api_server.py`, initialize the selected hosted LLM provider (Gemini, OpenAI, or Hugging Face Inference API) using their official SDKs or HTTPS endpoints.
* **Constraint:** This must be initialized **once** during server startup and stored in the application state.

### Task 3: Implement Cloud Vector Store (R-02)

* **Action:** Implement the necessary functions to connect to the external Vector Database (Pinecone/Weaviate) client and replace the use of `faiss.Index`.
* **Constraint:** The ingestion pipeline must successfully create embeddings (using the existing `create_embeddings` utility) and upload them to the cloud database upon file processing.

### Task 4: Define Core FastAPI Endpoints (D-01)

* **Action:** Define two primary asynchronous POST endpoints in `api_server.py`:
    1.  `/process_pdf`: Accepts file upload, runs `create_embeddings`, and pushes to the cloud Vector DB.
    2.  `/query`: Accepts question JSON, retrieves data from the Vector DB, and generates the answer via the hosted LLM.

### Task 5: Initial Governance and Metrics (S-01/M-01)

* **Action:** Implement a middleware or dependency function in FastAPI to perform an initial **API Key check** on the `/query` endpoint.
* **Action:** Implement basic **Cost Logging** to track the success/failure of the LLM API calls in the console output.