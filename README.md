# ASK MY PDF – AI PDF Assistant

Retrieval-Augmented Generation product that parses PDFs, generates embeddings, stores them in a local FAISS index, and exposes secured APIs plus a Streamlit UI for conversational querying.

![Architecture](docs/architecture.png)

## Key Capabilities
- **Model/Logic Layer:** Multi-pass PDF extraction (pypdf, pdfplumber, PyMuPDF), configurable chunking (500–1000 chars, overlap 100–150), LangChain-managed embeddings + FAISS retrieval, and pluggable LLM reasoning (OpenAI by default, Gemini or Hugging Face optional) with latency + retrieval metrics.
- **Backend API Layer (FastAPI):** `/upload`, `/extract-text`, `/query`, and `/history` endpoints with validation, error handling, and history logging.
- **Frontend UI (Streamlit):** Upload PDF, preview extracted text, chat with the document, send manual relevance feedback, and view previous queries.
- **Security & Packaging:** Size/type validation, sanitized filenames, `.env` secrets, threat model, sprint plan, appendix artifacts, requirements, and documented folder structure.

## Repository Structure
```
ASK-MY-PDF/
├── backend/
│   ├── main.py          # FastAPI app + endpoints
│   ├── extract.py       # Validation + multi-layer PDF extraction
│   ├── rag.py           # Chunking, embeddings, FAISS pipeline
│   └── vector_store/    # (reserved for future adapters)
├── frontend/
│   ├── app.py           # Streamlit UI
│   ├── components/      # UI helpers
│   └── (legacy static assets)
├── data/uploaded_files/ # Secure storage for uploads
├── models/              # Serialized FAISS index + metadata
├── docs/                # Architecture diagram, threat model, sprint plan, appendix
├── scripts/             # Automation helpers (generate_docs.py)
├── README.md
└── requirements.txt
```

## Setup
1. **Clone & create venv**
   ```bash
   git clone <repo>
   cd ASKMYPDF
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```
2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Environment variables (`.env`)**
   ```env
   LLM_PROVIDER=openai           # openai | gemini | huggingface
   OPENAI_API_KEY=your_openai_key
   OPENAI_MODEL=gpt-4o-mini
   GOOGLE_API_KEY=your_gemini_key   # only if LLM_PROVIDER=gemini
   GEMINI_MODEL=gemini-1.5-flash
   HUGGINGFACEHUB_API_TOKEN=your_hf_token   # only if LLM_PROVIDER=huggingface
   HUGGINGFACE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
   HUGGINGFACE_API_BASE=https://router.huggingface.co/v1
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   LLM_MAX_TOKENS=512
   ```
4. **Run backend**
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
5. **Run Streamlit UI**
   ```bash
   cd frontend
   streamlit run app.py
   ```

## API Reference
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check, doc + history counts |
| `/upload` | POST (multipart) | Validates + stores PDF (<20MB) and returns `doc_id` |
| `/extract-text` | POST (JSON) | Extracts text/tables, chunking + embeddings, indexes FAISS |
| `/query` | POST (JSON) | RAG query with latency/retrieval metrics + optional feedback |
| `/history` | GET | Recent Q/A pairs with metrics for auditing |

### Example: upload + extract + query
```powershell
curl.exe -F "file=@sample.pdf" http://localhost:8000/upload
curl.exe -X POST http://localhost:8000/extract-text ^
  -H "Content-Type: application/json" ^
  --data "{\"doc_id\":\"<id>\",\"chunk_size\":600,\"overlap\":120}"
curl.exe -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  --data "{\"doc_id\":\"<id>\",\"question\":\"Summarize the executive brief\",\"top_k\":5}"
```

## Evaluation Metrics
- **Latency (ms):** Wall-clock duration per query, displayed in UI metrics and history log.
- **Retrieval Accuracy:** Ratio of high-scoring chunks (score ≤ 1.0 distance) over returned chunks.
- **Manual Relevance Score:** Optional slider in UI, stored per query for manual QA review.

## Security Highlights
- File validation (extension, MIME, <20MB) and sanitised filenames.
- Text sanitization removes control characters and dangerous payloads.
- `.env` for API keys, no secrets rendered client-side.
- Full STRIDE threat model in `docs/threat_model.pdf`.
- History capped to last 20 entries to limit data exposure.

## Agile & Documentation Artifacts
- `docs/architecture.png`, `docs/threat_model.pdf`, `docs/sprint_plan.pdf`
- `docs/appendix.md` (product backlog, release plan, sprint plan, burndown, folder structure, etc.)
- `SPRINT_TRACKING.md` for detailed progress (updated alongside this release).

## Future Enhancements
- Multi-document collections + semantic reranking.
- Persistent history per user session + auth.
- Multi-lingual embeddings and translation support.
- Cloud deployment recipe (HF Spaces, Cloud Run) with CI/CD.

## License
MIT – contributions welcome via issues or pull requests.

