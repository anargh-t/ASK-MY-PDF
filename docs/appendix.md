# Appendix – ASK MY PDF Artifacts

## A. Product Backlog
- As a researcher, I want to upload lengthy PDFs so that I can explore them conversationally.
- As an analyst, I want the system to highlight which chunks powered each answer so I can trust the context.
- As a security reviewer, I want validated, size-limited uploads so malicious files are rejected.
- As a product owner, I want latency metrics so we can monitor responsiveness.
- As an end user, I want prior conversations to remain accessible so I can revisit insights.

## B. Release Plan
- **v0.1 – Chat Only:** UI hits Gemini directly without document context.
- **v0.2 – Text Extraction:** Add PDF ingestion, chunking, and manual query context.
- **v0.3 – RAG End-to-End:** Introduce FAISS index, retrieval, and evaluation metrics.
- **v1.0 – Secure Product:** Hardened FastAPI backend, Flask UI, documentation, and packaging assets.

## C. High-Level Sprint Plan
1. **Sprint 1:** Model development (extraction, chunking, embeddings, FAISS, metrics).
2. **Sprint 2:** Backend API + Flask UI, full CRUD endpoints, upload/extract/query/history.
3. **Sprint 3:** Security posture, STRIDE analysis, logging, input validation, failure simulations.
4. **Sprint 4:** Packaging, docs, release checklist, demo video hook, README polish.

## D. Sprint Planning Sheet (Story Points)
| Story | Description | Points |
|-------|-------------|--------|
| S1-1 | Implement multi-layer PDF extraction | 5 |
| S1-2 | Build chunker + embedding generator | 3 |
| S1-3 | Persist FAISS index + metadata | 5 |
| S2-1 | Design FastAPI endpoints (/upload,/extract-text,/query,/history) | 8 |
| S2-2 | Implement Flask UI with history | 5 |
| S3-1 | Add validation + exception handling + STRIDE docs | 5 |
| S4-1 | Produce appendix artifacts + README + packaging | 3 |

## E. Burndown (Sample)
Day 1: 34 pts → Day 2: 28 pts → Day 3: 21 pts → Day 4: 13 pts → Day 5: 5 pts → Demo: 0 pts.

## F. Folder Structure
```
ASKMYPDF/
│
├── backend/                          # FastAPI backend service
│   ├── __init__.py
│   ├── main.py                       # FastAPI app with /upload, /extract-text, /query, /history endpoints
│   ├── extract.py                    # Multi-layer PDF extraction (pypdf, pdfplumber, PyMuPDF)
│   ├── rag.py                        # RAG pipeline: chunking, embeddings, FAISS, Gemini integration
│   ├── vector_store/                 # Vector store utilities
│   └── __pycache__/                  # Python bytecode cache
│
├── frontend/                         # Flask web application
│   ├── app.py                        # Flask routes and UI logic
│   ├── templates/                    # Jinja2 HTML templates
│   │   ├── base.html                 # Base template with navigation
│   │   ├── index.html                # Main chat interface
│   │   └── landing.html              # Landing page
│   ├── static/                       # Static assets
│   │   ├── css/
│   │   │   └── style.css             # Custom styles
│   │   └── js/
│   │       └── app.js                # Frontend JavaScript
│   └── __pycache__/                  # Python bytecode cache
│
├── data/                             # Data storage directory
│   └── uploaded_files/               # User-uploaded PDF files
│       └── [UUID-based filenames]    # Sanitized PDF storage
│
├── models/                           # ML model artifacts
│   ├── embeddings.index              # FAISS vector index (persisted)
│   ├── embeddings_meta.json          # Embedding metadata
│   ├── documents_meta.json           # Document registry metadata
│   └── langchain_index/              # Alternative LangChain index
│       ├── index.faiss
│       └── index.pkl
│
├── docs/                             # Project documentation
│   ├── appendix.md                   # Complete appendix artifacts
│   ├── architecture.png              # System architecture diagram
│   ├── threat_model.pdf              # STRIDE threat model document
│   └── sprint_plan.pdf               # Sprint planning visualization
│
├── scripts/                          # Utility scripts
│   └── generate_docs.py              # Documentation generation automation
│
├── __init__.py                       # Package initialization
├── README.md                         # Project overview, setup, usage guide
├── ARCHITECTURE.md                   # Detailed system architecture documentation
├── SPRINT_TRACKING.md                # Sprint progress, burndown, metrics
├── TROUBLESHOOTING.md                # Common issues and solutions
├── requirements.txt                  # Python dependencies (FastAPI, Flask, FAISS, etc.)
├── test_client.py                    # API testing client
└── .gitignore                        # Git ignore patterns
```

## G. requirements.txt
All Python dependencies required to run the project.

```txt
# FastAPI backend
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
python-dotenv==1.0.0
pydantic==2.5.2

# PDF processing
pypdf==3.17.1
pdfplumber==0.11.4
PyMuPDF==1.23.7

# RAG + embeddings
sentence-transformers==2.3.1
faiss-cpu>=1.8.0

# LLM providers
google-generativeai==0.5.2
openai==1.54.1
huggingface-hub==0.25.2

# LangChain stack
langchain==0.2.6
langchain-core==0.2.10
langchain-community==0.2.6
langchain-openai==0.1.7
langchain-google-genai==1.0.5

# Frontend
flask==3.0.0
werkzeug==3.0.1
requests==2.31.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

## C. README.md (Project Documentation)

### Project Overview

**ASK MY PDF** is a production-ready Retrieval-Augmented Generation (RAG) system that enables natural language conversations with PDF documents. The system features:

- **Multi-layer PDF Extraction:** Robust parsing with automatic fallback chain (pdfplumber → PyPDF → PyMuPDF)
- **Advanced RAG Pipeline:** LangChain-powered chunking, local HuggingFace embeddings, and FAISS vector search
- **Flexible LLM Support:** Support for OpenAI GPT, Google Gemini, and Hugging Face models
- **Conversation-Aware:** Maintains conversation context for multi-turn dialogues
- **Real-time Metrics:** Live performance tracking (latency, retrieval accuracy, chunk relevance)
- **Modern UI/UX:** Responsive Flask web interface with drag-and-drop upload and chat history
- **Security-First:** Input validation, file sanitization, STRIDE threat modeling

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/anargh-t/ASK-MY-PDF.git
   cd ASK-MY-PDF
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/Mac
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # LLM Provider (Choose ONE)
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-your-api-key-here
   OPENAI_MODEL=gpt-4o-mini
   
   # OR use Gemini
   # LLM_PROVIDER=gemini
   # GOOGLE_API_KEY=your-gemini-api-key-here
   
   # Embedding Configuration
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   
   # Flask Configuration
   SECRET_KEY=your-secret-key-for-flask-sessions
   BACKEND_URL=http://localhost:8000
   ```

### Running Backend + Frontend

**Terminal 1 - Start Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```
Backend available at `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
python app.py
```
Frontend available at `http://localhost:5000`

**Access the application:** Open browser at `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```http
GET /
```
**Response:**
```json
{
  "status": "healthy",
  "documents": 5,
  "history": 23
}
```

#### 2. Upload PDF
```http
POST /upload
Content-Type: multipart/form-data
```
**Parameters:** `file` (PDF file, max 20MB)

**Response:**
```json
{
  "doc_id": "uuid-string",
  "filename": "document.pdf",
  "size_mb": 2.5,
  "uploaded_at": "2025-11-25T10:30:00"
}
```

#### 3. Extract and Index Text
```http
POST /extract-text
Content-Type: application/json
```
**Request Body:**
```json
{
  "doc_id": "uuid-string",
  "chunk_size": 500,
  "overlap": 100
}
```

**Response:**
```json
{
  "doc_id": "uuid-string",
  "filename": "document.pdf",
  "chunks_indexed": 45,
  "processed_at": "2025-11-25T10:31:15"
}
```

#### 4. Query Document
```http
POST /query
Content-Type: application/json
```
**Request Body:**
```json
{
  "doc_id": "uuid-string",
  "question": "What are the key findings?",
  "top_k": 5,
  "conversation_history": []
}
```

**Response:**
```json
{
  "answer": "Based on the document, the key findings are...",
  "relevant_chunks": ["Chunk 1 text...", "Chunk 2 text..."],
  "references": [{"page": 5, "text": "Reference snippet..."}],
  "metrics": {
    "latency_ms": 1250.5,
    "retrieved": 5,
    "retrieval_accuracy": 0.92
  }
}
```

#### 5. Get Query History
```http
GET /history
```
**Response:**
```json
{
  "history": [
    {
      "doc_id": "uuid-string",
      "question": "What are the conclusions?",
      "answer": "The main conclusions are...",
      "metrics": {
        "latency_ms": 980.2,
        "retrieved": 5,
        "retrieval_accuracy": 0.88
      }
    }
  ]
}
```

### Screenshots

*Note: Screenshots of the working system should be added here, including:*
- Landing page with upload interface
- PDF preview and extraction status
- Chat interface with question/answer display
- Metrics dashboard showing latency and accuracy
- Query history sidebar

*QR Code to GitHub Repository:*
[![QR Code](https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https://github.com/anargh-t/ASK-MY-PDF)](https://github.com/anargh-t/ASK-MY-PDF)

### Contributors

| Member | Role | GitHub Handle |
|--------|------|---------------|
| Anargh | Product & Delivery Lead | [@anargh-t](https://github.com/anargh-t) |
| Anurag | Backend & RAG Engineer | [GitHub Handle] |
| Justin | Frontend & UX Lead | [GitHub Handle] |
| Afnan | Security & DevOps Specialist | [GitHub Handle] |

**Repository:** [ASK-MY-PDF](https://github.com/anargh-t/ASK-MY-PDF)

---

*For complete documentation, see [README.md](../README.md) in the repository root.*

