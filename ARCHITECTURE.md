# Architecture Documentation

## System Overview

ASK MY PDF is a full-stack Retrieval-Augmented Generation (RAG) application that enables natural language querying of PDF documents. This document provides detailed technical architecture information about the actual implementation.

## Table of Contents

1. [Technology Stack](#technology-stack)
2. [System Architecture](#system-architecture)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Storage Architecture](#storage-architecture)
6. [API Architecture](#api-architecture)
7. [Security Architecture](#security-architecture)
8. [Deployment Architecture](#deployment-architecture)

## Technology Stack

### Backend Services

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **API Framework** | FastAPI | 0.104.1 | RESTful API endpoints, async support |
| **Server** | Uvicorn | 0.24.0 | ASGI server with auto-reload |
| **Data Validation** | Pydantic | 2.5.2 | Request/response schema validation |
| **Environment** | python-dotenv | 1.0.0 | Configuration management |

### PDF Processing

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **Primary Extractor** | pdfplumber | 0.11.4 | Text and table extraction |
| **Fallback #1** | PyPDF | 3.17.1 | Standard PDF text extraction |
| **Fallback #2** | PyMuPDF | 1.23.7 | Complex layout handling |

### RAG Pipeline

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **Framework** | LangChain | 0.2.6 | RAG orchestration and chains |
| **LangChain Core** | langchain-core | 0.2.10 | Core abstractions |
| **LangChain Community** | langchain-community | 0.2.6 | FAISS and HF integrations |
| **Embeddings** | sentence-transformers | 2.3.1 | Local text embedding generation |
| **Vector Store** | faiss-cpu | ≥1.8.0 | Similarity search index |

### LLM Providers

| Provider | Technology | Version | Models Supported |
|----------|-----------|---------|------------------|
| **OpenAI** | langchain-openai | 0.1.7 | GPT-4, GPT-4o-mini, GPT-3.5-turbo |
| **Google** | langchain-google-genai | 1.0.5 | Gemini 1.5 Flash, Gemini 1.5 Pro |
| **Hugging Face** | openai (compatible) | 1.54.1 | Llama 3, Gemma 2, Mistral (via router) |
| **HF Hub** | huggingface-hub | 0.25.2 | Model management |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|----------|
| **Web Framework** | Flask | 3.0.0 | Web application server |
| **WSGI** | Werkzeug | 3.0.1 | HTTP utilities |
| **HTTP Client** | requests | 2.31.0 | Backend API communication |
| **Templates** | Jinja2 | (Flask bundled) | HTML templating |
| **UI** | Vanilla JavaScript | - | Frontend logic |
| **Styling** | Custom CSS | - | Responsive design |

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Browser                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  HTML/CSS/JS (app.js) - PDF Viewer - Chat UI - Metrics  │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│         Flask Frontend (Port 5000)                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  app.py                                           │  │
│  │  - Session Management                             │  │
│  │  - Template Rendering                             │  │
│  │  - Backend API Proxy                              │  │
│  │  - PDF File Serving                               │  │
│  │  - Conversation History                           │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ REST API
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│         FastAPI Backend (Port 8000)                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  main.py - API Endpoints & Registry              │  │
│  │  │                                               │  │
│  │  ├─── extract.py - PDF Processing              │  │
│  │  │    - Multi-layer extraction                   │  │
│  │  │    - Validation & sanitization                │  │
│  │  │                                               │  │
│  │  └─── rag.py - RAG Pipeline                   │  │
│  │       - Text chunking                            │  │
│  │       - Embedding generation (LOCAL)             │  │
│  │       - FAISS vector search                      │  │
│  │       - LLM chain construction                   │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┼────────────────┐
                │               │                │
         ┌──────┴──────┐      ┌────┴────┐      ┌────┴────┐
         │  Local      │      │  LLM    │      │ File   │
         │  Storage    │      │  APIs   │      │Storage│
         └────────────┘      └─────────┘      └─────────┘
         FAISS Index       OpenAI        uploaded_files/
         documents_meta    Gemini
                          HF Router
```

### Component Responsibilities

#### Flask Frontend (`frontend/app.py`)
- **Session Management:** Server-side session storage for user state
- **Template Rendering:** Jinja2 templates for HTML generation
- **API Proxy:** Routes frontend requests to FastAPI backend
- **PDF Serving:** Streams PDF files for in-browser preview
- **State Persistence:** Maintains conversation history and document state

#### FastAPI Backend (`backend/main.py`)
- **Request Validation:** Pydantic models for type safety
- **Document Registry:** In-memory tracking of uploaded documents
- **Endpoint Implementation:** RESTful API for all operations
- **Error Handling:** Comprehensive exception management
- **CORS Configuration:** Cross-origin resource sharing for frontend

#### PDF Extraction (`backend/extract.py`)
- **Multi-layer Extraction:** Three-tier fallback for maximum compatibility
- **Validation:** File type, size, and integrity checks
- **Sanitization:** Removes harmful characters and control codes
- **Table Detection:** Extracts tabular data when available
- **Page Tracking:** Maintains page numbers for citations

#### RAG Pipeline (`backend/rag.py`)
- **Text Chunking:** LangChain RecursiveCharacterTextSplitter
- **Embedding Generation:** Local HuggingFace sentence transformers
- **Vector Storage:** FAISS index with persistence
- **Retrieval:** Similarity search with document filtering
- **LLM Chain:** Dynamic prompt construction with conversation history
- **Metrics Collection:** Latency, accuracy, and performance tracking

## Component Details

### 1. PDF Extraction Pipeline (`extract.py`)

```python
class PDFExtractionPipeline:
    Primary:  pdfplumber  # Text + tables, preserves formatting
    Fallback1: PyPDF      # Standard text extraction
    Fallback2: PyMuPDF    # Complex layouts, scanned docs
```

**Features:**
- **Page Markers:** Injects `<<PAGE_N>>` markers to track source pages
- **Table Extraction:** Converts tables to pipe-separated text
- **Text Cleaning:** Removes control characters (\x00-\x1F)
- **Validation:** Extension check, MIME type verification, size limit (20MB)
- **Error Handling:** Custom exceptions (PDFValidationError, PDFExtractionError)

**File Location:** `backend/extract.py`  
**Key Functions:**
- `validate_pdf_file()` - Pre-upload validation
- `extract_pdf_contents()` - Main extraction orchestrator
- `_clean_text()` - Sanitization and normalization

### 2. RAG Pipeline (`rag.py`)

**A. Text Chunking**
```python
RecursiveCharacterTextSplitter(
    chunk_size=500,        # Configurable: 200-1500
    chunk_overlap=100,     # Configurable: 50-400
    separators=["\n\n", "\n", " ", ""]
)
```

- Preserves semantic boundaries (paragraphs > sentences > words)
- Maintains page numbers from extraction markers
- Generates unique UUID for each chunk
- Stores metadata: `{doc_id, page, chunk_id}`

**B. Embedding Generation (LOCAL)**
```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

- **Runs locally** - no API calls
- Downloads model on first use (~90MB)
- Cached in `~/.cache/huggingface/`
- 384-dimensional dense vectors
- Average processing: ~50 chunks/second on CPU

**C. FAISS Vector Store**
```python
FAISS.from_texts(
    texts=chunk_texts,
    embedding=HuggingFaceEmbeddings(),
    metadatas=chunk_metadata
)
```

- **Index Type:** IndexFlatL2 (exact L2 distance)
- **Storage:** `models/langchain_index/` directory
- **Persistence:** Auto-save after indexing
- **Loading:** Deserialization on startup
- **Search:** `similarity_search_with_score(query, k=top_k)`

**D. LLM Integration**
```python
# Dynamic provider selection
if LLM_PROVIDER == "openai":
    ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
elif LLM_PROVIDER == "gemini":
    ChatGoogleGenerativeAI(model="gemini-1.5-flash")
elif LLM_PROVIDER == "huggingface":
    ChatOpenAI(base_url="https://router.huggingface.co/v1")
```

- **Prompt Template:** System message + conversation history + user question
- **Context Injection:** Top-k retrieved chunks
- **Output Parsing:** StrOutputParser for clean text responses
- **Error Handling:** API failures, rate limits, invalid responses

**File Location:** `backend/rag.py`  
**Key Classes:**
- `RAGPipeline` - Main orchestrator
- `Chunk` - Chunk data model
- `QueryMetrics` - Performance metrics

### 3. FastAPI Backend (`main.py`)

**A. Document Registry**
```python
class DocumentRegistry:
    _records: Dict[str, DocumentRecord]
    
    create(filename, path) -> DocumentRecord
    get(doc_id) -> DocumentRecord
    update(doc_id, **kwargs) -> DocumentRecord
```

- In-memory document tracking
- UUID-based document IDs
- Metadata: filename, upload time, chunk count, processing params

**B. API Endpoints**

| Endpoint | Method | Input | Output | Purpose |
|----------|--------|-------|--------|----------|
| `/` | GET | - | Health status | System check |
| `/upload` | POST | PDF file | doc_id, metadata | File upload |
| `/extract-text` | POST | doc_id, params | Indexing results | Text extraction |
| `/query` | POST | doc_id, question | Answer, chunks, metrics | RAG query |
| `/history` | GET | - | Query history | Audit log |

**C. Request/Response Models**
```python
class QueryRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=3, max_length=1500)
    top_k: int = Field(default=5, ge=1, le=10)
    relevance_score: Optional[float] = Field(ge=0.0, le=1.0)
    conversation_history: Optional[List[Dict[str, str]]]
```

**File Location:** `backend/main.py`  
**Key Features:**
- Auto-indexing on first query
- Conversation history support (last 10 messages)
- Comprehensive error responses
- CORS middleware for frontend access

### 4. Flask Frontend (`app.py`)

**A. Session Management**
```python
session['doc_id']       # Current document UUID
session['extracted']    # Indexing status
session['messages']     # Conversation history
session['filename']     # Original filename
session['history']      # Saved sessions
session['last_metrics'] # Performance metrics
```

**B. Route Structure**

| Route | Method | Purpose |
|-------|--------|----------|
| `/` | GET | Landing page |
| `/chat` | GET | Main chat interface |
| `/api/upload` | POST | Proxy to backend |
| `/api/extract` | POST | Proxy to backend |
| `/api/query` | POST | Proxy + session update |
| `/api/history` | GET | Backend history |
| `/api/clear` | POST | Clear current chat |
| `/api/new-session` | POST | Save & reset session |
| `/api/load-session` | POST | Restore saved session |
| `/api/pdf/<doc_id>` | GET | Serve PDF file |

**C. Frontend JavaScript (`static/js/app.js`)**
- Drag-and-drop file upload
- PDF preview with iframe
- Real-time chat interface
- Metrics display and updates
- Session persistence
- Error handling and toasts

**File Location:** `frontend/app.py`, `frontend/static/js/app.js`

## Data Flow

### 1. Document Upload Flow

```
User selects PDF
    ↓
JavaScript validates file (client-side)
    ↓
POST /api/upload (Flask proxy)
    ↓
POST /upload (FastAPI)
    ↓
Validate file (size, type, extension)
    ↓
Sanitize filename
    ↓
Save to data/uploaded_files/{uuid}_{filename}
    ↓
Create DocumentRecord with doc_id
    ↓
Return {doc_id, filename, size_mb, uploaded_at}
    ↓
Flask stores doc_id in session
    ↓
JavaScript displays file info + extract button
```

### 2. Text Extraction & Indexing Flow

```
User clicks "Extract & Index"
    ↓
POST /api/extract (Flask proxy)
    ↓
POST /extract-text (FastAPI)
    ↓
Retrieve file path from DocumentRegistry
    ↓
extract_pdf_contents()
    ├── Try pdfplumber (text + tables)
    ├── Fallback to PyPDF
    └── Fallback to PyMuPDF
    ↓
Clean text, preserve page markers
    ↓
RAGPipeline.register_document()
    ├── chunk_text() - split with page tracking
    ├── Generate embeddings (LOCAL)
    ├── Add to FAISS index
    └── Persist index to disk
    ↓
Update DocumentRegistry metadata
    ↓
Return {chunks_indexed, tables_detected, preview}
    ↓
Flask marks session['extracted'] = True
    ↓
JavaScript enables chat input
```

### 3. Query Processing Flow

```
User types question + clicks send
    ↓
JavaScript: POST /api/query
    ↓
Flask: Retrieve conversation_history from session
    ↓
POST /query (FastAPI)
    ↓
Auto-index check (if not already indexed)
    ↓
RAGPipeline.query()
    ├── Generate query embedding (LOCAL)
    ├── FAISS similarity_search_with_score()
    ├── Filter chunks by doc_id
    ├── Take top_k chunks
    ├── Build context string
    ├── Construct prompt with conversation history
    ├── LLM API call (OpenAI/Gemini/HF)
    ├── Parse response
    └── Calculate metrics (latency, accuracy)
    ↓
Store query in history log
    ↓
Return {answer, chunks, references, metrics}
    ↓
Flask: Update session['messages'] and session['last_metrics']
    ↓
JavaScript: Render message + update metrics display
```

### 4. Conversation Context Flow

```
Query N (with history)
    ↓
Flask: Get session['messages'] (last 10)
    ↓
Format as conversation_history: [{role, content}, ...]
    ↓
Send to backend with query
    ↓
RAG Pipeline:
    Build messages = [
        ("system", system_prompt + context),
        ("user", history_msg_1),
        ("assistant", history_msg_1_response),
        ...
        ("user", current_question)
    ]
    ↓
ChatPromptTemplate.from_messages(messages)
    ↓
LLM processes full conversation context
    ↓
Response considers previous exchanges
```

## Storage Architecture

### File System Layout

```
ASK-MY-PDF/
├── data/
│   └── uploaded_files/
│       ├── {uuid}_document1.pdf
│       ├── {uuid}_document2.pdf
│       └── ...
└── models/
    ├── langchain_index/           # FAISS persistent storage
    │   ├── index.faiss               # Vector index file
    │   └── index.pkl                 # Metadata pickle
    ├── documents_meta.json        # Document registry
    └── embeddings_meta.json       # Legacy (optional)
```

### Storage Components

#### 1. Uploaded PDFs (`data/uploaded_files/`)
- **Naming:** `{uuid}_{sanitized_filename}.pdf`
- **Permissions:** Read-only after upload
- **Retention:** Manual cleanup (no auto-delete)
- **Size Limit:** 20MB per file
- **Access:** Served via Flask `/api/pdf/<doc_id>` endpoint

#### 2. FAISS Index (`models/langchain_index/`)
- **Format:** Binary FAISS index + pickle metadata
- **Persistence:** Saved after each `register_document()` call
- **Loading:** Lazy load on first query or startup
- **Size:** ~1KB per chunk (varies with embedding model)
- **Backup:** Recommend periodic copies for production

#### 3. Document Metadata (`models/documents_meta.json`)
```json
{
  "doc_id_uuid": {
    "filename": "report.pdf",
    "text": "full extracted text...",
    "chunk_size": 500,
    "overlap": 100,
    "chunks": 45
  }
}
```
- **Format:** JSON with UTF-8 encoding
- **Updates:** After extraction and indexing
- **Size:** Grows with document collection (~10KB per doc)

#### 4. Flask Sessions (Server-Side)
- **Storage:** In-memory (default) or file-based
- **Data:** doc_id, messages, history, metrics
- **TTL:** Session expires on browser close
- **Security:** Encrypted with SECRET_KEY

### Data Persistence

| Component | Persistent | Location | Backup Recommended |
|-----------|-----------|----------|--------------------|
| PDF Files | Yes | `data/uploaded_files/` | Yes |
| FAISS Index | Yes | `models/langchain_index/` | Yes |
| Doc Metadata | Yes | `models/documents_meta.json` | Yes |
| Document Registry | No | In-memory (backend) | N/A - rebuilt from metadata |
| Flask Sessions | Configurable | In-memory or disk | Optional |
| Query History | No | In-memory (backend) | N/A - logged per session |

### Scaling Considerations

**Current Limits:**
- Single FAISS index for all documents
- In-memory document registry
- No database for structured data
- Local file storage

**Future Improvements:**
- Document namespaces/collections
- PostgreSQL for metadata
- S3/cloud storage for PDFs
- Redis for session management
- Distributed FAISS or vector DB (Weaviate, Qdrant)

## API Architecture

### Request/Response Cycle

```
Client Request
    ↓
Flask Route Handler
    ├── Session validation
    ├── Input sanitization
    └── Backend API call (requests library)
        ↓
    FastAPI Endpoint
        ├── Pydantic validation
        ├── Business logic
        ├── RAG pipeline
        └── Error handling
            ↓
        Response Model
            ↓
    Flask Response Handler
        ├── Session update
        ├── Response formatting
        └── JSON serialization
            ↓
Client receives JSON
```

### Error Handling Strategy

**Backend (FastAPI):**
```python
try:
    # Business logic
except PDFValidationError as e:
    raise HTTPException(status_code=400, detail=str(e))
except PDFExtractionError as e:
    raise HTTPException(status_code=400, detail=str(e))
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    LOGGER.exception("Unexpected error")
    raise HTTPException(status_code=500, detail="Internal error")
```

**Frontend (Flask):**
```python
try:
    response = requests.post(backend_url, json=data)
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        error = response.json().get('detail', 'Unknown error')
        return jsonify({'error': error}), response.status_code
except requests.exceptions.ConnectionError:
    return jsonify({'error': 'Backend unavailable'}), 503
except Exception as e:
    return jsonify({'error': str(e)}), 500
```

**JavaScript (Frontend):**
```javascript
fetch('/api/query', {method: 'POST', body: JSON.stringify(data)})
    .then(response => {
        if (!response.ok) throw new Error(response.statusText);
        return response.json();
    })
    .then(data => updateUI(data))
    .catch(error => showToast('error', error.message));
```

## Security Architecture

### STRIDE Threat Model Summary

| Threat | Mitigation | Implementation |
|--------|-----------|----------------|
| **Spoofing** | Session-based auth | Flask sessions with SECRET_KEY |
| **Tampering** | Input validation | Pydantic models, file checks |
| **Repudiation** | Audit logging | Query history, timestamps |
| **Info Disclosure** | Env variables | .env for API keys, .gitignore |
| **DoS** | Rate limiting | File size limits (20MB) |
| **Elevation** | Least privilege | No admin endpoints, read-only files |

### Input Validation Layers

**Layer 1: Client-Side (JavaScript)**
- File type check (`.pdf` extension)
- File size check (<20MB)
- Question length validation

**Layer 2: Flask Proxy**
- Request format validation
- Session state checks
- Content-Type verification

**Layer 3: FastAPI Backend**
- Pydantic schema validation
- File validation (extension, MIME, size)
- Filename sanitization
- Text content sanitization

**Layer 4: RAG Pipeline**
- Document ID verification
- Chunk count validation
- Query length limits

### Sanitization Functions

```python
# Filename sanitization
def sanitize_filename(filename: str) -> str:
    keepchars = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepchars).strip()

# Text sanitization
def _clean_text(text: str) -> str:
    # Remove control characters
    safe_text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", text)
    # Normalize whitespace
    safe_text = re.sub(r"\s+", " ", safe_text)
    return safe_text.strip()
```

### API Key Security

- **Storage:** `.env` file (never committed to Git)
- **Access:** `os.getenv()` with validation
- **Exposure:** Backend only, never sent to frontend
- **Rotation:** Environment variable update + restart

**Example:**
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaxxxxxxxxxxxxx
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxx
SECRET_KEY=random-secret-for-flask-sessions
```

## Deployment Architecture

### Local Development

```
Terminal 1: Backend
$ uvicorn backend.main:app --reload --port 8000

Terminal 2: Frontend
$ cd frontend
$ python app.py
```

**Ports:**
- Backend: 8000
- Frontend: 5000
- Access: http://localhost:5000

### Production Deployment (Recommendations)

#### Option 1: Traditional Server

```
Nginx (Reverse Proxy)
    ├── :80/:443 → Gunicorn (Flask) :5000
    └── /api/* → Uvicorn (FastAPI) :8000

Systemd Services
    ├── askmypdf-backend.service
    └── askmypdf-frontend.service

File Storage
    └── /var/lib/askmypdf/
        ├── data/
        └── models/
```

#### Option 2: Docker Containers

```yaml
# docker-compose.yml
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    env_file: .env
  
  frontend:
    build: ./frontend
    ports:
      - "5000:5000"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend
```

#### Option 3: Cloud Platforms

**Hugging Face Spaces:**
- Gradio wrapper for UI
- GPU support for local LLM inference
- Free tier available

**Google Cloud Run:**
- Containerized deployment
- Auto-scaling
- Pay-per-use

**AWS Lambda + API Gateway:**
- Serverless backend
- S3 for file storage
- DynamoDB for metadata

### Environment Configuration

**Development:**
```env
FLASK_DEBUG=True
LOG_LEVEL=DEBUG
```

**Production:**
```env
FLASK_DEBUG=False
LOG_LEVEL=INFO
SECRET_KEY=<strong-random-key>
BACKEND_URL=https://api.yourdomain.com
```

### Monitoring & Logging

**Backend Logs:**
- Request/response logging via FastAPI middleware
- Error tracking with Python logging module
- Query metrics stored in history

**Frontend Logs:**
- Flask request logs
- Session activity tracking
- Error reporting to console

**Metrics to Monitor:**
- Query latency (p50, p95, p99)
- Retrieval accuracy trends
- API error rates
- File upload sizes
- Active sessions
- FAISS index size

---

## Summary of Key Architectural Decisions

### ✅ What IS Used

1. **FAISS Vector Store (Local)**
   - Implementation: `langchain_community.vectorstores.FAISS`
   - Storage: Local disk at `models/langchain_index/`
   - Persistence: Saved/loaded from disk
   - Location: `backend/rag.py`

2. **HuggingFace Embeddings (Local)**
   - Implementation: `langchain_community.embeddings.HuggingFaceEmbeddings`
   - Model: `sentence-transformers/all-MiniLM-L6-v2` (configurable)
   - Execution: Runs locally (no API calls)
   - Location: `backend/rag.py`

3. **LLM via API (External)**
   - Providers: OpenAI, Google Gemini, Hugging Face Router
   - All via API calls (not local inference)
   - Location: `backend/main.py`

4. **Flask + FastAPI**
   - Separation of concerns: UI vs API
   - Session management in Flask
   - Business logic in FastAPI

### ❌ What is NOT Used

1. **Pinecone/Cloud Vector DBs** - Using local FAISS instead
2. **Local LLM Inference** - All LLM calls via API
3. **Database (SQL/NoSQL)** - Using JSON files and in-memory storage
4. **Authentication System** - Session-based, no user accounts
5. **Celery/Background Tasks** - Synchronous processing only

### Design Philosophy

- **Simplicity:** Single-machine deployment, no microservices
- **Privacy:** Local embeddings, minimal cloud dependencies
- **Flexibility:** Pluggable LLM providers via environment config
- **Performance:** FAISS for fast similarity search
- **Security:** Multiple validation layers, sanitization
- **Developer Experience:** Hot reload, comprehensive logging, clear errors

---

**Last Updated:** November 25, 2025  
**Version:** 1.0.0  
**Maintainer:** anargh-t

