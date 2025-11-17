# AskMyPDF: Cloud-Native RAG Agent

A scalable, production-ready RAG (Retrieval-Augmented Generation) application that enables natural language question answering over PDF documents using cloud-native architecture.

## Architecture

This project follows a **cloud-native architecture** where:

- **FastAPI**: Lightweight Python service framework
- **External LLM**: Hosted providers (Gemini, OpenAI, Hugging Face)
- **External Vector DB**: Pinecone (no local FAISS)
- **Embeddings**: Generated locally using sentence-transformers

## Features

- **PDF Processing**: Extract and chunk PDF documents
- **Cloud Vector Store**: Store embeddings in Pinecone
- **RAG Pipeline**: Retrieve relevant context and generate answers
- **API Security**: API key authentication for protected endpoints
- **Cost Logging**: Track LLM API usage and costs
- **Production-Ready**: Designed for deployment to Hugging Face Spaces or Cloud Run

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd ASKMYPDF
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```env
# Core settings
LLM_PROVIDER=gemini   # gemini | openai | huggingface
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=us-east-1-aws
PINECONE_INDEX_NAME=askmypdf

# LLM credentials (only set the one you plan to use)
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
HUGGINGFACE_REPO_ID=mistralai/Mistral-7B-Instruct-v0.1
```

### 5. Get API Keys

- **Hugging Face** *(only if `LLM_PROVIDER=huggingface`)*: API token with Inference access
- **Google Gemini** *(if `LLM_PROVIDER=gemini`)*: API key from [ai.google.dev](https://ai.google.dev)
- **OpenAI** *(if `LLM_PROVIDER=openai`)*: API key from [platform.openai.com](https://platform.openai.com)
- **Pinecone**: Vector DB API key and environment

### Windows (PowerShell) quickstart

```powershell
# 1) Create and activate venv
py -3 -m venv venv
.\venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Copy env template and edit credentials
Copy-Item .env.example .env
# Open .env in your editor and set the provider + keys you plan to use

# 4) Run the API server
python .\api_server.py
```

Notes:
- If you see a script execution error when activating venv, run once in the same PowerShell session:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

## Usage

### Start the server

```bash
python api_server.py
```

The server will start at `http://localhost:8000`

### Use the built-in web UI

- Navigate to `http://localhost:8000/app` for the guided interface
- Step 1: Upload and process a PDF (default chunk settings are applied automatically)
- Step 2: Ask your question—no API key or advanced parameters required
- Responses display the final answer and the context chunks that were used

### API Endpoints

#### 1. Health Check

```powershell
# PowerShell (ensure you call Windows curl.exe, not Invoke-WebRequest alias)
curl.exe http://localhost:8000/

# Or using Invoke-WebRequest
Invoke-WebRequest http://localhost:8000/ | Select-Object -ExpandProperty Content
```

#### 2. Process PDF

Upload a PDF file for processing:

```powershell
# Using curl.exe (recommended on Windows)
curl.exe -X POST "http://localhost:8000/process_pdf" ^
  -F "file=@document.pdf" ^
  -F "chunk_size=1000" ^
  -F "overlap=200"

# Or using Invoke-WebRequest with -Form (PowerShell 7+)
Invoke-WebRequest -Uri "http://localhost:8000/process_pdf" -Method Post `
  -Form @{ file = Get-Item .\document.pdf; chunk_size = 1000; overlap = 200 } | `
  Select-Object -ExpandProperty Content
```

#### 3. Query PDF

Ask questions about the uploaded PDF:

```powershell
curl.exe -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  --data-raw "{\"question\": \"What is the main topic of this document?\", \"top_k\": 5}"

# Or using Invoke-RestMethod
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post `
  -ContentType "application/json" `
  -Body (@{ question = "What is the main topic of this document?"; top_k = 5 } | ConvertTo-Json)
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
ASKMYPDF/
├── api_server.py          # FastAPI server with cloud integrations
├── rag_utils.py          # Core RAG utility functions
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
└── context.md            # Project specification
```

## Core Components

### `rag_utils.py`

Contains reusable RAG functions:
- `open_and_read_pdf()`: Extract text from PDFs
- `create_chunks_from_pages()`: Split text into overlapping chunks
- `text_formatter()`: Clean and format text
- `create_embeddings()`: Generate embeddings using sentence-transformers
- Retrieval is handled via the Pinecone index at query time

### `api_server.py`

FastAPI server with cloud-native integrations:
- **Startup**: Initializes the selected LLM provider and Pinecone
- **`/process_pdf`**: Accepts PDF upload, creates embeddings, uploads to Pinecone
- **`/query`**: Retrieves relevant chunks and generates answers using the configured LLM
- **Logging**: Cost tracking and detailed startup diagnostics

## Deployment

### Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Push this code to the Space repository
3. Add secrets: the Pinecone key plus whichever LLM provider keys you enabled
4. The Space will automatically build and deploy

### Deploy to Cloud Run

```bash
gcloud run deploy askmypdf \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars LLM_PROVIDER=gemini,PINECONE_API_KEY=your_key,PINECONE_ENV=us-east-1-aws,GOOGLE_API_KEY=your_gemini_key
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LLM_PROVIDER` | `gemini`, `openai`, or `huggingface` | No (default: gemini) |
| `GOOGLE_API_KEY` | Needed when `LLM_PROVIDER=gemini` | Conditional |
| `OPENAI_API_KEY` | Needed when `LLM_PROVIDER=openai` | Conditional |
| `HUGGINGFACEHUB_API_TOKEN` | Needed when `LLM_PROVIDER=huggingface` | Conditional |
| `HUGGINGFACE_REPO_ID` | Hugging Face repo ID (if using HF) | No (default: gpt2) |
| `PINECONE_API_KEY` | Pinecone API key | Yes |
| `PINECONE_INDEX_NAME` | Pinecone index name | No (default: askmypdf) |
| `PINECONE_ENV` | Pinecone environment (e.g. `us-east-1-aws`) | No (default: us-east-1-aws) |

### Parameters

- **Chunk Size**: Default 1000 characters (adjustable via `/process_pdf`)
- **Overlap**: Default 200 characters for context preservation
- **Top K**: Default 5 most relevant chunks for retrieval
- **LLM Model**: Default Mistral-7B-Instruct (configurable via environment)

## Sprint 1 Status

✅ **Task 1**: Environment setup and dependencies  
✅ **Task 2**: Hosted LLM integration (Gemini / OpenAI / Hugging Face APIs)  
✅ **Task 3**: Cloud vector store (Pinecone integration)  
✅ **Task 4**: Core API endpoints (`/process_pdf`, `/query`)  
✅ **Task 5**: Cost logging and simplified UX  

## Future Enhancements

- Support for multiple vector databases (Weaviate, Qdrant)
- Batch processing for multiple PDFs
- Advanced chunking strategies
- Multi-turn conversation support
- Enhanced cost tracking and analytics
- Rate limiting and request throttling

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

