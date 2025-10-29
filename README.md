# AskMyPDF: Cloud-Native RAG Agent

A scalable, production-ready RAG (Retrieval-Augmented Generation) application that enables natural language question answering over PDF documents using cloud-native architecture.

## Architecture

This project follows a **cloud-native architecture** where:

- **FastAPI**: Lightweight Python service framework
- **External LLM**: Hugging Face models via LangChain (no local inference)
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
HUGGINGFACEHUB_API_TOKEN=your_token_here
PINECONE_API_KEY=your_key_here
API_KEY=your_api_key_here
```

### 5. Get API Keys

- **Hugging Face**: Sign up at [huggingface.co](https://huggingface.co) and create an API token
- **Pinecone**: Sign up at [pinecone.io](https://pinecone.io) and create an API key
- **API Key**: Generate a secure random string for API authentication

## Usage

### Start the server

```bash
python api_server.py
```

The server will start at `http://localhost:8000`

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/
```

#### 2. Process PDF

Upload a PDF file for processing:

```bash
curl -X POST "http://localhost:8000/process_pdf" \
  -F "file=@document.pdf" \
  -F "chunk_size=1000" \
  -F "overlap=200"
```

#### 3. Query PDF

Ask questions about the uploaded PDF (requires API key):

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?", "top_k": 5}'
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
- `retrieve_relevant_chunks()`: Find relevant chunks using vector similarity

### `api_server.py`

FastAPI server with cloud-native integrations:
- **Startup**: Initializes LLM (HuggingFace) and Vector DB (Pinecone)
- **`/process_pdf`**: Accepts PDF upload, creates embeddings, uploads to Pinecone
- **`/query`**: Retrieves relevant chunks and generates answers using LLM
- **Security**: API key middleware for protected endpoints
- **Logging**: Cost tracking for LLM API calls

## Deployment

### Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Push this code to the Space repository
3. Add secrets: `HUGGINGFACEHUB_API_TOKEN`, `PINECONE_API_KEY`, `API_KEY`
4. The Space will automatically build and deploy

### Deploy to Cloud Run

```bash
gcloud run deploy askmypdf \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars HUGGINGFACEHUB_API_TOKEN=your_token,PINECONE_API_KEY=your_key,API_KEY=your_key
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face API token | Yes |
| `PINECONE_API_KEY` | Pinecone API key | Yes |
| `PINECONE_INDEX_NAME` | Pinecone index name | No (default: askmypdf) |
| `PINECONE_ENV` | Pinecone environment | No (default: us-west1-gcp) |
| `API_KEY` | API key for authentication | Yes |

### Parameters

- **Chunk Size**: Default 1000 characters (adjustable via `/process_pdf`)
- **Overlap**: Default 200 characters for context preservation
- **Top K**: Default 5 most relevant chunks for retrieval
- **LLM Model**: Default Mistral-7B-Instruct (configurable via environment)

## Sprint 1 Status

✅ **Task 1**: Environment setup and dependencies  
✅ **Task 2**: Hosted LLM integration (HuggingFace via LangChain)  
✅ **Task 3**: Cloud vector store (Pinecone integration)  
✅ **Task 4**: Core API endpoints (`/process_pdf`, `/query`)  
✅ **Task 5**: API security and cost logging  

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

