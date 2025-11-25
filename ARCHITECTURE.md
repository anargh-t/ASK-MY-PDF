# Architecture Documentation

## Current Implementation

This document accurately describes what the project **actually uses** based on the codebase.

### ✅ What IS Used

1. **FAISS Vector Store (Local)**
   - **Location**: `backend/rag.py` line 13
   - **Implementation**: `langchain_community.vectorstores.FAISS`
   - **Storage**: Local disk at `models/langchain_index/`
   - **Persistence**: Index is saved/loaded from disk
   - **Status**: ✅ Fully implemented and working

2. **Hugging Face Embeddings (Local)**
   - **Location**: `backend/rag.py` line 12, 83
   - **Implementation**: `langchain_community.embeddings.HuggingFaceEmbeddings`
   - **Model**: `sentence-transformers/all-MiniLM-L6-v2` (default, configurable via `EMBEDDING_MODEL`)
   - **Execution**: Runs locally on your machine (downloads model on first use)
   - **Status**: ✅ Fully implemented and working

3. **LLM via API (External)**
   - **Location**: `backend/main.py` lines 50-88
   - **Providers Supported**:
     - **OpenAI**: Via `langchain_openai.ChatOpenAI` (API calls)
     - **Google Gemini**: Via `langchain_google_genai.ChatGoogleGenerativeAI` (API calls)
     - **Hugging Face**: Via `langchain_openai.ChatOpenAI` with `base_url=https://router.huggingface.co/v1` (API calls)
   - **Status**: ✅ Fully implemented - requires API keys

### ❌ What is NOT Used

1. **Pinecone**
   - **Status**: ❌ Not used anywhere in the codebase
   - **Note**: The project uses local FAISS instead

2. **Local LLM Inference**
   - **Status**: ❌ Not currently implemented
   - **Note**: All LLM calls go through external APIs (OpenAI, Gemini, or Hugging Face router)

## Architecture Flow

```
PDF Upload
    ↓
PDF Extraction (pypdf/pdfplumber)
    ↓
Text Chunking (LangChain RecursiveCharacterTextSplitter)
    ↓
Embedding Generation (HuggingFaceEmbeddings - LOCAL)
    ↓
FAISS Index Storage (LOCAL - models/langchain_index/)
    ↓
Query Processing:
    - Generate query embedding (LOCAL)
    - Search FAISS index (LOCAL)
    - Retrieve top-k chunks
    - Send to LLM via API (OpenAI/Gemini/HF Router)
    - Return answer
```

## Key Files

- **`backend/rag.py`**: FAISS vector store + Hugging Face embeddings
- **`backend/main.py`**: FastAPI endpoints + LLM provider selection
- **`models/langchain_index/`**: Persistent FAISS index storage
- **`models/documents_meta.json`**: Document metadata

## Dependencies

- **`faiss-cpu>=1.8.0`**: Local vector search
- **`sentence-transformers==2.3.1`**: Local embedding generation
- **`langchain-community==0.2.6`**: FAISS wrapper + HuggingFace embeddings
- **`langchain-openai==0.1.7`**: OpenAI API client
- **`langchain-google-genai==1.0.5`**: Gemini API client
- **`huggingface-hub==0.25.2`**: Model downloading for embeddings

## Configuration

### Required for Embeddings (Local - No API Key Needed)
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Required for LLM (Choose One Provider)
```env
# Option 1: OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini

# Option 2: Gemini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_key
GEMINI_MODEL=gemini-1.5-flash

# Option 3: Hugging Face Router (API)
LLM_PROVIDER=huggingface
HUGGINGFACEHUB_API_TOKEN=your_token
HUGGINGFACE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
HUGGINGFACE_API_BASE=https://router.huggingface.co/v1
```

## Summary

- ✅ **FAISS**: Local vector database (no cloud service)
- ✅ **Hugging Face Embeddings**: Local model execution (no API)
- ✅ **LLM**: External API calls (OpenAI/Gemini/HF Router)
- ❌ **Pinecone**: Not used
- ❌ **Local LLM**: Not implemented (would require transformers + GPU)

