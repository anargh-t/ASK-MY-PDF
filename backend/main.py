import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional

os.environ.setdefault("HF_INFERENCE_ENDPOINT", "https://router.huggingface.co/hf-inference")

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from backend.extract import (
    ExtractedContent,
    MAX_FILE_SIZE_MB,
    PDFExtractionError,
    PDFValidationError,
    extract_pdf_contents,
)
from backend.rag import QueryMetrics, RAGPipeline

load_dotenv()

LOGGER = logging.getLogger("askmypdf.backend")
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploaded_files"
MODELS_DIR = PROJECT_ROOT / "models"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 500))
DEFAULT_OVERLAP = int(os.getenv("DEFAULT_OVERLAP", 100))
MAX_LLM_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 512))


def build_langchain_llm():
    if LLM_PROVIDER == "gemini":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini provider.")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=google_api_key,
            temperature=0.3,
            max_output_tokens=MAX_LLM_TOKENS,
        )

    if LLM_PROVIDER == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for OpenAI provider.")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.3,
            max_tokens=MAX_LLM_TOKENS,
            api_key=openai_api_key,
        )

    if LLM_PROVIDER == "huggingface":
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN for Hugging Face provider.")
        hf_base = os.getenv("HUGGINGFACE_API_BASE", "https://router.huggingface.co/v1")
        return ChatOpenAI(
            model=HUGGINGFACE_MODEL,
            api_key=hf_token,
            base_url=hf_base,
            temperature=0.3,
            max_tokens=MAX_LLM_TOKENS,
        )

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openai', 'gemini', or 'huggingface'."
    )


pipeline = RAGPipeline(
    model_name=EMBEDDING_MODEL,
    storage_dir=MODELS_DIR,
    llm=build_langchain_llm(),
)


def sanitize_filename(filename: str) -> str:
    keepchars = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepchars).strip() or "document.pdf"


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    size_mb: float
    uploaded_at: datetime


class ExtractRequest(BaseModel):
    doc_id: str = Field(..., description="Document identifier returned by /upload")
    chunk_size: int = Field(500, ge=200, le=1500)
    overlap: int = Field(100, ge=50, le=400)


class ExtractResponse(BaseModel):
    doc_id: str
    filename: str
    text_preview: str
    tables_detected: int
    chunks_indexed: int
    processed_at: datetime


class QueryRequest(BaseModel):
    doc_id: str
    question: str = Field(..., min_length=3, max_length=1500)
    top_k: int = Field(5, ge=1, le=10)
    relevance_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Optional manual rating supplied by the UI."
    )


class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]
    metrics: Dict[str, float]
    timestamp: datetime


class HistoryResponse(BaseModel):
    history: List[Dict]


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    path: Path
    uploaded_at: datetime
    text: Optional[str] = None
    tables: Optional[List[List[str]]] = None
    chunks_indexed: int = 0
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None


class DocumentRegistry:
    def __init__(self) -> None:
        self._records: Dict[str, DocumentRecord] = {}

    def create(self, filename: str, storage_path: Path) -> DocumentRecord:
        doc_id = str(uuid.uuid4())
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            path=storage_path,
            uploaded_at=datetime.utcnow(),
        )
        self._records[doc_id] = record
        return record

    def get(self, doc_id: str) -> DocumentRecord:
        if doc_id not in self._records:
            raise KeyError("Document not found.")
        return self._records[doc_id]

    def update(self, doc_id: str, **kwargs) -> DocumentRecord:
        record = self.get(doc_id)
        data = record.dict()
        data.update(kwargs)
        updated = DocumentRecord(**data)
        self._records[doc_id] = updated
        return updated


registry = DocumentRegistry()


def process_document(record: DocumentRecord, chunk_size: int, overlap: int) -> ExtractResponse:
    contents: ExtractedContent = extract_pdf_contents(record.path)
    result = pipeline.register_document(
        filename=record.filename,
        text=contents.text,
        chunk_size=chunk_size,
        overlap=overlap,
        doc_id=record.doc_id,
    )
    registry.update(
        record.doc_id,
        text=contents.text,
        tables=contents.tables,
        chunks_indexed=result["chunks"],
        chunk_size=chunk_size,
        overlap=overlap,
    )
    preview = contents.text[:500] + ("..." if len(contents.text) > 500 else "")
    return ExtractResponse(
        doc_id=record.doc_id,
        filename=record.filename,
        text_preview=preview,
        tables_detected=len(contents.tables or []),
        chunks_indexed=result["chunks"],
        processed_at=datetime.utcnow(),
    )


def ensure_document_indexed(doc_id: str) -> None:
    try:
        record = registry.get(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found. Upload first.")

    if record.chunks_indexed > 0:
        return

    chunk_size = record.chunk_size or DEFAULT_CHUNK_SIZE
    overlap = record.overlap or DEFAULT_OVERLAP
    process_document(record, chunk_size, overlap)


app = FastAPI(
    title="ASK MY PDF – AI PDF Assistant",
    description="Full-stack PDF RAG system with FAISS vector search.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "documents": len(pipeline.documents),
        "history": len(pipeline.history),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    filename = sanitize_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    destination = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        size_mb = destination.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
            )

        record = registry.create(filename=filename, storage_path=destination)

        return UploadResponse(
            doc_id=record.doc_id,
            filename=filename,
            size_mb=round(size_mb, 2),
            uploaded_at=record.uploaded_at,
        )
    except PDFValidationError as err:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        LOGGER.exception("Upload failed: %s", err)
        raise HTTPException(status_code=500, detail="Unable to upload PDF.")


@app.post("/extract-text", response_model=ExtractResponse)
async def extract_text(payload: ExtractRequest):
    try:
        record = registry.get(payload.doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found. Upload first.")

    try:
        return process_document(record, payload.chunk_size, payload.overlap)
    except (PDFValidationError, PDFExtractionError) as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        LOGGER.exception("Extraction failed: %s", err)
        raise HTTPException(status_code=500, detail="Unable to extract text from PDF.")


@app.post("/query", response_model=QueryResponse)
async def query_pdf(payload: QueryRequest):
    if len(payload.question.split()) > 250:
        raise HTTPException(status_code=400, detail="Question is too long. Keep it under 250 words.")

    try:
        ensure_document_indexed(payload.doc_id)
    except HTTPException:
        raise
    except Exception as err:
        LOGGER.exception("Auto-indexing failed: %s", err)
        raise HTTPException(status_code=500, detail="Unable to prepare document for querying.")

    try:
        answer, chunks, metrics = pipeline.query(
            doc_id=payload.doc_id,
            question=payload.question,
            top_k=payload.top_k,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        LOGGER.exception("Query failed: %s", err)
        raise HTTPException(status_code=500, detail="Query pipeline failed.")

    metrics_dict = asdict(metrics)
    if payload.relevance_score is not None and pipeline.history:
        pipeline.history[-1]["metrics"]["relevance_score"] = payload.relevance_score
        metrics_dict["relevance_score"] = payload.relevance_score

    return QueryResponse(
        answer=answer,
        relevant_chunks=[chunk["text"][:200] + "..." for chunk in chunks],
        metrics=metrics_dict,
        timestamp=datetime.utcnow(),
    )


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    return HistoryResponse(history=pipeline.history[-20:])

