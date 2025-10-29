"""
AskMyPDF: Cloud-Native RAG Agent - FastAPI Server
Connects to external LLM (HuggingFace) and Vector DB (Pinecone/Weaviate)
"""
import os
import logging
import traceback
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import uvicorn

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone
from rag_utils import (
    create_chunks_from_pages,
    text_formatter,
    create_embeddings,
    open_and_read_pdf
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Settings(BaseSettings):
    """Application settings from environment variables"""
    # Hugging Face
    huggingface_api_token: str = Field(..., env="HUGGINGFACEHUB_API_TOKEN")
    huggingface_repo_id: str = Field(default="mistralai/Mistral-7B-Instruct-v0.1")
    
    # Pinecone (default vector DB)
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="askmypdf")
    pinecone_env: str = Field(default="us-west1-gcp", env="PINECONE_ENV")
    
    # API Security
    api_key: str = Field(..., env="API_KEY")
    
    class Config:
        env_file = ".env"


settings = Settings()

# Initialize Pinecone
pinecone.init(
    api_key=settings.pinecone_api_key,
    environment=settings.pinecone_env
)

# Global state
app_state = {
    'llm': None,
    'vector_store': None,
}

security = HTTPBearer()


# Dependency for API key validation
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for protected endpoints"""
    if credentials.credentials != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LLM and Vector DB on startup"""
    logger.info("Starting AskMyPDF Cloud-Native RAG Agent...")
    
    try:
        # Initialize LLM (HuggingFace Endpoint)
        logger.info(f"Connecting to HuggingFace model: {settings.huggingface_repo_id}")
        app_state['llm'] = HuggingFaceEndpoint(
            repo_id=settings.huggingface_repo_id,
            task="text-generation",
            temperature=0.7,
            max_length=512,
            huggingfacehub_api_token=settings.huggingface_api_token
        )
        logger.info("✓ LLM initialized successfully")
        
        # Initialize Vector DB connection
        if settings.pinecone_index_name in pinecone.list_indexes():
            app_state['vector_store'] = pinecone.Index(settings.pinecone_index_name)
            logger.info("✓ Connected to existing Pinecone index")
        else:
            # Create index if it doesn't exist
            logger.info("Creating new Pinecone index...")
            pinecone.create_index(
                name=settings.pinecone_index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric="cosine"
            )
            app_state['vector_store'] = pinecone.Index(settings.pinecone_index_name)
            logger.info("✓ Created and connected to Pinecone index")
        
        logger.info("✓ Server ready to accept requests")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    logger.info("Shutting down server...")


# Initialize FastAPI app
app = FastAPI(
    title="AskMyPDF API",
    description="Cloud-Native RAG Agent for PDF Question Answering",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the PDF")
    top_k: int = Field(default=5, description="Number of relevant chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]
    timestamp: str
    cost: Optional[str] = None


class ProcessResponse(BaseModel):
    message: str
    chunks_created: int
    timestamp: str


# Utility Functions
def log_llm_cost(success: bool, tokens_used: int = 0, error: str = None):
    """Log LLM API call costs"""
    timestamp = datetime.now().isoformat()
    if success:
        logger.info(f"[COST LOG] {timestamp} - Success - Tokens: {tokens_used}")
    else:
        logger.error(f"[COST LOG] {timestamp} - Failure - Error: {error}")


def upload_vectors_to_pinecone(chunks: List[dict], embeddings: List[list]):
    """Upload embeddings to Pinecone"""
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            'id': chunk['id'],
            'values': embedding,
            'metadata': {
                'text': chunk['text'],
                'page': chunk['page']
            }
        })
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        app_state['vector_store'].upsert(vectors=batch)
    
    logger.info(f"Uploaded {len(vectors)} vectors to Pinecone")


def retrieve_from_pinecone(query_embedding: List[float], top_k: int = 5) -> List[dict]:
    """Retrieve relevant chunks from Pinecone"""
    results = app_state['vector_store'].query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    chunks = []
    for match in results.matches:
        chunks.append({
            'score': match.score,
            'text': match.metadata['text'],
            'page': match.metadata.get('page', 0)
        })
    
    return chunks


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AskMyPDF",
        "version": "1.0.0",
        "components": {
            "llm": "connected" if app_state['llm'] else "disconnected",
            "vector_db": "connected" if app_state['vector_store'] else "disconnected"
        }
    }


@app.post("/process_pdf", response_model=ProcessResponse)
async def process_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    overlap: int = 200
):
    """
    Process a PDF file and upload embeddings to the cloud vector database.
    
    Args:
        file: PDF file to process
        chunk_size: Size of text chunks (default: 1000 characters)
        overlap: Overlap between chunks (default: 200 characters)
    
    Returns:
        ProcessResponse with processing details
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        logger.info(f"Processing PDF: {file.filename}")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text from PDF
        pages = open_and_read_pdf(temp_path)
        
        # Create chunks
        chunks = create_chunks_from_pages(pages, chunk_size, overlap)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = create_embeddings(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Upload to Pinecone
        upload_vectors_to_pinecone(chunks, embeddings)
        logger.info("Uploaded vectors to Pinecone")
        
        # Cleanup
        os.remove(temp_path)
        
        return ProcessResponse(
            message=f"Successfully processed {file.filename}",
            chunks_created=len(chunks),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query_pdf(request: QueryRequest):
    """
    Query the PDF using RAG pipeline.
    
    Requires API key authentication via Authorization header.
    
    Args:
        request: QueryRequest with question and top_k parameters
    
    Returns:
        QueryResponse with answer and relevant chunks
    """
    try:
        logger.info(f"Processing query: {request.question}")
        
        # Generate query embedding
        query_embeddings = create_embeddings([request.question])
        query_embedding = query_embeddings[0]
        
        # Retrieve relevant chunks
        retrieved_chunks = retrieve_from_pinecone(query_embedding, top_k=request.top_k)
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Prepare context
        context = "\n\n".join([f"[Chunk {i+1}]: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)])
        
        # Create prompt
        prompt_template = """Answer the following question based on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
        
        # Generate answer using LLM
        full_prompt = prompt.format(context=context, question=request.question)
        
        logger.info("Calling LLM to generate answer...")
        start_time = datetime.now()
        
        try:
            answer = app_state['llm'](full_prompt)
            end_time = datetime.now()
            
            tokens_used = len(full_prompt.split()) + len(answer.split())
            log_llm_cost(success=True, tokens_used=tokens_used)
            
            logger.info(f"LLM call successful (duration: {(end_time-start_time).total_seconds():.2f}s)")
            
        except Exception as e:
            log_llm_cost(success=False, error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"LLM API call failed: {str(e)}"
            )
        
        return QueryResponse(
            answer=answer.strip(),
            relevant_chunks=[chunk['text'][:200] + "..." for chunk in retrieved_chunks],
            timestamp=datetime.now().isoformat(),
            cost=f"{tokens_used} tokens"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

