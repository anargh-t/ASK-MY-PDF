"""
AskMyPDF: Cloud-Native RAG Agent - FastAPI Server
Connects to external LLM (HuggingFace) and Vector DB (Pinecone/Weaviate)
"""
import os
import logging
import traceback
import httpx
import asyncio
from typing import Optional, List, Any, Dict
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import uvicorn

from pinecone import Pinecone, ServerlessSpec
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM Provider - choose "gemini", "openai", or "huggingface"
    llm_provider: str = Field(default="gemini", env="LLM_PROVIDER")
    
    # Google Gemini (recommended - free tier available)
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash-latest", env="GEMINI_MODEL")
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    
    # Hugging Face (requires Inference Providers permissions)
    huggingface_api_token: Optional[str] = Field(default=None, env="HUGGINGFACEHUB_API_TOKEN")
    huggingface_repo_id: str = Field(
        default="gpt2",
        env="HUGGINGFACE_REPO_ID"
    )

    # Pinecone (default vector DB)
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="askmypdf")
    # Use a known serverless-supported default
    pinecone_env: str = Field(default="us-east-1-aws", env="PINECONE_ENV")

    @model_validator(mode="before")
    @classmethod
    def _normalize_env_keys(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Support alternate env variable casing and common mistakes."""
        if not isinstance(data, dict):
            return data

        # Accept lowercase key from .env (huggingfacehub_api_token)
        token = data.get("huggingfacehub_api_token")
        if token and "huggingface_api_token" not in data:
            data["huggingface_api_token"] = token

        # Some users accidentally paste the token into huggingface_repo_id
        repo_id = data.get("huggingface_repo_id")
        if repo_id and isinstance(repo_id, str) and repo_id.startswith("hf_"):
            if "huggingface_api_token" not in data:
                data["huggingface_api_token"] = repo_id
            data["huggingface_repo_id"] = "mistralai/Mistral-7B-Instruct-v0.1"

        return data


settings = Settings()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Initialize Pinecone (v3 client)
def _parse_pinecone_env(env_str: str):
    """
    Parse env string into (cloud, region). Accepts both 'region-cloud' and 'cloud-region'.
    Falls back to a known serverless-supported default if unsupported.
    """
    supported = {
        ("aws", "us-east-1"),
        ("aws", "us-west-2"),
        ("aws", "eu-west-1"),
        ("aws", "ap-southeast-1"),
        ("gcp", "us-central1"),
        ("gcp", "europe-west4"),
        ("gcp", "asia-southeast1"),
    }
    default = ("aws", "us-east-1")
    try:
        tokens = env_str.split("-")
        cloud_tokens = {"aws", "gcp", "azure"}
        cloud = None
        region_tokens = []
        for t in tokens:
            if t in cloud_tokens and cloud is None:
                cloud = t
            else:
                region_tokens.append(t)
        # If cloud not found, assume last token is cloud
        if cloud is None and tokens:
            if tokens[-1] in cloud_tokens:
                cloud = tokens[-1]
                region_tokens = tokens[:-1]
        region = "-".join([tok for tok in region_tokens if tok]) or default[1]
        cloud = cloud or default[0]
        pair = (cloud, region)
        if pair not in supported:
            return default
        return pair
    except Exception:
        return default

pc = Pinecone(api_key=settings.pinecone_api_key)
cloud, region = _parse_pinecone_env(settings.pinecone_env)

# Global state
app_state = {
    'llm_provider': None,  # "gemini", "openai", or "huggingface"
    'gemini_client': None,  # Gemini client
    'gemini_model': None,  # Gemini model name
    'openai_client': None,  # OpenAI client
    'openai_model': None,  # OpenAI model name
    'hf_model': None,  # HuggingFace model name
    'hf_token': None,  # HuggingFace API token
    'vector_store': None,
}

# Custom exceptions
class RateLimitExceededError(Exception):
    """Raised when an upstream LLM provider reports a rate-limit/429 error."""

    def __init__(self, provider: str, retry_after: Optional[int] = None, detail: Optional[str] = None):
        self.provider = provider
        self.retry_after = retry_after
        self.detail = detail or "Rate limit exceeded"
        super().__init__(self.__str__())

    def __str__(self) -> str:
        retry_hint = f" Retry after ~{self.retry_after}s." if self.retry_after else ""
        provider_name = self.provider.capitalize()
        return f"{provider_name} rate limit exceeded.{retry_hint} {self.detail}"


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LLM and Vector DB on startup"""
    logger.info("Starting AskMyPDF Cloud-Native RAG Agent...")
    
    try:
        # Initialize LLM based on provider
        app_state['llm_provider'] = settings.llm_provider.lower()
        
        if app_state['llm_provider'] == "gemini":
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            app_state['gemini_client'] = genai
            app_state['gemini_model'] = settings.gemini_model
            logger.info(f"✓ Google Gemini LLM initialized with model: {settings.gemini_model}")
        elif app_state['llm_provider'] == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            from openai import AsyncOpenAI
            app_state['openai_client'] = AsyncOpenAI(api_key=settings.openai_api_key)
            app_state['openai_model'] = settings.openai_model
            logger.info(f"✓ OpenAI LLM initialized with model: {settings.openai_model}")
        else:
            # HuggingFace
            if not settings.huggingface_api_token:
                raise ValueError("HUGGINGFACEHUB_API_TOKEN is required when LLM_PROVIDER=huggingface")
            app_state['hf_model'] = settings.huggingface_repo_id
            app_state['hf_token'] = settings.huggingface_api_token
            logger.info(f"✓ HuggingFace LLM configuration ready for model: {settings.huggingface_repo_id}")
        
        # Initialize Vector DB connection (Pinecone v3)
        existing_indexes = set(pc.list_indexes().names())
        if settings.pinecone_index_name not in existing_indexes:
            logger.info("Creating new Pinecone index...")
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            logger.info("✓ Created Pinecone index")

        app_state['vector_store'] = pc.Index(settings.pinecone_index_name)
        logger.info("✓ Connected to Pinecone index")
        
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

if os.path.isdir(FRONTEND_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static"
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


def _extract_gemini_text(response) -> str:
    """Normalize Gemini responses that may contain multi-part content."""
    def _get_text_from_part(part) -> Optional[str]:
        if part is None:
            return None
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            return part.get("text")
        return getattr(part, "text", None)

    # Fast path: simple responses expose `.text`
    text = None
    try:
        text = getattr(response, "text", None)
    except Exception:
        text = None
    if isinstance(text, str) and text.strip():
        return text.strip()

    collected: List[str] = []
    try:
        candidates = getattr(response, "candidates", None)
    except Exception:
        candidates = None

    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None) or getattr(candidate, "parts", None)
            parts = []
            if content is None:
                continue
            if isinstance(content, dict):
                parts = content.get("parts") or []
            elif isinstance(content, list):
                parts = content
            else:
                parts = getattr(content, "parts", []) or []

            for part in parts:
                part_text = _get_text_from_part(part)
                if part_text:
                    collected.append(part_text.strip())

    # Some SDK versions nest under response.result
    if not collected:
        result = getattr(response, "result", None)
        if result and isinstance(result, dict):
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    part_text = _get_text_from_part(part)
                    if part_text:
                        collected.append(part_text.strip())

    if collected:
        return "\n\n".join(chunk for chunk in collected if chunk)

    raise ValueError(
        "Gemini API returned a response without text parts. "
        "Try a different model or check the server logs for raw response details."
    )


async def call_gemini_api(prompt: str, genai_client, model: str) -> str:
    """Call Google Gemini API (async)"""
    try:
        # List available models to find one that works
        model_to_use = model.replace("models/", "").replace("-latest", "")
        try:
            available_models = await asyncio.to_thread(genai_client.list_models)
            valid_models = [
                m.name.replace("models/", "") 
                for m in available_models 
                if 'generateContent' in m.supported_generation_methods
            ]
            if valid_models:
                logger.info(f"Available Gemini models: {valid_models}")
                # Try to use specified model or fallback to first available
                if model_to_use not in valid_models:
                    # Try variations
                    for vm in valid_models:
                        if model_to_use in vm or vm in model_to_use:
                            model_to_use = vm
                            break
                    else:
                        model_to_use = valid_models[0]
                        logger.warning(f"Using available model: {model_to_use}")
        except Exception as list_error:
            logger.warning(f"Could not list models: {list_error}, using: {model_to_use}")
        
        # Create model instance (without models/ prefix)
        model_instance = genai_client.GenerativeModel(model_to_use)
        
        # Generate content with simplified config (as dict, not types.GenerationConfig)
        response = await asyncio.to_thread(
            model_instance.generate_content,
            prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 512,
            }
        )

        return _extract_gemini_text(response)
    except Exception as e:
        # Provide helpful error message
        error_msg = str(e)
        lower_msg = error_msg.lower()
        if "429" in error_msg or "quota" in lower_msg or "rate limit" in lower_msg:
            retry_after = None
            # Gemini often includes "Retry in Xs"
            for token in lower_msg.split():
                if token.endswith("s") and token[:-1].replace(".", "", 1).isdigit():
                    try:
                        retry_after = int(float(token[:-1]))
                        break
                    except ValueError:
                        continue
            raise RateLimitExceededError(
                provider="gemini",
                retry_after=retry_after,
                detail=error_msg
            )
        if "404" in error_msg or "not found" in lower_msg:
            raise Exception(
                f"Gemini API error: {error_msg}\n\n"
                "Try using one of these model names in your .env:\n"
                "- gemini-pro\n"
                "- gemini-1.5-flash\n"
                "- gemini-1.5-pro\n"
                "Or check available models at: https://ai.google.dev/models/gemini"
            )
        raise Exception(f"Gemini API error: {error_msg}")


async def call_openai_api(prompt: str, client, model: str) -> str:
    """Call OpenAI API (async)"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")


async def call_huggingface_api(prompt: str, model: str, token: str) -> str:
    """Call Hugging Face Inference API directly (async)"""
    # Try the router endpoint first (requires Inference Providers permissions)
    api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 403:
            # Permission error - provide helpful guidance
            error_detail = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            error_text = error_detail.get("error", response.text)
            
            raise Exception(
                f"HTTP 403: {error_text}\n\n"
                "Your Hugging Face token doesn't have 'Inference Providers' permissions.\n"
                "To fix this:\n"
                "1. Go to https://huggingface.co/settings/tokens\n"
                "2. Create a new token with 'Inference Providers' scope enabled\n"
                "3. Update your .env file with the new token\n"
                "4. Restart the server\n\n"
                "Alternatively, you can use local models or other LLM services."
            )
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            if response.status_code == 503:
                error_msg += "\nModel is loading, please wait a moment and try again."
            raise Exception(error_msg)
        
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                # Usually has 'generated_text' key
                return result[0].get('generated_text', str(result[0]))
            else:
                return str(result[0])
        elif isinstance(result, dict):
            if 'generated_text' in result:
                return result['generated_text']
            elif 'text' in result:
                return result['text']
            else:
                return str(result)
        else:
            return str(result)


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AskMyPDF",
        "version": "1.0.0",
        "components": {
            "llm": "connected" if (app_state['gemini_client'] or app_state['openai_client'] or app_state['hf_model']) else "disconnected",
            "vector_db": "connected" if app_state['vector_store'] else "disconnected"
        }
    }


@app.get("/app", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the lightweight web UI"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


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


@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """
    Query the PDF using RAG pipeline.
    
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
        formatted_context = context or "No relevant context could be retrieved from the document."
        question_text = request.question.strip()
        
        # Generate answer using LLM
        full_prompt = prompt_template.format(
            context=formatted_context,
            question=question_text
        )
        
        logger.info("Calling LLM to generate answer...")
        start_time = datetime.now()
        
        try:
            # Call LLM based on provider
            if app_state['llm_provider'] == "gemini":
                answer = await call_gemini_api(
                    full_prompt,
                    app_state['gemini_client'],
                    app_state['gemini_model']
                )
            elif app_state['llm_provider'] == "openai":
                answer = await call_openai_api(
                    full_prompt,
                    app_state['openai_client'],
                    app_state['openai_model']
                )
            else:
                # HuggingFace
                answer = await call_huggingface_api(
                    full_prompt,
                    app_state['hf_model'],
                    app_state['hf_token']
                )
            
            end_time = datetime.now()
            
            tokens_used = len(full_prompt.split()) + len(answer.split())
            log_llm_cost(success=True, tokens_used=tokens_used)
            
            logger.info(f"LLM call successful (duration: {(end_time-start_time).total_seconds():.2f}s)")
            
        except RateLimitExceededError as rate_error:
            log_llm_cost(success=False, error=str(rate_error))
            raise HTTPException(
                status_code=429,
                detail=str(rate_error)
            ) from rate_error
        except Exception as e:
            error_msg = str(e)
            log_llm_cost(success=False, error=error_msg)
            
            # Error message already contains helpful guidance from call_huggingface_api
            detail = f"LLM API call failed: {error_msg}"
            
            raise HTTPException(status_code=500, detail=detail)
        
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

