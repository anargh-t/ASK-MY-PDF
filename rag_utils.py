"""
RAG Utilities: Core reusable functions for PDF processing and embeddings.
"""
import re
from typing import List, Dict
from pypdf import PdfReader


def open_and_read_pdf(pdf_path: str) -> List[str]:
    """
    Extract text from all pages of a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of page texts
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        pages.append(text)
    return pages


def create_chunks_from_pages(pages: List[str], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Split pages into overlapping chunks for better context preservation.
    
    Args:
        pages: List of page texts
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []
    chunk_id = 0
    
    for page_num, page_text in enumerate(pages):
        if not page_text.strip():
            continue
            
        # Clean text
        clean_text = text_formatter(page_text)
        
        # Create chunks with overlap
        start = 0
        while start < len(clean_text):
            end = start + chunk_size
            chunk_text = clean_text[start:end]
            
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': chunk_text.strip(),
                'page': page_num + 1,
                'start': start,
                'end': min(end, len(clean_text))
            })
            
            chunk_id += 1
            start = end - overlap  # Overlap for context
            
            # Avoid infinite loop
            if start >= len(clean_text):
                break
    
    return chunks


def text_formatter(text: str) -> str:
    """
    Clean and format text by removing extra whitespace and special characters.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text


def create_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Generate embeddings for a list of texts using sentence transformers.
    This still runs locally as per the constraint.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the sentence transformer model
        
    Returns:
        List of embedding vectors
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings.tolist()


