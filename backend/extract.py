import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pdfplumber
import fitz  # PyMuPDF
from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB = 20


class PDFValidationError(Exception):
    """Raised when the uploaded file fails validation."""


class PDFExtractionError(Exception):
    """Raised when text or tables cannot be extracted from a PDF."""


@dataclass
class ExtractedContent:
    text: str
    tables: List[List[str]]


def validate_pdf_file(file_path: Path) -> None:
    """Validate extension and size constraints."""
    suffix = file_path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise PDFValidationError("Only PDF files are allowed.")

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise PDFValidationError(f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove harmful characters."""
    safe_text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", text)
    safe_text = re.sub(r"\s+", " ", safe_text)
    return safe_text.strip()


def _extract_with_pdfplumber(file_path: Path) -> Tuple[str, List[List[str]]]:
    with pdfplumber.open(file_path) as pdf:
        pages_text = []
        tables: List[List[str]] = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
            for table in page.extract_tables() or []:
                flattened = [" | ".join(filter(None, row)) for row in table if any(row)]
                if flattened:
                    tables.append(flattened)
        return "\n".join(pages_text), tables


def _extract_with_pypdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    text_segments = []
    for page in reader.pages:
        text_segments.append(page.extract_text() or "")
    return "\n".join(text_segments)


def _extract_with_pymupdf(file_path: Path) -> str:
    document = fitz.open(file_path)
    text_segments = []
    for page in document:
        text_segments.append(page.get_text("text") or "")
    return "\n".join(text_segments)


def extract_pdf_contents(file_path: Path) -> ExtractedContent:
    """
    Extract text (and tables when available) from a PDF using multiple fallbacks.
    """
    validate_pdf_file(file_path)
    LOGGER.info("Extracting text from %s", file_path.name)

    text: str = ""
    tables: List[List[str]] = []

    try:
        text, tables = _extract_with_pdfplumber(file_path)
    except Exception as err:
        LOGGER.warning("pdfplumber failed (%s). Falling back to PyPDF.", err)

    if not text.strip():
        try:
            text = _extract_with_pypdf(file_path)
        except Exception as err:
            LOGGER.warning("PyPDF failed (%s). Falling back to PyMuPDF.", err)

    if not text.strip():
        try:
            text = _extract_with_pymupdf(file_path)
        except Exception as err:
            LOGGER.error("PyMuPDF failed (%s).", err)
            raise PDFExtractionError("Unable to extract text from PDF.")

    cleaned_text = _clean_text(text)
    cleaned_tables = [[_clean_text(row) for row in table if row] for table in tables]

    if not cleaned_text:
        raise PDFExtractionError("PDF did not contain readable text.")

    return ExtractedContent(text=cleaned_text, tables=cleaned_tables)

