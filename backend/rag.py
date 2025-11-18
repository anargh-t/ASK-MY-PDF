import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS

LOGGER = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    page: int


@dataclass
class QueryMetrics:
    latency_ms: float
    retrieved: int
    retrieval_accuracy: float
    relevance_score: Optional[float] = None


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: List[Chunk] = []
    for idx, segment in enumerate(splitter.split_text(text)):
        if segment.strip():
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_id="",
                    text=segment.strip(),
                    page=idx + 1,
                )
            )
    return chunks


class RAGPipeline:
    """LangChain-driven RAG pipeline with FAISS vector store."""

    def __init__(
        self,
        model_name: str,
        storage_dir: Path,
        llm,
    ):
        self.storage_dir = storage_dir
        self.index_path = storage_dir / "langchain_index"
        self.doc_meta_path = storage_dir / "documents_meta.json"
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store: Optional[LCFAISS] = self._load_vector_store()
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI PDF analyst. Answer strictly using the provided context. "
                    "If the answer is missing, reply with 'I need more information.'\nContext:\n{context}",
                ),
                ("user", "{question}"),
            ]
        )
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser
        self.documents: Dict[str, Dict] = self._load_documents()
        self.history: List[Dict] = []

    def _load_vector_store(self) -> Optional[LCFAISS]:
        if self.index_path.exists():
            try:
                LOGGER.info("Loading LangChain FAISS index from disk.")
                return LCFAISS.load_local(
                    str(self.index_path),
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            except Exception as err:
                LOGGER.error("Failed to load FAISS index: %s", err)
        return None

    def _persist_vector_store(self) -> None:
        if self.vector_store is None:
            return
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.index_path))

    def _load_documents(self) -> Dict[str, Dict]:
        if self.doc_meta_path.exists():
            try:
                with open(self.doc_meta_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as err:
                LOGGER.warning("Unable to load document metadata (%s). Starting fresh.", err)
        return {}

    def _persist_documents(self) -> None:
        try:
            self.doc_meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.doc_meta_path, "w", encoding="utf-8") as fh:
                json.dump(self.documents, fh, ensure_ascii=False, indent=2)
        except Exception as err:
            LOGGER.error("Failed to persist document metadata: %s", err)

    def register_document(
        self,
        filename: str,
        text: str,
        chunk_size: int,
        overlap: int,
        doc_id: Optional[str] = None,
    ) -> Dict:
        doc_id = (doc_id or str(uuid.uuid4())).strip()
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            chunk.doc_id = doc_id

        texts = [chunk.text for chunk in chunks]
        metadatas = [{"doc_id": chunk.doc_id, "page": chunk.page, "chunk_id": chunk.id} for chunk in chunks]

        if self.vector_store is None:
            self.vector_store = LCFAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas,
            )
        else:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

        self._persist_vector_store()

        self.documents[doc_id] = {
            "filename": filename,
            "text": text,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunks": len(chunks),
        }
        self._persist_documents()

        return {"doc_id": doc_id, "chunks": len(chunks)}

    def query(self, doc_id: str, question: str, top_k: int = 5) -> Tuple[str, List[Dict], QueryMetrics]:
        if self.vector_store is None:
            raise ValueError("No documents indexed yet. Please extract a PDF first.")

        doc_id = doc_id.strip()
        doc_metadata = self.documents.get(doc_id)
        if not doc_metadata:
            raise ValueError("Document not found. Extract text before querying.")

        start = time.perf_counter()
        search_results = self.vector_store.similarity_search_with_score(question, k=top_k * 4)
        filtered = [
            {"text": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in search_results
            if doc.metadata.get("doc_id") == doc_id
        ][:top_k]

        context = "\n\n".join(item["text"] for item in filtered) or "No relevant context found."
        response = self.chain.invoke({"context": context, "question": question}).strip()
        latency_ms = (time.perf_counter() - start) * 1000

        metrics = QueryMetrics(
            latency_ms=latency_ms,
            retrieved=len(filtered),
            retrieval_accuracy=float(
                len([item for item in filtered if item["score"] <= 1.0])
            )
            / max(1, len(filtered)),
            relevance_score=None,
        )

        history_entry = {
            "doc_id": doc_id,
            "question": question,
            "answer": response,
            "chunks_returned": len(filtered),
            "metrics": asdict(metrics),
        }
        self.history.append(history_entry)
        return response, filtered, metrics

