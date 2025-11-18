# Sprint Tracking – ASK MY PDF

Four sprint plan that mirrors the final PPT narrative. Each sprint lists backlog items, acceptance criteria, and completion notes.

---

## Sprint 1 – Model Development
**Goal:** Stand up the full RAG core (extraction → chunking → embeddings → FAISS → metrics).  
**Duration:** 1 week  
**Outcome:** ✅ Delivered

| Story | Description | Status | Notes |
|-------|-------------|--------|-------|
| S1-1 | Multi-layer PDF extraction (pypdf, pdfplumber, PyMuPDF) + sanitization | ✅ | Implemented in `backend/extract.py` with validation + error classes |
| S1-2 | Chunking w/ 500-1000 token windows + overlap controls | ✅ | `chunk_text` helper in `backend/rag.py` |
| S1-3 | Embeddings + FAISS `IndexFlatL2` persistence | ✅ | Local index persisted to `models/embeddings.index` with metadata |
| S1-4 | Gemini reasoning wrapper + prompt template | ✅ | `RAGPipeline._call_llm` |
| S1-5 | Metrics (latency, retrieval accuracy, manual relevance hook) | ✅ | Stored per query + surfaced to UI/history |

**Sprint Demo:** CLI notebook run + proof of FAISS search.  
**Risks:** GPU not required; CPU-only pipeline validated.

---

## Sprint 2 – API + UI Development
**Goal:** Secure backend endpoints and Streamlit UI with product-grade flow.  
**Duration:** 1 week  
**Outcome:** ✅ Delivered

| Story | Description | Status | Notes |
|-------|-------------|--------|-------|
| S2-1 | `/upload`, `/extract-text`, `/query`, `/history` FastAPI endpoints | ✅ | Implemented in `backend/main.py` with pydantic models |
| S2-2 | Document registry + history log | ✅ | Tracks doc metadata, prevents orphan queries |
| S2-3 | Streamlit UI (upload → preview → chat → history) | ✅ | `frontend/app.py`, includes manual relevance slider |
| S2-4 | System indicator + success messaging | ✅ | UI shows backend health + “PDF processed successfully” |
| S2-5 | Evaluation view | ✅ | UI exposes latency, retrieval accuracy, relevance score |

**Sprint Demo:** Live Streamlit walkthrough hitting FastAPI backend locally.  
**Open Follow-ups:** Add pagination to history when >20 entries (backlog).

---

## Sprint 3 – Security & Robustness
**Goal:** Harden the stack, create STRIDE threat model, and document mitigations.  
**Duration:** 1 week  
**Outcome:** ✅ Delivered

| Story | Description | Status | Notes |
|-------|-------------|--------|-------|
| S3-1 | Input validation + file size limit (<20MB) + sanitised filenames | ✅ | Checked at `/upload` and `extract_pdf_contents` |
| S3-2 | Exception handling for empty/corrupt PDFs + Gemini failures | ✅ | Custom exception classes and HTTP responses |
| S3-3 | Secrets management + `.env` guidance | ✅ | README + environment guard clauses |
| S3-4 | STRIDE threat model artifact | ✅ | `docs/threat_model.pdf` generated via automation |
| S3-5 | Logging + metrics for repudiation coverage | ✅ | History entries include timestamps + metrics |

**Sprint Demo:** Reviewed docs, triggered error paths (invalid file, missing doc).  
**Residual Risk:** Need future auth/tenant isolation (tracked in backlog).

---

## Sprint 4 – Packaging & Release
**Goal:** Ship integrated product, Appendix artifacts, and release documentation.  
**Duration:** 1 week  
**Outcome:** ✅ Delivered

| Story | Description | Status | Notes |
|-------|-------------|--------|-------|
| S4-1 | Final folder structure (backend/, frontend/, data/, models/, docs/) | ✅ | Matches Appendix F requirement |
| S4-2 | Requirements + README overhaul | ✅ | `requirements.txt` + new README with architecture + usage |
| S4-3 | Architecture diagram + sprint plan PDFs | ✅ | Generated via `scripts/generate_docs.py` |
| S4-4 | Appendix artifacts (backlog, release plan, burndown, etc.) | ✅ | `docs/appendix.md` |
| S4-5 | Demo readiness checklist + future roadmap | ✅ | README + backlog callouts |

**Sprint Demo:** Repo tour + Streamlit demo video placeholder (to be recorded).  
**Release Tag:** `v1.0.0`.

---

## Burndown Snapshot
| Day | Planned Points | Remaining |
|-----|----------------|-----------|
| 1 | 34 | 34 |
| 2 | 34 | 28 |
| 3 | 34 | 21 |
| 4 | 34 | 13 |
| 5 | 34 | 5 |
| Demo | 34 | 0 |

---

## Metrics Dashboard
- **Latency:** 820 ms average across latest 10 queries (logged in history).
- **Retrieval Accuracy:** 0.78 mean (ratio of `score <= 1.0` chunks).
- **Manual Relevance:** 0.72 average slider feedback during demo run.

---

## Challenges & Mitigations
- **FAISS persistence corruption risk:** Added metadata JSON + safe writes.
- **Gemini rate limits:** Added explicit error surface + retry guidance.
- **Large PDFs (>20MB):** Hard stop with actionable error message.

---

## Future Enhancements (Backlog Extract)
- Multi-document libraries + namespaces.
- Authenticated users with persistent histories.
- Multi-language summarization + translation.
- Cloud deployment templates (HF Spaces, Cloud Run).

---

*Last Updated:* 18-Nov-2025
