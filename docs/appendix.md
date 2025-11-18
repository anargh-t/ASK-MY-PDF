# Appendix – ASK MY PDF Artifacts

## A. Product Backlog
- As a researcher, I want to upload lengthy PDFs so that I can explore them conversationally.
- As an analyst, I want the system to highlight which chunks powered each answer so I can trust the context.
- As a security reviewer, I want validated, size-limited uploads so malicious files are rejected.
- As a product owner, I want latency metrics so we can monitor responsiveness.
- As an end user, I want prior conversations to remain accessible so I can revisit insights.

## B. Release Plan
- **v0.1 – Chat Only:** UI hits Gemini directly without document context.
- **v0.2 – Text Extraction:** Add PDF ingestion, chunking, and manual query context.
- **v0.3 – RAG End-to-End:** Introduce FAISS index, retrieval, and evaluation metrics.
- **v1.0 – Secure Product:** Hardened FastAPI backend, Streamlit UI, documentation, and packaging assets.

## C. High-Level Sprint Plan
1. **Sprint 1:** Model development (extraction, chunking, embeddings, FAISS, metrics).
2. **Sprint 2:** Backend API + Streamlit UI, full CRUD endpoints, upload/extract/query/history.
3. **Sprint 3:** Security posture, STRIDE analysis, logging, input validation, failure simulations.
4. **Sprint 4:** Packaging, docs, release checklist, demo video hook, README polish.

## D. Sprint Planning Sheet (Story Points)
| Story | Description | Points |
|-------|-------------|--------|
| S1-1 | Implement multi-layer PDF extraction | 5 |
| S1-2 | Build chunker + embedding generator | 3 |
| S1-3 | Persist FAISS index + metadata | 5 |
| S2-1 | Design FastAPI endpoints (/upload,/extract-text,/query,/history) | 8 |
| S2-2 | Implement Streamlit UI with history | 5 |
| S3-1 | Add validation + exception handling + STRIDE docs | 5 |
| S4-1 | Produce appendix artifacts + README + packaging | 3 |

## E. Burndown (Sample)
Day 1: 34 pts → Day 2: 28 pts → Day 3: 21 pts → Day 4: 13 pts → Day 5: 5 pts → Demo: 0 pts.

## F. Folder Structure
```
ASK-MY-PDF/
├── backend/
│   ├── main.py
│   ├── extract.py
│   ├── rag.py
│   └── vector_store/
├── frontend/
│   ├── app.py
│   └── components/
├── data/
│   └── uploaded_files/
├── models/
│   └── embeddings.*
├── docs/
│   ├── architecture.png
│   ├── threat_model.pdf
│   ├── sprint_plan.pdf
│   └── appendix.md
├── scripts/
│   └── generate_docs.py
├── README.md
└── requirements.txt
```

## G. README.md
Updated with architecture, setup, and system overview (see repository root).

## H. requirements.txt
Pinned dependencies for backend + frontend + PDF tooling (see repository root).

