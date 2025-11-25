# Troubleshooting Guide

Use this guide when the FastAPI/Flask stack misbehaves. The most common issues are LLM configuration, missing indices, or port conflicts.

---

## 1. “Query pipeline failed”
| Symptom | Probable cause | Fix |
|---------|----------------|-----|
| 410 / 404 for Hugging Face | Using the deprecated `api-inference` host or an unsupported model | Ensure `.env` contains<br>`LLM_PROVIDER=huggingface`<br>`HUGGINGFACE_API_BASE=https://router.huggingface.co/v1`<br>`HUGGINGFACE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct` (or another chat model). |
| 400 `model_not_supported` | Hugging Face router requires chat-capable models | Pick a router-supported chat model (Meta Llama 3 Instruct, Gemma Instruct, etc.). |
| “Document not found” | Query called before extraction or index was lost | Either click “Extract & Index” in the UI or rely on auto-indexing by re-running `/extract-text`. |

**Tip:** read `server.log` or the FastAPI console to see the raw provider error.

---

## 2. LLM provider quick checks
- **Gemini**
  - 404 → update `GEMINI_MODEL` to `gemini-pro`, `gemini-1.5-flash`, or `gemini-1.5-pro`.
  - 401 → regenerate the key at <https://aistudio.google.com/app/apikey>.
- **OpenAI**
  - 429 → review billing & usage <https://platform.openai.com/account/usage>. Set `LLM_PROVIDER=gemini` temporarily if throttled.
  - 404 → ensure `OPENAI_MODEL` matches the account capability (e.g. `gpt-4o-mini`).
- **Hugging Face Router**
  - 403 → token missing “Inference Providers” scope. Create a new token at <https://huggingface.co/settings/tokens>.
  - 410/404 → outdated endpoint; set `HUGGINGFACE_API_BASE=https://router.huggingface.co/v1`.
  - 400 `model_not_supported` → choose a chat model exposed via router (see [HF Router docs](https://huggingface.co/docs/inference-endpoints/router/overview)).

---

## 3. Backend not reachable / “Failed to fetch”
1. Confirm FastAPI is running: you should see `Application startup complete` and `Uvicorn running on http://127.0.0.1:8000`.
2. If you see `Errno 10048` (port already in use), free the port:
   ```powershell
   netstat -ano | findstr 8000
   taskkill /PID <PID> /F
   ```
   Then restart with `uvicorn backend.main:app --port 8000`.
3. Ensure the Flask UI points to the same backend URL (`BACKEND_URL` env var defaults to `http://localhost:8000`).
4. Clear browser cache / hard refresh if the UI still shows stale errors.

---

## 4. PDF processing issues
- **“Only PDF files are supported”** → double-check the extension and MIME type.
- **No chunks created** → the PDF may be scanned without text. Run OCR externally or try another file.
- **Stalled extraction** → large PDFs (>20 MB) are rejected. Keep files below the configured limit.

---

## 5. Setup checklist
1. **Environment variables** (minimum)
   ```env
   LLM_PROVIDER=openai              # or gemini / huggingface
   OPENAI_API_KEY=...
   OPENAI_MODEL=gpt-4o-mini
   GOOGLE_API_KEY=...               # if using Gemini
   GEMINI_MODEL=gemini-1.5-flash
   HUGGINGFACEHUB_API_TOKEN=...     # if using Hugging Face router
   HUGGINGFACE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
   HUGGINGFACE_API_BASE=https://router.huggingface.co/v1
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```
2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run backend**
   ```powershell
   uvicorn backend.main:app --reload --port 8000
   ```
4. **Run Flask UI**
   ```powershell
   cd frontend
   python app.py
   ```

---

## 6. Quick recovery
1. Stop server (`Ctrl+C`), ensure no stray Uvicorn processes are running.
2. Remove `__pycache__` folders (optional) and re-install requirements if dependencies changed.
3. Delete temporary uploads in `data/uploaded_files/` if space is an issue (the backend will recreate them).
4. Restart Uvicorn, then refresh the Flask page.

---

## 7. Need more help?
- Re-run recent requests with `uvicorn ... > server.log 2>&1` to capture stack traces.
- Switch LLM providers to isolate whether the issue is model-specific.
- Share the error snippet (including HTTP code) when filing an issue so we can reproduce quickly.

