# Troubleshooting Guide for AskMyPDF

## Common Issues and Solutions

### 1. "LLM API call failed" Errors

#### Gemini API Issues

**Problem**: Model not found (404 error)
- **Solution**: The code automatically detects available models. Check server logs for "Available Gemini models" message
- **Manual fix**: Update `.env` with a valid model name:
  ```
  GEMINI_MODEL=gemini-pro
  ```
  Or try: `gemini-1.5-flash`, `gemini-1.5-pro`

**Problem**: API key invalid
- **Solution**: 
  1. Get a new API key from https://aistudio.google.com/app/apikey
  2. Update `GOOGLE_API_KEY` in your `.env` file
  3. Restart the server

#### OpenAI API Issues

**Problem**: Quota exceeded (429 error)
- **Solution**: 
  - Check your OpenAI account billing at https://platform.openai.com/account/billing
  - Switch to Gemini by setting `LLM_PROVIDER=gemini` in `.env`

#### Hugging Face API Issues

**Problem**: 403 Forbidden - Insufficient permissions
- **Solution**: 
  1. Go to https://huggingface.co/settings/tokens
  2. Create a new token with "Inference Providers" scope
  3. Update `HUGGINGFACEHUB_API_TOKEN` in `.env`
  4. Restart server

### 2. "Failed to fetch" Error

**Problem**: Frontend can't connect to backend
- **Solutions**:
  1. **Check if server is running**: Look for "Application startup complete" in terminal
  2. **Check the URL**: Make sure you're accessing `http://localhost:8000/app`
  3. **Check CORS**: The server has CORS enabled, but if issues persist, check browser console
  4. **Restart server**: Stop (Ctrl+C) and restart with `python api_server.py`

### 3. Server Won't Start

**Problem**: Import errors or missing packages
- **Solution**: Install all dependencies:
  ```powershell
  pip install -r requirements.txt
  ```

**Problem**: Environment variables not found
- **Solution**: 
  1. Check that `.env` file exists in the project root
  2. Verify all required variables are set (see `.env` example below)
  3. Restart the server after changing `.env`

### 4. PDF Processing Fails

**Problem**: "Only PDF files are supported"
- **Solution**: Make sure you're uploading a `.pdf` file, not other formats

**Problem**: No chunks created
- **Solution**: 
  - Check if PDF has extractable text (some scanned PDFs need OCR)
  - Try adjusting `chunk_size` and `overlap` parameters

### 5. Query Returns No Results

**Problem**: No relevant chunks found
- **Solutions**:
  1. **Upload a PDF first**: Make sure you've processed a PDF before querying
  2. **Check Pinecone**: Verify your Pinecone API key is valid
  3. **Increase top_k**: Try setting `top_k=10` in the query to get more results

## Step-by-Step Setup Verification

### 1. Verify Environment Variables

Your `.env` file should have:
```env
# LLM Provider (choose one)
LLM_PROVIDER=gemini  # or "openai" or "huggingface"

# For Gemini
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-pro

# For OpenAI (if using)
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-3.5-turbo

# For Hugging Face (if using)
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
HUGGINGFACE_REPO_ID=gpt2

# Required
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=askmypdf
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start Server

```powershell
python api_server.py
```

**Expected output**:
```
INFO:     Starting AskMyPDF Cloud-Native RAG Agent...
INFO:     ✓ Google Gemini LLM initialized with model: gemini-pro
INFO:     ✓ Connected to Pinecone index
INFO:     ✓ Server ready to accept requests
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Test the Application

1. Open browser: `http://localhost:8000/app`
2. Upload a PDF file
3. Wait for "Successfully processed" message
4. Ask a question
5. Check the answer

## Quick Fixes

### Reset Everything

1. **Stop the server**: Press `Ctrl+C`
2. **Clear Python cache**: 
   ```powershell
   Remove-Item -Recurse -Force __pycache__
   ```
3. **Reinstall dependencies**:
   ```powershell
   pip install -r requirements.txt --upgrade
   ```
4. **Restart server**:
   ```powershell
   python api_server.py
   ```

### Check Logs

Look for error messages in the terminal where the server is running. Common patterns:
- `ERROR` - Something failed
- `WARNING` - Non-critical issue
- `INFO` - Normal operation

### Verify API Keys

**Gemini**: Test at https://aistudio.google.com/app/apikey
**OpenAI**: Test at https://platform.openai.com/api-keys
**Pinecone**: Test at https://app.pinecone.io/

## Still Having Issues?

1. **Check server logs** for specific error messages
2. **Verify all API keys** are valid and have proper permissions
3. **Ensure all packages are installed**: `pip list | findstr "google-generativeai openai httpx"`
4. **Try a different LLM provider** by changing `LLM_PROVIDER` in `.env`

## Recommended Configuration

For easiest setup, use **Gemini**:
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-pro
```

Gemini offers:
- ✅ Free tier available
- ✅ No special permissions needed
- ✅ Easy setup
- ✅ Good performance




