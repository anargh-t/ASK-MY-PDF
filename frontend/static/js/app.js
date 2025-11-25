// Global state
let currentDocId = null;
let currentFilename = null;
let isExtracted = false;
let messages = [];

// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const uploadProgress = document.getElementById('upload-progress');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const removeFileBtn = document.getElementById('remove-file');
const extractSection = document.getElementById('extract-section');
const extractBtn = document.getElementById('extract-btn');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatContainer = document.getElementById('chat-container');
const pdfContainer = document.getElementById('pdf-container');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const clearChatBtn = document.getElementById('clear-chat-btn');
const newSessionBtn = document.getElementById('new-session-btn');
const historyList = document.getElementById('history-list');
const chunkSizeSlider = document.getElementById('chunk-size');
const chunkSizeValue = document.getElementById('chunk-size-value');
const chunkOverlapSlider = document.getElementById('chunk-overlap');
const chunkOverlapValue = document.getElementById('chunk-overlap-value');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initialize state from server-side data
    const mainContent = document.querySelector('[data-doc-id]');
    
    if (mainContent) {
        const docId = mainContent.dataset.docId;
        const extracted = mainContent.dataset.extracted === 'true';
        
        if (docId) {
            currentDocId = docId;
        }
        if (extracted) {
            isExtracted = extracted;
        }
    }
    
    // Show extract button if doc_id exists
    if (currentDocId && extractSection) {
        extractSection.style.display = 'block';
    }
    
    initializeEventListeners();
    initializeSettings();
    checkBackendStatus();
    updateChatInputState();
});

// Event Listeners
function initializeEventListeners() {
    // File upload
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    fileInput.addEventListener('change', handleFileSelect);
    removeFileBtn.addEventListener('click', removeFile);
    
    // Extract
    extractBtn.addEventListener('click', handleExtract);

    
    // Chat
    sendBtn.addEventListener('click', handleSendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    
    // Session management
    clearChatBtn.addEventListener('click', clearChat);
    newSessionBtn.addEventListener('click', newSession);
    
    // History items
    document.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => loadSession(item.dataset.sessionId));
    });
}

function initializeSettings() {
    if (chunkSizeSlider) {
        chunkSizeSlider.addEventListener('input', (e) => {
            chunkSizeValue.textContent = e.target.value;
        });
    }
    
    if (chunkOverlapSlider) {
        chunkOverlapSlider.addEventListener('input', (e) => {
            chunkOverlapValue.textContent = e.target.value;
        });
    }
}

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.5)';
    uploadArea.style.background = 'rgba(255, 255, 255, 0.1)';
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.style.borderColor = 'rgba(255, 255, 255, 0.3)';
    uploadArea.style.background = 'rgba(255, 255, 255, 0.05)';
}

function handleDrop(e) {
    e.preventDefault();
    handleDragLeave(e);
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === 'application/pdf') {
        handleFile(files[0]);
    } else {
        showToast('error', 'Invalid file type', 'Please upload a PDF file');
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (file.size > 20 * 1024 * 1024) {
        showToast('error', 'File too large', 'Maximum file size is 20MB');
        return;
    }
    
    currentFilename = file.name;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'block';
    uploadArea.querySelector('.upload-placeholder').style.display = 'none';
    uploadProgress.style.display = 'block';
    
    uploadFile(file);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading('Uploading PDF...');
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(async response => {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return response.json();
        } else {
            const text = await response.text();
            throw new Error(text || `Server returned status ${response.status}`);
        }
    })
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showToast('error', 'Upload Failed', data.error);
            removeFile();
            return;
        }
        
        currentDocId = data.doc_id;
        isExtracted = false;
        
        showToast('success', 'Upload Successful', `${data.filename} uploaded successfully`);
        
        // Show extract button
        extractSection.style.display = 'block';
        
        // Update UI
        uploadProgress.style.display = 'none';
        uploadArea.querySelector('.upload-placeholder').style.display = 'block';
        
        // Update chat input state
        updateChatInputState();
        
        // Load PDF preview - will be loaded from server if page reloads
        loadPDFPreview(file);
        
        // Also set up server-side PDF URL for reloads
        if (data.doc_id) {
            const pdfViewer = document.getElementById('pdf-viewer');
            if (pdfViewer) {
                pdfViewer.src = `/api/pdf/${data.doc_id}`;
                pdfViewer.style.display = 'block';
                const placeholder = pdfContainer.querySelector('.pdf-placeholder');
                if (placeholder) placeholder.style.display = 'none';
            }
        }
    })
    .catch(error => {
        hideLoading();
        showToast('error', 'Upload Error', error.message);
        removeFile();
    });
}

function removeFile() {
    fileInput.value = '';
    fileInfo.style.display = 'none';
    extractSection.style.display = 'none';
    currentDocId = null;
    currentFilename = null;
    isExtracted = false;
    uploadArea.querySelector('.upload-placeholder').style.display = 'block';
    uploadProgress.style.display = 'none';
    pdfContainer.innerHTML = '<div class="pdf-empty"><div class="empty-icon">📂</div><p>No PDF uploaded yet</p></div>';
    updateChatInputState();
}

function loadPDFPreview(file) {
    if (!pdfContainer) return;

    const ensureViewer = () => {
        let pdfViewer = document.getElementById('pdf-viewer');
        if (!pdfViewer) {
            pdfContainer.innerHTML = `
                <div class="pdf-viewer">
                    <iframe id="pdf-viewer" style="display:block;width:100%;height:100%;border:none;"></iframe>
                </div>
            `;
            pdfViewer = document.getElementById('pdf-viewer');
        }
        const placeholder = pdfContainer.querySelector('.pdf-placeholder, .pdf-empty');
        if (placeholder) placeholder.remove();
        return pdfViewer;
    };

    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const pdfData = e.target.result;
            const pdfViewer = ensureViewer();
            if (pdfViewer) {
                pdfViewer.src = pdfData;
            }
        };
        reader.readAsDataURL(file);
    } else if (currentDocId) {
        const pdfViewer = ensureViewer();
        if (pdfViewer) {
            pdfViewer.src = `/api/pdf/${currentDocId}`;
        }
    }
}

// Extract Handler
function handleExtract() {
    if (!currentDocId) {
        showToast('error', 'No Document', 'Please upload a PDF first');
        return;
    }
    
    const chunkSize = parseInt(chunkSizeSlider.value);
    const overlap = parseInt(chunkOverlapSlider.value);
    
    showLoading('Extracting and indexing PDF...');
    
    fetch('/api/extract', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            doc_id: currentDocId,
            chunk_size: chunkSize,
            overlap: overlap
        })
    })
    .then(async response => {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return response.json();
        } else {
            const text = await response.text();
            throw new Error(text || `Server returned status ${response.status}`);
        }
    })
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showToast('error', 'Extraction Failed', data.error);
            return;
        }
        
        isExtracted = true;
        showToast('success', 'Indexing Complete', `Indexed ${data.chunks_indexed} chunks`);
        updateChatInputState();
    })
    .catch(error => {
        hideLoading();
        showToast('error', 'Extraction Error', error.message);
    });
}

// Chat Handlers
function handleSendMessage() {
    const question = chatInput.value.trim();
    
    if (!question) return;
    
    if (!currentDocId) {
        showToast('error', 'No Document', 'Please upload a PDF first');
        return;
    }
    
    // Get doc_id from session if not set in JS
    const docIdToUse = currentDocId || (document.querySelector('[data-doc-id]')?.dataset.docId);
    
    if (!docIdToUse) {
        showToast('error', 'No Document', 'Please upload a PDF first');
        return;
    }
    
    // Check extraction status from server-side data
    const extractedFromServer = document.querySelector('[data-extracted]')?.dataset.extracted === 'true';
    if (!isExtracted && !extractedFromServer) {
        showToast('error', 'Not Indexed', 'Please extract and index the PDF first');
        return;
    }
    
    // Clear input immediately for better UX
    chatInput.value = '';
    
    // Add user message to UI
    addMessage('user', question);
    
    // Disable input while processing
    chatInput.disabled = true;
    sendBtn.disabled = true;
    
    // Query backend
    const topK = parseInt(document.getElementById('top-k').value) || 5;
    
    fetch('/api/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            doc_id: docIdToUse,
            question: question,
            top_k: topK
        })
    })
    .then(async response => {
        // Log response for debugging
        console.log('Query response status:', response.status);
        console.log('Query response headers:', response.headers.get('content-type'));
        
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            // Log the response data
            console.log('Query response data:', data);
            return { data, status: response.status };
        } else {
            const text = await response.text();
            console.error('Non-JSON response:', text);
            throw new Error(text || `Server returned status ${response.status}`);
        }
    })
    .then(({ data, status }) => {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        
        if (status !== 200 || data.error) {
            if (data.error && data.error.toLowerCase().includes('re-upload')) {
                resetClientSessionState();
            }
            showToast('error', 'Query Failed', data.error || 'Unknown error occurred');
            return;
        }
        
        // Update currentDocId if returned
        if (data.doc_id) {
            currentDocId = data.doc_id;
        }
        
        // Add assistant message
        addMessage('assistant', data.answer, data.references);
        
        // Update metrics
        if (data.metrics) {
            updateMetrics(data.metrics);
        }
    })
    .catch(error => {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        console.error('Query error:', error);
        
        // Remove user message if query failed
        const messages = chatContainer.querySelectorAll('.message');
        if (messages.length > 0 && messages[messages.length - 1].classList.contains('message-user')) {
            messages[messages.length - 1].remove();
        }
        
        const errorMsg = error.message || 'Failed to query PDF. Please check the console for details.';
        showToast('error', 'Query Error', errorMsg);
    });
}

function addMessage(role, content, references = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = `<span class="message-role">${role === 'user' ? 'You' : 'AI Assistant'}</span>`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = content;
    
    messageDiv.appendChild(header);
    messageDiv.appendChild(messageContent);
    
    if (references && references.length > 0) {
        const refsDiv = document.createElement('div');
        refsDiv.className = 'message-references';
        refsDiv.innerHTML = '<strong>📎 References:</strong> ' + 
            references.map(ref => `<span class="reference-tag">Page ${ref.page}</span>`).join(' ');
        messageDiv.appendChild(refsDiv);
    }
    
    // Remove empty state if present
    const emptyState = chatContainer.querySelector('.chat-empty');
    if (emptyState) {
        emptyState.remove();
    }
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    messages.push({ role, content, references });
}

function updateChatInputState() {
    const isReady = currentDocId && isExtracted;
    chatInput.disabled = !isReady;
    sendBtn.disabled = !isReady;
    
    const hintEl = document.getElementById('chat-input-hint');
    if (hintEl) {
        hintEl.style.display = isReady ? 'none' : 'block';
    }
}

// Session Management
function clearChat() {
    fetch('/api/clear', {
        method: 'POST'
    })
    .then(() => {
        messages = [];
        chatContainer.innerHTML = '<div class="chat-empty"><div class="empty-icon">👋</div><p>Upload a PDF, extract it, and start asking questions!</p></div>';
        updateMetrics({
            latency_ms: 0,
            retrieved: 0,
            retrieval_accuracy: 0,
            relevance_score: 0
        });
    })
    .catch(error => {
        showToast('error', 'Error', 'Failed to clear chat');
    });
}

function newSession() {
    showLoading('Saving session...');
    
    fetch('/api/new-session', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showToast('error', 'Error', data.error);
            return;
        }
        
        // Reset current session
        currentDocId = null;
        currentFilename = null;
        isExtracted = false;
        messages = [];
        
        // Clear UI
        removeFile();
        clearChat();
        
        // Reload page to refresh history
        window.location.reload();
    })
    .catch(error => {
        hideLoading();
        showToast('error', 'Error', 'Failed to create new session');
    });
}

function loadSession(sessionId) {
    showLoading('Loading session...');
    
    fetch('/api/load-session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ session_id: sessionId })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            showToast('error', 'Error', data.error);
            return;
        }
        
        // Reload page to show loaded session
        window.location.reload();
    })
    .catch(error => {
        hideLoading();
        showToast('error', 'Error', 'Failed to load session');
    });
}

// Metrics
function updateMetrics(metrics) {
    if (!metrics) return;
    
    const latencyEl = document.getElementById('metric-latency');
    const chunksEl = document.getElementById('metric-chunks');
    const accuracyEl = document.getElementById('metric-accuracy');
    const relevanceEl = document.getElementById('metric-relevance');
    
    if (latencyEl) latencyEl.textContent = `${Math.round(metrics.latency_ms || 0)} ms`;
    if (chunksEl) chunksEl.textContent = metrics.retrieved || 0;
    if (accuracyEl) accuracyEl.textContent = (metrics.retrieval_accuracy || 0).toFixed(2);
    if (relevanceEl) relevanceEl.textContent = (metrics.relevance_score || 0).toFixed(2);
}

// Backend Status
function checkBackendStatus() {
    fetch('/api/history')
    .then(response => {
        if (response.ok) {
            const statusCard = document.getElementById('backend-status');
            if (statusCard) {
                const indicator = statusCard.querySelector('.status-indicator');
                if (indicator) {
                    indicator.className = 'status-indicator status-online';
                }
            }
        }
    })
    .catch(() => {
        // Backend might be offline, status will show from server-side rendering
    });
}

// Utility Functions
function showLoading(text = 'Processing...') {
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showToast(type, title, message) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-title">${title}</div>
        <div class="toast-message">${message}</div>
    `;
    
    const container = document.getElementById('toast-container');
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'toastSlideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function resetClientSessionState() {
    currentDocId = null;
    isExtracted = false;
    
    const mainContent = document.querySelector('[data-doc-id]');
    if (mainContent) {
        mainContent.dataset.docId = '';
        mainContent.dataset.extracted = 'false';
    }
    
    if (chatContainer) {
        chatContainer.innerHTML = `
            <div class="chat-empty">
                <div class="empty-icon">📂</div>
                <p>Please re-upload your PDF to continue.</p>
            </div>
        `;
    }
    
    if (pdfContainer) {
        pdfContainer.innerHTML = `
            <div class="pdf-empty">
                <div class="empty-icon">📂</div>
                <p>No PDF uploaded yet</p>
                <small>Upload a file from the sidebar to preview it here</small>
            </div>
        `;
    }
    
    if (extractSection) {
        extractSection.style.display = 'none';
    }
    
    if (fileInfo) {
        fileInfo.style.display = 'none';
    }
    
    const placeholder = uploadArea?.querySelector('.upload-placeholder');
    if (placeholder) {
        placeholder.style.display = 'block';
    }
    
    showToast('info', 'Session reset', 'Backend restarted. Please upload and index your PDF again.');
    updateChatInputState();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

