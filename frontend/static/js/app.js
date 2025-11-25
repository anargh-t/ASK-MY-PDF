// Global state
let currentDocId = null;
let currentFilename = null;
let isExtracted = false;
let messages = [];

// DOM Elements
const fileInput = document.getElementById('file-input');
const chatUploadArea = document.getElementById('chat-upload-area');
const uploadArea = document.getElementById('upload-area');
const uploadProgress = document.getElementById('upload-progress');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
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
    
    // Extract button is now in Advanced Settings, no need to show/hide it
    
    initializeEventListeners();
    initializeSettings();
    checkBackendStatus();
    updateChatInputState();
    
    // Only show upload area if template doesn't already have content and no PDF is loaded
    // The template will handle showing upload area vs ready state based on session
    // We only need to initialize if the upload area exists in the template
    const existingUploadArea = chatContainer?.querySelector('#chat-upload-area');
    if (existingUploadArea) {
        // Upload area exists in template, just initialize event listeners
        const newFileInput = existingUploadArea.querySelector('#file-input');
        const newUploadArea = existingUploadArea.querySelector('#upload-area');
        if (newFileInput && newUploadArea) {
            newUploadArea.addEventListener('click', () => newFileInput.click());
            newUploadArea.addEventListener('dragover', handleDragOver);
            newUploadArea.addEventListener('drop', handleDrop);
            newUploadArea.addEventListener('dragleave', handleDragLeave);
            newFileInput.addEventListener('change', handleFileSelect);
        }
    }
});

// Event Listeners
function initializeEventListeners() {
    // File upload
    if (uploadArea && fileInput) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('drop', handleDrop);
        uploadArea.addEventListener('dragleave', handleDragLeave);
    }
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Extract
    if (extractBtn) {
        extractBtn.addEventListener('click', handleExtract);
    }

    
    // Chat
    if (sendBtn) {
        sendBtn.addEventListener('click', handleSendMessage);
    }
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
    }
    
    // Session management
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', clearChat);
    }
    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', newSession);
    }
    
    // History items - click on info area to load session
    document.querySelectorAll('.history-info').forEach(info => {
        const item = info.closest('.history-item');
        if (item) {
            info.addEventListener('click', () => loadSession(item.dataset.sessionId));
        }
    });
    
    // History edit buttons
    document.querySelectorAll('.history-edit-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sessionId = btn.dataset.sessionId;
            renameSession(sessionId);
        });
    });
    
    // History delete buttons
    document.querySelectorAll('.history-delete-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sessionId = btn.dataset.sessionId;
            deleteSession(sessionId);
        });
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
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);
    if (fileInfo) fileInfo.style.display = 'block';
    if (uploadArea) {
        const placeholder = uploadArea.querySelector('.upload-placeholder');
        if (placeholder) placeholder.style.display = 'none';
    }
    if (uploadProgress) uploadProgress.style.display = 'block';
    
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
        
        // Update UI
        if (uploadProgress) uploadProgress.style.display = 'none';
        
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
        
        // Automatically extract and index the PDF
        autoExtract(data.doc_id);
    })
    .catch(error => {
        hideLoading();
        showToast('error', 'Upload Error', error.message);
        removeFile();
    });
}

function removeFile() {
    fileInput.value = '';
    if (fileInfo) fileInfo.style.display = 'none';
    currentDocId = null;
    currentFilename = null;
    isExtracted = false;
    
    if (uploadArea) {
        const placeholder = uploadArea.querySelector('.upload-placeholder');
        if (placeholder) placeholder.style.display = 'flex';
        if (uploadProgress) uploadProgress.style.display = 'none';
    }
    
    // Show upload area again
    if (chatUploadArea) {
        chatUploadArea.style.display = 'flex';
    }
    
    // Clear chat messages
    if (chatContainer) {
        chatContainer.innerHTML = '';
        const uploadAreaClone = chatUploadArea ? chatUploadArea.cloneNode(true) : null;
        if (uploadAreaClone) {
            chatContainer.appendChild(uploadAreaClone);
            // Re-initialize event listeners for the new upload area
            const newUploadArea = chatContainer.querySelector('#upload-area');
            const newFileInput = chatContainer.querySelector('#file-input');
            if (newUploadArea && newFileInput) {
                newUploadArea.addEventListener('click', () => newFileInput.click());
                newUploadArea.addEventListener('dragover', handleDragOver);
                newUploadArea.addEventListener('drop', handleDrop);
                newUploadArea.addEventListener('dragleave', handleDragLeave);
                newFileInput.addEventListener('change', handleFileSelect);
            }
        }
    }
    
    pdfContainer.innerHTML = '<div class="pdf-empty"><div class="empty-icon">📂</div><p>No PDF uploaded yet</p></div>';
    updateChatInputState();
}

// Auto-extract function called after upload
function autoExtract(docId) {
    if (!docId) return;
    
    const chunkSize = parseInt(chunkSizeSlider.value) || 500;
    const overlap = parseInt(chunkOverlapSlider.value) || 100;
    
    showLoading('Extracting and indexing PDF...');
    
    fetch('/api/extract', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            doc_id: docId,
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
        showToast('success', 'Ready to Chat!', `Indexed ${data.chunks_indexed} chunks. You can now ask questions!`);
        
        // Hide upload area and show empty chat state
        if (chatUploadArea) {
            chatUploadArea.style.display = 'none';
        }
        
        // Show empty chat message
        if (chatContainer && chatContainer.querySelector('.chat-upload-area')) {
            chatContainer.innerHTML = '<div class="chat-empty"><div class="empty-icon">👋</div><p>Start asking questions about your PDF!</p></div>';
        }
        
        updateChatInputState();
    })
    .catch(error => {
        hideLoading();
        showToast('error', 'Extraction Error', error.message);
    });
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

// Extract Handler (manual extraction from Advanced Settings)
function handleExtract() {
    if (!currentDocId) {
        showToast('error', 'No Document', 'Please upload a PDF first');
        return;
    }
    
    const chunkSize = parseInt(chunkSizeSlider.value) || 500;
    const overlap = parseInt(chunkOverlapSlider.value) || 100;
    
    showLoading('Re-indexing PDF with new settings...');
    
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
        showToast('success', 'Re-indexing Complete', `Re-indexed ${data.chunks_indexed} chunks with new settings`);
        updateChatInputState();
    })
    .catch(error => {
        hideLoading();
        showToast('error', 'Extraction Error', error.message);
    });
}

// Chat Handlers
function handleSendMessage() {
    if (!chatInput) {
        console.error('Chat input not found');
        return;
    }
    
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
    if (chatInput) {
        chatInput.value = '';
    }
    
    // Add user message to UI
    addMessage('user', question);
    
    // Show thinking indicator
    showThinkingIndicator();
    
    // Disable input while processing
    if (chatInput) {
        chatInput.disabled = true;
    }
    if (sendBtn) {
        sendBtn.disabled = true;
    }
    
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
        // Remove thinking indicator
        removeThinkingIndicator();
        
        if (chatInput) {
            chatInput.disabled = false;
        }
        if (sendBtn) {
            sendBtn.disabled = false;
        }
        
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
        // Remove thinking indicator
        removeThinkingIndicator();
        
        if (chatInput) {
            chatInput.disabled = false;
        }
        if (sendBtn) {
            sendBtn.disabled = false;
        }
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
    
    // Remove thinking indicator if present
    removeThinkingIndicator();
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    messages.push({ role, content, references });
}

function showThinkingIndicator() {
    // Remove any existing thinking indicator
    removeThinkingIndicator();
    
    if (!chatContainer) return;
    
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'message message-assistant message-thinking';
    thinkingDiv.id = 'thinking-indicator';
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = '<span class="message-role">AI Assistant</span>';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = `
        <div class="thinking-content">
            <span class="thinking-dots">
                <span></span>
                <span></span>
                <span></span>
            </span>
            <span class="thinking-text">Thinking...</span>
        </div>
    `;
    
    thinkingDiv.appendChild(header);
    thinkingDiv.appendChild(messageContent);
    
    chatContainer.appendChild(thinkingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeThinkingIndicator() {
    const thinkingIndicator = document.getElementById('thinking-indicator');
    if (thinkingIndicator) {
        thinkingIndicator.remove();
    }
}

function updateChatInputState() {
    const isReady = currentDocId && isExtracted;
    if (chatInput) {
        chatInput.disabled = !isReady;
    }
    if (sendBtn) {
        sendBtn.disabled = !isReady;
    }
    
    const hintEl = document.getElementById('chat-input-hint');
    if (hintEl) {
        hintEl.style.display = isReady ? 'none' : 'block';
    }
}

// Session Management
function clearChat() {
    fetch('/api/clear', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
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
        if (data.error) {
            showToast('error', 'Error', data.error);
            return;
        }
        
        messages = [];
        // Only clear messages, keep upload area if PDF is still loaded
        if (currentDocId && isExtracted) {
            if (chatContainer) {
                chatContainer.innerHTML = '<div class="chat-empty"><div class="empty-icon">👋</div><p>Start asking questions about your PDF!</p></div>';
            }
        } else {
            // Show upload area if no PDF is loaded
            if (chatContainer) {
                chatContainer.innerHTML = '';
                const uploadAreaClone = chatUploadArea ? chatUploadArea.cloneNode(true) : null;
                if (uploadAreaClone) {
                    chatContainer.appendChild(uploadAreaClone);
                    // Re-initialize event listeners
                    const newUploadArea = chatContainer.querySelector('#upload-area');
                    const newFileInput = chatContainer.querySelector('#file-input');
                    if (newUploadArea && newFileInput) {
                        newUploadArea.addEventListener('click', () => newFileInput.click());
                        newUploadArea.addEventListener('dragover', handleDragOver);
                        newUploadArea.addEventListener('drop', handleDrop);
                        newUploadArea.addEventListener('dragleave', handleDragLeave);
                        newFileInput.addEventListener('change', handleFileSelect);
                    }
                } else {
                    chatContainer.innerHTML = '<div class="chat-empty"><div class="empty-icon">📂</div><p>Upload a PDF to get started</p></div>';
                }
            }
        }
        updateMetrics({
            latency_ms: 0,
            retrieved: 0,
            retrieval_accuracy: 0,
            relevance_score: 0
        });
    })
    .catch(error => {
        console.error('Clear chat error:', error);
        const errorMsg = error.message || 'Failed to clear chat';
        showToast('error', 'Error', errorMsg);
    });
}

function newSession() {
    showLoading('Saving session...');
    
    fetch('/api/new-session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
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
            showToast('error', 'Error', data.error);
            return;
        }
        
        // Reset current session state
        currentDocId = null;
        currentFilename = null;
        isExtracted = false;
        messages = [];
        
        // Clear UI elements directly without making additional API calls
        if (chatContainer) {
            chatContainer.innerHTML = '';
            // Show upload area
            if (chatUploadArea) {
                const uploadAreaClone = chatUploadArea.cloneNode(true);
                chatContainer.appendChild(uploadAreaClone);
                // Re-initialize event listeners for the new upload area
                const newUploadArea = chatContainer.querySelector('#upload-area');
                const newFileInput = chatContainer.querySelector('#file-input');
                if (newUploadArea && newFileInput) {
                    newUploadArea.addEventListener('click', () => newFileInput.click());
                    newUploadArea.addEventListener('dragover', handleDragOver);
                    newUploadArea.addEventListener('drop', handleDrop);
                    newUploadArea.addEventListener('dragleave', handleDragLeave);
                    newFileInput.addEventListener('change', handleFileSelect);
                }
            } else {
                chatContainer.innerHTML = '<div class="chat-empty"><div class="empty-icon">📂</div><p>Upload a PDF to get started</p></div>';
            }
        }
        
        // Clear PDF preview
        if (pdfContainer) {
            pdfContainer.innerHTML = '<div class="pdf-empty"><div class="empty-icon">📂</div><p>No PDF uploaded yet</p></div>';
        }
        
        // Reset file input
        if (fileInput) {
            fileInput.value = '';
        }
        if (fileInfo) {
            fileInfo.style.display = 'none';
        }
        
        // Show upload area
        if (chatUploadArea) {
            chatUploadArea.style.display = 'flex';
        }
        
        // Reset metrics
        updateMetrics({
            latency_ms: 0,
            retrieved: 0,
            retrieval_accuracy: 0,
            relevance_score: 0
        });
        
        // Update chat input state
        updateChatInputState();
        
        // Reload page to refresh history in sidebar
        window.location.reload();
    })
    .catch(error => {
        hideLoading();
        console.error('New session error:', error);
        const errorMsg = error.message || 'Failed to create new session';
        showToast('error', 'Error', errorMsg);
    });
}

function loadSession(sessionId) {
    if (!sessionId) {
        showToast('error', 'Error', 'Invalid session ID');
        return;
    }
    
    showLoading('Loading session...');
    
    fetch('/api/load-session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ session_id: sessionId })
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
            showToast('error', 'Error', data.error);
            return;
        }
        
        // Reload page to show loaded session
        window.location.reload();
    })
    .catch(error => {
        hideLoading();
        console.error('Load session error:', error);
        const errorMsg = error.message || 'Failed to load session';
        showToast('error', 'Error', errorMsg);
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
    
    // Extract button is in Advanced Settings, no need to hide it
    
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

// Session Management - Rename and Delete
function renameSession(sessionId) {
    if (!sessionId) {
        showToast('error', 'Error', 'Invalid session ID');
        return;
    }
    
    // Find the session name
    const historyItem = document.querySelector(`.history-item[data-session-id="${sessionId}"]`);
    if (!historyItem) {
        showToast('error', 'Error', 'Session not found');
        return;
    }
    
    const currentName = historyItem.querySelector('.history-name').textContent;
    
    // Prompt for new name
    const newName = prompt('Enter new session name:', currentName);
    if (!newName || newName.trim() === '') {
        return;
    }
    
    if (newName.trim() === currentName) {
        return; // No change
    }
    
    fetch('/api/rename-session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: sessionId,
            name: newName.trim()
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
        if (data.error) {
            showToast('error', 'Error', data.error);
            return;
        }
        
        showToast('success', 'Success', 'Session renamed successfully');
        
        // Update the UI
        if (historyItem) {
            historyItem.querySelector('.history-name').textContent = newName.trim();
        }
        
        // Reload page to refresh history
        window.location.reload();
    })
    .catch(error => {
        console.error('Rename session error:', error);
        const errorMsg = error.message || 'Failed to rename session';
        showToast('error', 'Error', errorMsg);
    });
}

function deleteSession(sessionId) {
    if (!sessionId) {
        showToast('error', 'Error', 'Invalid session ID');
        return;
    }
    
    // Confirm deletion
    if (!confirm('Are you sure you want to delete this session? This action cannot be undone.')) {
        return;
    }
    
    fetch('/api/delete-session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: sessionId
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
        if (data.error) {
            showToast('error', 'Error', data.error);
            return;
        }
        
        showToast('success', 'Success', 'Session deleted successfully');
        
        // Remove the item from UI
        const historyItem = document.querySelector(`.history-item[data-session-id="${sessionId}"]`);
        if (historyItem) {
            historyItem.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                historyItem.remove();
                
                // Check if history is empty
                const historyList = document.getElementById('history-list');
                if (historyList && historyList.querySelectorAll('.history-item').length === 0) {
                    historyList.innerHTML = '<p class="history-empty">No saved sessions yet</p>';
                }
            }, 300);
        } else {
            // Reload page to refresh history
            window.location.reload();
        }
    })
    .catch(error => {
        console.error('Delete session error:', error);
        const errorMsg = error.message || 'Failed to delete session';
        showToast('error', 'Error', errorMsg);
    });
}

