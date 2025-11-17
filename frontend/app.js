// DOM Elements
const uploadForm = document.querySelector("#upload-form");
const uploadStatus = document.querySelector("#upload-status");
const queryForm = document.querySelector("#query-form");
const chatMessages = document.querySelector("#chat-messages");
const questionInput = document.querySelector("#question");
const sendBtn = document.querySelector("#send-btn");
const uploadBtn = document.querySelector("#upload-btn");
const fileInput = document.querySelector("#pdf-file");
const fileUploadArea = document.querySelector("#file-upload-area");
const filePreview = document.querySelector("#file-preview");
const fileRemove = document.querySelector("#file-remove");
const fileName = document.querySelector("#file-name");
const fileSize = document.querySelector("#file-size");
const connectionStatus = document.querySelector("#connection-status");

const API_BASE = window.location.origin;
const DEFAULT_CHUNK_SIZE = 1000;
const DEFAULT_OVERLAP = 200;
const DEFAULT_TOP_K = 5;

// State
let chatHistory = [];
let isProcessing = false;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  initializeApp();
  checkConnection();
  setupEventListeners();
});

// Initialize App
function initializeApp() {
  // Auto-resize textarea
  if (questionInput) {
    questionInput.addEventListener("input", autoResizeTextarea);
    questionInput.addEventListener("keydown", handleTextareaKeydown);
  }

  // File upload drag and drop
  setupFileUpload();
}

// Check Server Connection
async function checkConnection() {
  try {
    const response = await fetch(`${API_BASE}/`);
    const data = await response.json();
    
    if (data.status === "healthy") {
      updateConnectionStatus(true, "Connected");
      
      // Check component status
      const components = data.components || {};
      const llmStatus = components.llm === "connected" ? "✅" : "⚠️";
      const dbStatus = components.vector_db === "connected" ? "✅" : "⚠️";
      
      connectionStatus.querySelector(".status-text").textContent = 
        `${llmStatus} LLM ${dbStatus} Vector DB`;
    } else {
      updateConnectionStatus(false, "Disconnected");
    }
  } catch (error) {
    updateConnectionStatus(false, "Connection failed");
    console.error("Connection check failed:", error);
  }
}

function updateConnectionStatus(connected, text) {
  connectionStatus.classList.remove("connected", "disconnected");
  connectionStatus.classList.add(connected ? "connected" : "disconnected");
  if (text) {
    connectionStatus.querySelector(".status-text").textContent = text;
  }
}

// Setup Event Listeners
function setupEventListeners() {
  // Upload form
  if (uploadForm) {
    uploadForm.addEventListener("submit", handleUpload);
  }

  // Query form
  if (queryForm) {
    queryForm.addEventListener("submit", handleQuery);
  }

  // File removal
  if (fileRemove) {
    fileRemove.addEventListener("click", clearFileSelection);
  }
}

// File Upload Setup
function setupFileUpload() {
  if (!fileUploadArea || !fileInput) return;

  // Click to upload
  fileUploadArea.addEventListener("click", (e) => {
    if (e.target.closest(".file-preview")) return;
    fileInput.click();
  });

  // Drag and drop
  fileUploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    fileUploadArea.classList.add("drag-over");
  });

  fileUploadArea.addEventListener("dragleave", () => {
    fileUploadArea.classList.remove("drag-over");
  });

  fileUploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove("drag-over");
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === "application/pdf") {
      fileInput.files = files;
      handleFileSelect();
    }
  });

  // File input change
  fileInput.addEventListener("change", handleFileSelect);
}

function handleFileSelect() {
  const file = fileInput.files[0];
  if (!file) return;

  if (file.type !== "application/pdf") {
    showStatus(uploadStatus, "Please select a PDF file", "error");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showStatus(uploadStatus, "File size must be less than 10MB", "error");
    return;
  }

  // Show file preview
  fileName.textContent = file.name;
  fileSize.textContent = formatFileSize(file.size);
  filePreview.classList.remove("hidden");
  fileUploadArea.querySelector(".upload-placeholder").classList.add("hidden");
}

function clearFileSelection() {
  fileInput.value = "";
  filePreview.classList.add("hidden");
  fileUploadArea.querySelector(".upload-placeholder").classList.remove("hidden");
  uploadStatus.textContent = "";
  uploadStatus.className = "status-message";
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// Upload Handler
async function handleUpload(event) {
  event.preventDefault();

  if (!fileInput?.files?.length) {
    showStatus(uploadStatus, "Please select a PDF file first", "error");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);
  formData.append("chunk_size", DEFAULT_CHUNK_SIZE);
  formData.append("overlap", DEFAULT_OVERLAP);

  setButtonLoading(uploadBtn, true);
  showStatus(uploadStatus, "Processing PDF... This may take a moment", "info");

  try {
    const response = await fetch(`${API_BASE}/process_pdf`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Failed to process PDF");
    }

    showStatus(
      uploadStatus,
      `✅ Successfully processed! Created ${result.chunks_created} chunks`,
      "success"
    );

    // Clear welcome message and show success in chat
    clearWelcomeMessage();
    addSystemMessage(`PDF "${file.name}" has been processed and is ready for questions!`);

    // Clear file selection after successful upload
    setTimeout(() => {
      clearFileSelection();
    }, 2000);
  } catch (error) {
    console.error("Upload error:", error);
    showStatus(uploadStatus, `❌ ${error.message}`, "error");
  } finally {
    setButtonLoading(uploadBtn, false);
  }
}

// Query Handler
async function handleQuery(event) {
  event.preventDefault();

  if (isProcessing) return;

  const question = questionInput?.value.trim();
  const topK = DEFAULT_TOP_K;

  if (!question) {
    return;
  }

  isProcessing = true;
  setButtonLoading(sendBtn, true);
  questionInput.disabled = true;

  // Add user message to chat
  addMessage("user", question);

  // Clear input
  questionInput.value = "";
  autoResizeTextarea();
  updateSendButtonState();

  // Add loading message
  const loadingId = addLoadingMessage();

  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, top_k: topK }),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Query failed");
    }

    // Remove loading message
    removeLoadingMessage(loadingId);

    // Add assistant message
    addMessage("assistant", result.answer, result.relevant_chunks || []);

    // Save to history
    chatHistory.push({
      question,
      answer: result.answer,
      chunks: result.relevant_chunks || [],
      timestamp: new Date(),
    });
  } catch (error) {
    console.error("Query error:", error);
    
    // Remove loading message
    removeLoadingMessage(loadingId);

    let errorMessage = error.message;
    if (error.message === "Failed to fetch") {
      errorMessage = "Failed to connect to server. Please check your connection.";
    }

    addMessage("assistant", `❌ Error: ${errorMessage}`, [], true);
  } finally {
    isProcessing = false;
    setButtonLoading(sendBtn, false);
    questionInput.disabled = false;
    questionInput.focus();
  }
}

// Chat Functions
function clearWelcomeMessage() {
  const welcome = chatMessages.querySelector(".welcome-message");
  if (welcome) {
    welcome.remove();
  }
}

function addSystemMessage(text) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "message system";
  messageDiv.innerHTML = `
    <div class="message-content" style="width: 100%; text-align: center;">
      <div class="message-bubble" style="background: rgba(99, 102, 241, 0.1); color: var(--primary); border: 1px solid rgba(99, 102, 241, 0.2);">
        <div class="message-text">${escapeHtml(text)}</div>
      </div>
    </div>
  `;
  chatMessages.appendChild(messageDiv);
  scrollToBottom();
}

function addMessage(role, text, chunks = [], isError = false) {
  clearWelcomeMessage();

  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${role}`;

  const avatar = role === "user" ? "👤" : "🤖";
  const timestamp = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  let chunksHtml = "";
  if (chunks.length > 0 && !isError) {
    chunksHtml = `
      <div class="chunks-section">
        <div class="chunks-title">📚 Relevant Sources (${chunks.length}):</div>
        ${chunks
          .map(
            (chunk, idx) => `
          <div class="chunk-item">
            <strong>Source ${idx + 1}:</strong> ${escapeHtml(chunk.substring(0, 200))}${chunk.length > 200 ? "..." : ""}
          </div>
        `
          )
          .join("")}
      </div>
    `;
  }

  messageDiv.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-bubble ${isError ? "error" : ""}">
        <div class="message-text">${escapeHtml(text)}</div>
        ${chunksHtml}
      </div>
      <div class="message-time">${timestamp}</div>
    </div>
  `;

  chatMessages.appendChild(messageDiv);
  scrollToBottom();
}

function addLoadingMessage() {
  clearWelcomeMessage();

  const loadingId = `loading-${Date.now()}`;
  const messageDiv = document.createElement("div");
  messageDiv.id = loadingId;
  messageDiv.className = "message assistant";

  messageDiv.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-content">
      <div class="message-bubble">
        <div class="loading-message">
          <span>Thinking</span>
          <div class="loading-dots">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
          </div>
        </div>
      </div>
    </div>
  `;

  chatMessages.appendChild(messageDiv);
  scrollToBottom();
  return loadingId;
}

function removeLoadingMessage(loadingId) {
  const loadingElement = document.getElementById(loadingId);
  if (loadingElement) {
    loadingElement.remove();
  }
}

function scrollToBottom() {
  chatMessages.scrollTo({
    top: chatMessages.scrollHeight,
    behavior: "smooth",
  });
}

// Utility Functions
function autoResizeTextarea() {
  if (!questionInput) return;
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 150)}px`;
  updateSendButtonState();
}

function handleTextareaKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!sendBtn.disabled && !isProcessing) {
      queryForm.dispatchEvent(new Event("submit"));
    }
  }
}

function updateSendButtonState() {
  if (!sendBtn || !questionInput) return;
  const hasQuestion = questionInput.value.trim().length > 0;
  sendBtn.disabled = !hasQuestion || isProcessing;
}

function setButtonLoading(button, loading) {
  if (!button) return;
  const text = button.querySelector(".btn-text, .send-icon");
  const loader = button.querySelector(".btn-loader, .send-loader");

  if (loading) {
    if (text) text.classList.add("hidden");
    if (loader) loader.classList.remove("hidden");
    button.disabled = true;
  } else {
    if (text) text.classList.remove("hidden");
    if (loader) loader.classList.add("hidden");
    button.disabled = false;
    updateSendButtonState();
  }
}

function showStatus(element, message, type = "") {
  if (!element) return;
  element.textContent = message;
  element.classList.remove("success", "error", "info");
  if (type) {
    element.classList.add(type);
  }
  element.style.display = "block";
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Update send button state on input
if (questionInput) {
  questionInput.addEventListener("input", () => {
    autoResizeTextarea();
    updateSendButtonState();
  });
}

// Initial button state
updateSendButtonState();
