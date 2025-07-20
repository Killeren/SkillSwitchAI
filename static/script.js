/**
 * Together.ai Multi-Agent Chat Frontend
 * Handles WebSocket communication, UI interactions, and real-time messaging
 */

class ChatApplication {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.currentTheme = 'light';
        this.isProcessing = false;

        // Initialize the application
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadTheme();
        this.connect();
        this.setupAutoResize();
    }

    // WebSocket Connection Management
    connect() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            this.updateConnectionStatus('connecting', 'Connecting...');
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = (event) => this.onWebSocketOpen(event);
            this.ws.onmessage = (event) => this.onWebSocketMessage(event);
            this.ws.onclose = (event) => this.onWebSocketClose(event);
            this.ws.onerror = (event) => this.onWebSocketError(event);

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.updateConnectionStatus('error', 'Connection failed');
            this.scheduleReconnect();
        }
    }

    onWebSocketOpen(event) {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.updateConnectionStatus('connected', 'Connected');
        this.enableInput();
    }

    onWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }

    onWebSocketClose(event) {
        console.log('WebSocket disconnected:', event.code, event.reason);
        this.isConnected = false;
        this.updateConnectionStatus('error', 'Disconnected');
        this.disableInput();

        if (!event.wasClean) {
            this.scheduleReconnect();
        }
    }

    onWebSocketError(event) {
        console.error('WebSocket error:', event);
        this.updateConnectionStatus('error', 'Connection error');
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            this.updateConnectionStatus('error', 'Connection failed');
            this.showError('Connection lost. Please refresh the page to reconnect.');
        }
    }

    // Message Handling
    handleMessage(data) {
        switch (data.type) {
            case 'connection':
                this.sessionId = data.session_id;
                this.addSystemMessage(data.message);
                break;

            case 'processing':
                this.handleProcessingUpdate(data);
                break;

            case 'response':
                this.handleResponse(data);
                break;

            case 'stream_chunk':
                this.handleStreamChunk(data);
                break;

            case 'error':
                this.handleError(data.message);
                break;

            case 'history_cleared':
                this.clearMessages();
                this.addSystemMessage(data.message);
                break;

            case 'pong':
                console.log('Received pong');
                break;

            default:
                console.log('Unknown message type:', data.type);
        }
    }

    handleProcessingUpdate(data) {
        switch (data.status) {
            case 'started':
                this.showProcessing(data.message);
                this.addTypingIndicator();
                break;

            case 'selecting_models':
                this.updateProcessing(data.message);
                break;

            case 'completed':
                this.hideProcessing();
                this.removeTypingIndicator();
                break;

            case 'error':
                this.hideProcessing();
                this.removeTypingIndicator();
                this.showError(data.message);
                break;
        }
    }

    handleResponse(data) {
        this.removeTypingIndicator();
        this.addMessage('assistant', data.content, {
            generator: data.metadata.generator_model,
            critic: data.metadata.critic_model,
            iterations: data.metadata.iterations,
            quality: data.metadata.quality_score,
            time: data.metadata.processing_time
        });

        this.hideProcessing();
        this.enableInput();
        this.isProcessing = false;

        // Update model info
        this.updateModelInfo(data.metadata.generator_model, data.metadata.critic_model);
    }

    handleStreamChunk(data) {
        // Handle streaming response chunks if implemented
        console.log('Stream chunk:', data.chunk);
    }

    handleError(message) {
        this.removeTypingIndicator();
        this.hideProcessing();
        this.enableInput();
        this.isProcessing = false;
        this.showError(message);
    }

    // UI Management
    setupEventListeners() {
        // Send button
        const sendButton = document.getElementById('send-button');
        sendButton.addEventListener('click', () => this.sendMessage());

        // Message input
        const messageInput = document.getElementById('message-input');
        messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        messageInput.addEventListener('input', () => this.handleInput());

        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        themeToggle.addEventListener('click', () => this.toggleTheme());

        // Clear chat
        const clearChat = document.getElementById('clear-chat');
        clearChat.addEventListener('click', () => this.clearChat());

        // Error modal
        const errorModalClose = document.getElementById('error-modal-close');
        const errorModalOk = document.getElementById('error-modal-ok');
        errorModalClose.addEventListener('click', () => this.hideError());
        errorModalOk.addEventListener('click', () => this.hideError());

        // Click outside modal to close
        const errorModal = document.getElementById('error-modal');
        errorModal.addEventListener('click', (e) => {
            if (e.target === errorModal) {
                this.hideError();
            }
        });
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    handleInput() {
        const input = document.getElementById('message-input');
        const charCount = document.querySelector('.char-count');
        const sendButton = document.getElementById('send-button');

        const length = input.value.length;
        charCount.textContent = `${length}/4000`;

        // Enable/disable send button
        const canSend = length > 0 && length <= 4000 && this.isConnected && !this.isProcessing;
        sendButton.disabled = !canSend;

        // Auto-resize textarea
        this.autoResize(input);
    }

    setupAutoResize() {
        const input = document.getElementById('message-input');
        input.style.height = 'auto';
        input.style.height = input.scrollHeight + 'px';
    }

    autoResize(element) {
        element.style.height = 'auto';
        element.style.height = Math.min(element.scrollHeight, 200) + 'px';
    }

    // Message UI
    sendMessage() {
        const input = document.getElementById('message-input');
        const message = input.value.trim();

        if (!message || !this.isConnected || this.isProcessing) {
            return;
        }

        // Add user message to UI
        this.addMessage('user', message);

        // Send to WebSocket
        this.ws.send(JSON.stringify({
            type: 'chat',
            content: message
        }));

        // Clear input and disable
        input.value = '';
        input.style.height = 'auto';
        this.handleInput();
        this.disableInput();
        this.isProcessing = true;
    }

    addMessage(role, content, metadata = null) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        // Create avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'AI';

        // Create content container
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Create message text
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        if (role === 'assistant') {
            // Basic Markdown rendering for images and links
            textDiv.innerHTML = this.renderMarkdown(content);
        } else {
            textDiv.textContent = content;
        }

        contentDiv.appendChild(textDiv);

        // Add metadata for assistant messages
        if (role === 'assistant' && metadata) {
            const metadataDiv = document.createElement('div');
            metadataDiv.className = 'message-metadata';

            const items = [
                { icon: 'fas fa-robot', label: `Generator: ${metadata.generator}` },
                { icon: 'fas fa-search', label: `Critic: ${metadata.critic}` },
                { icon: 'fas fa-sync-alt', label: `Iterations: ${metadata.iterations}` },
                { icon: 'fas fa-star', label: `Quality: ${metadata.quality}/10` }
            ];

            if (metadata.time) {
                items.push({ icon: 'fas fa-clock', label: `Time: ${metadata.time}` });
            }

            items.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'metadata-item';
                itemDiv.innerHTML = `<i class="${item.icon}"></i> ${item.label}`;
                metadataDiv.appendChild(itemDiv);
            });

            contentDiv.appendChild(metadataDiv);
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);

        // Remove welcome message if present
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    // Add a simple Markdown renderer for images and links
    renderMarkdown(text) {
        // Images: ![alt](url)
        text = text.replace(/!\[([^\]]*)\]\(([^\)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%;border-radius:8px;margin:0.5rem 0;" />');
        // Links: [text](url) but not images (skip if preceded by '!')
        text = text.replace(/(^|[^!])\[([^\]]+)\]\(([^\)]+)\)/g, '$1<a href="$3" target="_blank" rel="noopener">$2</a>');
        // Line breaks
        text = text.replace(/\n/g, '<br/>');
        return text;
    }

    addSystemMessage(content) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'system-message';
        messageDiv.style.cssText = `
            text-align: center;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            background-color: var(--border-color);
            border-radius: var(--border-radius);
            font-size: var(--font-size-sm);
            color: var(--text-secondary);
        `;
        messageDiv.textContent = content;

        messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        const existingIndicator = messagesContainer.querySelector('.typing-indicator');
        if (existingIndicator) {
            return; // Already exists
        }

        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.style.background = 'linear-gradient(135deg, var(--primary-color), var(--primary-hover))';
        avatar.style.color = 'white';
        avatar.textContent = 'AI';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'typing-content';

        const dotsDiv = document.createElement('div');
        dotsDiv.className = 'typing-dots';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            dotsDiv.appendChild(dot);
        }

        const textSpan = document.createElement('span');
        textSpan.textContent = 'AI is thinking...';
        textSpan.style.marginLeft = '0.5rem';
        textSpan.style.fontSize = 'var(--font-size-sm)';
        textSpan.style.color = 'var(--text-secondary)';

        contentDiv.appendChild(dotsDiv);
        contentDiv.appendChild(textSpan);

        typingDiv.appendChild(avatar);
        typingDiv.appendChild(contentDiv);

        messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    clearMessages() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = '';

        // Add welcome message back
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <div class="welcome-icon">
                <i class="fas fa-robot"></i>
            </div>
            <h2>Welcome to Together.ai Multi-Agent Chat</h2>
            <p>This intelligent chat system uses multiple specialized AI models to provide high-quality responses through iterative refinement.</p>
            <div class="feature-highlights">
                <div class="feature">
                    <i class="fas fa-brain"></i>
                    <span>Intelligent Model Selection</span>
                </div>
                <div class="feature">
                    <i class="fas fa-sync-alt"></i>
                    <span>Iterative Refinement</span>
                </div>
                <div class="feature">
                    <i class="fas fa-rocket"></i>
                    <span>High-Quality Responses</span>
                </div>
            </div>
        `;
        messagesContainer.appendChild(welcomeDiv);
    }

    // Connection Status
    updateConnectionStatus(status, text) {
        const indicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.status-text');

        indicator.className = `status-indicator ${status}`;
        statusText.textContent = text;
    }

    // Input State Management
    enableInput() {
        const input = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');

        input.disabled = false;
        input.placeholder = 'Ask me anything... (Press Shift+Enter for new line, Enter to send)';
        this.handleInput(); // Update send button state

        // Focus input
        setTimeout(() => input.focus(), 100);
    }

    disableInput() {
        const input = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');

        input.disabled = true;
        input.placeholder = 'Processing...';
        sendButton.disabled = true;
    }

    // Processing Overlay
    showProcessing(message) {
        const overlay = document.getElementById('processing-overlay');
        const text = document.getElementById('processing-text');

        text.textContent = message;
        overlay.classList.add('show');
    }

    updateProcessing(message) {
        const text = document.getElementById('processing-text');
        text.textContent = message;
    }

    hideProcessing() {
        const overlay = document.getElementById('processing-overlay');
        overlay.classList.remove('show');
    }

    // Error Handling
    showError(message) {
        const modal = document.getElementById('error-modal');
        const errorMessage = document.getElementById('error-message');

        errorMessage.textContent = message;
        modal.classList.add('show');
    }

    hideError() {
        const modal = document.getElementById('error-modal');
        modal.classList.remove('show');
    }

    // Theme Management
    loadTheme() {
        const savedTheme = localStorage.getItem('chat-theme') || 'light';
        this.setTheme(savedTheme);
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }

    setTheme(theme) {
        this.currentTheme = theme;
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('chat-theme', theme);

        const themeIcon = document.querySelector('#theme-toggle i');
        themeIcon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }

    // Utility Functions
    clearChat() {
        if (this.isConnected) {
            this.ws.send(JSON.stringify({
                type: 'clear_history'
            }));
        }
    }

    updateModelInfo(generator, critic) {
        const modelInfo = document.getElementById('model-info');
        modelInfo.textContent = `${generator} ↔ ${critic}`;
    }

    // Ping to keep connection alive
    startHeartbeat() {
        setInterval(() => {
            if (this.isConnected) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // Ping every 30 seconds
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApplication();

    // Start heartbeat
    setTimeout(() => {
        window.chatApp.startHeartbeat();
    }, 1000);
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && window.chatApp && !window.chatApp.isConnected) {
        // Try to reconnect when page becomes visible
        window.chatApp.connect();
    }
});

// Handle beforeunload
window.addEventListener('beforeunload', () => {
    if (window.chatApp && window.chatApp.ws) {
        window.chatApp.ws.close();
    }
});