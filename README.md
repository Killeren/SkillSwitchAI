# SkillSwitchAI
Context based LLM selection for agentic workflow to provide more refined outputs

---

# Together.ai Multi-Agent Chat System

An intelligent chat application that leverages Together.ai's free models through a sophisticated multi-agent architecture. The system uses Model Context Protocol (MCP) for smart model selection, LangGraph for iterative refinement workflows, and provides a modern ChatGPT-like interface.

![System Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-blue)
![Models](https://img.shields.io/badge/Models-8%20Free%20Together.ai-green)
![Framework](https://img.shields.io/badge/Backend-FastAPI-red)
![Frontend](https://img.shields.io/badge/Frontend-WebSocket%20Chat-yellow)

## 🚀 Features

### **Intelligent Multi-Agent Architecture**
- **Base Agent**: Uses LLM-based selection to choose optimal generator/critic model pairs
- **Generator Agent**: Creates responses using specialized Together.ai models
- **Critic Agent**: Reviews and suggests improvements through iterative refinement
- **Model Context Protocol (MCP)**: Provides model metadata for LLM-based selection

### **8 Specialized Together.ai Free Models**
- **Meta-Llama-3.3-70B-Instruct-Turbo**: General chat, multilingual, reasoning
- **DeepSeek-R1-Distill-70B/14B/1.5B**: Chain-of-thought reasoning, problem-solving
- **DeepSeek-Coder-V2-Lite**: Code generation, programming, debugging
- **Llama-3.2-11B-Vision**: Image analysis, visual reasoning
- **Mistral-7B-Instruct**: Creative writing, balanced tasks
- **FLUX.1-schnell**: Text-to-image generation (3 months free)

### **Modern Web Interface**
- **Real-time WebSocket Communication**: Instant message delivery
- **ChatGPT-inspired Design**: Professional, responsive interface
- **Dark/Light Theme**: Automatic theme switching with persistence
- **Model Transparency**: Shows selected models, iterations, and quality scores
- **Mobile Responsive**: Works seamlessly across all device sizes

### **Production-Ready Backend**
- **FastAPI WebSocket Server**: High-performance async communication
- **Session Management**: Persistent conversation history
- **Connection Resilience**: Automatic reconnection with fallback handling
- **Health Monitoring**: Real-time connection status and session tracking

## 📋 System Requirements

- **Python 3.8+**
- **Together.ai Free Account** (Sign up at [together.ai](https://api.together.xyz))
- **4GB RAM minimum** (for running multiple models)
- **Modern web browser** with WebSocket support

## 🛠️ Installation & Setup

### 1. **Clone/Extract Project**
```bash
# If you have the zip file, extract it
unzip together-ai-multi-agent-chat.zip
cd together-ai-multi-agent-chat

# Or if cloning from repository
git clone <repository-url>
cd together-ai-multi-agent-chat
```

### 2. **Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your Together.ai API key
# Get your free API key from: https://api.together.xyz/settings/api-keys
```

Edit `.env` file:
```bash
TOGETHER_API_KEY=your_actual_api_key_here
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### 5. **Run the Application**

**Option A: Single Command (Recommended)**
```bash
python main.py
```

**Option B: Step-by-Step**
```bash
# 1. Start MCP server (in terminal 1)
python mcp_server.py

# 2. Start FastAPI backend (in terminal 2)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. **Access the Application**
Open your web browser and navigate to:
```
http://localhost:8000
```

## 🎯 Usage

### **Starting a Conversation**
1. Open the web interface in your browser
2. Wait for the "Connected" status indicator
3. Type your message in the input box
4. Press **Enter** to send (or **Shift+Enter** for new line)

### **Understanding the Process**
1. **LLM-Based Model Selection**: Base agent uses LLM to analyze query and select optimal models from all available options
2. **Generation**: Selected generator model creates initial response
3. **Criticism**: Critic model reviews and suggests improvements
4. **Refinement**: Iterative improvement until quality threshold is met
5. **Delivery**: Final response with metadata (models used, iterations, quality score, selection reasoning)

### **Example Queries**
```
Code Generation:
"Write a Python function to calculate factorial recursively"
→ Uses: DeepSeek-Coder + DeepSeek-R1 for logic validation

Mathematical Reasoning:
"Solve: If 2x + 5 = 15, what is x? Show your work."
→ Uses: DeepSeek-R1-70B + Llama-3.3-70B for verification

Creative Writing:
"Write a short story about time travel in 200 words"
→ Uses: Llama-3.3-70B + Mistral-7B for style critique

General Questions:
"Explain quantum computing in simple terms"
→ Uses: Optimized model pairing based on content
```

### **Interface Features**
- **Theme Toggle**: Click moon/sun icon to switch dark/light theme
- **Clear Chat**: Click trash icon to clear conversation history
- **Connection Status**: Green dot = connected, red dot = disconnected
- **Model Info**: Shows which models were selected for each response
- **Quality Metrics**: Displays iteration count and quality score

## 📁 Project Structure

```
together-ai-multi-agent-chat/
│
├── main.py                 # FastAPI backend server
├── mcp_server.py          # Model Context Protocol server
├── langgraph_workflow.py  # Multi-agent workflow logic
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── README.md            # This file
│
└── static/              # Frontend files
    ├── index.html      # Main HTML interface
    ├── style.css       # CSS styling
    └── script.js       # JavaScript functionality
```

## 🔧 Configuration Options

### **Model Selection Tuning**
The system now uses LLM-based selection for optimal model choice:
- **Dynamic Selection**: LLM analyzes each query and selects best models from all available options
- **Context-Aware**: Considers query type, complexity, and conversation history
- **Fallback System**: Rule-based fallback if LLM selection fails
- **Transparency**: Shows selection reasoning and confidence scores

### **Workflow Parameters**
Edit `langgraph_workflow.py` to adjust:
- Maximum iterations (default: 3)
- Quality score thresholds
- Timeout values

### **Frontend Customization**
Edit `static/style.css` to customize:
- Color schemes and themes
- Layout and spacing
- Animation speeds

## 🚨 Troubleshooting

### **Connection Issues**
```bash
# Check if FastAPI is running
curl http://localhost:8000/health

# Verify WebSocket connection
# Open browser developer tools → Network → WS tab
```

### **API Key Issues**
```bash
# Test your Together.ai API key
curl -H "Authorization: Bearer your_api_key" https://api.together.xyz/models
```

### **Model Selection Problems**
```bash
# Test MCP server independently
python mcp_server.py
```

### **Memory Issues**
- Reduce `max_iterations` in workflow
- Use smaller models (DeepSeek-1.5B instead of 70B)
- Increase system swap space

### **Common Error Messages**
- **"TOGETHER_API_KEY not found"**: Check your `.env` file
- **"Connection failed"**: Verify internet connection and API key
- **"Model not available"**: Try different time or check Together.ai status

## 🔄 API Reference

### **WebSocket Messages**
```javascript
// Send chat message
{
  "type": "chat",
  "content": "Your message here"
}

// Clear chat history
{
  "type": "clear_history"
}

// Ping for connection keepalive
{
  "type": "ping"
}
```

### **HTTP Endpoints**
```bash
# Health check
GET /health

# Simple chat (alternative to WebSocket)
POST /chat
{
  "content": "Your message",
  "session_id": "optional_session_id"
}

# Get session history
GET /sessions/{session_id}/history

# Clear session history
DELETE /sessions/{session_id}/history
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature-name`
3. **Make changes and test thoroughly**
4. **Update documentation** if needed
5. **Submit pull request** with detailed description

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black isort flake8 pytest

# Run code formatting
black .
isort .

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Together.ai** for providing free access to powerful LLMs
- **LangGraph** for the multi-agent workflow framework
- **FastAPI** for the high-performance web framework
- **OpenAI** for inspiration on chat interface design

## 📞 Support

For issues and questions:
1. **Check this README** for common solutions
2. **Review error messages** in browser console
3. **Test API connectivity** with provided curl commands
4. **Open GitHub issue** with detailed error information

---

**Built with ❤️ using Together.ai's free models and modern web technologies**

*Last updated: July 2025*
