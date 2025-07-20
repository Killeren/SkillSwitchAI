# File Overview - Together.ai Multi-Agent Chat System

## 📁 Complete Project Structure

```
together-ai-multi-agent-chat/
│
├── 🚀 SETUP FILES
│   ├── setup.sh              # Linux/Mac setup script
│   ├── setup.bat             # Windows setup script
│   ├── requirements.txt      # Python dependencies
│   ├── .env.example         # Environment template
│   └── README.md           # Complete documentation
│
├── 🤖 BACKEND CORE
│   ├── main.py              # FastAPI server + WebSocket
│   ├── mcp_server.py        # Model Context Protocol server
│   └── langgraph_workflow.py # Multi-agent workflow engine
│
└── 🎨 FRONTEND
    ├── static/index.html    # Chat interface HTML
    ├── static/style.css     # Modern ChatGPT-like styling
    └── static/script.js     # WebSocket client + UI logic
```

## 🔧 Core Components Explained

### **1. mcp_server.py - The Brain** 🧠
- **8 Together.ai Free Models Database**: Complete model context with specialties
- **Intelligent Task Classification**: Analyzes queries to determine task type (code, reasoning, creative, etc.)
- **Smart Model Selection**: Chooses optimal generator/critic pairs based on query analysis
- **Decision Logic**: Rules for pairing models (e.g., DeepSeek-Coder + DeepSeek-R1 for coding tasks)

**Key Features:**
- Task pattern matching with confidence scoring
- Model metadata (capabilities, strengths, use cases)
- Dynamic model pairing algorithms
- Fallback handling for model availability

### **2. langgraph_workflow.py - The Orchestrator** ⚡
- **Three-Agent Architecture**: Base → Generator → Critic workflow
- **Iterative Refinement**: Generator and Critic collaborate until quality threshold met
- **Quality Scoring**: Automatic response evaluation with 1-10 scoring
- **State Management**: Maintains conversation context and workflow state

**Agent Responsibilities:**
- **Base Agent**: Query analysis + model selection via MCP
- **Generator Agent**: Response creation using selected Together.ai model
- **Critic Agent**: Response evaluation + improvement suggestions

### **3. main.py - The Communication Hub** 🌐
- **FastAPI WebSocket Server**: Real-time bidirectional communication
- **Session Management**: Persistent conversation history per user
- **Connection Resilience**: Auto-reconnection + error handling
- **Health Monitoring**: Connection status + performance metrics

**API Endpoints:**
- `WebSocket /ws`: Real-time chat communication
- `POST /chat`: HTTP alternative for simple queries
- `GET /health`: System health check
- `GET/DELETE /sessions/{id}/history`: Session management

### **4. static/index.html - The Interface** 🎨
- **ChatGPT-Inspired Design**: Professional, clean layout
- **Responsive Layout**: Optimal screen utilization with reasonable padding
- **Feature Highlights**: Welcome screen explaining system capabilities
- **Accessibility**: Semantic markup + keyboard navigation

**UI Components:**
- Header with logo, controls, and connection status
- Scrollable message area with typing indicators
- Input area with character count and send button
- Processing overlay with status updates
- Error modal for user feedback

### **5. static/style.css - The Beauty** ✨
- **Modern Design System**: CSS custom properties for theming
- **Dark/Light Themes**: Automatic switching with localStorage persistence
- **Responsive Grid**: Mobile-first design with breakpoints
- **Smooth Animations**: Transitions, typing indicators, slide-ins

**Design Features:**
- Symmetrical layout maximizing screen usage
- Consistent spacing and typography
- Accessible color contrast ratios
- Smooth theme transitions

### **6. static/script.js - The Brain Frontend** ⚡
- **WebSocket Management**: Connection handling + auto-reconnection
- **Real-time Messaging**: Streaming message delivery
- **UI State Management**: Theme, input validation, connection status
- **Error Handling**: Graceful degradation + user feedback

**Key Functionality:**
- Message composition with auto-resize textarea
- Real-time typing indicators
- Connection heartbeat for stability
- Session persistence across page refreshes

## 🎯 How It All Works Together

### **Complete Flow:**
1. **User Opens Browser** → `index.html` loads + `script.js` establishes WebSocket
2. **User Types Message** → Frontend validates + sends via WebSocket to `main.py`
3. **FastAPI Receives** → Creates session + forwards to LangGraph workflow
4. **Base Agent Activates** → Queries `mcp_server.py` for optimal model selection
5. **MCP Analyzes Query** → Classifies task + returns generator/critic pair
6. **Generator Creates Response** → Uses selected Together.ai model via API
7. **Critic Evaluates Response** → Reviews quality + suggests improvements
8. **Iterative Refinement** → Generator/Critic collaborate until threshold met
9. **Final Response Delivered** → WebSocket streams result to frontend
10. **UI Updates** → Message appears with model metadata + quality metrics

### **Intelligence at Every Layer:**
- **Task Classification**: Understands query intent (coding, reasoning, creative)
- **Model Selection**: Matches specialized models to specific tasks
- **Quality Assurance**: Multi-agent review ensures high-quality responses
- **User Experience**: Transparent process with real-time status updates

## 💡 Quick Start Instructions

### **Windows Users:**
```bash
1. Extract together-ai-multi-agent-chat.zip
2. Double-click setup.bat
3. Edit .env file with your Together.ai API key
4. Run: python main.py
5. Open: http://localhost:8000
```

### **Mac/Linux Users:**
```bash
1. Extract together-ai-multi-agent-chat.zip
2. chmod +x setup.sh && ./setup.sh
3. Edit .env file with your Together.ai API key  
4. Run: python main.py
5. Open: http://localhost:8000
```

### **Manual Setup:**
```bash
1. Extract project files
2. python -m venv venv
3. source venv/bin/activate (or venv\Scripts\activate on Windows)
4. pip install -r requirements.txt
5. cp .env.example .env
6. Edit .env with your API key
7. python main.py
```

## 🔧 Customization Options

### **Model Selection Tuning** (`mcp_server.py`):
- Add new task patterns in `_initialize_task_patterns()`
- Modify selection rules in `select_models()`
- Adjust confidence thresholds in `classify_task()`

### **Workflow Parameters** (`langgraph_workflow.py`):
- Change `max_iterations` for more/fewer refinement cycles
- Modify quality thresholds in `_should_continue_refining()`
- Adjust model timeout values

### **UI Customization** (`static/style.css`):
- Update CSS custom properties for color schemes
- Modify layout dimensions and spacing
- Customize animations and transitions

## 🚨 Troubleshooting Quick Fixes

- **"API Key Error"**: Check `.env` file has correct `TOGETHER_API_KEY`
- **"Connection Failed"**: Verify internet connection + Together.ai service status
- **"Models Not Available"**: Try different models or check Together.ai rate limits
- **"Port Already in Use"**: Change PORT in `.env` or kill existing processes

## 🎉 What Makes This Special

✅ **Completely Free**: Uses only Together.ai's free tier models
✅ **Production Ready**: Error handling, reconnection, session management
✅ **Intelligent**: Smart model selection based on query analysis  
✅ **Modern UI**: ChatGPT-quality interface with dark/light themes
✅ **Transparent**: Shows model selection reasoning and quality metrics
✅ **Scalable**: Modular architecture for easy extension
✅ **Cross-Platform**: Works on Windows, Mac, Linux
✅ **Mobile Responsive**: Great experience on all devices

---

**You now have a complete, professional-grade AI chat system ready to run! 🚀**

*Built with ❤️ using Together.ai free models and modern web technologies*
