"""
FastAPI Backend for Together.ai Multi-Agent Chat System
Provides WebSocket endpoints for real-time communication
"""

import asyncio
import json
import os
from typing import List, Dict, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from langgraph_workflow import TogetherAIWorkflow

# Load environment variables
load_dotenv()

app = FastAPI(title="Together.ai Multi-Agent Chat API", version="1.0.0")

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class ChatMessage(BaseModel):
    content: str
    timestamp: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    metadata: Dict
    session_id: str
    timestamp: str

# Global state
class ConnectionManager:
    """Manages WebSocket connections and sessions"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = websocket
        self.sessions[session_id] = {
            "chat_history": [],
            "connected_at": datetime.now().isoformat(),
            "message_count": 0,
            "last_image_url": None  # Track last image per session
        }
        return session_id

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.sessions:
            del self.sessions[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text(json.dumps(message))

    async def send_stream_chunk(self, session_id: str, chunk: str, chunk_type: str = "content"):
        message = {
            "type": "stream_chunk",
            "chunk": chunk,
            "chunk_type": chunk_type,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_message(session_id, message)

    def add_to_history(self, session_id: str, user_message: str, ai_response: str):
        if session_id in self.sessions:
            self.sessions[session_id]["chat_history"].extend([user_message, ai_response])
            self.sessions[session_id]["message_count"] += 2

            # Keep only last 20 messages to manage memory
            if len(self.sessions[session_id]["chat_history"]) > 20:
                self.sessions[session_id]["chat_history"] = self.sessions[session_id]["chat_history"][-20:]

    def get_chat_history(self, session_id: str) -> List[str]:
        if session_id in self.sessions:
            return self.sessions[session_id]["chat_history"]
        return []

# Initialize global objects
manager = ConnectionManager()
workflow = None

def init_workflow():
    """Initialize the Together.ai workflow"""
    global workflow
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment variables")
    workflow = TogetherAIWorkflow(api_key)

# Routes
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "total_sessions": len(manager.sessions)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat communication"""
    session_id = await manager.connect(websocket)

    try:
        # Send connection confirmation
        await manager.send_message(session_id, {
            "type": "connection",
            "status": "connected",
            "session_id": session_id,
            "message": "Connected to Together.ai Multi-Agent Chat System"
        })

        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            if message_data.get("type") == "chat":
                await handle_chat_message(session_id, message_data.get("content", ""))

            elif message_data.get("type") == "ping":
                await manager.send_message(session_id, {"type": "pong"})

            elif message_data.get("type") == "clear_history":
                manager.sessions[session_id]["chat_history"] = []
                await manager.send_message(session_id, {
                    "type": "history_cleared",
                    "message": "Chat history has been cleared"
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"Client {session_id} disconnected")

    except Exception as e:
        print(f"WebSocket error for {session_id}: {str(e)}")
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"An error occurred: {str(e)}"
        })
        manager.disconnect(session_id)

async def handle_chat_message(session_id: str, user_input: str):
    """Process chat message through the multi-agent workflow"""
    if not user_input.strip():
        await manager.send_message(session_id, {
            "type": "error",
            "message": "Please provide a non-empty message"
        })
        return

    try:
        # Send processing status
        await manager.send_message(session_id, {
            "type": "processing",
            "status": "started",
            "message": "🤖 Processing your message..."
        })

        # Get chat history for context
        chat_history = manager.get_chat_history(session_id)
        last_image_url = manager.sessions[session_id].get("last_image_url")

        # Send model selection status
        await manager.send_message(session_id, {
            "type": "processing",
            "status": "selecting_models",
            "message": "🔍 Selecting optimal AI models..."
        })

        # Process through workflow with streaming updates
        result = await process_with_streaming_updates(session_id, user_input, chat_history, last_image_url)

        if result:
            # Send final response
            await manager.send_message(session_id, {
                "type": "response",
                "content": result["response"],
                "metadata": {
                    "generator_model": result["generator_model"],
                    "critic_model": result["critic_model"],
                    "iterations": result["iterations"],
                    "quality_score": result["quality_score"],
                    "processing_time": result.get("processing_time", "N/A")
                },
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })

            # Add to chat history
            manager.add_to_history(session_id, user_input, result["response"])

            # Send completion status
            await manager.send_message(session_id, {
                "type": "processing",
                "status": "completed",
                "message": f"✅ Response generated using {result['iterations']} iteration(s)"
            })

            # If a new image was generated, update last_image_url in session
            if result.get("last_image_url"):
                manager.sessions[session_id]["last_image_url"] = result["last_image_url"]

    except Exception as e:
        print(f"Error processing message for {session_id}: {str(e)}")
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"I apologize, but I encountered an error while processing your message: {str(e)}"
        })

async def process_with_streaming_updates(session_id: str, user_input: str, chat_history: List[str], last_image_url: str = None) -> Dict:
    """Process query with streaming status updates"""
    start_time = datetime.now()

    try:
        # Create a custom workflow that sends updates
        result = await workflow.process_query(user_input, chat_history, max_iterations=2, last_image_url=last_image_url)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        result["processing_time"] = f"{processing_time:.2f}s"
        return result

    except Exception as e:
        await manager.send_message(session_id, {
            "type": "processing",
            "status": "error",
            "message": f"❌ Processing failed: {str(e)}"
        })
        raise e

# Alternative HTTP endpoint for simple queries
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """HTTP endpoint for simple chat (alternative to WebSocket)"""
    try:
        session_id = message.session_id or str(uuid.uuid4())
        chat_history = manager.get_chat_history(session_id) if message.session_id else []
        last_image_url = manager.sessions[session_id].get("last_image_url") if message.session_id else None

        result = await workflow.process_query(message.content, chat_history, last_image_url=last_image_url)

        response = ChatResponse(
            response=result["response"],
            metadata={
                "generator_model": result["generator_model"],
                "critic_model": result["critic_model"],
                "iterations": result["iterations"],
                "quality_score": result["quality_score"]
            },
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

        # Add to history if session exists
        if message.session_id:
            manager.add_to_history(session_id, message.content, result["response"])

        # If a new image was generated, update last_image_url in session
        if result.get("last_image_url"):
            manager.sessions[session_id]["last_image_url"] = result["last_image_url"]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "chat_history": manager.get_chat_history(session_id),
        "session_info": manager.sessions[session_id]
    }

@app.delete("/sessions/{session_id}/history")
async def clear_session_history(session_id: str):
    """Clear chat history for a session"""
    if session_id not in manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    manager.sessions[session_id]["chat_history"] = []
    return {"message": "Chat history cleared", "session_id": session_id}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    try:
        init_workflow()
        print("✅ Together.ai workflow initialized successfully")
        print("🚀 FastAPI server starting...")
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {str(e)}")
        print("Please check your TOGETHER_API_KEY environment variable")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
