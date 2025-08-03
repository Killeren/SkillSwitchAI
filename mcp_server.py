"""
Model Context Protocol Server for SkillSwitchAI
Provides model metadata and context for dynamic LLM-based selection
"""

import json
from typing import Dict, List
from pydantic import BaseModel

class ModelInfo(BaseModel):
    name: str
    model_id: str
    specialties: List[str]
    capabilities: List[str]
    strengths: str
    use_cases: List[str]
    performance_tier: str
    context_length: int

class MCPServer:
    """MCP Server for SkillSwitchAI model metadata and context"""

    def __init__(self):
        self.models = self._initialize_models()

    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize the database of available models"""
        return {
            "meta-llama-3.3-70b-instruct-turbo": ModelInfo(
                name="Meta Llama 3.3 70B Instruct Turbo",
                model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                specialties=["general_chat", "multilingual", "reasoning", "analysis"],
                capabilities=["conversation", "translation", "summarization", "creative_writing"],
                strengths="Excellent general-purpose model with strong multilingual support and balanced performance",
                use_cases=["customer_support", "content_creation", "general_conversation", "translation"],
                performance_tier="high",
                context_length=131072
            ),
            "deepseek-r1-distill-70b": ModelInfo(
                name="DeepSeek R1 Distill 70B",
                model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                specialties=["reasoning", "chain_of_thought", "problem_solving", "analysis"],
                capabilities=["complex_reasoning", "mathematical_problem_solving", "logical_analysis", "critical_thinking"],
                strengths="Superior chain-of-thought reasoning and complex problem-solving capabilities",
                use_cases=["mathematical_problems", "logical_puzzles", "research_analysis", "strategic_planning"],
                performance_tier="high",
                context_length=32768
            ),
            "deepseek-r1-distill-14b": ModelInfo(
                name="DeepSeek R1 Distill 14B",
                model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                specialties=["reasoning", "analysis", "problem_solving"],
                capabilities=["structured_reasoning", "problem_decomposition", "analytical_thinking"],
                strengths="Balanced reasoning capabilities with good performance and efficiency",
                use_cases=["homework_help", "business_analysis", "decision_support", "research_assistance"],
                performance_tier="medium",
                context_length=32768
            ),
            "deepseek-r1-distill-1.5b": ModelInfo(
                name="DeepSeek R1 Distill 1.5B",
                model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                specialties=["lightweight_reasoning", "fast_responses", "basic_analysis"],
                capabilities=["quick_reasoning", "simple_problem_solving", "basic_analysis"],
                strengths="Fast and efficient for basic reasoning tasks with low latency",
                use_cases=["quick_questions", "simple_math", "basic_explanations", "fast_responses"],
                performance_tier="light",
                context_length=32768
            ),
            "deepseek-coder-v2-lite": ModelInfo(
                name="DeepSeek Coder V2 Lite",
                model_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                specialties=["coding", "programming", "debugging", "code_review"],
                capabilities=["code_generation", "debugging", "code_explanation", "refactoring"],
                strengths="Specialized for programming tasks with excellent code generation and debugging",
                use_cases=["software_development", "code_review", "debugging", "programming_education"],
                performance_tier="medium",
                context_length=16384
            ),
            "llama-3.2-11b-vision": ModelInfo(
                name="Llama 3.2 11B Vision",
                model_id="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                specialties=["vision", "image_analysis", "multimodal", "visual_reasoning"],
                capabilities=["image_understanding", "visual_question_answering", "scene_description", "visual_reasoning"],
                strengths="Advanced vision capabilities for image analysis and multimodal tasks",
                use_cases=["image_analysis", "visual_qa", "content_moderation", "visual_assistance"],
                performance_tier="medium",
                context_length=131072
            ),
            "mistral-7b": ModelInfo(
                name="Mistral 7B Instruct",
                model_id="mistralai/Mistral-7B-Instruct-v0.3",
                specialties=["general_purpose", "creative_writing", "balanced_tasks"],
                capabilities=["creative_writing", "general_conversation", "balanced_analysis", "content_generation"],
                strengths="Well-balanced model for general tasks with good creative capabilities",
                use_cases=["creative_writing", "content_creation", "general_assistance", "balanced_tasks"],
                performance_tier="medium",
                context_length=32768
            ),
            "flux-1-schnell": ModelInfo(
                name="FLUX.1 Schnell",
                model_id="black-forest-labs/FLUX.1-schnell-Free",
                specialties=["image_generation", "text_to_image", "creative_visuals"],
                capabilities=["image_generation", "artistic_creation", "visual_content"],
                strengths="High-quality text-to-image generation (free for 3 months)",
                use_cases=["image_creation", "visual_design", "artistic_projects", "content_illustration"],
                performance_tier="special",
                context_length=0  # Not applicable for image generation
            )
        }

    def get_model_context(self, model_ids: List[str]) -> Dict[str, ModelInfo]:
        """Get detailed context for specific models"""
        result = {}
        for model_id in model_ids:
            for key, model in self.models.items():
                if model.model_id == model_id or key == model_id:
                    result[key] = model
                    break
        return result

    def list_all_models(self) -> Dict[str, ModelInfo]:
        """Get all available models"""
        return self.models

    async def handle_request(self, request_type: str, **kwargs) -> Dict:
        """Handle MCP requests for model metadata and context"""
        if request_type == "list_models":
            return {model_key: model.dict() for model_key, model in self.models.items()}
        elif request_type == "get_model_context":
            model_ids = kwargs.get("model_ids", [])
            return {key: model.dict() for key, model in self.get_model_context(model_ids).items()}
        elif request_type == "select_models":
            return await self._select_models(**kwargs)
        else:
            return {"error": f"Unknown request type: {request_type}"}

    async def _select_models(self, user_input: str, chat_history: List[str] = None) -> Dict:
        """Select appropriate generator and critic models based on user input"""
        # Simple rule-based model selection
        user_input_lower = user_input.lower()
        
        # Default models
        generator = self.models["meta-llama-3.3-70b-instruct-turbo"]
        critic = self.models["deepseek-r1-distill-70b"]
        
        # Task classification
        task_type = "general_conversation"
        reasoning = "Selected general-purpose models for balanced performance"
        
        # Check for specific task types
        if any(word in user_input_lower for word in ["code", "program", "debug", "function", "class", "algorithm"]):
            generator = self.models["deepseek-coder-v2-lite"]
            critic = self.models["deepseek-r1-distill-14b"]
            task_type = "coding"
            reasoning = "Detected coding-related query, selected specialized coding model"
        
        elif any(word in user_input_lower for word in ["image", "picture", "photo", "draw", "generate", "create image"]):
            generator = self.models["flux-1-schnell"]
            critic = self.models["llama-3.2-11b-vision"]
            task_type = "image_generation"
            reasoning = "Detected image generation request, selected FLUX.1 Schnell for image creation"
        
        elif any(word in user_input_lower for word in ["math", "calculate", "equation", "solve", "problem", "reasoning"]):
            generator = self.models["deepseek-r1-distill-70b"]
            critic = self.models["deepseek-r1-distill-14b"]
            task_type = "mathematical_reasoning"
            reasoning = "Detected mathematical/reasoning task, selected high-performance reasoning model"
        
        elif any(word in user_input_lower for word in ["creative", "story", "write", "poem", "artistic"]):
            generator = self.models["mistral-7b"]
            critic = self.models["meta-llama-3.3-70b-instruct-turbo"]
            task_type = "creative_writing"
            reasoning = "Detected creative writing task, selected Mistral for creative capabilities"
        
        return {
            "generator": generator.dict(),
            "critic": critic.dict(),
            "task_classification": {
                "task_type": task_type,
                "confidence": 0.8
            },
            "reasoning": reasoning
        }

# Global MCP server instance
mcp_server = MCPServer()
