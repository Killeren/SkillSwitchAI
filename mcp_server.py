"""
Model Context Protocol Server for Together.ai Free Models
Provides intelligent model selection based on task classification
"""

import json
import asyncio
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import re

class ModelInfo(BaseModel):
    name: str
    model_id: str
    specialties: List[str]
    capabilities: List[str]
    strengths: str
    use_cases: List[str]
    performance_tier: str
    context_length: int

class TaskClassification(BaseModel):
    task_type: str
    confidence: float
    keywords: List[str]

class MCPServer:
    """MCP Server for Together.ai model selection and context"""

    def __init__(self):
        self.models = self._initialize_models()
        self.task_patterns = self._initialize_task_patterns()

    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize the database of Together.ai free models"""
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
                model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-70B",
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

    def _initialize_task_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for task classification"""
        return {
            "coding": [
                "code", "programming", "function", "algorithm", "bug", "debug", 
                "python", "javascript", "java", "c++", "html", "css", "sql",
                "implement", "script", "class", "method", "variable", "syntax"
            ],
            "reasoning": [
                "solve", "analyze", "logic", "problem", "mathematical", "calculate",
                "reasoning", "think", "explain why", "how does", "prove", "derive",
                "step by step", "chain of thought", "because", "therefore"
            ],
            "creative": [
                "write", "creative", "story", "poem", "article", "blog", "essay",
                "imagination", "fiction", "character", "plot", "narrative", "style"
            ],
            "vision": [
                "image", "picture", "visual", "photo", "see", "look", "describe",
                "analyze image", "what's in", "identify", "detect", "visual",
                "generate an image", "create an image", "draw", "paint", "artwork", "illustration",
                "make an image", "produce an image", "show me an image", "render an image", "image of", "create artwork"
            ],
            "general": [
                "explain", "what is", "how to", "help me", "information",
                "question", "chat", "talk", "discuss", "tell me"
            ],
            "translation": [
                "translate", "translation", "language", "convert", "from english",
                "to spanish", "french", "german", "chinese", "multilingual"
            ]
        }

    def classify_task(self, user_input: str, chat_history: List[str] = None) -> TaskClassification:
        """Classify the user's task based on input and chat history"""
        user_input_lower = user_input.lower()

        # Combine user input with recent chat history for context
        full_context = user_input_lower
        if chat_history:
            recent_history = " ".join(chat_history[-3:]).lower()  # Last 3 messages
            full_context += " " + recent_history

        task_scores = {}

        # Score each task type based on keyword matches
        for task_type, keywords in self.task_patterns.items():
            score = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword in full_context:
                    score += 1
                    matched_keywords.append(keyword)

                # Bonus for exact phrase matches
                if len(keyword.split()) > 1 and keyword in full_context:
                    score += 0.5

            if score > 0:
                task_scores[task_type] = {
                    'score': score / len(keywords),  # Normalize by total keywords
                    'keywords': matched_keywords
                }

        # Determine the primary task type
        if not task_scores:
            return TaskClassification(
                task_type="general",
                confidence=0.5,
                keywords=[]
            )

        best_task = max(task_scores.items(), key=lambda x: x[1]['score'])
        task_type = best_task[0]
        confidence = min(best_task[1]['score'] * 2, 1.0)  # Scale confidence
        keywords = best_task[1]['keywords']

        return TaskClassification(
            task_type=task_type,
            confidence=confidence,
            keywords=keywords
        )

    def select_models(self, user_input: str, chat_history: List[str] = None) -> Tuple[ModelInfo, ModelInfo]:
        """Select the best generator and critic models for the task"""
        task_classification = self.classify_task(user_input, chat_history)
        task_type = task_classification.task_type

        # Model selection logic based on task type
        selection_rules = {
            "coding": {
                "primary": ["deepseek-coder-v2-lite", "deepseek-r1-distill-70b"],
                "critic": ["deepseek-r1-distill-70b", "meta-llama-3.3-70b-instruct-turbo"]
            },
            "reasoning": {
                "primary": ["deepseek-r1-distill-70b", "deepseek-r1-distill-14b"],
                "critic": ["meta-llama-3.3-70b-instruct-turbo", "deepseek-r1-distill-14b"]
            },
            "creative": {
                "primary": ["meta-llama-3.3-70b-instruct-turbo", "mistral-7b"],
                "critic": ["mistral-7b", "meta-llama-3.3-70b-instruct-turbo"]
            },
            "vision": {
                "primary": ["flux-1-schnell"],
                "critic": ["meta-llama-3.3-70b-instruct-turbo", "deepseek-r1-distill-14b"]
            },
            "translation": {
                "primary": ["meta-llama-3.3-70b-instruct-turbo"],
                "critic": ["mistral-7b", "deepseek-r1-distill-14b"]
            },
            "general": {
                "primary": ["meta-llama-3.3-70b-instruct-turbo", "mistral-7b"],
                "critic": ["deepseek-r1-distill-14b", "mistral-7b"]
            }
        }

        # Get the selection rule for the task type
        rule = selection_rules.get(task_type, selection_rules["general"])

        # Select generator model (first choice from primary)
        generator_key = rule["primary"][0]
        generator_model = self.models[generator_key]

        # Select critic model (prefer different model than generator)
        critic_options = rule["critic"]
        critic_key = critic_options[0]

        # If critic would be same as generator, try second option
        if len(critic_options) > 1 and critic_key == generator_key:
            critic_key = critic_options[1]

        critic_model = self.models[critic_key]

        return generator_model, critic_model

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
        """Handle MCP requests"""
        if request_type == "select_models":
            user_input = kwargs.get("user_input", "")
            chat_history = kwargs.get("chat_history", [])

            generator, critic = self.select_models(user_input, chat_history)
            task_classification = self.classify_task(user_input, chat_history)

            return {
                "generator": generator.dict(),
                "critic": critic.dict(),
                "task_classification": task_classification.dict(),
                "reasoning": f"Selected {generator.name} for generation and {critic.name} for criticism based on {task_classification.task_type} task classification"
            }

        elif request_type == "list_models":
            return {model_key: model.dict() for model_key, model in self.models.items()}

        elif request_type == "get_model_context":
            model_ids = kwargs.get("model_ids", [])
            return {key: model.dict() for key, model in self.get_model_context(model_ids).items()}

        else:
            return {"error": f"Unknown request type: {request_type}"}

# Global MCP server instance
mcp_server = MCPServer()

# Example usage functions
async def test_model_selection():
    """Test the model selection functionality"""
    test_queries = [
        "Write a Python function to sort a list",
        "Solve this math problem: 2x + 5 = 15",
        "Write a creative story about a robot",
        "Analyze this image and describe what you see",
        "What is artificial intelligence?"
    ]

    for query in test_queries:
        result = await mcp_server.handle_request("select_models", user_input=query)
        print(f"\nQuery: {query}")
        print(f"Generator: {result['generator']['name']}")
        print(f"Critic: {result['critic']['name']}")
        print(f"Task: {result['task_classification']['task_type']} (confidence: {result['task_classification']['confidence']:.2f})")
        print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    asyncio.run(test_model_selection())
