"""
LangGraph Multi-Agent Workflow for Together.ai Models
Implements Base Agent -> Generator -> Critic iterative refinement
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import together
from mcp_server import mcp_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('skillswitch.log')
    ]
)

logger = logging.getLogger(__name__)

# Fallback model selection configuration
FALLBACK_RULES = {
    "coding": {
        "keywords": ["code", "program", "function", "debug", "algorithm", "python", "javascript", "java", "programming", "development"],
        "generator": "deepseek-coder-v2-lite",
        "critic": "deepseek-r1-distill-14b"
    },
    "reasoning": {
        "keywords": ["explain", "analyze", "reason", "think", "logic", "problem", "why", "how", "compare"],
        "generator": "deepseek-r1-distill-14b", 
        "critic": "deepseek-r1-distill-70b"
    },
    "image": {
        "keywords": ["image", "picture", "photo", "generate image", "create image", "draw", "visual"],
        "generator": "flux-1-schnell",
        "critic": "llama-3.2-11b-vision"
    },
    "default": {
        "generator": "mistral-7b",
        "critic": "deepseek-r1-distill-14b"
    }
}

# State definition for the workflow
class AgentState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    user_input: str
    chat_history: List[str]
    selected_generator: Optional[Dict]
    selected_critic: Optional[Dict]
    current_response: str
    critic_feedback: str
    iteration_count: int
    max_iterations: int
    final_response: str
    metadata: Dict
    last_image_url: Optional[str]  # Track last generated image

class TogetherAIWorkflow:
    """LangGraph workflow for multi-agent conversation system"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.graph = self._build_graph()
        
        # Verify API key is set
        if not api_key:
            raise ValueError("API key is required")
        logger.info("API key initialized successfully")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("base_agent", self.base_agent)
        workflow.add_node("generator_agent", self.generator_agent)
        workflow.add_node("critic_agent", self.critic_agent)
        workflow.add_node("refine_response", self.refine_response)
        workflow.add_node("finalize", self.finalize_response)

        # Add edges
        workflow.set_entry_point("base_agent")
        workflow.add_edge("base_agent", "generator_agent")
        workflow.add_edge("generator_agent", "critic_agent")
        workflow.add_conditional_edges(
            "critic_agent",
            self._should_continue_refining,
            {
                "continue": "refine_response",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("refine_response", "critic_agent")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _prepare_models_data(self, all_models: Dict) -> str:
        """Convert models to JSON format for LLM consumption"""
        return json.dumps(
            {k: {
                "name": v.name,
                "model_id": v.model_id,
                "specialties": v.specialties,
                "capabilities": v.capabilities,
                "strengths": v.strengths,
                "use_cases": v.use_cases,
                "performance_tier": v.performance_tier,
                "context_length": v.context_length
            } for k, v in all_models.items()}, 
            indent=2
        )

    def _create_selection_prompt(self, state: AgentState, models_json: str) -> str:
        """Create selection prompt for the LLM"""
        return f"""You are an expert AI agent orchestrator. Your task is to select the best generator and critic models for a given user query.

Here is the context describing all available Together.ai models:
{models_json}

Given the user's query and conversation history, select the best model for generation and for criticism. The critic can be the same as the generator or different.

Consider:
- The user's query type and complexity
- Each model's specialties and strengths
- Performance tiers (high, medium, light, special)
- Context length requirements
- The conversation history for context

Respond in this exact JSON format:
{{
  "generator": "model_key_here",
  "critic": "model_key_here", 
  "reasoning": "Brief explanation of your selection decision",
  "confidence": 0.85
}}

USER QUERY: {state["user_input"]}
CHAT HISTORY: {state.get("chat_history", [])}

Remember: Use the exact model keys from the JSON above (e.g., "deepseek-coder-v2-lite", "mistral-7b", etc.)"""

    async def _call_llm_for_selection(self, selection_prompt: str) -> dict:
        """Call Together.ai LLM for model selection"""
        formatted_prompt = f"System: You are an expert AI model selector. Provide only valid JSON responses.\n\nUser: {selection_prompt}\n\nAssistant: "
        
        try:
            response = together.Complete.create(
                prompt=formatted_prompt,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                stop=["User:", "System:"]
            )
        except Exception as explicit_error:
            logger.warning(f"Explicit API key failed: {str(explicit_error)}")
            # Fallback to global API key only
            response = together.Complete.create(
                prompt=formatted_prompt,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                stop=["User:", "System:"]
            )
        
        return response

    def _parse_selection_response(self, response: dict) -> tuple:
        """Parse and validate LLM selection response"""
        selection_text = response["choices"][0]["text"].strip()
        selection_result = json.loads(selection_text)
        
        # Validate the response
        if "generator" not in selection_result or "critic" not in selection_result:
            raise ValueError("Missing generator or critic in selection result")
        
        return selection_result, selection_text

    def _validate_and_build_state(self, state: AgentState, selection_result: dict, all_models: Dict) -> None:
        """Validate selected models and build state"""
        generator_key = selection_result["generator"]
        critic_key = selection_result["critic"]
        
        # Get the selected models from our database
        if generator_key not in all_models:
            raise ValueError(f"Invalid generator key: {generator_key}")
        if critic_key not in all_models:
            raise ValueError(f"Invalid critic key: {critic_key}")
        
        selected_generator = all_models[generator_key]
        selected_critic = all_models[critic_key]
        
        state["selected_generator"] = {
            "name": selected_generator.name,
            "model_id": selected_generator.model_id,
            "specialties": selected_generator.specialties,
            "capabilities": selected_generator.capabilities,
            "strengths": selected_generator.strengths,
            "use_cases": selected_generator.use_cases,
            "performance_tier": selected_generator.performance_tier,
            "context_length": selected_generator.context_length
        }
        
        state["selected_critic"] = {
            "name": selected_critic.name,
            "model_id": selected_critic.model_id,
            "specialties": selected_critic.specialties,
            "capabilities": selected_critic.capabilities,
            "strengths": selected_critic.strengths,
            "use_cases": selected_critic.use_cases,
            "performance_tier": selected_critic.performance_tier,
            "context_length": selected_critic.context_length
        }
        
        state["metadata"] = {
            "selection_reasoning": selection_result.get("reasoning", "No reasoning provided"),
            "selection_confidence": selection_result.get("confidence", 0.5),
            "selection_method": "llm_based",
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"LLM Selected Generator: {selected_generator.name}")
        logger.info(f"LLM Selected Critic: {selected_critic.name}")
        logger.info(f"Selection Reasoning: {selection_result.get('reasoning', 'No reasoning provided')}")
        logger.info(f"Selection Confidence: {selection_result.get('confidence', 0.5)}")

    async def base_agent(self, state: AgentState) -> AgentState:
        """Base agent selects appropriate generator and critic models using LLM-based selection"""
        logger.info("Base Agent: Selecting optimal models using LLM")
        
        # Ensure we have access to the API key
        if not hasattr(self, 'api_key') or not self.api_key:
            raise ValueError("API key not available in base_agent")

        # Get all available models from MCP server
        all_models = mcp_server.list_all_models()
        
        try:
            logger.info("Using API key for model selection")
            
            # Prepare data and prompts
            models_json = self._prepare_models_data(all_models)
            selection_prompt = self._create_selection_prompt(state, models_json)
            
            # Call LLM for selection
            response = await self._call_llm_for_selection(selection_prompt)
            
            # Parse and validate response
            try:
                selection_result, selection_text = self._parse_selection_response(response)
                self._validate_and_build_state(state, selection_result, all_models)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Failed to parse or process LLM selection output: {e}")
                if 'selection_text' in locals():
                    logger.debug(f"Raw response: {selection_text}")
                # Fallback to default selection
                await self._fallback_model_selection(state, all_models)
                
        except Exception as e:
            logger.error(f"LLM selection failed: {str(e)}")
            # Fallback to default selection
            await self._fallback_model_selection(state, all_models)

        return state

    async def _fallback_model_selection(self, state: AgentState, all_models: Dict):
        """Fallback model selection when LLM-based selection fails"""
        logger.info("Using fallback model selection")
        
        # Rule-based fallback using configuration
        user_input_lower = state["user_input"].lower()
        selected_category = "default"
        
        # Check each category for keyword matches
        for category, rules in FALLBACK_RULES.items():
            if category == "default":
                continue
            if any(keyword in user_input_lower for keyword in rules["keywords"]):
                selected_category = category
                break
        
        # Get the appropriate model keys
        if selected_category == "default":
            generator_key = FALLBACK_RULES["default"]["generator"]
            critic_key = FALLBACK_RULES["default"]["critic"]
        else:
            generator_key = FALLBACK_RULES[selected_category]["generator"]
            critic_key = FALLBACK_RULES[selected_category]["critic"]
        
        selected_generator = all_models[generator_key]
        selected_critic = all_models[critic_key]
        
        state["selected_generator"] = {
            "name": selected_generator.name,
            "model_id": selected_generator.model_id,
            "specialties": selected_generator.specialties,
            "capabilities": selected_generator.capabilities,
            "strengths": selected_generator.strengths,
            "use_cases": selected_generator.use_cases,
            "performance_tier": selected_generator.performance_tier,
            "context_length": selected_generator.context_length
        }
        
        state["selected_critic"] = {
            "name": selected_critic.name,
            "model_id": selected_critic.model_id,
            "specialties": selected_critic.specialties,
            "capabilities": selected_critic.capabilities,
            "strengths": selected_critic.strengths,
            "use_cases": selected_critic.use_cases,
            "performance_tier": selected_critic.performance_tier,
            "context_length": selected_critic.context_length
        }
        
        state["metadata"] = {
            "selection_reasoning": f"Fallback selection ({selected_category}): {generator_key} for generation, {critic_key} for criticism",
            "selection_confidence": 0.6,
            "selection_method": "fallback_rule_based",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Fallback Generator: {selected_generator.name}")
        logger.info(f"Fallback Critic: {selected_critic.name}")
        logger.info(f"Fallback Category: {selected_category}")

    async def generator_agent(self, state: AgentState) -> AgentState:
        print(f"✍️  Generator Agent: Creating response with {state['selected_generator']['name']}...")
        
        # Ensure we have access to the API key
        if not hasattr(self, 'api_key') or not self.api_key:
            raise ValueError("API key not available in generator_agent")

        generator_name = state['selected_generator']['name'].lower()
        if "flux" in generator_name:
            # Handle image generation with FLUX.1 Schnell
            try:
                prompt = state["user_input"]
                
                # Try image generation with global API key
                response = together.Image.create(
                    prompt=prompt,
                    model=state["selected_generator"]["model_id"],
                    results=1,
                    width=1024,
                    height=1024,
                    steps=3
                )
                # Validate response structure
                if not response or "data" not in response or not response["data"] or "url" not in response["data"][0]:
                    raise ValueError("Invalid response structure from image generation API")
                
                image_url = response["data"][0]["url"]
                state["current_response"] = f"![Generated Image]({image_url})\n\n[View Image]({image_url})"
                state["last_image_url"] = image_url  # Store last image URL
                print(f"🖼️ Generated image: {image_url}")
            except Exception as e:
                print(f"❌ Image generation error: {str(e)}")
                state["current_response"] = f"Sorry, I couldn't generate the image: {str(e)}"
            return state

        # Prepare prompt with context
        system_prompt = f"""You are an AI assistant specialized in {', '.join(state['selected_generator']['specialties'])}. 
        Your strengths include: {state['selected_generator']['strengths']}

        Provide a helpful, accurate, and detailed response to the user's query. 
        Consider the conversation history to maintain context and coherence."""

        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]

        # Add chat history
        if state.get("chat_history"):
            for i, msg in enumerate(state["chat_history"][-6:]):  # Last 6 messages for context
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": msg})

        # Add current user input
        messages.append({"role": "user", "content": state["user_input"]})

        # If this is a refinement iteration, add previous response and feedback
        if state.get("current_response") and state.get("critic_feedback"):
            refinement_prompt = f"""
            Previous response: {state['current_response']}

            Feedback for improvement: {state['critic_feedback']}

            Please provide an improved version of the response that addresses the feedback while maintaining accuracy and helpfulness.
            """
            messages.append({"role": "user", "content": refinement_prompt})

        try:
            # Generate response using Together.ai
            # Convert messages to prompt format
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n\n"
            
            prompt += "Assistant: "
            
            response = together.Complete.create(
                prompt=prompt,
                model=state["selected_generator"]["model_id"],
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                stop=["User:", "System:"]
            )

            # Validate response structure
            if not response or "choices" not in response or not response["choices"] or "text" not in response["choices"][0]:
                raise ValueError("Invalid response structure from text generation API")
            
            generated_response = response["choices"][0]["text"].strip()
            state["current_response"] = generated_response

            print(f"📝 Generated response ({len(generated_response)} chars)")

        except Exception as e:
            print(f"❌ Generation error: {str(e)}")
            state["current_response"] = f"I apologize, but I encountered an error while generating a response: {str(e)}"

        return state

    async def critic_agent(self, state: AgentState) -> AgentState:
        """Critic agent evaluates and provides feedback on the response"""
        print(f"🔍 Critic Agent: Evaluating with {state['selected_critic']['name']}...")
        
        # Ensure we have access to the API key
        if not hasattr(self, 'api_key') or not self.api_key:
            raise ValueError("API key not available in critic_agent")

        critic_name = state['selected_critic']['name'].lower()
        is_vision_critic = "vision" in critic_name or "image" in critic_name
        image_url = state.get("last_image_url")

        if is_vision_critic and image_url:
            critic_prompt = f"""You are an expert vision model. Analyze the following image and answer the user's question or provide the requested information.

Image URL: {image_url}

User Query: {state['user_input']}
"""
        else:
            critic_prompt = f"""You are an expert critic specialized in {', '.join(state['selected_critic']['specialties'])}.
Your role is to evaluate the quality of AI-generated responses and provide constructive feedback.

Evaluate the following response to the user query:

Original Query: {state['user_input']}

Response to Evaluate: {state['current_response']}

Please provide:
1. A quality score from 1-10 (where 10 is excellent)
2. Specific strengths of the response
3. Areas for improvement (if any)
4. Suggestions for enhancement
5. Whether the response adequately addresses the user's query

Format your feedback as constructive criticism that can help improve the response.
If the response is already high-quality (score 8+), you may indicate that minimal changes are needed.
"""

        try:
            # Get critic evaluation
            prompt = f"System: You are a helpful and constructive AI response critic.\n\nUser: {critic_prompt}\n\nAssistant: "
            
            response = together.Complete.create(
                prompt=prompt,
                model=state["selected_critic"]["model_id"],
                max_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                stop=["User:", "System:"]
            )

            # Validate response structure
            if not response or "choices" not in response or not response["choices"] or "text" not in response["choices"][0]:
                raise ValueError("Invalid response structure from critic API")
            
            critic_feedback = response["choices"][0]["text"].strip()
            state["critic_feedback"] = critic_feedback

            # Extract quality score for decision making
            quality_score = self._extract_quality_score(critic_feedback)
            state["metadata"]["quality_score"] = quality_score

            print(f"📊 Critic feedback received (Quality: {quality_score}/10)")

        except Exception as e:
            print(f"❌ Critic error: {str(e)}")
            state["critic_feedback"] = "Unable to provide detailed feedback due to an error."
            state["metadata"]["quality_score"] = 5  # Default score

        return state

    def _extract_quality_score(self, feedback: str) -> int:
        """Extract quality score from critic feedback"""
        import re

        # Look for patterns like "8/10", "score: 7", "quality: 9"
        patterns = [
            r"(\d+)/10",
            r"score[:\s]+(\d+)",
            r"quality[:\s]+(\d+)",
            r"rating[:\s]+(\d+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, feedback.lower())
            if match:
                score = int(match.group(1))
                return min(max(score, 1), 10)  # Ensure score is between 1-10

        # Default score if no pattern found
        return 6

    def _should_continue_refining(self, state: AgentState) -> str:
        """Decide whether to continue refining or finalize the response"""
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        quality_score = state.get("metadata", {}).get("quality_score", 5)

        # Continue if we haven't reached max iterations and quality could be better
        if iteration_count < max_iterations and quality_score < 8:
            print(f"🔄 Continuing refinement (iteration {iteration_count + 1}/{max_iterations}, quality: {quality_score}/10)")
            return "continue"
        else:
            print(f"✅ Finalizing response (iterations: {iteration_count + 1}, quality: {quality_score}/10)")
            return "finalize"

    async def refine_response(self, state: AgentState) -> AgentState:
        """Prepare for next refinement iteration"""
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        print(f"🔄 Starting refinement iteration {state['iteration_count']}")
        return state

    async def finalize_response(self, state: AgentState) -> AgentState:
        """Finalize the response with metadata"""
        state["final_response"] = state["current_response"]

        # Add metadata for transparency
        iterations = state.get("iteration_count", 0) + 1
        quality_score = state.get("metadata", {}).get("quality_score", "N/A")

        state["metadata"]["final_iterations"] = iterations
        state["metadata"]["processing_complete"] = True

        print(f"🎉 Response finalized after {iterations} iteration(s)")
        print(f"📈 Final quality score: {quality_score}/10")

        return state

    async def process_query(
        self, 
        user_input: str, 
        chat_history: List[str] = None, 
        max_iterations: int = 3,
        last_image_url: Optional[str] = None
    ) -> Dict:
        """Process a user query through the multi-agent workflow"""

        initial_state = AgentState(
            messages=[],
            user_input=user_input,
            chat_history=chat_history or [],
            selected_generator=None,
            selected_critic=None,
            current_response="",
            critic_feedback="",
            iteration_count=0,
            max_iterations=max_iterations,
            final_response="",
            metadata={},
            last_image_url=last_image_url
        )

        print(f"🚀 Starting multi-agent processing for: '{user_input[:50]}...'")

        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)

        return {
            "response": final_state["final_response"],
            "metadata": final_state["metadata"],
            "generator_model": final_state["selected_generator"]["name"],
            "critic_model": final_state["selected_critic"]["name"],
            "iterations": final_state["metadata"].get("final_iterations", 1),
            "quality_score": final_state["metadata"].get("quality_score", "N/A"),
            "last_image_url": final_state.get("last_image_url")
        }

# Example usage
async def test_workflow():
    """Test the workflow with sample queries"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")

    if not api_key:
        print("Please set TOGETHER_API_KEY in your .env file")
        return

    workflow = TogetherAIWorkflow(api_key)

    test_queries = [
        "Write a Python function to calculate factorial",
        "Explain quantum computing in simple terms",
        "Create a short story about time travel"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        result = await workflow.process_query(query)
        print(f"Query: {query}")
        print(f"Generator: {result['generator_model']}")
        print(f"Critic: {result['critic_model']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Quality: {result['quality_score']}/10")
        print(f"Response: {result['response'][:200]}...")

if __name__ == "__main__":
    asyncio.run(test_workflow())
