"""
LangGraph Multi-Agent Workflow for Together.ai Models
Implements Base Agent -> Generator -> Critic iterative refinement
"""

import asyncio
import json
from typing import Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from together import Together
from mcp_server import mcp_server
import os
from datetime import datetime

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
        self.client = Together(api_key=api_key)
        self.graph = self._build_graph()

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

    async def base_agent(self, state: AgentState) -> AgentState:
        """Base agent selects appropriate generator and critic models"""
        print("🤖 Base Agent: Selecting optimal models...")

        try:
            # Use MCP server to select models
            selection_result = await mcp_server.handle_request(
                "select_models",
                user_input=state["user_input"],
                chat_history=state.get("chat_history", [])
            )

            # Ensure we have valid selection results
            if "generator" not in selection_result or "critic" not in selection_result:
                print("❌ Invalid model selection result, using defaults")
                # Fallback to default models using MCP server
                all_models = await mcp_server.handle_request("list_models")
                selection_result = {
                    "generator": all_models["meta-llama-3.3-70b-instruct-turbo"],
                    "critic": all_models["deepseek-r1-distill-70b"],
                    "task_classification": {"task_type": "general_conversation", "confidence": 0.5},
                    "reasoning": "Using default models due to selection error"
                }

            state["selected_generator"] = selection_result["generator"]
            state["selected_critic"] = selection_result["critic"]
            state["metadata"] = {
                "task_classification": selection_result.get("task_classification", {"task_type": "unknown"}),
                "selection_reasoning": selection_result.get("reasoning", "Default selection"),
                "timestamp": datetime.now().isoformat()
            }

            print(f"✅ Selected Generator: {selection_result['generator']['name']}")
            print(f"✅ Selected Critic: {selection_result['critic']['name']}")
            print(f"📋 Task Type: {selection_result['task_classification']['task_type']}")

        except Exception as e:
            print(f"❌ Error in base agent: {str(e)}")
            # Fallback to default models using MCP server
            try:
                all_models = await mcp_server.handle_request("list_models")
                state["selected_generator"] = all_models["meta-llama-3.3-70b-instruct-turbo"]
                state["selected_critic"] = all_models["deepseek-r1-distill-70b"]
            except:
                # Ultimate fallback with hardcoded defaults
                state["selected_generator"] = {
                    "name": "Meta Llama 3.3 70B Instruct Turbo",
                    "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "specialties": ["general_chat", "multilingual", "reasoning", "analysis"],
                    "strengths": "Excellent general-purpose model with strong multilingual support"
                }
                state["selected_critic"] = {
                    "name": "DeepSeek R1 Distill 70B",
                    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-70B",
                    "specialties": ["reasoning", "chain_of_thought", "problem_solving", "analysis"],
                    "strengths": "Superior chain-of-thought reasoning and complex problem-solving capabilities"
                }
            
            state["metadata"] = {
                "task_classification": {"task_type": "general_conversation", "confidence": 0.5},
                "selection_reasoning": f"Fallback due to error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            print("✅ Using fallback models")

        return state

    async def generator_agent(self, state: AgentState) -> AgentState:
        print(f"✍️  Generator Agent: Creating response with {state['selected_generator']['name']}...")

        generator_name = state['selected_generator']['name'].lower()
        if "flux" in generator_name:
            # Handle image generation with FLUX.1 Schnell
            try:
                prompt = state["user_input"]
                response = self.client.images.generate(
                    model=state["selected_generator"]["model_id"],
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                    steps=3
                )
                image_url = response.data[0].url
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
            response = self.client.chat.completions.create(
                model=state["selected_generator"]["model_id"],
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                stream=False
            )

            generated_response = response.choices[0].message.content
            state["current_response"] = generated_response

            print(f"📝 Generated response ({len(generated_response)} chars)")

        except Exception as e:
            print(f"❌ Generation error: {str(e)}")
            state["current_response"] = f"I apologize, but I encountered an error while generating a response: {str(e)}"

        return state

    async def critic_agent(self, state: AgentState) -> AgentState:
        """Critic agent evaluates and provides feedback on the response"""
        print(f"🔍 Critic Agent: Evaluating with {state['selected_critic']['name']}...")

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
            response = self.client.chat.completions.create(
                model=state["selected_critic"]["model_id"],
                messages=[
                    {"role": "system", "content": "You are a helpful and constructive AI response critic."},
                    {"role": "user", "content": critic_prompt}
                ],
                max_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                stream=False
            )

            critic_feedback = response.choices[0].message.content
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
            "generator_model": final_state.get("selected_generator", {}).get("name", "Unknown"),
            "critic_model": final_state.get("selected_critic", {}).get("name", "Unknown"),
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
