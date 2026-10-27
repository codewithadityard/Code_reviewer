from langchain_core.messages import SystemMessage, HumanMessage
from scheduled_client import get_scheduled_chat_client
from state import ReviewerState
from langsmith import traceable

PLANNER_SYSTEM_PROMPT = """You are an expert software architect and technical planner.
Your job:
1. Break down the user's software requirement into a concise, actionable, step-by-step technical plan.
2. Outline key components, functions, data structures, and edge cases.
3. Be concise and technical (under 350 words). Do NOT write the final implementation code. Produce a clear, efficient architectural blueprint for the Coding Engineer.
"""

@traceable(name="Planner Agent Node", run_type="chain")
async def planner_node(state: ReviewerState) -> dict:
    """LangGraph node that generates the architectural plan from user request."""
    user_request = state.get("user_request") or state.get("context") or ""
    client = get_scheduled_chat_client(priority=1, temperature=0.0, provider=state.get("provider"))

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Requirement to plan:\n{user_request}")
    ]

    response = await client.ainvoke(messages)
    plan_text = response.content if hasattr(response, "content") else str(response)

    return {
        "saved_plan": plan_text,
        "current_state": "PLANNER",
        "iteration_count": state.get("iteration_count", 0) + 1
    }
