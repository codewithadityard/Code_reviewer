import re
from langchain_core.messages import SystemMessage, HumanMessage
from scheduled_client import get_scheduled_chat_client
from state import ReviewerState
from langsmith import traceable

REVIEW_SYSTEM_PROMPT = """You are a senior code reviewer and software architect node.
Your job:
1. Review the tested code for Readability, Performance, Security, and Maintainability.
2. Ensure variable names are explicit, types are properly handled, and comments explain complex logic.
3. Provide a concise bulleted summary of optimizations and review findings.
4. If minor optimizations or docstring improvements are needed, provide the complete revised code inside a markdown block:
```python
# Refactored optimized code
```
5. If the code is already optimal, state that it is optimal and keep the code unchanged.
"""

@traceable(name="Review Agent Node", run_type="chain")
async def review_node(state: ReviewerState) -> dict:
    """LangGraph node that performs automated static code review and optimization."""
    code = state.get("saved_code", "")
    qa_results = state.get("test_results", {})
    client = get_scheduled_chat_client(priority=10, temperature=0.0, provider=state.get("provider"))

    messages = [
        SystemMessage(content=REVIEW_SYSTEM_PROMPT),
        HumanMessage(content=f"Code to Review:\n```python\n{code}\n```\n\nQA Test Results:\n{qa_results.get('output', 'All tests passed.')}")
    ]

    response = await client.ainvoke(messages)
    raw_text = response.content if hasattr(response, "content") else str(response)

    # Check if reviewer updated the code
    match = re.search(r'```(?:python|py)?\s*\n(.*?)(\n```|$)', raw_text, re.DOTALL | re.IGNORECASE)
    refined_code = match.group(1).strip() if match else code
    refined_code = re.sub(r'^```(?:python|py)?\s*|```$', '', refined_code, flags=re.MULTILINE).strip()
    if not refined_code:
        refined_code = code

    return {
        "saved_code": refined_code,
        "review_notes": raw_text,
        "current_state": "REVIEW",
        "iteration_count": state.get("iteration_count", 0) + 1
    }
