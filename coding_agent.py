import os
import re
from langchain_core.messages import SystemMessage, HumanMessage
from scheduled_client import get_scheduled_chat_client
from state import ReviewerState
from langsmith import traceable

CODER_SYSTEM_PROMPT = """You are a senior Python developer.
Your job:
1. Write clean, complete, production-ready Python code based on the technical plan.
2. If QA failures, error tracebacks, or human feedback are provided, diagnose and fix the bugs directly.
3. Always wrap your code inside a single standard markdown block:
```python
# Your complete python code here
```
4. Do NOT use placeholders, ellipsis (...), or incomplete implementations.
"""

@traceable(name="Coder Agent Node", run_type="chain")
async def coder_node(state: ReviewerState) -> dict:
    """LangGraph node that generates or fixes Python code."""
    client = get_scheduled_chat_client(priority=3, temperature=0.0, provider=state.get("provider"))
    
    plan = state.get("saved_plan", "")
    current_code = state.get("saved_code", "")
    qa_feedback = state.get("qa_feedback", "")
    human_feedback = state.get("human_feedback")

    # Prune lengthy components to fit well within Groq TPM limits
    trimmed_plan = plan[:1800] if len(plan) > 1800 else plan
    trimmed_feedback = qa_feedback[-1500:] if len(qa_feedback) > 1500 else qa_feedback

    prompt_content = f"Architectural Blueprint:\n{trimmed_plan}\n\n"
    if current_code:
        prompt_content += f"Existing Code:\n```python\n{current_code}\n```\n\n"
    if trimmed_feedback:
        prompt_content += f"QA Test Failures to Fix:\n{trimmed_feedback}\n\n"
    if human_feedback:
        prompt_content += f"Human Feedback / Rejection Reason:\n{human_feedback}\n\n"

    prompt_content += "Provide the updated, complete Python code:"

    messages = [
        SystemMessage(content=CODER_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content)
    ]

    response = await client.ainvoke(messages)
    raw_text = response.content if hasattr(response, "content") else str(response)

    # Extract python code block safely
    match = re.search(r'```(?:python|py)?\s*\n(.*?)(\n```|$)', raw_text, re.DOTALL | re.IGNORECASE)
    extracted_code = match.group(1).strip() if match else raw_text.strip()
    extracted_code = re.sub(r'^```(?:python|py)?\s*|```$', '', extracted_code, flags=re.MULTILINE).strip()

    if not extracted_code and current_code:
        extracted_code = current_code

    # Save to working_space for test executions
    os.makedirs("working_space", exist_ok=True)
    with open(os.path.join("working_space", "solution.py"), "w", encoding="utf-8") as f:
        f.write(extracted_code)

    return {
        "saved_code": extracted_code,
        "current_state": "CODER",
        "human_feedback": None,
        "iteration_count": state.get("iteration_count", 0) + 1
    }
