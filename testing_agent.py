import os
import re
import sys
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from scheduled_client import get_scheduled_chat_client
from state import ReviewerState
from langsmith import traceable

QA_SYSTEM_PROMPT = """You are a strict, uncompromising Python QA Automation Engineer.
Your job:
1. Write a complete pytest script (using `pytest` and `assert`) that imports and validates the provided code.
2. Always import functions or classes directly from `solution` (e.g. `from solution import *`).
3. Cover edge cases, typical usage, and type checks concisely.
4. You must write complete, executable test code inside a single standard markdown block:
```python
# Complete pytest test code here
```
5. Do NOT use dummy pass statements, external uninstalled packages, time.sleep delays, or mock the implementation.
6. Keep tests concise and fast without long essay docstrings so that test suites are complete and syntax-valid.
"""

@traceable(name="QA Agent Node", run_type="chain")
async def qa_node(state: ReviewerState) -> dict:
    """LangGraph node that synthesizes pytest suites and runs tests on generated code."""
    code = state.get("saved_code", "")
    client = get_scheduled_chat_client(priority=5, temperature=0.0, provider=state.get("provider"))

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=f"Write pytest tests for this code:\n```python\n{code}\n```")
    ]

    response = await client.ainvoke(messages)
    raw_text = response.content if hasattr(response, "content") else str(response)

    # Extract test code block safely
    match = re.search(r'```(?:python|py)?\s*\n(.*?)(\n```|$)', raw_text, re.DOTALL | re.IGNORECASE)
    test_code = match.group(1).strip() if match else raw_text.strip()
    test_code = re.sub(r'^```(?:python|py)?\s*|```$', '', test_code, flags=re.MULTILINE).strip()

    import ast

    # Validate solution code syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        current_retries = state.get("retries", 0)
        return {
            "qa_passed": False,
            "qa_feedback": f"SyntaxError in solution code: {e}",
            "test_results": {"exit_code": -1, "output": f"SyntaxError in solution code: {e}"},
            "retries": current_retries + 1,
            "current_state": "QA",
            "iteration_count": state.get("iteration_count", 0) + 1
        }

    # Validate test code syntax
    try:
        ast.parse(test_code)
    except SyntaxError as e:
        current_retries = state.get("retries", 0)
        return {
            "qa_passed": False,
            "qa_feedback": f"SyntaxError in generated test code (output was likely truncated): {e}. Regenerating concise tests.",
            "test_results": {"exit_code": -1, "output": f"SyntaxError in test code: {e}"},
            "retries": current_retries + 1,
            "current_state": "QA",
            "iteration_count": state.get("iteration_count", 0) + 1
        }

    # Save solution and test files into working_space
    work_dir = os.path.abspath("working_space")
    os.makedirs(work_dir, exist_ok=True)
    solution_path = os.path.join(work_dir, "solution.py")
    test_path = os.path.join(work_dir, "test_solution.py")

    with open(solution_path, "w", encoding="utf-8") as f:
        f.write(code)

    # Prepend import of solution if not present
    if "from solution import" not in test_code and "import solution" not in test_code:
        test_file_content = f"import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\nfrom solution import *\n\n{test_code}"
    else:
        test_file_content = f"import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n{test_code}"

    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_file_content)

    # Execute pytest in subprocess with 15s timeout
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pytest", test_path, "-v",
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")
        full_output = f"{stdout_str}\n{stderr_str}".strip()
        exit_code = proc.returncode

        qa_passed = (exit_code == 0)
    except asyncio.TimeoutError:
        qa_passed = False
        full_output = "Execution timed out after 15 seconds."
        exit_code = -1
    except Exception as exc:
        qa_passed = False
        full_output = f"Execution failed: {exc}"
        exit_code = -1

    current_retries = state.get("retries", 0)
    new_retries = current_retries if qa_passed else current_retries + 1

    return {
        "qa_passed": qa_passed,
        "qa_feedback": "" if qa_passed else full_output,
        "test_results": {"exit_code": exit_code, "output": full_output},
        "retries": new_retries,
        "current_state": "QA",
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def should_retry_qa(state: ReviewerState) -> str:
    """Conditional edge routing based on test pass/fail status."""
    if state.get("qa_passed"):
        return "reviewer"
    if state.get("retries", 0) >= state.get("max_retries", 3):
        return "reviewer"
    return "coder"
