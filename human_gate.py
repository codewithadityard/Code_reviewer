from langgraph.types import interrupt
from state import ReviewerState


def human_gate_node(state: ReviewerState) -> dict:
    """
    Suspends execution via LangGraph interrupt.
    Returns state payload to the checkpointer and halts until
    resumed via /api/tasks/{task_id}/approve, /reject, or /review/resume/{thread_id}.
    """
    approval_payload = interrupt({
        "type": "NEEDS_HUMAN_APPROVAL",
        "code_to_review": state.get("saved_code", ""),
        "review_notes": state.get("review_notes", ""),
        "test_results": state.get("test_results", {}),
        "retries": state.get("retries", 0),
        "modified_files": state.get("modified_files", []),
        "estimated_cost": state.get("cost_estimate", 0.0)
    })

    # approval_payload is injected when execution resumes
    approved = False
    feedback = None

    if isinstance(approval_payload, dict):
        approved = approval_payload.get("approved", False)
        feedback = approval_payload.get("feedback")
    elif isinstance(approval_payload, str):
        if approval_payload.strip().lower() in ("yes", "approved", "true"):
            approved = True
        else:
            feedback = approval_payload
    elif isinstance(approval_payload, bool):
        approved = approval_payload

    if approved:
        return {
            "status": "FINISHED",
            "current_state": "FINISHED",
            "human_feedback": None,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
    else:
        return {
            "status": "WAITING_FOR_HUMAN",
            "current_state": "CODER",
            "human_feedback": feedback or "Human rejected code. Please revise implementation.",
            "iteration_count": state.get("iteration_count", 0) + 1
        }


def route_after_human(state: ReviewerState) -> str:
    """Routes execution to END if approved, or routes back to coder on feedback."""
    if state.get("status") == "FINISHED":
        return "END"
    return "coder"