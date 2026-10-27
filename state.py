from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator


def merge_dict_add(left: Optional[Dict[str, int]], right: Optional[Dict[str, int]]) -> Dict[str, int]:
    """Custom reducer for dictionary token usage summing in LangGraph."""
    merged = (left or {}).copy()
    for k, v in (right or {}).items():
        merged[k] = merged.get(k, 0) + v
    return merged


class AgentStepRecord(TypedDict):
    step: str
    agent: str
    timestamp: float


class ReviewerState(TypedDict, total=False):
    # Core Pipeline Fields
    task_id: str
    user_request: str
    current_state: str
    saved_plan: str
    saved_code: str
    test_results: Dict[str, Any]         # exit_code, stdout, stderr
    qa_passed: bool
    qa_feedback: str
    review_notes: str
    retries: int
    max_retries: int
    human_feedback: Optional[Any]
    status: str                         # RUNNING, WAITING_FOR_HUMAN, FINISHED, ABORTED

    # GitHub / Repository Fields
    repo_url: Optional[str]
    pr_id: Optional[int]
    commit_sha: Optional[str]
    modified_files: List[Dict[str, Any]]  # path, diff, extracted_ast_context
    syntax_errors: List[str]
    token_usage: Annotated[Dict[str, int], merge_dict_add]
    cost_estimate: Annotated[float, operator.add]
    step_history: Annotated[List[AgentStepRecord], operator.add]
    iteration_count: int


class SupervisorDecision(BaseModel):
    next_node: Literal["planner", "coder", "tester", "reviewer", "human_gate", "FINISH"]
    rationale: str = Field(description="Deterministic reasoning for step routing.")