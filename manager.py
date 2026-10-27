import asyncio
import json
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langsmith import traceable
from langfuse.langchain import CallbackHandler
from langfuse import get_client

from state import ReviewerState
from planner_agent import planner_node
from coding_agent import coder_node
from testing_agent import qa_node, should_retry_qa
from review_agent import review_node
from human_gate import human_gate_node, route_after_human

# Initialize in-memory checkpointer for thread persistence
checkpointer = MemorySaver()

# Build the LangGraph State Graph
workflow = StateGraph(ReviewerState)

# Add Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("tester", qa_node)
workflow.add_node("reviewer", review_node)
workflow.add_node("human_gate", human_gate_node)

# Add Edges
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "tester")
workflow.add_conditional_edges(
    "tester",
    should_retry_qa,
    {
        "coder": "coder",
        "reviewer": "reviewer"
    }
)
workflow.add_edge("reviewer", "human_gate")
workflow.add_conditional_edges(
    "human_gate",
    route_after_human,
    {
        "END": END,
        "coder": "coder"
    }
)

# Compile the runnable graph with checkpointer
app_graph = workflow.compile(checkpointer=checkpointer)


@traceable(name="AutoDev LangGraph Pipeline", run_type="chain")
async def run_pipeline_stream(
    user_request: str,
    task_id: str = "default",
    initial_state: str = "PLANNER",
    preloaded_context: dict = None,
    provider: str = "auto"
):
    """
    Async generator that executes the LangGraph State Machine.
    Instruments execution with Langfuse CallbackHandler for end-to-end tracing.
    Yields structured JSON events for FastAPI SSE streaming.
    """
    langfuse_handler = CallbackHandler()
    config = {
        "configurable": {"thread_id": task_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": task_id,
            "langfuse_user_id": "aditya",
            "langfuse_tags": ["autodev", "langgraph", "multi-agent"],
            "langfuse_trace_name": f"AutoDev Pipeline - {task_id[:8]}"
        }
    }
    preloaded_context = preloaded_context or {}

    # Check if this thread already has an active, halted state
    existing_state = await app_graph.aget_state(config)
    is_resuming = bool(existing_state and existing_state.next)

    try:
        if is_resuming:
            # Graph is halted at human_gate and waiting for resume
            human_feedback = preloaded_context.get("context") or preloaded_context.get("feedback") or ""
            resume_payload = {"approved": False, "feedback": human_feedback}
            stream_input = Command(resume=resume_payload)
        else:
            stream_input = {
                "task_id": task_id,
                "user_request": user_request,
                "current_state": initial_state,
                "saved_plan": preloaded_context.get("saved_plan", ""),
                "saved_code": preloaded_context.get("saved_code", ""),
                "retries": preloaded_context.get("retries", 0),
                "max_retries": 3,
                "status": "RUNNING",
                "iteration_count": 0,
                "token_usage": {},
                "cost_estimate": 0.0,
                "step_history": [],
                "modified_files": [],
                "syntax_errors": [],
                "provider": provider or preloaded_context.get("provider", "auto")
            }

        yield json.dumps({
            "type": "STATE_CHANGE",
            "state": "INITIALIZING",
            "message": "LangGraph state machine started."
        })

        async for chunk in app_graph.astream(stream_input, config=config, stream_mode="updates"):
            for node_name, updates in chunk.items():
                if node_name == "__interrupt__":
                    # Execution suspended at human_gate
                    current_snap = await app_graph.aget_state(config)
                    vals = current_snap.values if current_snap else {}
                    yield json.dumps({
                        "type": "NEEDS_HUMAN_APPROVAL",
                        "code_to_review": vals.get("saved_code", ""),
                        "langgraph_notes": vals.get("review_notes", ""),
                        "context": vals.get("saved_plan", ""),
                        "retries": vals.get("retries", 0),
                        "qa_passed": vals.get("qa_passed", False),
                        "qa_feedback": vals.get("qa_feedback", "")
                    })
                    return

                # Normal node execution update
                yield json.dumps({
                    "type": "STATE_CHANGE",
                    "state": node_name.upper(),
                    "message": f"Phase '{node_name}' completed."
                })

                if node_name == "planner":
                    yield json.dumps({
                        "type": "AGENT_THOUGHT",
                        "agent": "Planner",
                        "content": updates.get("saved_plan", "")
                    })
                elif node_name == "coder":
                    yield json.dumps({
                        "type": "AGENT_THOUGHT",
                        "agent": "Coder",
                        "content": f"Generated code snippet ({len(updates.get('saved_code', ''))} characters)"
                    })
                elif node_name == "tester":
                    test_info = updates.get("test_results", {})
                    passed = updates.get("qa_passed", False)
                    yield json.dumps({
                        "type": "AGENT_THOUGHT",
                        "agent": "QA_Engineer",
                        "content": f"Tests {'PASSED' if passed else 'FAILED'}.\n{test_info.get('output', '')}"
                    })
                elif node_name == "reviewer":
                    yield json.dumps({
                        "type": "AGENT_THOUGHT",
                        "agent": "Reviewer",
                        "content": updates.get("review_notes", "")
                    })

        # Pipeline finished to END
        final_state = await app_graph.aget_state(config)
        final_vals = final_state.values if final_state else {}
        yield json.dumps({
            "type": "PIPELINE_COMPLETE",
            "final_code": final_vals.get("saved_code", "")
        })

    except Exception as exc:
        yield json.dumps({
            "type": "PIPELINE_FAILED",
            "message": str(exc)
        })
    finally:
        # Flush traces to Langfuse
        try:
            get_client().flush()
        except Exception:
            pass


        