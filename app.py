from dotenv import load_dotenv
load_dotenv()

import uuid
import json
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langgraph.types import Command
from langfuse.langchain import CallbackHandler
from langfuse import get_client

from manager import run_pipeline_stream, app_graph
from api_gateway import global_gateway


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when FastAPI starts its event loop to initialize
    the global API gateway queues, locks, and concurrency limits.
    """
    await global_gateway._initialize()
    yield
    print("Gateway is active")


app = FastAPI(title="AutoDev Multi-Agent API (LangGraph + Langfuse)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

task_db: Dict[str, Dict[str, Any]] = {}


class CreateTaskRequest(BaseModel):
    prompt: str
    provider: Optional[str] = "auto"


class RejectRequest(BaseModel):
    feedback: str


# -----------------------------------------------------------------------------
# 1. Create a New Task
# -----------------------------------------------------------------------------
@app.post("/api/tasks/create")
async def create_task(request: CreateTaskRequest):
    task_id = str(uuid.uuid4())
    task_db[task_id] = {
        "task_id": task_id,
        "prompt": request.prompt,
        "provider": request.provider or "auto",
        "status": "CREATED",
        "initial_state": "PLANNER",
        "preloaded_context": {},
        "final_code": None
    }
    return {"task_id": task_id, "status": "CREATED", "message": "Task created successfully"}


# -----------------------------------------------------------------------------
# 2. Stream Task Pipeline over SSE
# -----------------------------------------------------------------------------
@app.get("/api/tasks/{task_id}/stream")
async def stream_task_pipeline(task_id: str):
    """
    Streams the LangGraph execution events over Server-Sent Events (SSE).
    Supports initial execution as well as resuming after human feedback.
    Traces execution in real-time to Langfuse.
    """
    if task_id not in task_db:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    task = task_db[task_id]
    task["status"] = "RUNNING"

    async def event_generator():
        user_prompt = task["prompt"]
        provider = task.get("provider", "auto")
        initial_state = task.get("initial_state", "PLANNER")
        preloaded_context = task.get("preloaded_context", {})

        async for event_raw in run_pipeline_stream(
            user_request=user_prompt,
            task_id=task_id,
            initial_state=initial_state,
            preloaded_context=preloaded_context,
            provider=provider
        ):
            event = json.loads(event_raw)
            event_type = event.get("type")

            if event_type == "NEEDS_HUMAN_APPROVAL":
                task["status"] = "WAITING_FOR_HUMAN"
                task["saved_code"] = event.get("code_to_review", "")
                task["preloaded_context"] = {
                    "saved_plan": event.get("context", ""),
                    "saved_code": event.get("code_to_review", ""),
                    "context": event.get("context", ""),
                    "retries": event.get("retries", 0)
                }

            elif event_type == "PIPELINE_COMPLETE":
                task["status"] = "FINISHED"
                task["final_code"] = event.get("final_code")

            elif event_type == "PIPELINE_FAILED":
                task["status"] = "ABORTED"

            yield f"data: {event_raw}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# 3. Approve Task
# -----------------------------------------------------------------------------
@app.post("/api/tasks/{task_id}/approve")
async def approve_task(task_id: str):
    """Called by the user to mark the generated code as approved in LangGraph."""
    if task_id not in task_db:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    task = task_db[task_id]
    if task["status"] != "WAITING_FOR_HUMAN":
        raise HTTPException(status_code=400, detail=f"Task is not waiting for approval (Status: {task['status']})")

    # Resume halted LangGraph thread if interrupted, instrumented with Langfuse
    langfuse_handler = CallbackHandler()
    config = {
        "configurable": {"thread_id": task_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": task_id,
            "langfuse_tags": ["autodev", "human-approval"],
            "langfuse_trace_name": f"Approve Task - {task_id[:8]}"
        }
    }
    try:
        graph_state = await app_graph.aget_state(config)
        if graph_state and graph_state.next:
            await app_graph.ainvoke(Command(resume={"approved": True}), config=config)
            get_client().flush()
    except Exception as exc:
        print(f"[APPROVE WARNING]: Could not resume LangGraph: {exc}")

    task["status"] = "FINISHED"
    return {
        "status": "SUCCESS",
        "message": "Task approved successfully!",
        "final_code": task.get("saved_code")
    }


# -----------------------------------------------------------------------------
# 4. Reject Task with Feedback
# -----------------------------------------------------------------------------
@app.post("/api/tasks/{task_id}/reject")
async def reject_task(task_id: str, request: RejectRequest):
    """
    Called when human rejects code. Updates context with feedback
    and resumes LangGraph thread back to the coder node.
    """
    if task_id not in task_db:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    task = task_db[task_id]
    if task["status"] != "WAITING_FOR_HUMAN":
        raise HTTPException(status_code=400, detail="Task is not waiting for approval")

    prev_context = task.get("preloaded_context", {})
    saved_plan = prev_context.get("saved_plan", "")
    saved_code = prev_context.get("saved_code", task.get("saved_code", ""))

    new_context = (
        f"Plan: {saved_plan}\n"
        f"Your Code:\n{saved_code}\n"
        f"Human Feedback / Rejection Reason: {request.feedback}\n"
        f"Fix the code based on the feedback."
    )

    task["status"] = "READY_TO_RESUME"
    task["initial_state"] = "CODER"
    task["preloaded_context"] = {
        "saved_plan": saved_plan,
        "saved_code": saved_code,
        "context": new_context,
        "feedback": request.feedback,
        "retries": prev_context.get("retries", 0)
    }

    # Resume halted LangGraph thread with rejection feedback
    langfuse_handler = CallbackHandler()
    config = {
        "configurable": {"thread_id": task_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": task_id,
            "langfuse_tags": ["autodev", "human-rejection"],
            "langfuse_trace_name": f"Reject Task - {task_id[:8]}"
        }
    }
    try:
        graph_state = await app_graph.aget_state(config)
        if graph_state and graph_state.next:
            await app_graph.ainvoke(
                Command(resume={"approved": False, "feedback": request.feedback}),
                config=config
            )
            get_client().flush()
    except Exception as exc:
        print(f"[REJECT WARNING]: Could not resume LangGraph directly: {exc}")

    return {
        "status": "SUCCESS",
        "message": "Feedback recorded. Re-connect to the stream endpoint to view resumed execution."
    }


# -----------------------------------------------------------------------------
# 5. Task Status & Direct LangGraph Resume
# -----------------------------------------------------------------------------
@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Returns the current state and data for a given task."""
    if task_id not in task_db:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return task_db[task_id]


@app.post("/review/resume/{thread_id}")
async def resume_review(thread_id: str, payload: dict):
    """Direct LangGraph checkpointer resume endpoint for GitHub review threads."""
    langfuse_handler = CallbackHandler()
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["autodev", "github-review-resume"],
            "langfuse_trace_name": f"Resume Review - {thread_id[:8]}"
        }
    }
    state = await app_graph.aget_state(config)
    if not state.next:
        raise HTTPException(status_code=400, detail="Graph is not awaiting interruption/input.")

    resumed_run = await app_graph.ainvoke(
        Command(resume=payload),
        config=config
    )
    get_client().flush()
    return {"status": "resumed", "state": resumed_run}