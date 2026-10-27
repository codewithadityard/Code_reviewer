import streamlit as st
import requests
import json
import time

st.set_page_config(
    page_title="AutoDev - Multi-Agent Code Reviewer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "task_id" not in st.session_state:
    st.session_state.task_id = None
if "status" not in st.session_state:
    st.session_state.status = "IDLE"
if "saved_plan" not in st.session_state:
    st.session_state.saved_plan = ""
if "saved_code" not in st.session_state:
    st.session_state.saved_code = ""
if "qa_output" not in st.session_state:
    st.session_state.qa_output = ""
if "qa_passed" not in st.session_state:
    st.session_state.qa_passed = None
if "review_notes" not in st.session_state:
    st.session_state.review_notes = ""
if "retries" not in st.session_state:
    st.session_state.retries = 0
if "logs" not in st.session_state:
    st.session_state.logs = []


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Quick Templates
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    backend_url = st.text_input(
        "FastAPI Backend URL",
        value="http://127.0.0.1:8000",
        key="backend_url_input",
        help="Address of the running FastAPI AutoDev backend"
    ).rstrip("/")

    # Health check
    if st.button("📡 Test Backend Connection", key="btn_test_backend"):
        try:
            res = requests.get(f"{backend_url}/docs", timeout=3)
            if res.status_code == 200:
                st.success("Connected to FastAPI Backend!")
            else:
                st.warning(f"Backend responded with code {res.status_code}")
        except Exception as e:
            st.error(f"Could not connect: {e}")

    st.markdown("---")
    st.subheader("🧠 Model Engine")
    provider_choice = st.radio(
        "Execution Provider",
        options=["Auto (Groq + Ollama Fallback)", "Ollama (100% Local, Zero Limits)", "Groq Cloud Only"],
        index=0,
        key="provider_choice_radio",
        help="In 'Auto' mode, Groq is used for high speed and falls back to local Ollama if rate or token limits hit. In 'Ollama' mode, 100% of execution stays local."
    )
    provider_map = {
        "Auto (Groq + Ollama Fallback)": "auto",
        "Ollama (100% Local, Zero Limits)": "ollama",
        "Groq Cloud Only": "groq"
    }
    selected_provider = provider_map.get(provider_choice, "auto")

    st.markdown("---")
    st.subheader("💡 Sample Prompts")
    sample_1 = "Write a thread-safe rate limiter class in Python with sliding window counter."
    sample_2 = "Create an LRU Cache class with get and put methods having O(1) complexity."
    sample_3 = "Implement a binary search tree with insert, search, and in-order traversal."

    if st.button("📌 Rate Limiter", key="btn_sample_rate_limiter"):
        st.session_state.user_prompt_input = sample_1
        st.rerun()
    if st.button("📌 LRU Cache", key="btn_sample_lru_cache"):
        st.session_state.user_prompt_input = sample_2
        st.rerun()
    if st.button("📌 Binary Search Tree", key="btn_sample_bst"):
        st.session_state.user_prompt_input = sample_3
        st.rerun()

    st.markdown("---")
    st.caption("AutoDev v2.0 • Powered by LangGraph (Groq + Ollama Fallback)")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("🤖 AutoDev — Multi-Agent SDLC & Code Reviewer")
st.markdown(
    "Autonomous engineering team powered by **LangGraph**: "
    "`Planner` ➔ `Coder` ➔ `QA Engineer (Pytest)` ➔ `Reviewer` ➔ `Human Gate`"
)

# Pipeline Flow Status Indicator
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="1. Planner", value="Active" if st.session_state.saved_plan else "Idle")
with col2:
    st.metric(label="2. Coder", value="Generated" if st.session_state.saved_code else "Idle")
with col3:
    qa_val = "Idle"
    if st.session_state.qa_passed is True:
        qa_val = "Passed ✅"
    elif st.session_state.qa_passed is False:
        qa_val = f"Failed ({st.session_state.retries} retries)"
    st.metric(label="3. QA Tests", value=qa_val)
with col4:
    st.metric(label="4. Reviewer", value="Audited" if st.session_state.review_notes else "Idle")
with col5:
    st.metric(label="Status", value=st.session_state.status)

st.markdown("---")


# -----------------------------------------------------------------------------
# Helper: Stream Pipeline Execution
# -----------------------------------------------------------------------------
def stream_pipeline(task_id: str, backend: str):
    """Connects to FastAPI SSE stream and updates session state in real time."""
    stream_url = f"{backend}/api/tasks/{task_id}/stream"
    status_placeholder = st.empty()
    progress_bar = st.progress(10, text="Initializing LangGraph execution...")

    try:
        with requests.get(stream_url, stream=True, timeout=300) as response:
            if response.status_code != 200:
                st.error(f"Stream error: HTTP {response.status_code}")
                return

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue

                json_str = decoded[6:]
                try:
                    event = json.loads(json_str)
                except Exception:
                    continue

                event_type = event.get("type")

                # Handle events
                if event_type == "STATE_CHANGE":
                    state_name = event.get("state", "")
                    msg = event.get("message", "")
                    st.session_state.logs.append(f"🔄 **[{state_name}]**: {msg}")

                    if state_name == "PLANNER":
                        progress_bar.progress(25, text="Architectural Planning in progress...")
                    elif state_name == "CODER":
                        progress_bar.progress(50, text="Writing and fixing code...")
                    elif state_name == "QA" or state_name == "TESTER":
                        progress_bar.progress(70, text="Running automated pytest suites...")
                    elif state_name == "REVIEW" or state_name == "REVIEWER":
                        progress_bar.progress(90, text="Performing code optimization review...")

                elif event_type == "AGENT_THOUGHT":
                    agent = event.get("agent", "Agent")
                    content = event.get("content", "")
                    st.session_state.logs.append(f"💬 **{agent}**: {content[:150]}...")

                    if agent == "Planner":
                        st.session_state.saved_plan = content
                    elif agent == "QA_Engineer":
                        st.session_state.qa_output = content
                        st.session_state.qa_passed = "PASSED" in content
                    elif agent == "Reviewer":
                        st.session_state.review_notes = content

                elif event_type == "NEEDS_HUMAN_APPROVAL":
                    progress_bar.progress(95, text="Paused: Awaiting Human-in-the-Loop review")
                    st.session_state.status = "WAITING_FOR_HUMAN"
                    st.session_state.saved_code = event.get("code_to_review", "")
                    st.session_state.review_notes = event.get("langgraph_notes", "")
                    st.session_state.saved_plan = event.get("context", "")
                    st.session_state.retries = event.get("retries", 0)
                    if "qa_passed" in event:
                        st.session_state.qa_passed = event.get("qa_passed")
                    if event.get("qa_feedback"):
                        st.session_state.qa_output = event.get("qa_feedback")
                    st.session_state.logs.append("⏸️ **Pipeline Paused**: Waiting for human approval.")
                    st.rerun()

                elif event_type == "PIPELINE_COMPLETE":
                    progress_bar.progress(100, text="Pipeline Completed Successfully!")
                    st.session_state.status = "FINISHED"
                    if event.get("final_code"):
                        st.session_state.saved_code = event.get("final_code")
                    st.session_state.logs.append("✅ **Pipeline Complete!**")
                    st.rerun()

                elif event_type == "PIPELINE_FAILED":
                    st.session_state.status = "ABORTED"
                    st.error(f"Pipeline Failed: {event.get('message')}")
                    st.session_state.logs.append(f"❌ **Failed**: {event.get('message')}")
                    st.rerun()

    except Exception as exc:
        st.error(f"Error during streaming: {exc}")


# -----------------------------------------------------------------------------
# Input Prompt Area
# -----------------------------------------------------------------------------
if "user_prompt_input" not in st.session_state:
    st.session_state.user_prompt_input = "Write a Python class for a resilient Token Bucket rate limiter."

user_prompt = st.text_area(
    "What do you want the AI software team to build?",
    key="user_prompt_input",
    height=120,
    placeholder="Describe the algorithm, feature, or service you want implemented..."
)

col_run, col_reset = st.columns([1, 5])
with col_run:
    start_clicked = st.button("🚀 Start Pipeline", key="btn_start_pipeline", type="primary", use_container_width=True)
with col_reset:
    if st.button("🔄 Reset All", key="btn_reset_all", use_container_width=False):
        app_keys = [
            "task_id", "status", "saved_plan", "saved_code",
            "qa_output", "qa_passed", "review_notes", "retries", "logs", "user_prompt_input"
        ]
        for key in app_keys:
            st.session_state.pop(key, None)
        st.rerun()

if start_clicked:
    if not user_prompt.strip():
        st.warning("Please enter a requirement or prompt.")
    else:
        with st.spinner("Creating task on FastAPI backend..."):
            try:
                res = requests.post(
                    f"{backend_url}/api/tasks/create",
                    json={"prompt": user_prompt, "provider": selected_provider},
                    timeout=10
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.task_id = data.get("task_id")
                    st.session_state.status = "RUNNING"
                    st.session_state.saved_plan = ""
                    st.session_state.saved_code = ""
                    st.session_state.qa_output = ""
                    st.session_state.review_notes = ""
                    st.session_state.logs = [f"🚀 Task created with ID: `{st.session_state.task_id}`"]
                    st.rerun()
                else:
                    st.error(f"Failed to create task: HTTP {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# If task is RUNNING, trigger stream
if st.session_state.status == "RUNNING" and st.session_state.task_id:
    stream_pipeline(st.session_state.task_id, backend_url)


# -----------------------------------------------------------------------------
# Human-in-the-Loop Action Panel (When Waiting for Human)
# -----------------------------------------------------------------------------
if st.session_state.status == "WAITING_FOR_HUMAN":
    if st.session_state.qa_passed is False:
        st.error(f"### ❌ Automated Tests FAILED ({st.session_state.retries} retries exhausted)")
        st.warning(
            "The Coder Agent could not pass the QA tests within the retry limit. "
            "Please check the **🧪 QA Test Results** tab below, inspect the failure, and use **Option B** below to provide feedback."
        )
    else:
        st.warning("### ⚠️ Human-in-the-Loop Gate: Review & Approval Required")
        st.info("The code has been generated, tested with pytest, and audited by the Reviewer Agent. Please inspect the code below and approve or reject it.")

    col_app, col_rej = st.columns(2)

    with col_app:
        st.markdown("#### Option A: Approve Code")
        st.caption("Mark code as approved and finalize the pipeline.")
        if st.button("✅ Approve & Finalize", key="btn_approve_finalize", type="primary", use_container_width=True):
            try:
                res = requests.post(f"{backend_url}/api/tasks/{st.session_state.task_id}/approve", timeout=10)
                if res.status_code == 200:
                    st.session_state.status = "FINISHED"
                    st.success("Task approved successfully!")
                    st.rerun()
                else:
                    st.error(f"Approve error: {res.text}")
            except Exception as e:
                st.error(f"Failed to approve: {e}")

    with col_rej:
        st.markdown("#### Option B: Reject with Feedback")
        st.caption("Send specific corrective instructions back to the Coder Agent.")
        feedback_text = st.text_input("Feedback / Instructions for revision:", key="feedback_text_input", placeholder="e.g. Add validation for negative values...")
        if st.button("🔁 Send Feedback & Re-run", key="btn_reject_rerun", use_container_width=True):
            if not feedback_text.strip():
                st.warning("Please provide feedback before rejecting.")
            else:
                try:
                    res = requests.post(
                        f"{backend_url}/api/tasks/{st.session_state.task_id}/reject",
                        json={"feedback": feedback_text},
                        timeout=10
                    )
                    if res.status_code == 200:
                        st.session_state.status = "RUNNING"
                        st.session_state.logs.append(f"🔁 Rejected by human: '{feedback_text}'. Routing back to Coder.")
                        st.rerun()
                    else:
                        st.error(f"Reject error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to reject: {e}")

    st.markdown("---")


# -----------------------------------------------------------------------------
# Completed Pipeline Celebration
# -----------------------------------------------------------------------------
if st.session_state.status == "FINISHED":
    st.success("🎉 **Pipeline Complete! Your production-ready code is approved and ready.**")
    st.download_button(
        label="📥 Download solution.py",
        data=st.session_state.saved_code,
        file_name="solution.py",
        key="btn_download_solution",
        mime="text/x-python",
        type="primary"
    )
    st.markdown("---")


# -----------------------------------------------------------------------------
# Deep Dive Artifact Tabs
# -----------------------------------------------------------------------------
tab_code, tab_plan, tab_qa, tab_review, tab_logs = st.tabs([
    "💻 Generated Code",
    "📋 Architectural Plan",
    "🧪 QA Test Results",
    "🔍 Review Audit",
    "📜 Live Thought Log"
])

with tab_code:
    if st.session_state.saved_code:
        st.code(st.session_state.saved_code, language="python", line_numbers=True)
    else:
        st.info("Code will appear here once the Coder Agent finishes implementation.")

with tab_plan:
    if st.session_state.saved_plan:
        st.markdown(st.session_state.saved_plan)
    else:
        st.info("Architectural blueprint will appear here once the Planner Agent runs.")

with tab_qa:
    if st.session_state.qa_output:
        st.markdown(f"**Status**: {'✅ Tests Passed' if st.session_state.qa_passed else '❌ Tests Failed'}")
        st.code(st.session_state.qa_output, language="bash")
    else:
        st.info("Pytest execution results will appear here once the QA Agent validates the code.")

with tab_review:
    if st.session_state.review_notes:
        st.markdown(st.session_state.review_notes)
    else:
        st.info("Code quality and refactoring notes will appear here once the Reviewer Agent audits the code.")

with tab_logs:
    if st.session_state.logs:
        for log in reversed(st.session_state.logs):
            st.markdown(log)
    else:
        st.info("Agent execution logs will appear here during pipeline runs.")
