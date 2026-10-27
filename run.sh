#!/usr/bin/env bash
# Exit script if any command fails
set -e

# Trap exit signal to cleanly kill both processes when pressing Ctrl+C
trap 'kill $(jobs -p) 2>/dev/null' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/autodev-env/bin/python"
UVICORN="$SCRIPT_DIR/autodev-env/bin/uvicorn"
STREAMLIT="$SCRIPT_DIR/autodev-env/bin/streamlit"

echo "=========================================================="
echo "🚀 Launching AutoDev Platform (LangGraph + Streamlit)"
echo "=========================================================="

echo "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000..."
"$PYTHON" -m uvicorn app:app --host 127.0.0.1 --port 8000 &

# Wait for backend to be ready
echo "Waiting for API gateway to activate..."
sleep 2

echo "[2/2] Starting Streamlit Web Frontend on http://localhost:8501..."
"$PYTHON" -m streamlit run frontend.py --server.port 8501
