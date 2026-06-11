#!/usr/bin/env bash
# run.sh — launch any part of the Setu project
#
# Usage:
#   ./run.sh                         # smoke-test all Sarvam API calls
#   ./run.sh scratch                 # framework-free agent (text query)
#   ./run.sh graph                   # LangGraph agent (text query)
#   ./run.sh app                     # voice app: mic -> agent -> speaker
#   ./run.sh app --agent graph       # voice app with LangGraph agent
#   ./run.sh app --text "..."        # voice app, skip mic, use text
#   ./run.sh app --chat              # multi-turn chat REPL
#   ./run.sh app --demo              # scripted 3-turn demo, no mic needed
#   ./run.sh eval                    # run the eval pipeline
#   ./run.sh mcp                     # start MCP inspector (interactive)

set -euo pipefail
cd "$(dirname "$0")"

# ── environment ───────────────────────────────────────────────────────────────
export PYTHONIOENCODING=utf-8

if [ ! -f ".env" ]; then
  echo "ERROR: .env not found. Copy .env.example to .env and add your SARVAM_API_KEY."
  exit 1
fi

# ── virtual environment ───────────────────────────────────────────────────────
if [ -d ".venv" ]; then
  if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate          # Windows Git Bash / MSYS2
  else
    source .venv/bin/activate              # Linux / macOS
  fi
else
  echo "No .venv found. Creating one and installing dependencies..."
  python -m venv .venv
  if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
  else
    source .venv/bin/activate
  fi
  pip install -q -r requirements.txt
  echo "Dependencies installed."
fi

# ── dispatch ──────────────────────────────────────────────────────────────────
CMD="${1:-smoke}"
shift || true   # remaining args forwarded to the target script

case "$CMD" in
  smoke|test|"")
    echo "Running Sarvam API smoke tests (all 4 functions)..."
    python sarvam_client.py
    ;;
  scratch)
    echo "Running scratch agent (framework-free)..."
    python scratch_agent.py "$@"
    ;;
  graph)
    echo "Running LangGraph agent..."
    python graph_agent.py "$@"
    ;;
  app)
    python app.py "$@"
    ;;
  eval)
    echo "Running eval pipeline..."
    python eval/run_eval.py "$@"
    ;;
  mcp)
    echo "Starting MCP inspector (Ctrl-C to stop)..."
    mcp dev mcp_server.py
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Available: smoke | scratch | graph | app | eval | mcp"
    exit 1
    ;;
esac
