#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PID_FILE="${OLLAMA_PID_FILE:-$HOME/.ollama/ollama.pid}"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Stopped Ollama (pid $pid)"
  fi
  rm -f "$PID_FILE"
else
  pkill -u "$USER" -x ollama 2>/dev/null && echo "Stopped Ollama" || echo "Ollama is not running"
fi
