#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PID_FILE="${OLLAMA_PID_FILE:-$HOME/.ollama/ollama.pid}"
LOG_FILE="${OLLAMA_LOG_FILE:-$HOME/.ollama/ollama.log}"

mkdir -p "$(dirname "$PID_FILE")" "$OLLAMA_MODELS"

if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already running at ${OLLAMA_HOST}"
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Ollama is already running (pid $(cat "$PID_FILE"))"
  exit 0
fi

echo "Starting Ollama server..."
nohup ollama serve >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

for _ in $(seq 1 30); do
  if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ollama ready at ${OLLAMA_HOST} (pid $(cat "$PID_FILE"))"
    echo "Log: $LOG_FILE"
    exit 0
  fi
  sleep 1
done

echo "Ollama failed to start. Check $LOG_FILE" >&2
exit 1
