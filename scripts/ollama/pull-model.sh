#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

MODEL="${OLLAMA_MODEL:-qwen3:4b}"

"$SCRIPT_DIR/start.sh"
echo "Pulling model: $MODEL"
ollama pull "$MODEL"
ollama list
