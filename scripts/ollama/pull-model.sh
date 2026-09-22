#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

MODEL="${OLLAMA_MODEL:-qwen3:4b}"
EMBED_MODEL="${MEMORY_EMBED_MODEL-nomic-embed-text}"

"$SCRIPT_DIR/start.sh"
echo "Pulling model: $MODEL"
ollama pull "$MODEL"
if [[ -n "$EMBED_MODEL" ]]; then
  echo "Pulling embedding model: $EMBED_MODEL"
  ollama pull "$EMBED_MODEL"
fi
ollama list
