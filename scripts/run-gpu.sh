#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"your prompt here\"" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_SCRIPTS="$ROOT_DIR/scripts/ollama"

module load python/3.13.0-gcc-13.1.0-7ypl2

# shellcheck source=ollama/env.sh
source "$OLLAMA_SCRIPTS/env.sh"
"$OLLAMA_SCRIPTS/pull-model.sh"

cd "$ROOT_DIR"
source .venv/bin/activate
python -m app "$@"
