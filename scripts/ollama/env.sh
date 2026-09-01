#!/usr/bin/env bash
# Source this file before running Ollama or the app on Libra GPU nodes.

INSTALL_DIR="${OLLAMA_INSTALL_DIR:-$HOME/.local/ollama}"

export PATH="$INSTALL_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
