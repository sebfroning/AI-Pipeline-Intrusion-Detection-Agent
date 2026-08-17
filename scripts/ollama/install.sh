#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${OLLAMA_INSTALL_DIR:-$HOME/.local/ollama}"
ARCHIVE="$HOME/tmp/ollama-linux-amd64.tar.zst"

mkdir -p "$INSTALL_DIR" "$HOME/.ollama/models" "$HOME/tmp"

if [[ -x "$INSTALL_DIR/bin/ollama" ]]; then
  echo "Ollama already installed at $INSTALL_DIR/bin/ollama"
  "$INSTALL_DIR/bin/ollama" --version
  exit 0
fi

echo "Downloading Ollama..."
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst -o "$ARCHIVE"

echo "Extracting to $INSTALL_DIR..."
tar --zstd -xf "$ARCHIVE" -C "$INSTALL_DIR"

echo "Installed:"
"$INSTALL_DIR/bin/ollama" --version
