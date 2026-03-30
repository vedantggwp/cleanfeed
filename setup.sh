#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Project Resonance setup for macOS Apple Silicon (MPS)..."

if [[ -f pyproject.toml ]]; then
  echo "pyproject.toml already exists, skipping uv init."
else
  echo "Initializing uv project..."
  uv init --bare --no-workspace
fi

echo "Adding torch..."
uv add 'torch>=2.1.0'

echo "Adding torchaudio..."
uv add 'torchaudio>=2.1.0'

echo "Adding torchcodec (audio backend for torchaudio 2.9+)..."
uv add torchcodec

echo "Pinning numpy<2.0 (resemble-enhance requires NumPy 1.x)..."
uv add 'numpy<2.0'

echo "Adding resemble-enhance from GitHub..."
uv add "resemble-enhance @ git+https://github.com/resemble-ai/resemble-enhance.git"

echo "Adding gradio..."
uv add gradio

echo "Setup complete."
