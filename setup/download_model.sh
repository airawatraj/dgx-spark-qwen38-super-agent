#!/usr/bin/env bash
# setup/download_model.sh
# Downloads the Qwen3.8-27B NVFP4 weights and DSpark drafter to local HF cache.
# Run this before docker/start.sh — avoids the download happening cold inside Docker.
# TIP: run inside tmux if on SSH (total ~27 GB, takes 20-30 min on a good connection)
set -euo pipefail

MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead}"
DRAFTER_ID="${DRAFTER_ID:-RadixArk/Qwen3.8-27B-DSpark}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

echo "=== Downloading Qwen3.8-27B NVFP4 + DSpark Drafter ==="
echo "  Main model:  $MODEL_ID  (~24 GB)"
echo "  Drafter:     $DRAFTER_ID  (~2.7 GB)"
echo "  Destination: $HF_CACHE_DIR"
echo

if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

mkdir -p "$HF_CACHE_DIR"

echo "[1/2] Downloading main model: $MODEL_ID ..."
HF_HOME="$HF_CACHE_DIR" uvx huggingface_hub download "$MODEL_ID"

echo
echo "[2/2] Downloading DSpark drafter: $DRAFTER_ID ..."
HF_HOME="$HF_CACHE_DIR" uvx huggingface_hub download "$DRAFTER_ID"

echo
echo "Download complete. Start the server with:"
echo "  bash docker/start.sh"
