#!/usr/bin/env bash
# setup/download_model.sh
# Downloads Qwen3.8-27B NVFP4 weights to local HF cache.
# Safe to run multiple times — skips models already cached.
# Run inside tmux if on SSH (~24 GB total).
set -euo pipefail

MODEL_ID="${MODEL_ID:-unsloth/Qwen3.8-27B-NVFP4}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_CACHE_DIR="$HF_HOME/hub/models--unsloth--Qwen3.8-27B-NVFP4"

echo "=== Downloading Qwen3.8-27B NVFP4 (with built-in MTP) ==="
echo "  Main model:  $MODEL_ID"
echo "  HF cache:    $HF_HOME"
echo

if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. May fail for gated models."
  echo
fi

# ── Check Main Model Cache ────────────────────────────────────────────────────
if [[ -d "$MODEL_CACHE_DIR/snapshots" ]]; then
  SNAP_COUNT=$(find "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [[ "$SNAP_COUNT" -gt 0 ]]; then
    echo "✓ Main model already present at $MODEL_CACHE_DIR"
    echo "  All weights present. Skipping download."
    echo
    echo "Next: bash docker/start.sh"
    exit 0
  fi
fi

# ── Download Model ────────────────────────────────────────────────────────────
echo "Downloading main model (~24 GB)..."
uvx hf download "$MODEL_ID"

echo
echo "✓ Models verified and ready in $HF_HOME"
echo "Next: bash docker/start.sh"
