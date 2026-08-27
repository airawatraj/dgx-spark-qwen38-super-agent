#!/usr/bin/env bash
# setup/download_model.sh
# Downloads Qwen3.8-27B NVFP4 weights and DSpark drafter to local HF cache.
# Safe to run multiple times — skips models already cached.
# Run inside tmux if on SSH (~27 GB total).
set -euo pipefail

MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead}"
DRAFTER_ID="${DRAFTER_ID:-RadixArk/Qwen3.8-27B-DSpark}"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_CACHE_DIR="$HF_HOME/hub/models--RadixArk--Qwen3.8-27B-NVFP4-BF16-LMHead"
DRAFTER_CACHE_DIR="$HF_HOME/hub/models--RadixArk--Qwen3.8-27B-DSpark"

echo "=== Downloading Qwen3.8-27B NVFP4 + DSpark Drafter ==="
echo "  Main model:  $MODEL_ID"
echo "  Drafter:     $DRAFTER_ID"
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
BASE_CACHED=false
if [[ -d "$MODEL_CACHE_DIR/snapshots" ]]; then
  SNAP_COUNT=$(find "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [[ "$SNAP_COUNT" -gt 0 ]]; then
    echo "✓ Main model already present at $MODEL_CACHE_DIR"
    BASE_CACHED=true
  fi
fi

# ── Check Drafter Cache ───────────────────────────────────────────────────────
DRAFTER_CACHED=false
if [[ -d "$DRAFTER_CACHE_DIR/snapshots" ]]; then
  DRAFT_SNAP_COUNT=$(find "$DRAFTER_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [[ "$DRAFT_SNAP_COUNT" -gt 0 ]]; then
    echo "✓ DSpark drafter already present at $DRAFTER_CACHE_DIR"
    DRAFTER_CACHED=true
  fi
fi

if [[ "$BASE_CACHED" == "true" && "$DRAFTER_CACHED" == "true" ]]; then
  echo "  All weights present. Skipping download."
  echo
  echo "Next: bash docker/start.sh"
  exit 0
fi

# ── Download Missing Models ───────────────────────────────────────────────────
if [[ "$BASE_CACHED" != "true" ]]; then
  echo "Downloading main model (~24 GB)..."
  uvx hf download "$MODEL_ID"
fi

if [[ "$DRAFTER_CACHED" != "true" ]]; then
  echo "Downloading DSpark drafter (~2.7 GB)..."
  uvx hf download "$DRAFTER_ID"
fi

echo
echo "✓ Download complete. Models cached in $HF_HOME"
echo "Next: bash docker/start.sh"
