#!/usr/bin/env bash
# setup/download_model.sh
# Downloads Qwen3.8-27B NVFP4 weights and DFlash2 drafter to local HF cache.
# Safe to run multiple times — skips models already cached.
# Run inside tmux if on SSH (~27 GB total).
set -euo pipefail

MODEL_ID="${MODEL_ID:-unsloth/Qwen3.8-27B-NVFP4}"
DRAFTER_ID="${DRAFTER_ID:-z-lab/Qwen3.8-27B-DFlash2}"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_CACHE_DIR="$HF_HOME/hub/models--unsloth--Qwen3.8-27B-NVFP4"
DRAFTER_CACHE_DIR="$HF_HOME/hub/models--z-lab--Qwen3.8-27B-DFlash2"

echo "=== Downloading Qwen3.8-27B NVFP4 + DFlash2 Drafter ==="
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
    echo "✓ DFlash2 drafter already present at $DRAFTER_CACHE_DIR"
    DRAFTER_CACHED=true
  fi
fi

# ── Download Missing Models ───────────────────────────────────────────────────
if [[ "$BASE_CACHED" != "true" ]]; then
  echo "Downloading main model (~24 GB)..."
  uvx hf download "$MODEL_ID"
fi

if [[ "$DRAFTER_CACHED" != "true" ]]; then
  echo "Downloading DFlash2 drafter (~2.7 GB)..."
  uvx hf download "$DRAFTER_ID"
fi

# ── Ensure Drafter Architecture Compatibility with vLLM ────────────────────────
# vLLM model registry registers DFlash under 'DFlashDraftModel'
python3 -c '
import glob, json, os, sys
hf_dir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/.cache/huggingface")
for cfg in glob.glob(f"{hf_dir}/**/*Qwen3.8-27B-DFlash2*/**/config.json", recursive=True):
    real_path = os.path.realpath(cfg)
    try:
        with open(real_path, "r") as f:
            data = json.load(f)
        if "architectures" in data and "DFlash2DraftModel" in data["architectures"]:
            data["architectures"] = ["DFlashDraftModel" if a == "DFlash2DraftModel" else a for a in data["architectures"]]
            with open(real_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✓ Patched {real_path} for vLLM compatibility (DFlash2DraftModel -> DFlashDraftModel)")
    except Exception as e:
        pass
' "$HF_HOME" 2>/dev/null || true

echo
echo "✓ Models verified and ready in $HF_HOME"
echo "Next: bash docker/start.sh"
