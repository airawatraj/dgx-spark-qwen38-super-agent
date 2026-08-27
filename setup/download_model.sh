#!/usr/bin/env bash
# setup/download_model.sh
# Downloads Qwen3.8-27B NVFP4 weights and DSpark drafter to local HF cache.
# Idempotent — skips models already present. Set FORCE_DOWNLOAD=1 to re-download.
# Run inside tmux if on SSH (~27 GB total).
set -euo pipefail

MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead}"
DRAFTER_ID="${DRAFTER_ID:-RadixArk/Qwen3.8-27B-DSpark}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

repo_cache_dir() {
  local repo_id="$1"
  local repo_dir="models--${repo_id//\/\/--}"
  if [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
    printf '%s\n' "$HUGGINGFACE_HUB_CACHE/$repo_dir"
    return
  fi
  if [[ -n "${HF_HOME:-}" ]]; then
    printf '%s\n' "$HF_HOME/hub/$repo_dir"
    return
  fi
  printf '%s\n' "$HOME/.cache/huggingface/hub/$repo_dir"
}

repo_cache_exists() {
  local candidate
  candidate="$(repo_cache_dir "$1")"
  if [[ -d "$candidate" ]] && find "$candidate" -type f -print -quit | grep -q .; then
    echo "$candidate"
    return 0
  fi
  return 1
}

download_repo() {
  local repo_id="$1"
  local label="$2"
  echo
  echo "=== Downloading $label ==="
  echo "  Repo: $repo_id"
  if [[ "$FORCE_DOWNLOAD" != "1" ]]; then
    if found_dir="$(repo_cache_exists "$repo_id")"; then
      echo "  Already cached — skipping."
      echo "  $found_dir"
      return
    fi
  fi
  uvx hf download "$repo_id"
}

echo "=== Qwen3.8-27B NVFP4 + DSpark Drafter ==="
echo "  Model:   $MODEL_ID"
echo "  Drafter: $DRAFTER_ID"
echo

if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. May fail for gated models."
fi

download_repo "$MODEL_ID"   "NVFP4 BF16-LMHead main model (~24 GB)"
download_repo "$DRAFTER_ID" "DSpark drafter (~2.7 GB)"

echo
echo "Download complete. Next: bash docker/start.sh"
