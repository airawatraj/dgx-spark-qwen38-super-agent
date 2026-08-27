#!/usr/bin/env bash
set -euo pipefail

echo "=== DGX Spark Qwen3.8-27B Setup Check ==="

echo "[1/4] Checking Docker..."
docker version --format 'Docker {{.Server.Version}}' >/dev/null
docker version --format '  Server {{.Server.Version}}'

echo "[2/4] Checking uv / uvx..."
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is not installed."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
uv --version
uvx --version

echo "[3/4] Checking Hugging Face auth..."
uvx hf auth whoami

echo "[4/4] Checking Docker container image..."
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
if docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
  echo "  vLLM image already present: $VLLM_IMAGE"
else
  echo "  vLLM image not found locally: $VLLM_IMAGE"
  echo "  It will be pulled automatically on first start (~10 GB)."
  echo "  To pull now: docker pull $VLLM_IMAGE"
fi

echo
echo "Setup check complete."
echo "Next: bash setup/download_model.sh"
echo "      bash docker/start.sh         (vLLM DFlash2 — ~38–54 tok/s default)"
echo "      bash docker/start-sglang.sh  (SGLang DSpark — 262K)"
echo "      bash docker/start-vllm.sh    (vLLM eager fallback — 512K ctx)"
