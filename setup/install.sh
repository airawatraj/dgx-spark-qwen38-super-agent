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

echo "[4/4] Checking SGLang image availability..."
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:qwen38-27b}"
if docker image inspect "$SGLANG_IMAGE" >/dev/null 2>&1; then
  echo "  SGLang image already present: $SGLANG_IMAGE"
else
  echo "  SGLang image not found locally: $SGLANG_IMAGE"
  echo "  It will be pulled automatically on first start (~8–12 GB)."
  echo "  To pull now: docker pull $SGLANG_IMAGE"
fi

echo
echo "Setup check complete."
echo "Next: bash setup/download_model.sh"
echo "      bash docker/start.sh   (SGLang DSpark default)"
echo "      bash docker/start-vllm.sh  (vLLM fallback — 512K ctx)"
