#!/usr/bin/env bash
# start-vllm.sh — vLLM fallback / rollback script
#
# This is the original vLLM launch preserved as a rollback path.
# Use when you need:
#   - 512K context window (DSpark caps at 262K)
#   - Multimodal inputs (image × 4, video × 1)
#   - vLLM-specific debugging
#
# For the default high-speed SGLang DSpark stack, use docker/start.sh
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_ID="${MODEL_ID:-unsloth/Qwen3.8-27B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

echo "=== Qwen 3.8-27B vLLM preflight (rollback mode) ==="
echo "  Model ID:               $MODEL_ID"
echo "  Served model name:      $SERVED_MODEL_NAME"
echo "  Container:              $CONTAINER_NAME"
echo "  Image:                  $VLLM_IMAGE"
echo "  Port:                   $VLLM_PORT"
echo "  Max model len:          $MAX_MODEL_LEN  (512K — vLLM only)"
echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "  Max batched tokens:     $MAX_NUM_BATCHED_TOKENS"
echo "  HF cache:               $HF_CACHE_DIR"
echo

echo "Removing any existing container named $CONTAINER_NAME ..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting spark-brain (vLLM) ..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -p "${VLLM_PORT}:8000" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "$VLLM_IMAGE" \
  "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 4, "video": 1}'

echo
echo "Container started. Follow logs with:"
echo "  docker logs -f $CONTAINER_NAME"
echo "Ready check:"
echo "  curl -sf http://localhost:$VLLM_PORT/health && echo OK"
echo
echo "Speed note: vLLM delivers ~14 tok/s. For ~51 tok/s, use: bash docker/start.sh"
