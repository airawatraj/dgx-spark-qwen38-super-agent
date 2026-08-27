#!/usr/bin/env bash
# start.sh — vLLM DFlash2 (high-speed production stack)
#
# Runs unsloth/Qwen3.8-27B-NVFP4 (4-bit lm_head) with z-lab/Qwen3.8-27B-DFlash2
# speculative decoding (k=8) on a single NVIDIA DGX Spark node.
# Expected speed: ~38–54+ tok/s single stream | 100/100 Tool-Eval intelligence
set -euo pipefail

# ── Configuration (override via env vars before calling this script) ──────────
CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_ID="${MODEL_ID:-unsloth/Qwen3.8-27B-NVFP4}"
DFLASH_DRAFT="${DFLASH_DRAFT:-z-lab/Qwen3.8-27B-DFlash2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-8}"

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "=== Qwen3.8-27B vLLM DFlash2 preflight ==="
echo "  Model:                  $MODEL_ID (4-bit lm_head)"
echo "  Drafter:                $DFLASH_DRAFT (k=$NUM_SPECULATIVE_TOKENS)"
echo "  Served model name:      $SERVED_MODEL_NAME"
echo "  Container:              $CONTAINER_NAME"
echo "  Image:                  $VLLM_IMAGE"
echo "  Port:                   $VLLM_PORT"
echo "  Context window:         $MAX_MODEL_LEN (262K)"
echo "  GPU memory util:        $GPU_MEMORY_UTILIZATION"
echo "  Max concurrent seqs:    $MAX_NUM_SEQS"
echo "  HF cache:               $HF_CACHE_DIR"
echo

echo "Removing any existing container named $CONTAINER_NAME ..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting spark-brain (vLLM DFlash2) ..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --shm-size 32g \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -p "${VLLM_PORT}:8000" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL_ID" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --skip-mm-profiling \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --generation-config auto \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_DRAFT\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}" \
    --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}'

echo
echo "Container started. Follow logs with:"
echo "  docker logs -f $CONTAINER_NAME"
echo "Ready check:"
echo "  curl -sf http://localhost:$VLLM_PORT/health && echo OK"
echo
echo "Expected speed: ~38–54 tok/s | 100/100 Tool-Eval"
echo "SGLang DSpark:  bash docker/start-sglang.sh"
echo "vLLM 512K mode: bash docker/start-vllm.sh"
