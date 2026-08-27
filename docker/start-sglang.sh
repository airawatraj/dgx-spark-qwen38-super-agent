#!/usr/bin/env bash
# start-sglang.sh — SGLang DSpark alternate launch script
#
# Runs Qwen3.8-27B-NVFP4-BF16-LMHead via SGLang with DSpark speculative decoding.
# High concurrency throughput: ~60.5 tok/s (4-stream), 115.3 tok/s peak (10-stream).
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:qwen38-27b}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead}"
DSPARK_DRAFT="${DSPARK_DRAFT:-RadixArk/Qwen3.8-27B-DSpark}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-10}"
MAX_MAMBA_CACHE="${MAX_MAMBA_CACHE:-$(( MAX_CONCURRENT_REQUESTS * 4 ))}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"

echo "=== Qwen3.8-27B SGLang DSpark preflight ==="
echo "  Model:                  $MODEL_ID"
echo "  Drafter:                $DSPARK_DRAFT"
echo "  Served model name:      $SERVED_MODEL_NAME"
echo "  Container:              $CONTAINER_NAME"
echo "  Image:                  $SGLANG_IMAGE"
echo "  Port:                   $VLLM_PORT"
echo "  Context window:         262144 (262K)"
echo "  Mem fraction static:    $MEM_FRACTION"
echo "  Max concurrent:         $MAX_CONCURRENT_REQUESTS"
echo "  Mamba cache size:       $MAX_MAMBA_CACHE"
echo "  HF cache:               $HF_CACHE_DIR"
echo "  CPU pinning:            cores 5-9,15-19 (GB10 X5 only)"
echo

echo "Removing any existing container named $CONTAINER_NAME ..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting spark-brain (SGLang DSpark) ..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --shm-size 32g \
  --cpuset-cpus "5-9,15-19" \
  -p "${VLLM_PORT}:8000" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "$SGLANG_IMAGE" \
  python -m sglang.launch_server \
    --model-path "$MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tp-size 1 \
    --attention-backend flashinfer \
    --mem-fraction-static "$MEM_FRACTION" \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 8192 \
    --disable-prefill-cuda-graph \
    --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-mamba-cache-size "$MAX_MAMBA_CACHE" \
    --max-running-requests "$MAX_CONCURRENT_REQUESTS" \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "$DSPARK_DRAFT" \
    --speculative-dspark-block-size 7 \
    --speculative-num-draft-tokens 8 \
    --speculative-draft-model-quantization unquant \
    --cuda-graph-max-bs 4 \
    --enable-torch-compile \
    --torch-compile-max-bs 4 \
    --num-continuous-decode-steps 2 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --language-only

echo
echo "Container started. Follow logs with:"
echo "  docker logs -f $CONTAINER_NAME"
echo "Ready check (takes ~3–5 min for torch.compile warmup):"
echo "  curl -sf http://localhost:$VLLM_PORT/v1/models && echo OK"
echo
echo "Measured speed: 19–21 tok/s single-stream (60.5 tok/s 4-session concurrent)"
