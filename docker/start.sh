#!/usr/bin/env bash
# start.sh — SGLang DSpark (default high-speed stack)
#
# Runs Qwen3.8-27B-NVFP4 via SGLang with DSpark speculative decoding.
# Runs Qwen3.8-27B-NVFP4 via SGLang with DSpark speculative decoding.
#
# Key differences vs the old vLLM start (start-vllm.sh):
#   - Runtime:   vLLM → SGLang (lmsysorg/sglang:qwen38-27b)
#   - Model:     unsloth/Qwen3.8-27B-NVFP4 → RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead
#                (BF16 lm_head avoids hard-reboot bug with CUDA graph capture on GB10)
#   - Speculative decode: none → DSpark block-7 (RadixArk/Qwen3.8-27B-DSpark drafter)
#   - CPU pinning: X5 fast cores only (5-9,15-19 — A725 effi-cores excluded, +2–7%)
#   - Context:   512K → 262K (SGLang DSpark incompatible with YaRN rope scaling)
#   - Multimodal: disabled (--language-only, saves ~3GB; re-enable if needed)
#   - Tool calling and reasoning parser: identical flags, zero client changes needed
#
# Rollback to vLLM (512K context, multimodal):
#   bash docker/start-vllm.sh
set -euo pipefail

# ── Configuration (override via env vars before calling this script) ──────────
CONTAINER_NAME="${CONTAINER_NAME:-spark-brain}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:qwen38-27b}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead}"
DSPARK_DRAFT="${DSPARK_DRAFT:-RadixArk/Qwen3.8-27B-DSpark}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cogni-Brain}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Concurrency and GDN state pool sizing
# MAX_MAMBA_CACHE = MAX_CONCURRENT_REQUESTS * 4  (S=4 for extra_buffer_lazy + overlap)
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-10}"
MAX_MAMBA_CACHE="${MAX_MAMBA_CACHE:-$(( MAX_CONCURRENT_REQUESTS * 4 ))}"

# Memory: 0.90 is the measured optimum for DSpark on GB10.
# DSpark drafter needs ~2.7 GB on top of the ~24 GB main model.
MEM_FRACTION="${MEM_FRACTION:-0.90}"

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "=== Qwen3.8-27B SGLang DSpark preflight ==="
echo "  Model:                  $MODEL_ID"
echo "  Drafter:                $DSPARK_DRAFT"
echo "  Served model name:      $SERVED_MODEL_NAME"
echo "  Container:              $CONTAINER_NAME"
echo "  Image:                  $SGLANG_IMAGE"
echo "  Port:                   $VLLM_PORT"
echo "  Context window:         262144 (256K — DSpark limit)"
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
echo "Ready check (SGLang takes ~3–5 min for torch.compile warmup):"
echo "  curl -sf http://localhost:$VLLM_PORT/v1/models && echo OK"
echo
echo "Expected speed: ~51 tok/s (code/tools)  ~23 tok/s (chat)"
echo "Rollback to vLLM (512K ctx): bash docker/start-vllm.sh"
