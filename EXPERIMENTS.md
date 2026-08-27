# Experiments & Historical Results

This document archives benchmark results from configurations that are no longer
the default. The current default stack is documented in [README.md](README.md).

---

## Iteration 1 — vLLM Baseline (unsloth/Qwen3.8-27B-NVFP4)

**Config:** `docker/start-vllm.sh`  
**Model:** `unsloth/Qwen3.8-27B-NVFP4`  
**Runtime:** vLLM `vllm/vllm-openai:latest`  
**Date:** August 2026

### Configuration summary

| Parameter | Value |
|---|---|
| Runtime | vLLM OpenAI server |
| Model | unsloth/Qwen3.8-27B-NVFP4 |
| KV cache | fp8 |
| Max context | 512K (VLLM_ALLOW_LONG_MAX_MODEL_LEN=1) |
| Chunked prefill | 16384 tokens |
| GPU memory utilization | 0.85 |
| Speculative decode | None (--enforce-eager) |
| Multimodal | Yes (image ×4, video ×1) |
| Tool calling | 100/100 |

### Results

| Metric | Result |
|---|---|
| Average TPS (single session) | ~14 tok/s |
| Peak TPS | ~14 tok/s |
| TTFT (steady state) | ~6–7 s |
| Max context tested | 512K |
| Tool eval (smarts) | **100/100** |

### Speed benchmark screenshots

> Benchmarked with vLLM at 512K context window. Tests 1–5 from `benchmark_speed.py`.
> Note: Tests 2–5 used `enable_thinking=False` via `chat_template_kwargs` to avoid
> Qwen3 `<think>` block exhausting the `max_tokens` budget — see
> [benchmark_speed.py fix](https://github.com/airawatraj/dgx-spark-qwen38-super-agent/commit/77ce061).

<p align="center">
  <img src="./assets/benchmark_speed_test_524K.png" width="700" alt="vLLM speed benchmark — Tests 1-5 at 512K context">
</p>

### Smarts benchmark screenshots

> Tool-use evaluation via `benchmark_smarts.py`. 100/100 score confirmed.

<p align="center">
  <img src="./assets/benchmark_smarts_test_524K_1.png" width="700" alt="vLLM smarts benchmark — page 1">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_test_524K_2.png" width="700" alt="vLLM smarts benchmark — page 2">
</p>

### Notes

- `--enforce-eager` was required to prevent CUDA graph OOM on first boot. The flash-autotune
  cache (`/root/.cache/vllm/flashinfer_autotune_cache`) took ~2 min to warm up.
- Speed limited by vLLM's generic cuBLAS path — no GB10-native NVFP4 GEMM kernels.
- 14 tok/s is the true stock vLLM ceiling for this model on a single DGX Spark.
- Migrated to SGLang DSpark (~51 tok/s) while preserving this config as `docker/start-vllm.sh`.

---

*Add new experiment sections above this line in reverse-chronological order.*
