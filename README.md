# Running a Real Local AI Agent on DGX Spark: Qwen3.8-27B via vLLM MTP

I bought a DGX Spark to do real work: running serious local AI agents and training foundation models from scratch - not to run benchmarks.

*(If you are curious about the training side of this hardware, check out [SageGPT](https://github.com/airawatraj/sage-gpt), my 7.5M parameter Sanskrit SLM trained entirely from scratch on this same machine).*

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Base Model](https://img.shields.io/badge/base%20model-Qwen3.8--27B-limegreen)
![Runtime](https://img.shields.io/badge/runtime-vLLM%20MTP-orange)
![Hardware](https://img.shields.io/badge/hardware-NVIDIA%20DGX%20Spark-brightgreen?logo=nvidia&logoColor=white)
![Speed](https://img.shields.io/badge/speed-24--26%20tok%2Fs%20(119%20burst)-success)
![Smarts](https://img.shields.io/badge/smarts-100%2F100-brightgreen)
![Context](https://img.shields.io/badge/context-262K-blue)
![Tool Calling](https://img.shields.io/badge/tool--calling-native%20100%25-success)
![Mode](https://img.shields.io/badge/mode-reasoning%20%2B%20tools-black)

Part of the DGX Spark local agent series:
- [Cogni-Brain (Nemotron-120B)](https://github.com/airawatraj/dgx-spark-nemotron-super-agent) — deep reasoning, 23 TPS
- [Cogni-Brain-2 (Qwen 3.6-35B via Atlas)](https://github.com/airawatraj/dgx-spark-qwen-super-agent) — fast & agentic, 218 TPS
- More setups: [github.com/airawatraj](https://github.com/airawatraj)

This iteration runs **[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)** (with 4-bit `lm_head` and calibrated FP8 KV cache scaling) via **vLLM** with native **Multi-Token Prediction (MTP)** speculative decoding. Delivering **24–26 tok/s steady-state generation** (bursting to **119.1 tok/s**), a verified **100/100 Tool-Eval score (15/15 PASS, 0 errors)**, and a **262K token context window** on a single DGX Spark node.

> ⚠️ **Personal workstation setup. Not for enterprise use. Use at your own risk.**

---

## Why This Setup

### The Model & Speculative Architecture

[Qwen3.8](https://qwenlm.github.io/blog/qwen3/) is a hybrid thinking/non-thinking model series. The 27B variant offers a strong balance of reasoning depth and agentic speed. Running `unsloth/Qwen3.8-27B-NVFP4` with full 4-bit `lm_head` quantization saves ~1.1 GB of projection memory bandwidth per forward pass vs unquantized heads. Paired with native MTP speculative decoding in vLLM, it eliminates external drafter overhead while achieving perfect tool-calling determinism.

**Architecture Evolution:**

| Feature | vLLM Eager (Baseline) | SGLang DSpark (Iteration 2) | vLLM MTP (Current Default) |
|---|---|---|---|
| Runtime | vLLM `vllm-openai` | SGLang (`qwen38-27b`) | **vLLM + Native MTP** |
| Model | `unsloth/Qwen3.8-27B-NVFP4` | `RadixArk NVFP4 BF16-LMHead` | **`unsloth/Qwen3.8-27B-NVFP4` (4-bit head)** |
| Speculative Drafter | None | `RadixArk DSpark` (block-7) | **Built-in MTP Head ($k=2$)** |
| Single-Stream Decode | ~14 tok/s | ~19–21 tok/s | **24–26 tok/s (119.1 tok/s burst)** |
| Concurrency Decode | ~14 tok/s | 115.3 tok/s (10-stream) | **60.6 tok/s (3-stream)** |
| Context Window | **512K** | 262K | **262K (131K+ verified working)** |
| Tool-Eval Score | **100/100** | 97/100 | **100/100 (15/15 full passes)** |

### Architecture Overview

```mermaid
flowchart TD
    A["NVIDIA DGX Spark<br/>GB10 Grace-Blackwell<br/>128GB Unified Memory"]

    A --> B["vLLM Engine<br/>CUDA Architecture sm_121a · chunked prefill 16K"]
    A --> S["Built-in MTP Head<br/>Multi-Token Prediction k=2"]

    B --> C["Qwen3.8-27B-NVFP4<br/>Cogni-Brain (4-bit lm_head)"]
    B --> D["OpenAI-compatible API<br/>localhost:8000"]
    S --> C

    C --> E["Native Tool Calling<br/>qwen3_coder parser (100% precision)"]
    C --> F["Native Reasoning<br/>qwen3 parser"]
    C --> G["24-26 tok/s single stream<br/>100/100 Tool-Eval"]
```

---

## Quick Start

> ⚠️ **Note:** Run `download_model.sh` once before first launch (~24 GB total). Run in `tmux` if on SSH.

```bash
# 1. Verify prerequisites (Docker, uv/uvx, HF auth, vLLM image)
bash setup/install.sh

# 2. Download model — one-time, ~24 GB (run in tmux on SSH)
bash setup/download_model.sh

# 3. Launch spark-brain (vLLM MTP)
bash docker/start.sh

# 4. Follow logs (wait for "Application startup complete")
docker logs -f spark-brain

# 5. Ready check
curl -sf http://localhost:8000/health && echo OK
```

### Alternate Stack Toggles

- **SGLang DSpark** (262K context, high-concurrency batching):
  ```bash
  bash docker/stop.sh && bash docker/start-sglang.sh
  ```
- **vLLM Eager Fallback** (512K context, multimodal vision inputs):
  ```bash
  bash docker/stop.sh && bash docker/start-vllm.sh
  ```

---

## Benchmark Results

```bash
# Single-stream speed and context test
uv run benchmark/benchmark_speed.py

# Tool-use capability benchmark (15 agentic scenarios)
uv run benchmark/benchmark_smarts.py

# Full throughput sweep (spark-arena compatible format)
uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_arena.csv
```

### Speed & Context Results (24–26 tok/s steady, 119 tok/s burst)

<p align="center">
  <img src="./assets/benchmark_speed_qwen38_nvfp4.png" width="700" alt="vLLM MTP Speed & Context Benchmark Results">
</p>

### Smarts Results (Tool-Use Evaluation — 100/100 Perfect Score)

<p align="center">
  <img src="./assets/benchmark_smarts_qwen38_nvfp4_1.png" width="700" alt="Smarts benchmark 100/100 — scenarios 1-14">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_qwen38_nvfp4_2.png" width="700" alt="Smarts benchmark 100/100 — breakdown and points">
</p>

---

## Repository Structure

```text
dgx-spark-qwen38-super-agent/
├── README.md                ← this file
├── EXPERIMENTS.md           ← archived benchmark data from prior iterations
├── CITATION.cff             ← citation metadata
├── LICENSE                  ← MIT license
├── setup/
│   ├── install.sh           ← verify Docker, uv/uvx, and Hugging Face auth
│   └── download_model.sh    ← download unsloth NVFP4 (~24 GB)
├── docker/
│   ├── start.sh             ← vLLM MTP launch (100/100 smarts, default)
│   ├── start-sglang.sh      ← SGLang DSpark launch (high concurrency)
│   ├── start-vllm.sh        ← vLLM eager fallback (512K ctx, multimodal)
│   ├── stop.sh              ← stop and remove the container
│   └── status.sh            ← health check and metrics
└── benchmark/
    ├── benchmark_speed.py   ← TPS, TTFT, concurrency, and context benchmark
    ├── benchmark_smarts.py  ← tool-eval-bench wrapper for capability checks
    └── benchmark_speed_arena.py ← full throughput sweep (arena-compatible output)
```

## Compared to Prior Published Results

| Who | Model | Runtime | Single-Stream | Concurrency | Context | Tool-Eval |
|---|---|---|---|---|---|---|
| **[Cogni-Brain-2 (airawatraj)](https://spark-arena.com/benchmark/sub1779495971526)** | Qwen 3.6-35B | Atlas NVFP4 | **218.85 tok/s** | — | 131K | **100/100** |
| **[Cogni-Brain (airawatraj)](https://spark-arena.com/benchmark/sub1778644062716)** | Nemotron-120B | vLLM NVFP4 | **23.45 tok/s** | — | 131K | **100/100** |
| **[Cogni-Brain (SGLang)](https://spark-arena.com/benchmark/sub1787869873047)** | Qwen3.8-27B | SGLang DSpark | 19.6 tok/s | **115.3 tok/s** | 262K | 97/100 |
| **Cogni-Brain (this repo)** | Qwen3.8-27B | **vLLM MTP** | **24–26 tok/s** | **60.6 tok/s** | 262K | **100/100** |

> All historical baseline logs and arena sweep charts are archived in [EXPERIMENTS.md](EXPERIMENTS.md).
