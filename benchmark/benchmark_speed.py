#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
DGX Spark / Qwen 3.8-27B SGLang DSpark benchmark
Tests TPS, TTFT, concurrency, and maximum usable context window.
Usage: uv run benchmark/benchmark_speed.py [--host localhost] [--port 8000] [--model Cogni-Brain]
"""

import argparse
import json
import statistics
import sys
import threading
import time
from datetime import datetime

import requests


COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "reset": "\033[0m",
    "dim": "\033[2m",
}


def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def header(title):
    line = "─" * 60
    print(f"\n{c(line, 'cyan')}")
    print(f"{c('  ' + title, 'bold')}")
    print(f"{c(line, 'cyan')}")


def result_line(label, value, unit="", color="green"):
    print(f"  {c(label.ljust(30), 'dim')} {c(str(value), color)} {unit}")


def make_prompt(n_words):
    base = ("The quick brown fox jumps over the lazy dog. " * 50).split()
    words = (base * ((n_words // len(base)) + 1))[:n_words]
    return " ".join(words) + "\n\nSummarize the above text in one sentence."


def count_tokens_approx(text):
    return int(len(text.split()) * 1.33)


def stream_completion(
    host, port, model, prompt, max_tokens=200, timeout=120, debug=False,
    enable_thinking=True, thinking_budget=None
):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # Qwen3 thinking mode: disable or cap the <think> budget so it does not
    # consume the entire max_tokens allowance in throughput-focused tests.
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif thinking_budget is not None:
        payload["chat_template_kwargs"] = {"thinking_budget": thinking_budget}

    t_start = time.perf_counter()
    t_first = None
    full_text = ""
    usage_tokens = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return None, None, 0, "", f"HTTP {resp.status_code}: {resp.text[:200]}"

            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                if debug:
                    print(f"  RAW: {data[:200]}")
                try:
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage_tokens = chunk["usage"].get("completion_tokens")
                    delta = chunk["choices"][0]["delta"]
                    text = delta.get("content", "") or ""
                    think = delta.get("reasoning_content", "") or ""
                    combined = text + think
                    if combined:
                        if t_first is None:
                            t_first = time.perf_counter()
                        full_text += combined
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except requests.exceptions.Timeout:
        return None, None, 0, "", "Timeout"
    except requests.exceptions.ConnectionError:
        return None, None, 0, "", "Connection refused — is spark-brain running on the specified port?"
    except Exception as exc:
        return None, None, 0, "", str(exc)

    t_end = time.perf_counter()

    if t_first is None:
        return None, None, 0, full_text, "No tokens generated"

    ttft_ms = (t_first - t_start) * 1000
    generation_time = t_end - t_first
    tokens = usage_tokens if usage_tokens and usage_tokens > 0 else max(1, len(full_text) // 4)
    tps = tokens / generation_time if generation_time > 0 else 0
    return round(ttft_ms), round(tps, 1), tokens, full_text, None


def test_baseline_tps(host, port, model, debug=False):
    header("TEST 1 — Baseline TPS (single session, short prompt)")
    prompt = "Explain quantum entanglement in simple terms."
    runs = 3
    results = []

    print("  Running warmup request (not included in averages)...")
    warmup_ttft, warmup_tps, warmup_tokens, _, warmup_err = stream_completion(
        host, port, model, prompt, max_tokens=300, debug=debug
    )
    if warmup_err:
        print(c(f"  Warning: warmup failed: {warmup_err}", "yellow"))
    else:
        print(
            f"  Warmup: TTFT={c(str(warmup_ttft)+'ms', 'yellow')}  "
            f"TPS={c(str(warmup_tps), 'green')}  tokens={warmup_tokens}"
        )

    print(f"  Running {runs} consecutive requests...")
    for idx in range(runs):
        ttft, tps, tokens, _, err = stream_completion(
            host, port, model, prompt, max_tokens=300, debug=debug
        )
        if err:
            print(c(f"  Run {idx + 1} failed: {err}", "red"))
            continue
        results.append((ttft, tps, tokens))
        print(f"  Run {idx + 1}: TTFT={c(str(ttft)+'ms', 'yellow')}  TPS={c(str(tps), 'green')}  tokens={tokens}")
        time.sleep(1)

    if not results:
        return 0, 0

    avg_tps = round(statistics.mean([row[1] for row in results]), 1)
    avg_ttft = round(statistics.mean([row[0] for row in results]))
    peak_tps = max([row[1] for row in results])
    if warmup_ttft is not None and not warmup_err:
        result_line("Warmup TTFT", warmup_ttft, "ms", "yellow")
    result_line("Average TPS", avg_tps, "tok/s", "green")
    result_line("Peak TPS", peak_tps, "tok/s", "green")
    result_line("Average TTFT (steady state)", avg_ttft, "ms", "yellow")
    return avg_tps, peak_tps


def test_tps_vs_length(host, port, model):
    header("TEST 2 — TPS vs Output Length")
    # Thinking mode is disabled here: we want to measure raw generation throughput
    # at specific output lengths, not the thinking budget. With reasoning enabled,
    # small max_tokens limits are consumed entirely by <think> blocks.
    lengths = [50, 150, 300, 600, 1000]
    prompt = "Write a detailed explanation of how transformers work in machine learning."

    print(f"  {'Target tokens'.ljust(18)} {'Actual tok'.ljust(12)} {'TPS'.ljust(12)} {'TTFT'}")
    print(f"  {'─' * 56}")

    for max_tok in lengths:
        ttft, tps, tokens, _, err = stream_completion(
            host, port, model, prompt,
            max_tokens=max_tok,
            enable_thinking=False,
        )
        if err:
            print(f"  {str(max_tok).ljust(18)} {c('FAILED: ' + err, 'red')}")
        else:
            tps_color = "green" if tps >= 50 else "yellow" if tps >= 30 else "red"
            token_label = f"{tokens} tok"
            print(
                f"  {str(max_tok)+' tok':18} {token_label.ljust(12)} "
                f"{c(str(tps)+' tok/s', tps_color).ljust(20)} {ttft}ms"
            )
        time.sleep(1)


def test_concurrent(host, port, model, max_concurrent=4):
    header("TEST 3 — Concurrent Sessions TPS")
    prompts_list = [
        "Explain the history of the Roman Empire in detail.",
        "Describe how neural networks learn from data.",
        "What are the key principles of thermodynamics?",
        "Explain the causes and effects of the French Revolution.",
    ]

    for sessions in range(1, max_concurrent + 1):
        results = [None] * sessions
        errors = []

        def run_request(idx):
            ttft, tps, tokens, _, err = stream_completion(
                host, port, model, prompts_list[idx % len(prompts_list)],
                max_tokens=200, enable_thinking=False,
            )
            if err:
                errors.append(err)
            else:
                results[idx] = (tokens, tps, ttft)

        threads = [threading.Thread(target=run_request, args=(idx,)) for idx in range(sessions)]
        t_start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - t_start

        valid = [row for row in results if row is not None]
        if not valid:
            print(f"  {str(sessions)+' session(s)'.ljust(14)} {c('ALL FAILED: ' + str(errors[0]), 'red')}")
            time.sleep(3)
            continue

        total_tokens = sum(row[0] for row in valid)
        total_tps = round(total_tokens / elapsed, 1) if elapsed > 0 else 0
        per_session = round(total_tps / sessions, 1)
        color = "green" if total_tps >= 80 else "yellow" if total_tps >= 40 else "red"
        print(
            f"  {str(sessions)+' session(s)'.ljust(14)} "
            f"total={c(str(total_tps)+' tok/s', color).ljust(22)} "
            f"per-session={c(str(per_session)+' tok/s', color)}"
        )
        time.sleep(3)


def test_context_window(host, port, model):
    header("TEST 4 — Context Window Limits")
    print(f"  {c('Testing progressively larger contexts...', 'dim')}")
    print(f"  {'Context tokens'.ljust(20)} {'Result'.ljust(20)} {'TPS'}")
    print(f"  {'─' * 50}")

    # Qwen3.8-27B on SGLang DSpark supports up to 262K context
    sizes = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    last_working = 0

    for size in sizes:
        prompt = make_prompt(int(size * 0.75))
        actual_tokens = count_tokens_approx(prompt)

        ttft, tps, _, _, err = stream_completion(
            host, port, model, prompt, max_tokens=100, timeout=300, enable_thinking=False
        )
        if err:
            if "context" in err.lower() or "length" in err.lower() or "exceed" in err.lower():
                status = c("Context exceeded", "red")
            elif "timeout" in err.lower():
                status = c("Timeout", "red")
            else:
                status = c(err[:25], "red")
            print(f"  ~{str(actual_tokens)+' tok':15} {status}")
            break

        last_working = actual_tokens
        tps_color = "green" if tps >= 40 else "yellow" if tps >= 15 else "red"
        print(f"  ~{str(actual_tokens)+' tok':15} {c('OK', 'green'):20} {c(str(tps)+' tok/s', tps_color)}")
        time.sleep(2)

    if last_working:
        result_line("Max working context", f"~{last_working:,}", "tokens", "green")
    return last_working


def test_health(host, port):
    header("TEST 5 — Server Health")
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        result_line(
            "Health endpoint",
            "OK" if response.status_code == 200 else f"HTTP {response.status_code}",
            color="green" if response.status_code == 200 else "red",
        )
    except Exception as exc:
        result_line("Health endpoint", f"FAILED: {exc}", color="red")

    try:
        response = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json().get("data", [])
            served = ", ".join(item.get("id", "?") for item in data[:3]) if data else "none reported"
            result_line("Served model(s)", served, color="green")
    except Exception:
        result_line("Models endpoint", "not available", color="yellow")


def print_summary(avg_tps, peak_tps, max_context, host, port, model):
    header("SUMMARY")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_line("Timestamp", now)
    result_line("Endpoint", f"http://{host}:{port}")
    result_line("Model", model)
    print()
    result_line(
        "Average TPS (single session)",
        avg_tps,
        "tok/s",
        "green" if avg_tps >= 50 else "yellow" if avg_tps >= 30 else "red",
    )
    result_line(
        "Peak TPS (single session)",
        peak_tps,
        "tok/s",
        "green" if peak_tps >= 50 else "yellow" if peak_tps >= 30 else "red",
    )
    result_line("Max usable context", f"~{max_context:,}" if max_context else "not tested", "tokens")
    print()
    if avg_tps >= 50:
        print(f"  {c('Excellent — SGLang DSpark config is working well', 'green')}")
    elif avg_tps >= 30:
        print(f"  {c('Good — decent throughput, but there is room to tune', 'yellow')}")
    else:
        print(f"  {c('Below expectation — check logs and memory settings', 'red')}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DGX Spark Qwen 3.8-27B vLLM setup",
        epilog="Run with: uv run benchmark/benchmark_speed.py",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--debug", action="store_true", help="Print raw stream chunks for debugging")
    parser.add_argument("--skip-context", action="store_true", help="Skip context window test")
    parser.add_argument("--skip-concurrent", action="store_true", help="Skip concurrent session test")
    args = parser.parse_args()

    print(f"\n{c('DGX Spark Qwen3.8-27B SGLang DSpark Benchmark', 'bold')}")
    print(f"{c('Target: ', 'dim')}http://{args.host}:{args.port}  model={args.model}")

    try:
        response = requests.get(f"http://{args.host}:{args.port}/health", timeout=30)
        if response.status_code != 200:
            print(c(f"\nCannot reach healthy SGLang endpoint (HTTP {response.status_code})", "red"))
            sys.exit(1)
    except Exception as exc:
        print(c(f"\nCannot reach SGLang: {exc}", "red"))
        print(c("  Make sure spark-brain is running and the port is correct.", "dim"))
        sys.exit(1)

    print(c("  API is reachable\n", "green"))

    avg_tps, peak_tps = test_baseline_tps(args.host, args.port, args.model, debug=args.debug)
    test_tps_vs_length(args.host, args.port, args.model)

    if not args.skip_concurrent:
        test_concurrent(args.host, args.port, args.model)

    max_context = 0
    if not args.skip_context:
        max_context = test_context_window(args.host, args.port, args.model)

    test_health(args.host, args.port)
    print_summary(avg_tps, peak_tps, max_context, args.host, args.port, args.model)


if __name__ == "__main__":
    main()
