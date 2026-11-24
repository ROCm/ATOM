# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Benchmark baseline (no speculative decoding) throughput via vLLM."""

import argparse
import json
import os
import time

import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def get_model_name(base_url):
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        return resp.json()["data"][0]["id"]
    except (requests.RequestException, IndexError, KeyError, ValueError):
        return "/dev/shm/Kimi-K2.5-MXFP4"


def query_vllm(base_url, model_name, messages, max_tokens=2048):
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def load_mtbench(n):
    path = os.path.expanduser("~/.cache/benchmark_data/mtbench.jsonl")
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            prompts.append([{"role": "user", "content": q["turns"][0]}])
    return prompts[:n]


def load_gsm8k(n):
    path = os.path.expanduser("~/.cache/benchmark_data/gsm8k_test.jsonl")
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": "Question: " + q["question"] + "\nAnswer:",
                    }
                ]
            )
    return prompts[:n]


def load_humaneval(n):
    import gzip

    path = os.path.expanduser("~/.cache/benchmark_data/humaneval.jsonl.gz")
    prompts = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": "Complete the following Python function:\n\n"
                        + q["prompt"],
                    }
                ]
            )
    return prompts[:n]


def load_math500(n):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", token=HF_TOKEN)
    prompts = []
    for row in ds:
        prompts.append([{"role": "user", "content": row["problem"]}])
    return prompts[:n]


def load_ceval(n):
    from datasets import load_dataset

    ds = load_dataset(
        "ceval/ceval-exam", "computer_network", split="val", token=HF_TOKEN
    )
    prompts = []
    for row in ds:
        q = row["question"]
        choices = []
        for c in ["A", "B", "C", "D"]:
            if row.get(c):
                choices.append(f"{c}. {row[c]}")
        prompt = q + "\n" + "\n".join(choices) + "\n请选择正确答案："
        prompts.append([{"role": "user", "content": prompt}])
    return prompts[:n]


def load_aime(n):
    import pyarrow.parquet as pq

    path = os.path.expanduser("~/.cache/benchmark_data/aime2024.parquet")
    table = pq.read_table(path)
    prompts = []
    for row in table.to_pydict()["problem"][:n]:
        prompts.append([{"role": "user", "content": row}])
    return prompts[:n]


def run_benchmark(base_url, model_name, name, prompts, max_tokens=2048):
    print(f"\n{'='*60}")
    print(f"Running: {name} ({len(prompts)} samples)")
    print(f"{'='*60}")

    total_output_tokens = 0
    errors = 0
    start = time.time()

    for i, msgs in enumerate(prompts):
        try:
            result = query_vllm(base_url, model_name, msgs, max_tokens=max_tokens)
            usage = result.get("usage", {})
            total_output_tokens += usage.get("completion_tokens", 0)
        except (requests.RequestException, ValueError) as e:
            errors += 1
            if errors <= 3:
                print(f"  ERROR on sample {i}: {e}")

    elapsed = time.time() - start
    throughput = total_output_tokens / elapsed if elapsed > 0 else 0

    result = {
        "benchmark": name,
        "num_samples": len(prompts),
        "total_output_tokens": total_output_tokens,
        "throughput_tps": round(throughput, 1),
        "latency_s": round(elapsed, 1),
        "errors": errors,
    }

    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output tokens: {total_output_tokens}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--num-samples", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = get_model_name(args.base_url)
    print(f"Using model: {model_name}")
    print("Mode: BASELINE (no speculative decoding)")
    print(f"Samples per benchmark: {args.num_samples}")

    n = args.num_samples
    benchmark_configs = {
        "mtbench": (load_mtbench, min(n, 40), 2048),
        "gsm8k": (load_gsm8k, n, 512),
        "humaneval": (load_humaneval, n, 1024),
        "math500": (load_math500, n, 2048),
        "ceval": (load_ceval, min(n, 19), 512),
        "aime": (load_aime, min(n, 30), 4096),
    }

    all_results = {}
    for name, (loader, sample_n, max_tok) in benchmark_configs.items():
        try:
            prompts = loader(sample_n)
            result = run_benchmark(
                args.base_url, model_name, name, prompts, max_tokens=max_tok
            )
            all_results[name] = result
        except (OSError, requests.RequestException, KeyError, ValueError) as e:
            print(f"ERROR running {name}: {e}")
            import traceback

            traceback.print_exc()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.output_dir, f"baseline_no_draft_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    print(f"\n{'='*70}")
    print(f"{'Benchmark':<20} {'n':>5} {'Tok/s':>8}")
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(f"{name:<20} {r['num_samples']:>5} {r['throughput_tps']:>8.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
