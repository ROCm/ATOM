# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
Benchmark Eagle3 speculative decoding with vLLM.

Measures accept_length (average accepted tokens per speculation step) across
multiple benchmark datasets. Matches the reference performance table from
lightseekorg/kimi-k2.5-eagle3.
"""

import argparse
import json
import os
import re
import time

import requests

BFCL_BASE_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"


def download_file(url, filename):
    cache_dir = os.path.expanduser("~/.cache/benchmark_data")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        return path
    print(f"Downloading {filename}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def get_model_name(base_url):
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        data = resp.json()
        return data["data"][0]["id"]
    except (requests.RequestException, IndexError, KeyError, ValueError):
        return "/dev/shm/Kimi-K2.5-BF16"


def load_mtbench(n):
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    path = download_file(url, "mtbench.jsonl")
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            prompts.append([{"role": "user", "content": q["turns"][0]}])
    return prompts[:n]


def load_ceval(n):
    url = "https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/test/computer_network_test.csv"
    path = download_file(url, "ceval_cn_test.csv")
    import csv

    prompts = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", row.get("Question", ""))
            choices = []
            for c in ["A", "B", "C", "D"]:
                if c in row:
                    choices.append(f"{c}. {row[c]}")
            prompt = q + "\n" + "\n".join(choices) + "\nAnswer:"
            prompts.append([{"role": "user", "content": prompt}])
    if len(prompts) < n:
        prompts = prompts * (n // len(prompts) + 1)
    return prompts[:n]


def load_gsm8k(n):
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    path = download_file(url, "gsm8k_test.jsonl")
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
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    import gzip

    path = download_file(url, "humaneval.jsonl.gz")
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
    url = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test/data-00000-of-00001.parquet"
    path = download_file(url, "math500_test.parquet")
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    prompts = []
    for row in table.to_pydict()["problem"][:n]:
        prompts.append([{"role": "user", "content": row}])
    return prompts[:n]


def load_aime(n):
    url = "https://huggingface.co/datasets/HuggingFaceH4/aime_2024/resolve/main/data/train-00000-of-00001.parquet"
    path = download_file(url, "aime2024.parquet")
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    prompts = []
    for row in table.to_pydict()["problem"][:n]:
        prompts.append([{"role": "user", "content": row}])
    if len(prompts) < n:
        prompts = prompts * (n // len(prompts) + 1)
    return prompts[:n]


def load_bfcl_v3(variant, n):
    filemap = {
        "simple": "BFCL_v3_simple.json",
        "multiple": "BFCL_v3_multiple.json",
        "parallel": "BFCL_v3_parallel.json",
        "parallel_multiple": "BFCL_v3_parallel_multiple.json",
        "live_simple": "BFCL_v3_live_simple.json",
        "live_multiple": "BFCL_v3_live_multiple.json",
        "live_parallel": "BFCL_v3_live_parallel.json",
        "live_parallel_multiple": "BFCL_v3_live_parallel_multiple.json",
    }
    fname = filemap[variant]
    url = f"{BFCL_BASE_URL}/{fname}"
    path = download_file(url, fname)

    # BFCL files are JSONL (one JSON object per line)
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))

    prompts = []
    for entry in data[:n]:
        funcs = entry.get("function", [])
        messages = entry.get("question", [[]])
        if isinstance(messages[0], list):
            messages = messages[0]

        parts = []
        if funcs:
            parts.append(
                "Available functions:\n```json\n"
                + json.dumps(funcs, indent=2, ensure_ascii=False)
                + "\n```\n"
            )
        for msg in messages:
            if msg.get("role") == "user":
                parts.append(msg.get("content", ""))
        prompts.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to functions. Use them if required.",
                },
                {"role": "user", "content": "\n".join(parts)},
            ]
        )
    return prompts[:n]


def query_vllm(base_url, model_name, messages, max_tokens=2048):
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": False},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def get_prometheus_metrics(base_url):
    try:
        resp = requests.get(f"{base_url}/metrics", timeout=10)
        metrics = {}
        for line in resp.text.split("\n"):
            if "spec_decode" in line and not line.startswith("#"):
                # Handle labels: vllm:spec_decode_num_drafts_total{engine_id="0"} 123.0
                match = re.match(r"^([^\s{]+)(?:\{[^}]*\})?\s+([\d.eE+-]+)", line)
                if match:
                    key = match.group(1)
                    val = float(match.group(2))
                    metrics[key] = metrics.get(key, 0) + val
        return metrics
    except (requests.RequestException, ValueError):
        return {}


def run_benchmark(base_url, model_name, name, prompts, max_tokens=2048, concurrency=1):
    print(f"\n{'='*60}")
    print(f"Running: {name} ({len(prompts)} samples, concurrency={concurrency})")
    print(f"{'='*60}")

    # Get metrics before
    m_before = get_prometheus_metrics(base_url)
    drafts_before = m_before.get("vllm:spec_decode_num_drafts_total", 0)
    accepted_before = m_before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    draft_tokens_before = m_before.get("vllm:spec_decode_num_draft_tokens_total", 0)

    total_output_tokens = 0
    errors = 0
    start = time.time()

    if concurrency <= 1:
        for i, msgs in enumerate(prompts):
            try:
                result = query_vllm(base_url, model_name, msgs, max_tokens=max_tokens)
                usage = result.get("usage", {})
                total_output_tokens += usage.get("completion_tokens", 0)
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start
                    print(f"  Progress: {i+1}/{len(prompts)} ({elapsed:.1f}s)")
            except (requests.RequestException, ValueError) as e:
                errors += 1
                if errors <= 3:
                    print(f"  ERROR on sample {i}: {e}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {
                ex.submit(query_vllm, base_url, model_name, m, max_tokens): i
                for i, m in enumerate(prompts)
            }
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    total_output_tokens += r.get("usage", {}).get(
                        "completion_tokens", 0
                    )
                except (requests.RequestException, ValueError) as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  ERROR on sample {futs[fut]}: {e}")
                done += 1
                if done % 50 == 0 or done == len(prompts):
                    elapsed = time.time() - start
                    print(f"  Progress: {done}/{len(prompts)} ({elapsed:.1f}s)")

    elapsed = time.time() - start
    if errors > 3:
        print(f"  ... {errors} total errors")

    # Get metrics after
    m_after = get_prometheus_metrics(base_url)
    drafts_after = m_after.get("vllm:spec_decode_num_drafts_total", 0)
    accepted_after = m_after.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    draft_tokens_after = m_after.get("vllm:spec_decode_num_draft_tokens_total", 0)

    num_drafts = drafts_after - drafts_before
    num_accepted = accepted_after - accepted_before
    num_draft_tokens = draft_tokens_after - draft_tokens_before

    # accept_length = 1 + (accepted / drafts) per vLLM convention
    accept_length = 1.0 + (num_accepted / num_drafts) if num_drafts > 0 else 1.0
    acceptance_rate = (
        (num_accepted / num_draft_tokens * 100) if num_draft_tokens > 0 else 0
    )

    throughput = total_output_tokens / elapsed if elapsed > 0 else 0

    result = {
        "benchmark": name,
        "num_samples": len(prompts),
        "accept_length": round(accept_length, 3),
        "acceptance_rate": round(acceptance_rate, 1),
        "total_output_tokens": total_output_tokens,
        "throughput_tps": round(throughput, 1),
        "latency_s": round(elapsed, 1),
        "num_drafts": int(num_drafts),
        "num_accepted": int(num_accepted),
        "num_draft_tokens": int(num_draft_tokens),
        "errors": errors,
    }

    print(f"  Accept length: {accept_length:.3f}")
    print(f"  Acceptance rate: {acceptance_rate:.1f}%")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Total time: {elapsed:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of in-flight requests per benchmark (1 = sequential, backward-compatible).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect model name
    model_name = get_model_name(args.base_url)
    print(f"Using model: {model_name}")

    benchmark_configs = {
        "mtbench": (load_mtbench, 80, 2048),
        "ceval": (load_ceval, 212, 512),
        "gsm8k": (load_gsm8k, 500, 512),
        "humaneval": (load_humaneval, 164, 1024),
        "math500": (load_math500, 500, 2048),
        "aime": (load_aime, 30, 4096),
        "bfcl_v3_simple": (lambda n: load_bfcl_v3("simple", n), 400, 1024),
        "bfcl_v3_multiple": (lambda n: load_bfcl_v3("multiple", n), 200, 1024),
        "bfcl_v3_parallel": (lambda n: load_bfcl_v3("parallel", n), 200, 1024),
        "bfcl_v3_parallel_multiple": (
            lambda n: load_bfcl_v3("parallel_multiple", n),
            200,
            1024,
        ),
        "bfcl_v3_live_simple": (lambda n: load_bfcl_v3("live_simple", n), 1547, 1024),
        "bfcl_v3_live_multiple": (
            lambda n: load_bfcl_v3("live_multiple", n),
            1030,
            1024,
        ),
        "bfcl_v3_live_parallel": (lambda n: load_bfcl_v3("live_parallel", n), 97, 1024),
        "bfcl_v3_live_parallel_multiple": (
            lambda n: load_bfcl_v3("live_parallel_multiple", n),
            170,
            1024,
        ),
    }

    if "all" in args.benchmarks:
        run_list = list(benchmark_configs.keys())
    else:
        run_list = args.benchmarks

    all_results = {}
    for name in run_list:
        if name not in benchmark_configs:
            print(f"WARN: Unknown benchmark '{name}', skipping")
            continue
        loader, n, max_tok = benchmark_configs[name]
        try:
            prompts = loader(n)
            result = run_benchmark(
                args.base_url,
                model_name,
                name,
                prompts,
                max_tokens=max_tok,
                concurrency=args.concurrency,
            )
            all_results[name] = result
        except (OSError, requests.RequestException, KeyError, ValueError) as e:
            print(f"ERROR running {name}: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        args.output_dir, f"phase1_v2_vllm_results_{timestamp}.json"
    )
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Benchmark':<35} {'n':>5} {'Accept Len':>12} {'Tok/s':>8}")
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(
            f"{name:<35} {r['num_samples']:>5} {r['accept_length']:>12.3f} {r['throughput_tps']:>8.1f}"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
