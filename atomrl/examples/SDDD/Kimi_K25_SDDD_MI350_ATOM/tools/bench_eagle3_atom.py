# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
Benchmark Eagle3 speculative decoding with ATOM.

Measures accept_length (average accepted tokens per speculation step) across
multiple benchmark datasets using ATOM's /debug/mtp_stats endpoint.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
        return "/dev/shm/Kimi-K2.5-MXFP4"


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
    url = "https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/computer_network/test-00000-of-00001.parquet"
    path = download_file(url, "ceval_cn_test.parquet")
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    data = table.to_pydict()
    prompts = []
    for i in range(len(data["question"])):
        q = data["question"][i]
        choices = []
        for c in ["A", "B", "C", "D"]:
            if c in data:
                choices.append(f"{c}. {data[c][i]}")
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
    url = (
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
    )
    path = download_file(url, "math500_test.jsonl")
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            prompts.append([{"role": "user", "content": q["problem"]}])
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


def query_server(base_url, model_name, messages, max_tokens=2048):
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


def get_atom_mtp_stats(base_url):
    try:
        resp = requests.get(f"{base_url}/debug/mtp_stats", timeout=10)
        return resp.json()
    except (requests.RequestException, ValueError):
        return {}


def run_benchmark(base_url, model_name, name, prompts, max_tokens=2048, batch_size=1):
    print(f"\n{'='*60}")
    print(f"Running: {name} ({len(prompts)} samples, BS={batch_size})")
    print(f"{'='*60}")

    stats_before = get_atom_mtp_stats(base_url)
    draft_before = stats_before.get("total_draft_tokens", 0)
    accepted_before = stats_before.get("total_accepted_tokens", 0)

    total_output_tokens = 0
    errors = 0
    done_count = 0
    lock = Lock()
    start = time.time()

    def _worker(msgs):
        return query_server(base_url, model_name, msgs, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_worker, msgs): i for i, msgs in enumerate(prompts)}
        for future in as_completed(futures):
            try:
                result = future.result()
                usage = result.get("usage", {})
                with lock:
                    total_output_tokens += usage.get("completion_tokens", 0)
                    done_count += 1
                    if done_count % 50 == 0:
                        elapsed = time.time() - start
                        print(
                            f"  Progress: {done_count}/{len(prompts)} ({elapsed:.1f}s)"
                        )
            except (requests.RequestException, ValueError) as e:
                with lock:
                    errors += 1
                    done_count += 1
                    if errors <= 3:
                        print(f"  ERROR on sample {futures[future]}: {e}")

    elapsed = time.time() - start
    if errors > 3:
        print(f"  ... {errors} total errors")

    stats_after = get_atom_mtp_stats(base_url)
    draft_after = stats_after.get("total_draft_tokens", 0)
    accepted_after = stats_after.get("total_accepted_tokens", 0)

    num_draft_tokens = draft_after - draft_before
    num_accepted = accepted_after - accepted_before

    mtp_k = 3
    num_steps = num_draft_tokens // mtp_k if mtp_k > 0 else 0
    accept_length = 1.0 + (num_accepted / num_steps) if num_steps > 0 else 1.0
    acceptance_rate = (
        (num_accepted / num_draft_tokens * 100) if num_draft_tokens > 0 else 0
    )

    # Per-position acceptance from distribution
    dist_before = stats_before.get("distribution", {})
    dist_after = stats_after.get("distribution", {})
    distribution = {}
    for k in dist_after:
        distribution[k] = dist_after[k] - dist_before.get(k, 0)

    throughput = total_output_tokens / elapsed if elapsed > 0 else 0

    result = {
        "benchmark": name,
        "num_samples": len(prompts),
        "accept_length": round(accept_length, 3),
        "acceptance_rate": round(acceptance_rate, 1),
        "total_output_tokens": total_output_tokens,
        "throughput_tps": round(throughput, 1),
        "latency_s": round(elapsed, 1),
        "num_draft_tokens": int(num_draft_tokens),
        "num_accepted": int(num_accepted),
        "distribution": distribution,
        "errors": errors,
    }

    print(f"  Accept length: {accept_length:.3f}")
    print(f"  Acceptance rate: {acceptance_rate:.1f}%")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Total time: {elapsed:.1f}s")
    if distribution:
        print(f"  Distribution: {distribution}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    parser.add_argument(
        "--phase", default="phase2", help="Training phase (phase1 or phase2)"
    )
    parser.add_argument("--step", default="auto", help="Checkpoint step number")
    parser.add_argument(
        "--draft-model", default=None, help="Draft model path for metadata"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Concurrent requests")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_model_name(args.base_url)
    print(f"Using model: {model_name}")
    print("Backend: ATOM")
    print(f"Batch size: {args.batch_size}")

    initial_stats = get_atom_mtp_stats(args.base_url)
    if not initial_stats.get("enabled", False):
        print(
            "WARNING: MTP/Eagle3 stats not enabled — acceptance metrics will be unavailable"
        )

    mtp_k = len(initial_stats.get("distribution", {})) - 1
    if mtp_k <= 0:
        mtp_k = 3
    print(f"Detected num_speculative_tokens: {mtp_k}")

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
                batch_size=args.batch_size,
            )
            all_results[name] = result
        except (OSError, requests.RequestException, KeyError, ValueError) as e:
            print(f"ERROR running {name}: {e}")
            import traceback

            traceback.print_exc()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    step_str = args.step if args.step != "auto" else "latest"
    result_file = os.path.join(
        args.output_dir, f"{args.phase}_atom_eagle3_step{step_str}_{timestamp}.json"
    )
    draft_model = args.draft_model or f"/dev/shm/Kimi_K25_eagle3_v2_{args.phase}_HF"
    with open(result_file, "w") as f:
        json.dump(
            {
                "backend": "atom",
                "target_model": "/dev/shm/Kimi-K2.5-MXFP4",
                "draft_model": draft_model,
                "phase": args.phase,
                "checkpoint_step": step_str,
                "num_speculative_tokens": mtp_k,
                "benchmarks": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {result_file}")

    print(f"\n{'='*70}")
    print(
        f"{'Benchmark':<35} {'n':>5} {'Accept Len':>12} {'Acc Rate':>10} {'Tok/s':>8}"
    )
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(
            f"{name:<35} {r['num_samples']:>5} {r['accept_length']:>12.3f} {r['acceptance_rate']:>9.1f}% {r['throughput_tps']:>8.1f}"
        )
    print(f"{'='*70}")

    if all_results:
        avg_accept = sum(r["accept_length"] for r in all_results.values()) / len(
            all_results
        )
        avg_rate = sum(r["acceptance_rate"] for r in all_results.values()) / len(
            all_results
        )
        print(f"{'AVERAGE':<35} {'':>5} {avg_accept:>12.3f} {avg_rate:>9.1f}%")


if __name__ == "__main__":
    main()
