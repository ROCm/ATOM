# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
Benchmark baseline (no speculative decoding) with ATOM using SPEED-Bench dataset.
Measures throughput only — no accept_length since there's no drafting.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pyarrow.parquet as pq
import requests

PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED"


def get_model_name(base_url):
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        data = resp.json()
        return data["data"][0]["id"]
    except (requests.RequestException, IndexError, KeyError, ValueError):
        return "/dev/shm/Kimi-K2.5-MXFP4"


def load_speedbench(parquet_path):
    table = pq.read_table(parquet_path)
    data = table.to_pydict()
    samples = []
    for i in range(len(data["question_id"])):
        turns = data["turns"][i]
        if any(PLACEHOLDER in t for t in turns):
            continue
        samples.append(
            {
                "question_id": data["question_id"][i],
                "category": data["category"][i],
                "turns": turns,
                "multiturn": data["multiturn"][i],
            }
        )
    return samples


def query_server(base_url, model_name, messages, max_tokens=2048):
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def run_single_sample(base_url, model_name, sample, max_tokens=2048):
    turns = sample["turns"]
    messages = []
    total_tokens = 0
    for turn in turns:
        messages.append({"role": "user", "content": turn})
        result = query_server(base_url, model_name, messages, max_tokens=max_tokens)
        usage = result.get("usage", {})
        total_tokens += usage.get("completion_tokens", 0)
        assistant_msg = result["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_msg})
    return total_tokens


def run_category_benchmark(
    base_url, model_name, category, samples, max_tokens=2048, batch_size=1
):
    print(f"\n{'='*60}")
    print(f"Running: {category} ({len(samples)} samples, BS={batch_size})")
    print(f"{'='*60}")

    total_output_tokens = 0
    errors = 0
    done_count = 0
    lock = Lock()
    start = time.time()

    def _worker(sample):
        return run_single_sample(base_url, model_name, sample, max_tokens)

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_worker, s): i for i, s in enumerate(samples)}
        for future in as_completed(futures):
            try:
                tokens = future.result()
                with lock:
                    total_output_tokens += tokens
                    done_count += 1
                    if done_count % 20 == 0:
                        elapsed = time.time() - start
                        print(
                            f"  Progress: {done_count}/{len(samples)} ({elapsed:.1f}s)"
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

    throughput = total_output_tokens / elapsed if elapsed > 0 else 0

    result = {
        "category": category,
        "num_samples": len(samples),
        "batch_size": batch_size,
        "total_output_tokens": total_output_tokens,
        "throughput_tps": round(throughput, 1),
        "latency_s": round(elapsed, 1),
        "errors": errors,
    }

    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Total time: {elapsed:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--phase", default="phase2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_model_name(args.base_url)
    print(f"Using model: {model_name}")
    print("Backend: ATOM (no speculative decoding)")
    print(f"Batch size: {args.batch_size}")

    samples = load_speedbench(args.data_path)
    print(f"Loaded {len(samples)} resolved samples")

    by_category = {}
    for s in samples:
        by_category.setdefault(s["category"], []).append(s)

    if "all" in args.categories:
        run_categories = sorted(by_category.keys())
    else:
        run_categories = args.categories

    all_results = {}
    for cat in run_categories:
        if cat not in by_category:
            continue
        cat_samples = by_category[cat]
        try:
            result = run_category_benchmark(
                args.base_url,
                model_name,
                cat,
                cat_samples,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
            )
            all_results[cat] = result
        except (OSError, requests.RequestException, KeyError, ValueError) as e:
            print(f"ERROR running {cat}: {e}")
            import traceback

            traceback.print_exc()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        args.output_dir,
        f"{args.phase}_speedbench_atom_baseline_bs{args.batch_size}_{timestamp}.json",
    )
    with open(result_file, "w") as f:
        json.dump(
            {
                "backend": "atom",
                "benchmark": "SPEED-Bench",
                "target_model": "/dev/shm/Kimi-K2.5-MXFP4",
                "speculative_decoding": False,
                "phase": args.phase,
                "batch_size": args.batch_size,
                "temperature": 0,
                "categories": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {result_file}")

    print(f"\n{'='*58}")
    print(f"{'Category':<20} {'n':>5} {'Tok/s':>8} {'Time':>8}")
    print(f"{'='*58}")
    for cat, r in all_results.items():
        print(
            f"{cat:<20} {r['num_samples']:>5} {r['throughput_tps']:>8.1f} {r['latency_s']:>7.1f}s"
        )
    print(f"{'='*58}")
    if all_results:
        avg_tps = sum(r["throughput_tps"] for r in all_results.values()) / len(
            all_results
        )
        print(f"{'MEAN':<20} {'':>5} {avg_tps:>8.1f}")


if __name__ == "__main__":
    main()
