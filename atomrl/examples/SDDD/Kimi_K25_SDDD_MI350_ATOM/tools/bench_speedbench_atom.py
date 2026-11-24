# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
Benchmark Eagle3 speculative decoding with ATOM using SPEED-Bench dataset.

Measures accept_length per category from the SPEED-Bench qualitative split,
matching the evaluation protocol in Table 1 of arXiv:2604.09557.
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


def query_server(base_url, model_name, messages, max_tokens=2048, no_think=False):
    body = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def get_atom_mtp_stats(base_url):
    try:
        resp = requests.get(f"{base_url}/debug/mtp_stats", timeout=10)
        return resp.json()
    except (requests.RequestException, ValueError):
        return {}


def run_single_sample(base_url, model_name, sample, max_tokens=2048, no_think=False):
    turns = sample["turns"]
    messages = []
    total_tokens = 0
    for turn in turns:
        messages.append({"role": "user", "content": turn})
        result = query_server(
            base_url, model_name, messages, max_tokens=max_tokens, no_think=no_think
        )
        usage = result.get("usage", {})
        total_tokens += usage.get("completion_tokens", 0)
        assistant_msg = result["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_msg})
    return total_tokens


def run_category_benchmark(
    base_url,
    model_name,
    category,
    samples,
    max_tokens=2048,
    batch_size=1,
    no_think=False,
):
    print(f"\n{'='*60}")
    print(f"Running: {category} ({len(samples)} samples, BS={batch_size})")
    print(f"{'='*60}")

    stats_before = get_atom_mtp_stats(base_url)
    draft_before = stats_before.get("total_draft_tokens", 0)
    accepted_before = stats_before.get("total_accepted_tokens", 0)
    dist_before = stats_before.get("distribution", {})

    total_output_tokens = 0
    errors = 0
    done_count = 0
    lock = Lock()
    start = time.time()

    def _worker(sample):
        return run_single_sample(
            base_url, model_name, sample, max_tokens, no_think=no_think
        )

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

    stats_after = get_atom_mtp_stats(base_url)
    draft_after = stats_after.get("total_draft_tokens", 0)
    accepted_after = stats_after.get("total_accepted_tokens", 0)
    dist_after = stats_after.get("distribution", {})

    num_draft_tokens = draft_after - draft_before
    num_accepted = accepted_after - accepted_before

    distribution = {}
    for k in dist_after:
        distribution[k] = dist_after[k] - dist_before.get(k, 0)

    num_steps = sum(distribution.values()) if distribution else 0
    accept_length = 1.0 + (num_accepted / num_steps) if num_steps > 0 else 1.0
    acceptance_rate = (
        (num_accepted / num_draft_tokens * 100) if num_draft_tokens > 0 else 0
    )
    throughput = total_output_tokens / elapsed if elapsed > 0 else 0

    result = {
        "category": category,
        "num_samples": len(samples),
        "batch_size": batch_size,
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

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--data-path", required=True, help="Path to SPEED-Bench qualitative parquet"
    )
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--phase", default="phase2")
    parser.add_argument("--step", default="latest")
    parser.add_argument("--draft-model", default=None)
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable thinking mode (enable_thinking=false)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_model_name(args.base_url)
    print(f"Using model: {model_name}")
    print("Backend: ATOM")
    print(f"Batch size: {args.batch_size}")
    if args.no_think:
        print("Thinking mode: DISABLED")

    initial_stats = get_atom_mtp_stats(args.base_url)
    if not initial_stats.get("enabled", False):
        print(
            "WARNING: MTP/Eagle3 stats not enabled — acceptance metrics will be unavailable"
        )

    mtp_k = len(initial_stats.get("distribution", {})) - 1
    if mtp_k <= 0:
        mtp_k = 3
    print(f"Detected num_speculative_tokens: {mtp_k}")

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
            print(f"WARN: Unknown category '{cat}', skipping")
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
                no_think=args.no_think,
            )
            all_results[cat] = result
        except (OSError, requests.RequestException, KeyError, ValueError) as e:
            print(f"ERROR running {cat}: {e}")
            import traceback

            traceback.print_exc()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    think_tag = "_nothink" if args.no_think else ""
    result_file = os.path.join(
        args.output_dir,
        f"{args.phase}_speedbench_atom_eagle3_bs{args.batch_size}{think_tag}_step{args.step}_{timestamp}.json",
    )
    draft_model = args.draft_model or f"/dev/shm/Kimi_K25_eagle3_v2_{args.phase}_HF"
    with open(result_file, "w") as f:
        json.dump(
            {
                "backend": "atom",
                "benchmark": "SPEED-Bench",
                "target_model": "/dev/shm/Kimi-K2.5-MXFP4",
                "draft_model": draft_model,
                "phase": args.phase,
                "checkpoint_step": args.step,
                "num_speculative_tokens": mtp_k,
                "batch_size": args.batch_size,
                "categories": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {result_file}")

    print(f"\n{'='*70}")
    print(f"{'Category':<20} {'n':>5} {'Accept Len':>12} {'Acc Rate':>10} {'Tok/s':>8}")
    print(f"{'='*70}")
    for cat, r in all_results.items():
        print(
            f"{cat:<20} {r['num_samples']:>5} {r['accept_length']:>12.3f} {r['acceptance_rate']:>9.1f}% {r['throughput_tps']:>8.1f}"
        )
    print(f"{'='*70}")

    if all_results:
        avg_al = sum(r["accept_length"] for r in all_results.values()) / len(
            all_results
        )
        avg_rate = sum(r["acceptance_rate"] for r in all_results.values()) / len(
            all_results
        )
        print(f"{'MEAN':<20} {'':>5} {avg_al:>12.3f} {avg_rate:>9.1f}%")


if __name__ == "__main__":
    main()
