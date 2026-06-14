#!/usr/bin/env python3
"""Extract and compare all experiment results vs baseline."""

import re
import glob
import os
import json

dirs = {
    "baseline": "/app/benchmark_results/baseline_pr473",
    "gpu_util_095": (
        sorted(glob.glob("/app/benchmark_results/gpu_util_095_*"))[-1]
        if glob.glob("/app/benchmark_results/gpu_util_095_*")
        else ""
    ),
    "max_batch_8k": (
        sorted(glob.glob("/app/benchmark_results/max_batch_tokens_8k_*"))[-1]
        if glob.glob("/app/benchmark_results/max_batch_tokens_8k_*")
        else ""
    ),
    "moe_tune": (
        sorted(glob.glob("/app/benchmark_results/moe_threshold_tune_*"))[-1]
        if glob.glob("/app/benchmark_results/moe_threshold_tune_*")
        else ""
    ),
    "block_32": (
        sorted(glob.glob("/app/benchmark_results/block_size_32_*"))[-1]
        if glob.glob("/app/benchmark_results/block_size_32_*")
        else ""
    ),
    "combined": (
        sorted(glob.glob("/app/benchmark_results/combined_*"))[-1]
        if glob.glob("/app/benchmark_results/combined_*")
        else ""
    ),
}


def parse(text):
    tput = re.search(r"Output token throughput.*?(\d+\.?\d*)", text)
    ttft = re.search(r"Mean TTFT.*?(\d+\.?\d*)", text)
    ttft99 = re.search(r"P99 TTFT.*?(\d+\.?\d*)", text)
    tpot = re.search(r"Mean TPOT.*?(\d+\.?\d*)", text)
    tpot99 = re.search(r"P99 TPOT.*?(\d+\.?\d*)", text)
    if all(v is not None for v in [tput, ttft, ttft99, tpot, tpot99]):
        return {
            "throughput": float(tput.group(1)),
            "ttft_mean": float(ttft.group(1)),
            "ttft_p99": float(ttft99.group(1)),
            "tpot_mean": float(tpot.group(1)),
            "tpot_p99": float(tpot99.group(1)),
        }
    return None


# Collect all results
all_results = {}
for label, d in dirs.items():
    if not d:
        continue
    all_results[label] = {}
    for f in sorted(glob.glob(os.path.join(d, "*.stdout"))):
        name = os.path.basename(f).replace(".stdout", "")
        r = parse(open(f).read())
        if r:
            all_results[label][name] = r

# Print comparison tables
bl = all_results.get("baseline", {})
combined = all_results.get("combined", {})

print("=" * 100)
print(
    "FINAL PARETO COMPARISON: Baseline vs Combined (gpu_util_095 + max_batch_tokens_8k)"
)
print("=" * 100)

for scenario in ["1k_1k", "8k_1k"]:
    print(f"\n{'=' * 80}")
    print(
        f"  {scenario.upper()} (ISL={'1024' if '1k_1k' in scenario else '8192'}, OSL=1024)"
    )
    print(f"{'=' * 80}")
    print(
        f"  {'Conc':<6} {'BL Tput':>10} {'NEW Tput':>10} {'Delta':>8} {'BL TTFT':>10} {'NEW TTFT':>10} {'Delta':>8} {'BL TPOT':>10} {'NEW TPOT':>10} {'Delta':>8}"
    )
    print(f"  {'-' * 94}")

    for conc in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        key = f"{scenario}_c{conc}"
        b = bl.get(key)
        c = combined.get(key)
        if b and c:
            td = (c["throughput"] - b["throughput"]) / b["throughput"] * 100
            ttd = (b["ttft_mean"] - c["ttft_mean"]) / b["ttft_mean"] * 100
            tpd = (b["tpot_mean"] - c["tpot_mean"]) / b["tpot_mean"] * 100
            print(
                f"  {conc:<6} {b['throughput']:>10.1f} {c['throughput']:>10.1f} {td:>+7.1f}% "
                f"{b['ttft_mean']:>10.1f} {c['ttft_mean']:>10.1f} {ttd:>+7.1f}% "
                f"{b['tpot_mean']:>10.1f} {c['tpot_mean']:>10.1f} {tpd:>+7.1f}%"
            )
        elif b:
            print(
                f"  {conc:<6} {b['throughput']:>10.1f} {'N/A':>10} {'':>8} {b['ttft_mean']:>10.1f} {'N/A':>10}"
            )

# All experiment comparison at key points
print(f"\n\n{'=' * 100}")
print("ALL EXPERIMENTS AT KEY CONCURRENCY POINTS")
print(f"{'=' * 100}")

for scenario in ["1k_1k", "8k_1k"]:
    for conc in [1, 32, 64, 128, 256]:
        key = f"{scenario}_c{conc}"
        b = bl.get(key)
        if not b:
            continue
        print(f"\n  {key}:")
        print(
            f"    {'Label':<20} {'Throughput':>10} {'TTFT':>10} {'TPOT':>10} {'Tput %':>8} {'TTFT %':>8} {'TPOT %':>8}"
        )
        print(f"    {'-' * 78}")
        print(
            f"    {'baseline':<20} {b['throughput']:>10.1f} {b['ttft_mean']:>10.1f} {b['tpot_mean']:>10.1f} {'ref':>8} {'ref':>8} {'ref':>8}"
        )
        for label in [
            "gpu_util_095",
            "max_batch_8k",
            "moe_tune",
            "block_32",
            "combined",
        ]:
            r = all_results.get(label, {}).get(key)
            if r:
                td = (r["throughput"] - b["throughput"]) / b["throughput"] * 100
                ttd = (b["ttft_mean"] - r["ttft_mean"]) / b["ttft_mean"] * 100
                tpd = (b["tpot_mean"] - r["tpot_mean"]) / b["tpot_mean"] * 100
                print(
                    f"    {label:<20} {r['throughput']:>10.1f} {r['ttft_mean']:>10.1f} {r['tpot_mean']:>10.1f} {td:>+7.1f}% {ttd:>+7.1f}% {tpd:>+7.1f}%"
                )

# Output JSON summary
summary = {"baseline": bl, "combined": combined}
for label in ["gpu_util_095", "max_batch_8k", "moe_tune", "block_32"]:
    summary[label] = all_results.get(label, {})
json.dump(summary, open("/app/benchmark_results/final_comparison.json", "w"), indent=2)
print("\n\nSaved to /app/benchmark_results/final_comparison.json")
