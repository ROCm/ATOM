#!/usr/bin/env python3
import re
import glob
import sys
import os

results_dir = (
    sys.argv[1] if len(sys.argv) > 1 else "/app/benchmark_results/baseline_pr473"
)
files = sorted(glob.glob(os.path.join(results_dir, "*.stdout")))
print(
    f"{'Scenario':<20} {'Tput(tok/s)':>12} {'TTFT mean':>10} {'TTFT p99':>10} {'TPOT mean':>10} {'TPOT p99':>10}"
)
print("-" * 82)
for f in files:
    name = os.path.basename(f).replace(".stdout", "")
    text = open(f).read()
    tput = re.search(r"Output token throughput.*?(\d+\.?\d*)", text)
    ttft_mean = re.search(r"Mean TTFT.*?(\d+\.?\d*)", text)
    ttft_p99 = re.search(r"P99 TTFT.*?(\d+\.?\d*)", text)
    tpot_mean = re.search(r"Mean TPOT.*?(\d+\.?\d*)", text)
    tpot_p99 = re.search(r"P99 TPOT.*?(\d+\.?\d*)", text)
    vals = [tput, ttft_mean, ttft_p99, tpot_mean, tpot_p99]
    if all(v is not None for v in vals):
        print(
            f"{name:<20} {float(tput.group(1)):>12.1f} {float(ttft_mean.group(1)):>10.1f} {float(ttft_p99.group(1)):>10.1f} {float(tpot_mean.group(1)):>10.1f} {float(tpot_p99.group(1)):>10.1f}"
        )
