#!/usr/bin/env python3
"""Analyze CU-mask timing data across multiple disaggregated prefill/decode runs.

Reads iter_times_*.log files from the working directory (or --dir) and produces:
  1. Prefill P50/P90 latency by token bucket and CU config
  2. Decode P50/P90 latency by batch size and CU config
  3. Prefill/decode throughput comparison
  4. Tradeoff matrix (prefill speedup vs decode penalty)
  5. Concurrent bottleneck analysis (max of prefill, decode)
  6. Optimal CU config lookup table

Usage:
    python analyze_cu_partitioning.py [--dir /path/to/logs]
"""

import argparse
import os
import re
import statistics

PREFILL_RE = re.compile(r"prefill iter ([\d.]+)ms \| reqs=(\d+) \| tokens=(\d+)")
DECODE_RE = re.compile(
    r"(?<!prefill )iter ([\d.]+)ms \| reqs=(\d+) "
    r"\(prefill=(\d+) decode=(\d+)\) \| "
    r"tokens=(\d+) \(prefill=(\d+) decode=(\d+)\)"
)

TOKEN_BUCKETS = [
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
    (4096, 8192),
    (8192, 16384),
]
DECODE_BUCKETS = [
    (1, 16),
    (17, 32),
    (33, 64),
    (65, 96),
    (97, 128),
]

# Map config label -> (prefill CU%, decode CU%)
CU_SPLIT = {
    "30cu": (30, 70),
    "40cu": (40, 60),
    "50cu": (50, 50),
    "60cu": (60, 40),
    "70cu": (70, 30),
    "80cu": (80, 20),
}


def pctl(arr, p):
    """Return the p-th percentile of a sorted-able list."""
    if not arr:
        return None
    arr = sorted(arr)
    idx = min(int(len(arr) * p), len(arr) - 1)
    return arr[idx]


def parse_file(path):
    prefill, decode = [], []
    with open(path) as f:
        for line in f:
            m = PREFILL_RE.search(line)
            if m:
                prefill.append((float(m.group(1)), int(m.group(2)), int(m.group(3))))
                continue
            m = DECODE_RE.search(line)
            if m:
                decode.append(
                    (
                        float(m.group(1)),
                        int(m.group(2)),
                        int(m.group(3)),
                        int(m.group(4)),
                        int(m.group(5)),
                        int(m.group(6)),
                        int(m.group(7)),
                    )
                )
    return prefill, decode


def discover_logs(directory):
    """Find iter_times_*.log files and derive config labels."""
    logs = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.startswith("iter_times_") or not fname.endswith(".log"):
            continue
        label = fname.replace("iter_times_", "").replace(".log", "")
        logs[label] = os.path.join(directory, fname)
    return logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default=".", help="Directory containing iter_times_*.log files"
    )
    args = parser.parse_args()

    log_files = discover_logs(args.dir)
    if not log_files:
        print(f"No iter_times_*.log files found in {args.dir}")
        return

    results = {}
    for label, path in log_files.items():
        pf, dc = parse_file(path)
        if pf or dc:
            results[label] = {"prefill": pf, "decode": dc}

    labels_order = []
    # torch baselines first, then CU configs in ascending order
    for l in ["torch", "torch_i8k"]:
        if l in results:
            labels_order.append(l)
    for l in sorted(results.keys()):
        if l not in labels_order:
            labels_order.append(l)

    cu_configs = [l for l in labels_order if l not in ("torch", "torch_i8k")]
    baseline = "torch" if "torch" in results else labels_order[0]

    # ─── Build P50 lookup tables ───
    pf_p50 = {}
    for l in labels_order:
        pf_p50[l] = {}
        for lo, hi in TOKEN_BUCKETS:
            bucket = [ms for ms, _, tok in results[l]["prefill"] if lo <= tok < hi]
            pf_p50[l][(lo, hi)] = pctl(bucket, 0.5)

    dc_p50 = {}
    for l in labels_order:
        dc_p50[l] = {}
        pure = [
            (ms, d_r)
            for ms, tr, pr, d_r, tt, pt, dt in results[l]["decode"]
            if pr == 0 and d_r > 0
        ]
        for lo, hi in DECODE_BUCKETS:
            bucket = [ms for ms, d_r in pure if lo <= d_r <= hi]
            dc_p50[l][(lo, hi)] = pctl(bucket, 0.5)

    W = 130

    # ═══ SECTION 1: Prefill P50 ═══
    print("=" * W)
    print("PREFILL P50 LATENCY (ms) — rows=token bucket, cols=CU config")
    print("=" * W)
    active_pf = [l for l in labels_order if results[l]["prefill"]]
    hdr = f"{'Tokens':>15}"
    for l in active_pf:
        hdr += f" {l:>12}"
    print(hdr)
    for lo, hi in TOKEN_BUCKETS:
        row = f"{f'[{lo},{hi})':>15}"
        any_data = False
        for l in active_pf:
            bucket = sorted(
                [ms for ms, _, tok in results[l]["prefill"] if lo <= tok < hi]
            )
            if bucket:
                row += f" {pctl(bucket, .5):>12.2f}"
                any_data = True
            else:
                row += f" {'—':>12}"
        if any_data:
            print(row)

    # ═══ SECTION 2: Decode P50 ═══
    print("\n" + "=" * W)
    print("DECODE P50 LATENCY (ms) — pure decode only (prefill_reqs=0)")
    print("=" * W)
    active_dc = [l for l in labels_order if results[l]["decode"]]
    hdr = f"{'Decode Batch':>15}"
    for l in active_dc:
        hdr += f" {l:>12}"
    print(hdr)
    for lo, hi in DECODE_BUCKETS:
        row = f"{f'[{lo},{hi}]':>15}"
        any_data = False
        for l in active_dc:
            pure = [
                (ms, d_r)
                for ms, tr, pr, d_r, tt, pt, dt in results[l]["decode"]
                if pr == 0 and d_r > 0
            ]
            bucket = sorted([ms for ms, d_r in pure if lo <= d_r <= hi])
            if bucket:
                row += f" {pctl(bucket, .5):>12.2f}"
                any_data = True
            else:
                row += f" {'—':>12}"
        if any_data:
            print(row)

    # ═══ SECTION 3: Decode P90 ═══
    print("\n" + "=" * W)
    print("DECODE P90 LATENCY (ms)")
    print("=" * W)
    hdr = f"{'Decode Batch':>15}"
    for l in active_dc:
        hdr += f" {l:>12}"
    print(hdr)
    for lo, hi in DECODE_BUCKETS:
        row = f"{f'[{lo},{hi}]':>15}"
        any_data = False
        for l in active_dc:
            pure = [
                (ms, d_r)
                for ms, tr, pr, d_r, tt, pt, dt in results[l]["decode"]
                if pr == 0 and d_r > 0
            ]
            bucket = sorted([ms for ms, d_r in pure if lo <= d_r <= hi])
            if bucket:
                row += f" {pctl(bucket, .9):>12.2f}"
                any_data = True
            else:
                row += f" {'—':>12}"
        if any_data:
            print(row)

    # ═══ SECTION 4: Prefill P90 ═══
    print("\n" + "=" * W)
    print("PREFILL P90 LATENCY (ms)")
    print("=" * W)
    hdr = f"{'Tokens':>15}"
    for l in active_pf:
        hdr += f" {l:>12}"
    print(hdr)
    for lo, hi in TOKEN_BUCKETS:
        row = f"{f'[{lo},{hi})':>15}"
        any_data = False
        for l in active_pf:
            bucket = sorted(
                [ms for ms, _, tok in results[l]["prefill"] if lo <= tok < hi]
            )
            if bucket:
                row += f" {pctl(bucket, .9):>12.2f}"
                any_data = True
            else:
                row += f" {'—':>12}"
        if any_data:
            print(row)

    # ═══ SECTION 5: Throughput ═══
    print("\n" + "=" * W)
    print("PREFILL THROUGHPUT (tok/s) — P50")
    print("=" * W)
    hdr = f"{'Tokens':>15}"
    for l in active_pf:
        hdr += f" {l:>12}"
    print(hdr)
    for lo, hi in TOKEN_BUCKETS:
        row = f"{f'[{lo},{hi})':>15}"
        any_data = False
        for l in active_pf:
            bucket = [
                tok / (ms / 1000)
                for ms, _, tok in results[l]["prefill"]
                if lo <= tok < hi
            ]
            if bucket:
                row += f" {pctl(bucket, .5):>12.0f}"
                any_data = True
            else:
                row += f" {'—':>12}"
        if any_data:
            print(row)

    print("\n" + "=" * W)
    print("DECODE THROUGHPUT (tok/s) — P50 (pure decode)")
    print("=" * W)
    hdr = f"{'Decode Batch':>15}"
    for l in active_dc:
        hdr += f" {l:>12}"
    print(hdr)
    for lo, hi in DECODE_BUCKETS:
        row = f"{f'[{lo},{hi}]':>15}"
        any_data = False
        for l in active_dc:
            pure = [
                (d_r / (ms / 1000), d_r)
                for ms, tr, pr, d_r, tt, pt, dt in results[l]["decode"]
                if pr == 0 and d_r > 0
            ]
            bucket = [tps for tps, d_r in pure if lo <= d_r <= hi]
            if bucket:
                row += f" {pctl(bucket, .5):>12.0f}"
                any_data = True
            else:
                row += f" {'—':>12}"
        if any_data:
            print(row)

    # ═══ SECTION 6: Tradeoff matrix ═══
    if baseline in results:
        print("\n" + "=" * W)
        print(f"TRADEOFF MATRIX vs '{baseline}': prefill speedup / decode penalty")
        print("  Speedup > 1 = faster prefill; Penalty > 1 = slower decode")
        print("=" * W)

        print(f"\n{'Config':>8} {'PF%':>5} {'DC%':>5} | ", end="")
        for lo, hi in TOKEN_BUCKETS:
            print(f"PF[{lo}-{hi}) ", end="")
        print("| ", end="")
        for lo, hi in DECODE_BUCKETS:
            print(f"DC[{lo}-{hi}] ", end="")
        print()
        print("-" * W)

        for cfg in cu_configs:
            pf_cu, dc_cu = CU_SPLIT.get(cfg, (0, 0))
            row = f"{cfg:>8} {pf_cu:>4}% {dc_cu:>4}% | "
            for lo, hi in TOKEN_BUCKETS:
                t_base = pf_p50[baseline].get((lo, hi))
                t_cfg = pf_p50[cfg].get((lo, hi))
                if t_base and t_cfg:
                    row += f"  {t_base / t_cfg:>5.2f}x   "
                else:
                    row += "    —      "
            row += "| "
            for lo, hi in DECODE_BUCKETS:
                t_base = dc_p50[baseline].get((lo, hi))
                t_cfg = dc_p50[cfg].get((lo, hi))
                if t_base and t_cfg:
                    row += f"  {t_cfg / t_base:>5.2f}x  "
                else:
                    row += "    —     "
            print(row)

    # ═══ SECTION 7: Concurrent bottleneck ═══
    print("\n" + "=" * W)
    print("CONCURRENT BOTTLENECK: max(prefill_p50, decode_p50) — lower is better")
    print("  Prefill and decode run on separate streams; iteration = max of both")
    print("=" * W)

    all_configs = [baseline] + cu_configs if baseline in results else cu_configs

    print(f"{'PF Tokens':>12} {'DC Batch':>10} | ", end="")
    for cfg in all_configs:
        print(f" {cfg:>8}", end="")
    print(f" | {'BEST':>8} {'Speedup':>8}")
    print("-" * W)

    for pf_lo, pf_hi in TOKEN_BUCKETS:
        for dc_lo, dc_hi in DECODE_BUCKETS:
            row = f"{f'[{pf_lo},{pf_hi})':>12} {f'[{dc_lo},{dc_hi}]':>10} | "
            best_cfg = None
            best_val = float("inf")
            base_val = float("inf")

            for cfg in all_configs:
                cp = pf_p50.get(cfg, {}).get((pf_lo, pf_hi))
                cd = dc_p50.get(cfg, {}).get((dc_lo, dc_hi))
                if cp is not None and cd is not None:
                    bottleneck = max(cp, cd)
                    row += f" {bottleneck:>8.1f}"
                    if cfg == baseline:
                        base_val = bottleneck
                    if bottleneck < best_val:
                        best_val = bottleneck
                        best_cfg = cfg
                else:
                    row += f" {'—':>8}"

            speedup = (
                base_val / best_val if best_val > 0 and base_val < float("inf") else 0
            )
            row += f" | {best_cfg or '—':>8} {speedup:>7.2f}x"
            print(row)

    # ═══ SECTION 8: Summary ═══
    print("\n" + "=" * W)
    print("OVERALL SUMMARY")
    print("=" * W)
    for label in labels_order:
        pf = results[label]["prefill"]
        dc = results[label]["decode"]
        pure_dc = [
            (ms, d_r) for ms, tr, pr, d_r, tt, pt, dt in dc if pr == 0 and d_r > 0
        ]
        print(f"\n  {label}:")
        print(f"    Prefill iterations: {len(pf)}")
        if pf:
            pf_ms = [ms for ms, _, _ in pf]
            pf_tok = [t for _, _, t in pf]
            print(
                f"    Prefill tokens: min={min(pf_tok)} max={max(pf_tok)} "
                f"mean={statistics.mean(pf_tok):.0f}"
            )
            print(
                f"    Prefill latency: mean={statistics.mean(pf_ms):.1f}ms "
                f"p50={pctl(pf_ms, .5):.1f}ms p90={pctl(pf_ms, .9):.1f}ms"
            )
        print(f"    Decode iterations (pure): {len(pure_dc)}")
        if pure_dc:
            dc_ms = [ms for ms, _ in pure_dc]
            dc_bs = [bs for _, bs in pure_dc]
            print(
                f"    Decode batch: min={min(dc_bs)} max={max(dc_bs)} "
                f"mean={statistics.mean(dc_bs):.1f}"
            )
            print(
                f"    Decode latency: mean={statistics.mean(dc_ms):.1f}ms "
                f"p50={pctl(dc_ms, .5):.1f}ms p90={pctl(dc_ms, .9):.1f}ms"
            )

    # ═══ SECTION 9: Optimal config per scenario ═══
    if baseline in results:
        print("\n" + "=" * W)
        print("OPTIMAL CU CONFIG per workload (lowest P50)")
        print("=" * W)

        print("\nPrefill — best CU% per token bucket:")
        for lo, hi in TOKEN_BUCKETS:
            best_label, best_p = None, float("inf")
            for l in cu_configs:
                bucket = sorted(
                    [ms for ms, _, tok in results[l]["prefill"] if lo <= tok < hi]
                )
                if bucket:
                    v = pctl(bucket, 0.5)
                    if v < best_p:
                        best_p, best_label = v, l
            if best_label:
                base_bucket = sorted(
                    [
                        ms
                        for ms, _, tok in results[baseline]["prefill"]
                        if lo <= tok < hi
                    ]
                )
                base_p = pctl(base_bucket, 0.5) if base_bucket else float("inf")
                sp = base_p / best_p if best_p > 0 else 0
                print(
                    f"  [{lo},{hi}): best={best_label} ({best_p:.1f}ms) "
                    f"vs {baseline} ({base_p:.1f}ms) = {sp:.2f}x"
                )

        print("\nDecode — best CU% per batch size:")
        for lo, hi in DECODE_BUCKETS:
            best_label, best_p = None, float("inf")
            for l in cu_configs:
                pure = [
                    (ms, d_r)
                    for ms, tr, pr, d_r, tt, pt, dt in results[l]["decode"]
                    if pr == 0 and d_r > 0
                ]
                bucket = sorted([ms for ms, d_r in pure if lo <= d_r <= hi])
                if bucket:
                    v = pctl(bucket, 0.5)
                    if v < best_p:
                        best_p, best_label = v, l
            if best_label:
                base_pure = [
                    (ms, d_r)
                    for ms, tr, pr, d_r, tt, pt, dt in results[baseline]["decode"]
                    if pr == 0 and d_r > 0
                ]
                base_bucket = sorted([ms for ms, d_r in base_pure if lo <= d_r <= hi])
                base_p = pctl(base_bucket, 0.5) if base_bucket else float("inf")
                sp = base_p / best_p if best_p > 0 else 0
                print(
                    f"  [{lo},{hi}]: best={best_label} ({best_p:.1f}ms) "
                    f"vs {baseline} ({base_p:.1f}ms) = {sp:.2f}x"
                )


if __name__ == "__main__":
    main()
