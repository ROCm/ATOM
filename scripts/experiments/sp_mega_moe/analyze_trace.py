#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Count fusion kernels and retain per-chunk timings from a one-request trace."""

import argparse
import csv
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["native", "mega"], default="mega")
    parser.add_argument("--chunks", type=int, default=1)
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("chunks must be positive")
    paths = sorted(args.traces.glob("rank_*/*.pt.trace.json*"))
    if len(paths) != 4 or {p.parent.name for p in paths} != {
        f"rank_{i}" for i in range(4)
    }:
        raise RuntimeError(
            "Expected exactly one trace per rank; use a fresh trace directory"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    rows, counts = [], []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            trace = json.load(stream)
        kernels = sorted(
            (e for e in trace["traceEvents"] if e.get("cat") == "kernel"),
            key=lambda e: e["ts"],
        )
        names = Counter(e["name"] for e in kernels)
        count = {
            "rank": path.parent.name,
            "prefix": sum(
                n
                for name, n in names.items()
                if name.startswith("sp_rs_norm_h6144_tp1")
            ),
            "fp8_tail": sum(
                n
                for name, n in names.items()
                if name.startswith("mega_moe_tp_fused_") and "_tn1" in name
            ),
            "bf16_tail": sum(
                n
                for name, n in names.items()
                if name.startswith("mega_moe_tp_fused_") and "_tn2" in name
            ),
        }
        (args.output / f"{path.parent.name}-kernel-counts.json").write_text(
            json.dumps(names, indent=2)
        )
        counts.append(count)
        if args.mode == "mega" and (
            count["prefix"],
            count["fp8_tail"],
            count["bf16_tail"],
        ) != (57 * args.chunks, 56 * args.chunks, args.chunks):
            raise RuntimeError(f"Fusion count mismatch: {count}")
        marker = "mega_moe_tp_fused_" if args.mode == "mega" else "QuickReduceScatter"
        events = [e for e in kernels if marker in e["name"]]
        if len(events) != 57 * args.chunks:
            raise RuntimeError(
                f"Unexpected marker count: {path}, {marker}, {len(events)}"
            )
        for chunk in range(args.chunks):
            group = events[57 * chunk : 57 * (chunk + 1)]
            rows.append(
                {
                    "rank": path.parent.name,
                    "chunk": chunk + 1,
                    "marker": marker,
                    "mean_us": statistics.mean(e["dur"] for e in group),
                    "median_us": statistics.median(e["dur"] for e in group),
                    "first_marker_epoch_s": (
                        trace["baseTimeNanoseconds"] / 1e9 + group[0]["ts"] / 1e6
                    ),
                }
            )
    with (args.output / "per-chunk.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "fusion-counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    # Native markers cover RS only; Mega markers include compute and tail.
    # These columns diagnose drift within a mode, not the A/B regional speedup.


if __name__ == "__main__":
    main()
