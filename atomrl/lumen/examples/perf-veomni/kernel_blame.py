#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Attributes GPU kernels back to the host op that launched them, so a kernel
category can be turned into a list of operators to fuse rather than a total.

A kernel carries a correlation id shared with its cuda_runtime launch event.
The launch sits inside a stack of cpu_op frames on the launching thread; the
innermost one is the aten op responsible, and its ancestors give the module
context. Both are reported, because the innermost op alone ("aten::mul") does
not say which part of the model to change.

    kernel_blame.py <trace.json.gz> [--category elementwise] [--top 25]

Categories match trace_report.py's classify().
"""

import argparse
import gzip
import json
import re
from bisect import bisect_right
from collections import defaultdict

COMM_RX = re.compile(r"nccl|rccl", re.IGNORECASE)


def classify(nm):
    low = nm.lower()
    if COMM_RX.search(low):
        return "communication"
    if nm.startswith("Cijk_") or "gemm" in low or "hipblas" in low:
        return "GEMM"
    if "attn" in low or "bwd_kernel" in low or "softmax" in low or "flash" in low:
        return "attention"
    if "conv" in low or "miopen" in low or "implicit" in low:
        return "convolution"
    if "memcpy" in low:
        return "memcpy"
    if "adam" in low or "multi_tensor" in low:
        return "optimizer"
    if "norm" in low:
        return "normalisation"
    if "reduce" in low:
        return "reduction"
    if (
        "elementwise" in low
        or "vectorized" in low
        or "copy" in low
        or "cat" in low
        or "fill" in low
    ):
        return "elementwise"
    return "other"


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--category", default="elementwise")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    ev = load(args.trace)["traceEvents"]
    steps = [
        e
        for e in ev
        if e.get("name", "").startswith("ProfilerStep#")
        and e.get("cat") == "user_annotation"
    ]
    if not steps:
        raise SystemExit("no ProfilerStep# user_annotation events")
    n = len(steps)
    s0 = min(e["ts"] for e in steps)
    e0 = max(e["ts"] + e["dur"] for e in steps)
    wall = (e0 - s0) / n

    # kernels of interest, keyed by correlation
    want = defaultdict(float)
    kcount = defaultdict(int)
    kname = {}
    total_cat = 0.0
    for e in ev:
        if e.get("cat") != "kernel" or e.get("ph") != "X" or not (s0 <= e["ts"] <= e0):
            continue
        if classify(e["name"]) != args.category:
            continue
        c = (e.get("args") or {}).get("correlation")
        if c is None:
            continue
        want[c] += e["dur"]
        kcount[c] += 1
        kname[c] = e["name"]
        total_cat += e["dur"]

    # launches for those correlations, with their thread and timestamp
    launches = []
    for e in ev:
        if e.get("cat") == "cuda_runtime" and e.get("ph") == "X":
            c = (e.get("args") or {}).get("correlation")
            if c in want:
                launches.append((e.get("tid"), e["ts"], c))

    # cpu_op frames per thread, sorted; find the innermost frame covering a ts
    by_tid = defaultdict(list)
    for e in ev:
        if e.get("cat") == "cpu_op" and e.get("ph") == "X" and e.get("dur", 0) > 0:
            by_tid[e.get("tid")].append((e["ts"], e["ts"] + e["dur"], e["name"]))
    for v in by_tid.values():
        v.sort()
    starts = {tid: [f[0] for f in v] for tid, v in by_tid.items()}

    def frames_at(tid, ts):
        v = by_tid.get(tid)
        if not v:
            return []
        i = bisect_right(starts[tid], ts)
        return [f for f in v[:i] if f[1] >= ts]

    leaf = defaultdict(lambda: [0.0, 0])
    pair = defaultdict(lambda: [0.0, 0])
    unattributed = [0.0, 0]
    for tid, ts, c in launches:
        fr = frames_at(tid, ts)
        d, k = want[c], kcount[c]
        if not fr:
            unattributed[0] += d
            unattributed[1] += k
            continue
        fr.sort(key=lambda f: f[1] - f[0])
        innermost = fr[0][2]
        # outermost frame that is not the step annotation gives module context
        outer = max(fr, key=lambda f: f[1] - f[0])[2]
        leaf[innermost][0] += d
        leaf[innermost][1] += k
        pair[(outer, innermost)][0] += d
        pair[(outer, innermost)][1] += k

    print(f"=== {args.trace.split('/')[-2]}: category '{args.category}' ===")
    print(
        f"  {n} step(s), {wall / 1e3:.1f} ms/step; category total "
        f"{total_cat / 1e3 / n:.1f} GPU ms/step ({total_cat / (wall * n) * 100:.1f}% of wall), "
        f"{sum(kcount.values()) / n:.0f} kernels/step"
    )
    if unattributed[1]:
        print(
            f"  unattributed: {unattributed[0] / 1e3 / n:.1f} ms/step, {unattributed[1] / n:.0f} kernels/step"
        )

    print(f"\n  top {args.top} launching aten ops")
    print(f"  {'GPU ms':>8} {'% cat':>7} {'kernels':>9} {'us/kern':>8}  aten op")
    for nm, (d, k) in sorted(leaf.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(
            f"  {d / 1e3 / n:8.2f} {d / total_cat * 100:6.1f}% {k / n:9.0f} {d / k:8.1f}  {nm}"
        )

    print(f"\n  top {args.top} (outer frame -> aten op)")
    print(f"  {'GPU ms':>8} {'kernels':>9}  outer -> leaf")
    for (o, i), (d, k) in sorted(pair.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"  {d / 1e3 / n:8.2f} {k / n:9.0f}  {o}  ->  {i}")


if __name__ == "__main__":
    main()
