#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Kernel-level diff between two traces: which GPU kernels appeared, vanished or
changed cost, in ms per step.

This is the kernel-level evidence a replacement needs. A patch that reports
itself applied has proved nothing -- a counter-example on this workload logged
"846/846 nn.Linear patched" against a kernel list that did not move by a single
microsecond. Two
traces either differ in their kernels or the patch did not reach the GPU.

    kernel_diff.py <a.json.gz> <b.json.gz> [--top 25]

Costs are normalised per step (divided by the number of ProfilerStep spans in
each trace), so traces with different window lengths stay comparable.
"""

import argparse
import gzip
import json
from collections import defaultdict

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        doc = json.load(fh)
    events = doc.get("traceEvents", doc)
    # Count distinct step names, not events: a ProfilerStep span is emitted
    # once per profiler thread row, so counting occurrences divides the
    # per-step cost by an arbitrary factor (3 on these traces).
    steps = len(
        {
            e["name"]
            for e in events
            if str(e.get("name", "")).startswith("ProfilerStep#")
        }
    )
    per_kernel = defaultdict(lambda: [0.0, 0])
    for e in events:
        if e.get("cat") in GPU_CATS and e.get("dur"):
            slot = per_kernel[e["name"]]
            slot[0] += e["dur"] / 1000.0
            slot[1] += 1
    steps = max(steps, 1)
    return {k: (v[0] / steps, v[1] / steps) for k, v in per_kernel.items()}, steps


def short(name, width=62):
    return name if len(name) <= width else name[: width - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    ka, steps_a = load(args.a)
    kb, steps_b = load(args.b)
    print(f"A = {args.a}   ({steps_a} steps)")
    print(f"B = {args.b}   ({steps_b} steps)\n")

    names = set(ka) | set(kb)
    rows = []
    for n in names:
        ams, acalls = ka.get(n, (0.0, 0))
        bms, bcalls = kb.get(n, (0.0, 0))
        rows.append((bms - ams, n, ams, bms, acalls, bcalls))
    rows.sort(key=lambda r: -abs(r[0]))

    tot_a = sum(v[0] for v in ka.values())
    tot_b = sum(v[0] for v in kb.values())
    print(f"{'delta ms':>10}{'A ms':>10}{'B ms':>10}{'A n':>7}{'B n':>7}  kernel")
    print("-" * 110)
    for delta, n, ams, bms, acalls, bcalls in rows[: args.top]:
        tag = "  <- GONE" if bms == 0 else ("  <- NEW" if ams == 0 else "")
        print(
            f"{delta:>+10.2f}{ams:>10.2f}{bms:>10.2f}{acalls:>7.0f}{bcalls:>7.0f}  {short(n)}{tag}"
        )

    gone = [n for n in ka if n not in kb]
    new = [n for n in kb if n not in ka]
    print(
        f"\nkernels only in A: {len(gone)}   only in B: {len(new)}   shared: {len(set(ka) & set(kb))}"
    )
    print(
        f"total GPU kernel time: A {tot_a:.1f} ms/step -> B {tot_b:.1f} ms/step  ({tot_b - tot_a:+.1f} ms)"
    )
    if not gone and not new:
        print("NO kernel appeared or vanished -- if a patch claims to have replaced an")
        print("operator, this is the trace saying it did not reach the GPU.")


if __name__ == "__main__":
    main()
