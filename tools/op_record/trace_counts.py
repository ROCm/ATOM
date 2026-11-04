#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-step kernel counts and GPU time from an ATOM torch profiler trace, to
check `export.py`'s per-step op counts against what the GPU actually ran and
to see how much GPU time the exported tuner families cover.

    python tools/op_record/trace_counts.py TRACE.json.gz [--phase decode] [--top 30]

A step is one ``decode[...]`` (or ``prefill[...]``) CPU annotation; the GPU
copy of each label (``gpu_user_annotation``) is not counted, or every
per-step figure would halve. ``--match REGEX`` sums the kernels it matches,
e.g. one family's kernels.
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter


def load(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)["traceEvents"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("trace")
    p.add_argument("--phase", default="decode", choices=["decode", "prefill"])
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--match", action="append", default=[], help="regex, repeatable")
    a = p.parse_args(argv)

    events = load(a.trace)
    steps = sum(
        1
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "user_annotation"
        and str(e.get("name", "")).startswith(f"{a.phase}[")
    )
    if not steps:
        raise SystemExit(f"no {a.phase}[...] step annotation in {a.trace}")
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    total = sum(e["dur"] for e in kernels)
    count, time = Counter(), Counter()
    for e in kernels:
        count[e["name"]] += 1
        time[e["name"]] += e["dur"]

    print(
        f"{steps} {a.phase} steps; {len(kernels) / steps:.1f} kernels and "
        f"{total / steps:.0f} us of kernel time per step "
        "(kernels on concurrent streams add up)"
    )
    for pattern in a.match:
        names = [n for n in count if re.search(pattern, n)]
        n = sum(count[k] for k in names)
        t = sum(time[k] for k in names)
        print(
            f"match {pattern!r}: {len(names)} kernel names, {n / steps:.2f}/step, "
            f"{t / steps:.0f} us/step, {100 * t / total:.1f}%"
        )
    print(f"{'per step':>9} {'us/step':>8} {'share':>6}  kernel")
    for name, t in time.most_common(a.top):
        print(
            f"{count[name] / steps:9.2f} {t / steps:8.1f} {100 * t / total:5.1f}%  "
            f"{name[:100]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
