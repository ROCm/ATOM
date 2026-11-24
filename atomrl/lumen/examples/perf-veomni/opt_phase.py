#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Step time and optimizer-step cost from stack-free traces, read off PyTorch's
own "Optimizer.step#<cls>.step" profiler range rather than off Python frames,
so the figure is not inflated by stack sampling.

    opt_phase.py <trace dir or file> [...]
"""

import glob
import gzip
import json
import os
import sys


def main():
    print(f"{'trace':30}{'step ms':>9}{'opt wall':>10}{'opt GPU':>9}{'host-bound':>12}")
    for arg in sys.argv[1:]:
        path = (
            glob.glob(os.path.join(arg, "*.json.gz"))[0] if os.path.isdir(arg) else arg
        )
        with gzip.open(path, "rt") as fh:
            ev = json.load(fh)["traceEvents"]
        step_spans = [
            e
            for e in ev
            if str(e.get("name", "")).startswith("ProfilerStep#") and e.get("ph") == "X"
        ]
        steps = len({e["name"] for e in step_spans}) or 1
        # one span per step name; take the longest instance of each (the outer one)
        per_name = {}
        for e in step_spans:
            per_name[e["name"]] = max(per_name.get(e["name"], 0), e["dur"])
        step_ms = sum(per_name.values()) / 1000 / steps
        gpu = sorted(
            (e["ts"], e["ts"] + e["dur"])
            for e in ev
            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur")
        )

        def busy(a, b, gpu=gpu):
            return sum(min(e, b) - max(s, a) for s, e in gpu if e > a and s < b)

        spans = [
            e
            for e in ev
            if str(e.get("name", "")).startswith("Optimizer.step#")
            and e.get("ph") == "X"
        ]
        wall = sum(e["dur"] for e in spans) / 1000 / steps
        g = sum(busy(e["ts"], e["ts"] + e["dur"]) for e in spans) / 1000 / steps
        label = os.path.basename(os.path.dirname(path))
        print(f"{label:30}{step_ms:>9.1f}{wall:>10.1f}{g:>9.1f}{wall - g:>12.1f}")


if __name__ == "__main__":
    main()
