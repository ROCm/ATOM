#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Step-by-step loss and grad-norm comparison of runs against a reference run,
reporting exact equality separately from the size of any difference.

    loss_equal.py <reference run> <run> [<run> ...]
"""

import sys

import summarize as S

KEYS = ("training/total_loss", "training/grad_norm")


def series(run):
    rows = S.read_rows(run) or {}
    return {s: {k: rows[s].get(k) for k in KEYS} for s in sorted(rows)}


def main():
    ref_name, *others = sys.argv[1:]
    ref = series(ref_name)
    print(f"reference: {ref_name} ({len(ref)} steps)")
    for name in others:
        cur = series(name)
        steps = sorted(set(ref) & set(cur))
        for k in KEYS:
            pairs = [
                (ref[s][k], cur[s][k])
                for s in steps
                if ref[s][k] is not None and cur[s][k] is not None
            ]
            equal = sum(1 for a, b in pairs if a == b)
            dev = max((abs(b - a) / abs(a) for a, b in pairs if a), default=0.0)
            print(
                f"  {name:28} {k.split('/')[-1]:11} {equal:>2}/{len(pairs)} steps bit-identical   "
                f"max rel dev {dev:.3e}"
            )


if __name__ == "__main__":
    main()
