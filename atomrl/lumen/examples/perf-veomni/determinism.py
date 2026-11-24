#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Does a mode reproduce itself? Compares each mode's repeats against each other
rather than against the baseline.

This separates two things the main table cannot: a patch that changes the
arithmetic but does so deterministically (every repeat agrees to the last bit,
the loss simply sits somewhere else) from one that introduces run-to-run
nondeterminism (its own repeats disagree). Only the second kind makes a
training run unreproducible, and the deliverable has to grade them apart.

    determinism.py pvq-B3 pvq-L4 ...
"""

import sys

import summarize as S

REPS = ["r1", "r2", "r3"]


def main():
    modes = sys.argv[1:]
    if not modes:
        raise SystemExit("usage: determinism.py <mode> [<mode> ...]")
    print(f"{'mode':24}{'max self-deviation':>20}   note")
    print("-" * 70)
    for m in modes:
        series = {}
        for rep in REPS:
            rows = S.read_rows(f"{m}-{rep}")
            if rows:
                series[rep] = {
                    s: rows[s].get("training/total_loss") for s in sorted(rows)
                }
        if len(series) < 2:
            print(f"{m:24}{'insufficient data':>20}")
            continue
        values = [v for d in series.values() for v in d.values() if v is not None]
        ref = series[REPS[0]]
        devs = [
            abs(v - ref[s]) / abs(ref[s])
            for rep in list(series)[1:]
            for s, v in series[rep].items()
            if ref.get(s) and v is not None
        ]
        worst = max(devs) * 100 if devs else 0.0
        if values and all(v == 0 for v in values):
            note = (
                "loss is identically 0 -- this task computes none, so 0% means nothing"
            )
        elif worst == 0.0:
            note = "bit-identical across repeats"
        else:
            note = "repeats disagree: nondeterministic"
        print(f"{m:24}{worst:>19.5f}%   {note}")


if __name__ == "__main__":
    main()
