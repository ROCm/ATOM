# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""In which order production's LM head GEMM adds its fp32 terms, read off its
outputs, and where the step tail's head differs.

Every vocab row is one probe: x is all ones, the row holds 2^24 at K step a,
-2^24 at step c and 0.5 at step b (zero elsewhere). The 0.5 survives ('S')
unless it is added to a sum that holds the 2^24 at that moment; which probes
survive shows the partial sums' K ranges, the order inside them and the order
the partials are added in. Scans: A = (a, b, c) = (s, s + d, s + 2 d),
B = (b, a, c), C = (a, c, b); a grid of triples over the split boundaries.

    python tools/steptail_korder.py
    env: MS (8,16,32,64), MINE (1) run the step tail's check build beside it
"""

import itertools
import os
import sys

import torch
from aiter.tuned_gemm import tgemm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steptail_check as C

from atom.models.kimi_k3.step_tail import head_dispatch
from atom.models.kimi_k3.step_tail import layout as L

MS = [int(m) for m in os.environ.get("MS", "8,16,32,64").split(",")]
MINE = os.environ.get("MINE", "1") == "1"
BIG, SMALL = 16777216.0, 0.5
GRID = (
    0,
    1,
    2,
    3,
    10,
    37,
    38,
    39,
    74,
    75,
    76,
    77,
    78,
    111,
    112,
    113,
    150,
    151,
    152,
    153,
    200,
    222,
    223,
)
dev = C.dev


def probes():
    """[(label, a, b, c)]: 2^24 at step a, 0.5 at b, -2^24 at c."""
    out = []
    for d in (1, 2, 3, 4, 8, 16, 32):
        for s in range(L.STEPS - 2 * d):
            out.append((f"A{d}", s, s + d, s + 2 * d))
            out.append((f"B{d}", s + d, s, s + 2 * d))
            out.append((f"C{d}", s, s + 2 * d, s + d))
    for a, b, c in itertools.permutations(GRID, 3):
        out.append(("G", a, b, c))
    return out


def ranges(values):
    values = sorted(values)
    out, i = [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        out.append(str(values[i]) if i == j else f"{values[i]}-{values[j]}")
        i = j + 1
    return ",".join(out)


def main():
    plan = probes()
    assert len(plan) <= L.VR
    w = torch.zeros(L.VOCAB, L.HIDDEN, dtype=torch.bfloat16, device=dev)
    rows = torch.arange(len(plan), device=dev)
    for column, value in ((1, BIG), (2, SMALL), (3, -BIG)):
        at = torch.tensor([p[column] * L.K_STEP for p in plan], device=dev)
        w[rows, at] = value
    labels = sorted({p[0] for p in plan}, key=lambda s: (s[0], int(s[1:] or 0)))
    for m in MS:
        x = torch.ones(m, L.HIDDEN, dtype=torch.bfloat16, device=dev)
        splits = head_dispatch.head_splits(m)
        prod = tgemm.mm(x, C.shard(w, 0), None)
        if not all(torch.equal(prod[0], prod[r]) for r in range(m)):
            print(f"M {m}: production rows differ")
        got = {"prod": prod[0, : len(plan)].float().tolist()}
        if MINE:
            f = C.Fixture(m, 8 if m % 8 == 0 else m, 0, splits, tap=True)
            logits = C.production_logits(w, x)
            ids = torch.zeros(m, dtype=torch.int32, device=dev)
            f.stage(w, x, ids, logits)
            f.fire()
            torch.cuda.synchronize()
            got["mine"] = f.tap[0, : len(plan)].float().tolist()
            print(
                f"M {m} splits {splits}: mine == production on "
                f"{sum(a == b for a, b in zip(got['prod'], got['mine']))}/{len(plan)} probes"
            )
        for label in labels:
            if label == "G":
                continue
            for who, values in got.items():
                survived = [
                    p[1] if label[0] != "B" else p[2]
                    for p, v in zip(plan, values)
                    if p[0] == label and v == SMALL
                ]
                other = sorted(
                    {
                        v
                        for p, v in zip(plan, values)
                        if p[0] == label and v not in (0.0, SMALL)
                    }
                )
                print(
                    f"M {m} {who} scan {label}: survives at s = {ranges(survived)}"
                    + (f" other values {other[:6]}" if other else "")
                )
        grid = [(p, v) for p, v in zip(plan, got["prod"]) if p[0] == "G"]
        print(
            f"M {m} prod grid: {sum(v == SMALL for _, v in grid)}/{len(grid)} survive"
        )
        if MINE:
            diff = [
                (p[1:], a, b)
                for p, a, b in zip(plan, got["prod"], got["mine"])
                if p[0] == "G" and a != b
            ]
            print(
                f"M {m} grid: {len(diff)} differ; first (a b c) prod mine: {diff[:24]}"
            )


if __name__ == "__main__":
    main()
