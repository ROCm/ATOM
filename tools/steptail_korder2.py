# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The order of the 32 products inside one WMMA step of production's LM head
GEMM, and where the incoming sum joins them, against the step tail's head.

A probe row holds 2^24 at element i of K step STEP, -2^24 at element j and 0.5
at element k (x all ones): the 0.5 is lost when it is added to a sum holding
the 2^24. All ordered triples of the 32 elements; then the carry probes, 2^24
in the step before and (0.5, -2^24) at elements of the step.

    python tools/steptail_korder2.py
    env: MS (8,64), STEP (5)
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

MS = [int(m) for m in os.environ.get("MS", "8,64").split(",")]
STEP = int(os.environ.get("STEP", "5"))
BIG, SMALL = 16777216.0, 0.5
SHOW = (
    (0, 31),
    (31, 0),
    (0, 16),
    (15, 16),
    (0, 1),
    (1, 0),
    (0, 2),
    (7, 8),
    (8, 23),
    (16, 31),
    (16, 17),
    (3, 28),
)
dev = C.dev


def probe_sets():
    triples = [("T", i, k, j) for i, j, k in itertools.permutations(range(32), 3)]
    carry = [("K", -1, k, j) for j, k in itertools.permutations(range(32), 2)]
    carry += [("J", i, k, -1) for i, k in itertools.permutations(range(32), 2)]
    plan = triples + carry
    return [plan[at : at + L.VR] for at in range(0, len(plan), L.VR)]


def weights(plan):
    """(label, i, k, j): 2^24 at element i, 0.5 at k, -2^24 at j of STEP; -1:
    element 0 of the step before (i) or after (j)."""
    w = torch.zeros(L.VOCAB, L.HIDDEN, dtype=torch.bfloat16, device=dev)
    rows = torch.arange(len(plan), device=dev)

    def column(e, side):
        return STEP * L.K_STEP + e if e >= 0 else (STEP + side) * L.K_STEP

    for value, pick, side in ((BIG, 1, -1), (SMALL, 2, 0), (-BIG, 3, 1)):
        at = torch.tensor([column(p[pick], side) for p in plan], device=dev)
        w[rows, at] = value
    return w


def main():
    sets = probe_sets()
    for m in MS:
        x = torch.ones(m, L.HIDDEN, dtype=torch.bfloat16, device=dev)
        splits = head_dispatch.head_splits(m)
        f = C.Fixture(m, 8, 0, splits, tap=True)
        plan, prod, mine = [], [], []
        for part in sets:
            w = weights(part)
            p = tgemm.mm(x, C.shard(w, 0), None)
            logits = C.production_logits(w, x)
            f.stage(w, x, torch.zeros(m, dtype=torch.int32, device=dev), logits)
            f.fire()
            torch.cuda.synchronize()
            plan += part
            prod += p[0, : len(part)].float().tolist()
            mine += f.tap[0, : len(part)].float().tolist()
            del w, logits
        same = sum(a == b for a, b in zip(prod, mine))
        print(f"M {m} splits {splits}: mine == production on {same}/{len(plan)} probes")
        for label in "TKJ":
            sel = [(p, a, b) for p, a, b in zip(plan, prod, mine) if p[0] == label]
            print(
                f"M {m} {label}: production survives {sum(a == SMALL for _, a, _ in sel)}/{len(sel)}, "
                f"mine {sum(b == SMALL for _, _, b in sel)}; other values "
                f"{sorted({a for _, a, _ in sel if a not in (0.0, SMALL)})[:4]}"
            )
        table = {(p[1], p[3], p[2]): (a, b) for p, a, b in zip(plan, prod, mine)}
        for i, j in SHOW:
            for who in (0, 1):
                line = "".join(
                    (
                        "."
                        if k in (i, j)
                        else ("S" if table[(i, j, k)][who] == SMALL else "L")
                    )
                    for k in range(32)
                )
                print(
                    f"M {m} {'prod' if who == 0 else 'mine'} 2^24@{i:2d} -2^24@{j:2d}: {line}"
                )
        for label in "KJ":
            for j in (0, 1, 15, 16, 31):
                for who in (0, 1):
                    line = "".join(
                        (
                            "."
                            if k == j
                            else (
                                "S"
                                if table[(-1, j, k) if label == "K" else (j, -1, k)][
                                    who
                                ]
                                == SMALL
                                else "L"
                            )
                        )
                        for k in range(32)
                    )
                    what = (
                        f"2^24 before, -2^24@{j:2d}"
                        if label == "K"
                        else f"2^24@{j:2d}, -2^24 after"
                    )
                    print(f"M {m} {'prod' if who == 0 else 'mine'} {what}: {line}")


if __name__ == "__main__":
    main()
