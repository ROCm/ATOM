# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""In which order production's DSpark Markov op adds its fp32 terms
(``dspark_markov_argmax``: logits = base + W1[prev] . W2^T), read off its
outputs: what the step tail's Markov kernel has to reproduce.

A probe is one vocab column: W1's rows are all ones, the column's W2 row holds
2^24, -2^24 and 0.5 at chosen K positions, and the column is the only one of
its batch row with a finite base logit, so the row's maximum (the op's packed
``(max, id)``) is that column's logit. The 0.5 survives unless it is added to
a sum holding the 2^24. Families: every ordered triple of the 8 K steps; every
ordered triple of the 32 elements of one step; the carry into and out of a
step; whether the base logit is added before or after the dot product. Each is
compared with the model the LM head follows: 32-column steps in order from 0,
each step two fused groups (the incoming sum with K 0..15, then K 16..31), the
base logit added last.

    python tools/steptail_markov_order.py
    env: BATCH (8: requests a call; 16 and 32 pick the op's other tilings)
"""

import itertools
import os

import numpy as np
import torch

from atom.model_ops.dspark_markov_argmax import _launch

BATCH = [int(b) for b in os.environ.get("BATCH", "8,16").split(",")]
RANK, STEP = 256, 3
BIG, SMALL = 16777216.0, 0.5
dev = torch.device("cuda", torch.cuda.current_device())


def probes():
    """[(family, [(k, value), ..], base)]"""
    out = []
    for a, b, c in itertools.permutations(range(RANK // 32), 3):
        out.append(("steps", [(a * 32, BIG), (b * 32, SMALL), (c * 32, -BIG)], 0.0))
    at = STEP * 32
    for i, j, k in itertools.permutations(range(32), 3):
        out.append(("elements", [(at + i, BIG), (at + k, SMALL), (at + j, -BIG)], 0.0))
    for j, k in itertools.permutations(range(32), 2):
        out.append(("carry in", [(at - 32, BIG), (at + k, SMALL), (at + j, -BIG)], 0.0))
        out.append(
            ("carry out", [(at + j, BIG), (at + k, SMALL), (at + 32, -BIG)], 0.0)
        )
    for s in range(RANK // 32):
        out.append(("base", [(0, BIG), (s * 32 + 5, SMALL)], -BIG))
    return out


def model(terms, base):
    acc = 0.0
    for s in range(RANK // 32):
        for lo, hi in ((0, 16), (16, 32)):
            vals = [v for k, v in terms if k // 32 == s and lo <= k % 32 < hi]
            if not vals:
                continue
            bigs = [v for v in vals if abs(v) >= BIG]
            if bigs or abs(acc) >= BIG:
                acc = (acc if abs(acc) >= BIG else 0.0) + sum(bigs)
            else:
                acc += sum(vals)
    return float(np.float32(base) + np.float32(acc))


def main():
    plan = probes()
    n = len(plan)
    vocab = -(-n // 128) * 128
    w2 = torch.zeros(vocab, RANK, dtype=torch.bfloat16, device=dev)
    for slot in range(3):
        rows = [i for i, p in enumerate(plan) if len(p[1]) > slot]
        w2[
            torch.tensor(rows, device=dev),
            torch.tensor([plan[i][1][slot][0] for i in rows], device=dev),
        ] = torch.tensor([plan[i][1][slot][1] for i in rows], device=dev).to(
            torch.bfloat16
        )
    w1 = torch.ones(vocab, RANK, dtype=torch.bfloat16, device=dev)
    bias = torch.tensor([p[2] for p in plan], device=dev).to(torch.bfloat16)
    want = [model(p[1], p[2]) for p in plan]
    for batch in BATCH:
        base = torch.full(
            (batch, vocab), float("-inf"), dtype=torch.bfloat16, device=dev
        )
        prev = torch.arange(batch, dtype=torch.int32, device=dev)
        out = torch.empty(batch, dtype=torch.int32, device=dev)
        packed = torch.empty(batch, 2, dtype=torch.float32, device=dev)
        got = torch.empty(n, dtype=torch.float32, device=dev)
        wrong_id = 0
        for start in range(0, n, batch):
            cols = torch.arange(start, min(start + batch, n), device=dev)
            rows = torch.arange(cols.numel(), device=dev)
            base[rows, cols] = bias[cols]
            _launch(base, prev, w1, w2, out, 0, packed)
            got[cols] = packed[: cols.numel(), 0]
            wrong_id += int((packed[: cols.numel(), 1].long() != cols).sum())
            base[rows, cols] = float("-inf")
        got = got.tolist()
        print(f"batch {batch}: {n} probes, {wrong_id} answered another column")
        for family in ("steps", "elements", "carry in", "carry out", "base"):
            sel = [i for i, p in enumerate(plan) if p[0] == family]
            same = sum(got[i] == want[i] for i in sel)
            kept = sum(got[i] == SMALL for i in sel)
            bad = [(plan[i][1], got[i], want[i]) for i in sel if got[i] != want[i]][:3]
            print(
                f"  batch {batch} {family}: production == model on {same}/{len(sel)}; the 0.5 "
                f"survives in {kept}" + (f"; first differences {bad}" if bad else "")
            )


if __name__ == "__main__":
    main()
