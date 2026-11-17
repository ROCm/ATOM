# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A served step's outputs in one launch (``TkStepRunner._outputs``' GPU path), on TkRank's stream right after the step:
the rejection sampler's output ids and verdict from the logs the step appended to, and the drafts it left.

- ``a = ar_num[ar_num[0]]`` (the log's last entry: this step's accepted tokens), ``t = ar_acc[0]`` (tokens logged);
- ``out[i] = ar_acc[t - a + 1 + i]`` for ``i < a``, else -1 (``[1, rows]`` int32);
- ``verdict = [a - 1, rows - a, a - 1]`` (``[3, 1]`` int32: accepted drafts, rejected, anchor row);
- ``drafts = vtok[1 : 1 + k]`` (``[1, k]`` int64).

Same values as the torch ops it replaces (``runner.TkStepRunner._outputs_torch``, the CPU path), which were ~18 launches
between two steps.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kernel(acc, num, vtok, out, verdict, drafts, acc_n, ROWS: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    cnt = tl.load(num)
    a = tl.load(num + cnt)
    t = tl.load(acc)
    r = tl.arange(0, BLOCK)
    idx = tl.minimum(tl.maximum(t - a + 1 + r, 0), acc_n - 1)
    tok = tl.load(acc + idx, mask=r < ROWS, other=-1)
    tl.store(out + r, tl.where(r < a, tok, -1).to(tl.int32), mask=r < ROWS)
    tl.store(verdict + r, tl.where(r == 1, ROWS - a, a - 1).to(tl.int32), mask=r < 3)
    d = tl.load(vtok + 1 + r, mask=r < K, other=0)
    tl.store(drafts + r, d.to(tl.int64), mask=r < K)


def gather(acc: torch.Tensor, num: torch.Tensor, vtok: torch.Tensor, rows: int, k: int):
    """``(out [1, rows] int32, verdict [3, 1] int32, drafts [1, k] int64)`` on the current stream."""
    dev = acc.device
    out = torch.empty(rows, dtype=torch.int32, device=dev)
    verdict = torch.empty(3, dtype=torch.int32, device=dev)
    drafts = torch.empty(k, dtype=torch.int64, device=dev)
    block = triton.next_power_of_2(max(rows, k, 3))
    _gather_kernel[(1,)](
        acc, num, vtok, out, verdict, drafts, acc.numel(), ROWS=rows, K=k, BLOCK=block
    )
    return out.view(1, rows), verdict.view(3, 1), drafts.view(1, k)
