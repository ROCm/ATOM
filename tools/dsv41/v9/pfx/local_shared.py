# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""PF-SHS v3: the comm-fused prefill's local shared-expert sum.

``commfused_v13._local_shared`` sums the replica outputs as ``fp32.add_(bf16
.float())``; the mixed-dtype elementwise kernel costs ~587 us a call, three a
FFN. These Triton kernels accumulate in fp32 in the same order
``((o0 + o1) + o2) + o3`` and round once to bf16 (RTNE), zeroing the rows past
``valid``: bitwise equal to the original.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _pair(a_ptr, b_ptr, acc_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    a = tl.load(a_ptr + off, mask=m).to(tl.float32)
    b = tl.load(b_ptr + off, mask=m).to(tl.float32)
    tl.store(acc_ptr + off, a + b, mask=m)


@triton.jit
def _accum(acc_ptr, b_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    a = tl.load(acc_ptr + off, mask=m)
    b = tl.load(b_ptr + off, mask=m).to(tl.float32)
    tl.store(acc_ptr + off, a + b, mask=m)


@triton.jit
def _final(
    acc_ptr,
    b_ptr,
    out_ptr,
    n_valid,
    n_out,
    HAS_ACC: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mo = off < n_out
    mv = off < n_valid
    if HAS_ACC:
        v = tl.load(acc_ptr + off, mask=mv, other=0.0)
        if HAS_B:
            v = v + tl.load(b_ptr + off, mask=mv, other=0.0).to(tl.float32)
    else:
        v = tl.load(b_ptr + off, mask=mv, other=0.0).to(tl.float32)
    v = tl.where(mv, v, 0.0)
    tl.store(out_ptr + off, v.to(tl.bfloat16), mask=mo)


def _grid(n):
    return (triton.cdiv(n, _BLOCK),)


def combine_v3(outs_iter, valid, out_rows, hidden, device):
    """outs_iter yields bf16 [rows, hidden] contiguous replica outputs in order; returns bf16 [out_rows, hidden]."""
    sh = torch.empty((out_rows, hidden), dtype=torch.bfloat16, device=device)
    n_out = out_rows * hidden
    n_valid = valid * hidden
    it = iter(outs_iter)
    o0 = next(it).contiguous()
    o1 = next(it, None)
    if o1 is None:
        _final[_grid(n_out)](
            o0, o0, sh, n_valid, n_out, HAS_ACC=False, HAS_B=True, BLOCK=_BLOCK
        )
        return sh
    o1 = o1.contiguous()
    o2 = next(it, None)
    # Two replicas: out = bf16(f(o0) + f(o1)), still the fp32 path exactly.
    if o2 is None:
        acc = torch.empty(o0.shape, dtype=torch.float32, device=device)
        _pair[_grid(o0.numel())](o0, o1, acc, n_valid, BLOCK=_BLOCK)
        _final[_grid(n_out)](
            acc, acc, sh, n_valid, n_out, HAS_ACC=True, HAS_B=False, BLOCK=_BLOCK
        )
        return sh
    acc = torch.empty(o0.shape, dtype=torch.float32, device=device)
    _pair[_grid(n_valid)](o0, o1, acc, n_valid, BLOCK=_BLOCK)
    del o0, o1
    nxt = next(it, None)
    cur = o2.contiguous()
    while nxt is not None:
        _accum[_grid(n_valid)](acc, cur, n_valid, BLOCK=_BLOCK)
        cur = nxt.contiguous()
        nxt = next(it, None)
    _final[_grid(n_out)](
        acc, cur, sh, n_valid, n_out, HAS_ACC=True, HAS_B=True, BLOCK=_BLOCK
    )
    return sh


def local_shared(reps, value, scale, valid, out_rows, hidden):
    return combine_v3(
        (r(value, x_scale=scale) for r in reps), valid, out_rows, hidden, value.device
    )
