# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Renormalized softmax top-k routing, one Triton program per token.

Bit-identical to AITER's `topk_softmax` with `need_renorm` (ids, tie order,
weights, the sigmoid-scored shared-expert slots it appends, and the pad rows
of its optional `num_valid_rows` count), also on rows with fewer than k finite
logits. On MI355X at 512 experts it takes about 4 us per call where the HIP
kernel takes 9-11 us; past a few thousand rows the HIP kernel is faster, and
`topk_softmax_dispatch.py` picks it there. Each selection is a single int64
max over (value, -index).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _topk_softmax_kernel(
    logits_ptr,
    w_ptr,
    ids_ptr,
    num_valid_rows_ptr,
    stride_l,
    stride_w,
    stride_i,
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    NSH: tl.constexpr,
    HAS_NUM_VALID_ROWS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, E)
    x = tl.load(logits_ptr + row * stride_l + offs).to(tl.float32)
    # High word: the float bits made order-preserving as int32 (-0.0 folded
    # into +0.0, which compare equal). Low word: E - 1 - index, so equal
    # values resolve to the lowest expert id.
    bits = x.to(tl.int32, bitcast=True)
    b = tl.where(x == 0.0, 0, bits)
    low = (E - 1 - offs).to(tl.int64)
    # AITER picks with `>` and clears each pick to -inf: once the finite
    # logits run out the lowest index (expert 0) is picked again, its slot
    # already cleared. A NaN sorts below -inf, so it is never picked.
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    b = tl.where(nan, -2147483648, b ^ ((b >> 31) & 0x7FFFFFFF))
    key = b.to(tl.int64) * 4294967296 + low
    cleared = -2139095041 * 4294967296 + low  # -inf's high word
    kk = tl.arange(0, KP)
    wv = tl.zeros((KP,), dtype=tl.float32)
    iv = tl.zeros((KP,), dtype=tl.int32)
    total = 0.0
    for k in tl.static_range(K):
        best = tl.max(key, 0)
        idx = (E - 1 - (best & 0xFFFFFFFF)).to(tl.int32)
        hi = (best >> 32).to(tl.int32)
        v = (hi ^ ((hi >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)
        if k == 0:
            row_max = v
        e = libdevice.exp(v - row_max)
        wv = tl.where(kk == k, e, wv)
        iv = tl.where(kk == k, idx, iv)
        total += e
        key = tl.where(offs == idx, cleared, key)
    wv = wv * (1.0 / total)
    if HAS_NUM_VALID_ROWS:
        pad = row >= tl.load(num_valid_rows_ptr)
        wv = tl.where(pad, 0.0, wv)
        iv = tl.where(pad, -1, iv)
    tl.store(w_ptr + row * stride_w + kk, wv, mask=kk < K)
    tl.store(ids_ptr + row * stride_i + kk, iv, mask=kk < K)
    if NSH > 0:
        sh = tl.arange(0, NSH)
        lg = tl.load(logits_ptr + row * stride_l + E + sh).to(tl.float32)
        tl.store(w_ptr + row * stride_w + K + sh, 1.0 / (1.0 + libdevice.exp(-lg)))


def triton_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    num_shared_experts: int = 0,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    """Fill `topk_weights[:, :k]` / `topk_ids[:, :k]` like AITER's
    `topk_softmax(..., need_renorm=True, ...)` (k = `topk_ids.shape[1]`), and
    `topk_weights[:, k + s]` with sigmoid(shared logit s) for each of the
    `num_shared_experts` trailing logits.

    As in AITER, a row with fewer than k finite logits picks expert 0 for the
    rest, with weight 0. A NaN logit is never picked. AITER agrees unless the
    NaN is the first element of one of its per-thread vector loads (16 bf16
    below 512 experts, 32 at 512): there it may pick it, making every weight
    NaN, or misorder the row, so rows with NaNs are not bit-identical.

    `num_valid_rows`, as in AITER (ROCm/aiter#6916): an optional int32 device
    tensor whose first element is the count of real rows. Rows at or past it
    are padding: their routed ids are written -1 and their weights 0; shared
    columns are written as for any row. Read on device, so a replayed CUDA
    graph follows each step's count."""
    tokens, width = gating_output.shape
    k = topk_ids.shape[1]
    assert (
        gating_output.stride(-1) == topk_weights.stride(-1) == topk_ids.stride(-1) == 1
    ), "the logits and both outputs must have a unit last stride"
    if num_valid_rows is not None:
        assert (
            num_valid_rows.dtype == torch.int32 and num_valid_rows.numel() >= 1
        ), "num_valid_rows must be an int32 tensor of at least one element"
    if not tokens:
        return
    _topk_softmax_kernel[(tokens,)](
        gating_output,
        topk_weights,
        topk_ids,
        num_valid_rows,
        gating_output.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        E=width - num_shared_experts,
        K=k,
        KP=triton.next_power_of_2(k),
        NSH=num_shared_experts,
        HAS_NUM_VALID_ROWS=num_valid_rows is not None,
        num_warps=1,
        # The weights match AITER bit for bit only without FMA contraction.
        enable_fp_fusion=False,
    )
