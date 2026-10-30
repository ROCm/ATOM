# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Two-row FP8-model HC mix using the native HC kernels."""

import torch
import triton

from atom.model_ops.qwen4_exp.ops.hc_fused import _hc_mix_kernel, _hc_pre_kernel


def mix_rows(
    h: torch.Tensor,
    norm_weight: torch.Tensor,
    w_cat: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
    hc: int,
    has_inject: bool,
    block_out: torch.Tensor | None = None,
    raw: torch.Tensor | None = None,
):
    """Deferred combine + full mix, one program set per token.

    Returns (streams, mixed, inject logits or None). Rows re-read the weights,
    which stay cache resident, so this is for small batches.
    """
    m, width = h.shape
    assert m == 2
    hidden = width // hc
    nout = w_cat.shape[0]
    rank = w_up.shape[1]
    combine = block_out is not None
    normed = torch.empty_like(h)
    h_new = torch.empty_like(h) if combine else h
    part = torch.empty((m, hc, nout), dtype=torch.float32, device=h.device)
    bn = 8
    _hc_pre_kernel[(triton.cdiv(nout, bn), hc, m)](
        h,
        block_out if combine else h,
        raw if combine else h,
        norm_weight,
        w_cat,
        h_new,
        normed,
        part,
        eps,
        h.stride(0),
        block_out.stride(0) if combine else 0,
        raw.stride(0) if combine else 0,
        H=hidden,
        HC=hc,
        NOUT=nout,
        COMBINE=combine,
        BN=bn,
        BK=hidden // 5,
        num_warps=4,
    )
    mixed = torch.empty((m, hidden), dtype=h.dtype, device=h.device)
    raw_out = (
        torch.empty((m, hc), dtype=h.dtype, device=h.device) if has_inject else None
    )
    bh = 4
    _hc_mix_kernel[(triton.cdiv(hidden, bh), m)](
        part,
        normed,
        w_up,
        mixed,
        raw_out if has_inject else mixed,
        normed.stride(0),
        H=hidden,
        HC=hc,
        R=rank,
        RP=triton.next_power_of_2(rank),
        NOUT=nout,
        HAS_INJECT=has_inject,
        BH=bh,
        num_warps=4,
    )
    return h_new, mixed, raw_out
