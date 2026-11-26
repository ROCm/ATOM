# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The fallback draft's QSA norm, interleaved-MRoPE RoPE and BF16 K / V cache publication in one launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_norm_rope_cache_kernel(
    Q,
    K,
    V,
    QW,
    KW,
    OQ,
    KC,
    VC,
    SLOT,
    COS,
    SIN,
    POS,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    PS: tl.constexpr,
    PA: tl.constexpr,
    CS: tl.constexpr,
    BS: tl.constexpr,
    SLOTS: tl.constexpr,
    EPS: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // (QH + KH)
    head = row % (QH + KH)
    d = tl.arange(0, D)
    if head < QH:
        src = Q + token * QS + head * D
        weight = QW
    else:
        src = K + token * KS + (head - QH) * D
        weight = KW
    x = tl.load(src + d).to(tl.float32)
    w = tl.load(weight + d).to(tl.float32) + 1.0
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    # Match DualRMSNorm before RoPE consumes BF16 values.
    normed = (x * rstd * w).to(OQ.dtype.element_ty)
    half = RD // 2
    freq = d % half
    axis = tl.where((freq % 3 == 1) & (freq < 3 * SH), 1, 0)
    axis = tl.where((freq % 3 == 2) & (freq < 3 * SW), 2, axis)
    pos = tl.load(POS + axis * PA + token * PS)
    cos = tl.load(COS + pos * half + freq).to(normed.dtype).to(tl.float32)
    sin = tl.load(SIN + pos * half + freq).to(normed.dtype).to(tl.float32)
    paired = tl.where(d < half, d + half, d - half)
    partner = tl.gather(normed, paired, axis=0).to(tl.float32)
    partner = tl.where(d < half, -partner, partner)
    xc = (normed.to(tl.float32) * cos).to(normed.dtype).to(tl.float32)
    ys = (partner * sin).to(normed.dtype).to(tl.float32)
    out = tl.where(d < RD, (xc + ys).to(normed.dtype), normed)
    if head < QH:
        tl.store(OQ + (token * QH + head) * D + d, out)
    else:
        slot = tl.load(SLOT + token).to(tl.int64)
        address = (slot // BS) * CS + (slot % BS) * KH * D + (head - QH) * D + d
        value = tl.load(V + token * VS + (head - QH) * D + d)
        valid = (slot >= 0) & (slot < SLOTS)
        tl.store(KC + address, out, valid)
        tl.store(VC + address, value, valid)


def qsa_norm_rope_cache(q, k, value, norm, rope, positions, k_cache, v_cache, slots):
    """Normalize / rotate Q and K (interleaved MRoPE) and publish K / V once for decode or MTP verify rows.

    The per-head norm and both RoPE products retain their BF16 endpoints. Query is contiguous; K goes directly to the
    existing NHD paged BF16 cache. Negative slots preserve padded-request cache rows. Each CTA reads only its own query
    row; q4 has no inter-row dependency here.
    """
    if not getattr(rope, "mrope_interleaved", False):
        raise ValueError(
            "the fallback draft's QSA epilogue rotates interleaved MRoPE sections only"
        )
    tokens, q_heads, dim = q.shape
    query = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    sections = getattr(rope, "mrope_section", None) or [rope.rotary_dim // 2, 0, 0]
    _qsa_norm_rope_cache_kernel[(tokens * (q_heads + k.shape[1]),)](
        q,
        k,
        value,
        norm.q_norm.weight,
        norm.k_norm.weight,
        query,
        k_cache,
        v_cache,
        slots,
        rope.cos_cache,
        rope.sin_cache,
        positions,
        QS=q.stride(0),
        KS=k.stride(0),
        VS=value.stride(0),
        PS=positions.stride(-1),
        PA=(
            positions.stride(0)
            if positions.ndim == 2 and positions.shape[0] == 3
            else 0
        ),
        CS=k_cache.stride(0),
        BS=k_cache.shape[1],
        SLOTS=k_cache.shape[0] * k_cache.shape[1],
        EPS=norm._eps,
        QH=q_heads,
        KH=k.shape[1],
        D=dim,
        RD=rope.rotary_dim,
        SH=sections[1],
        SW=sections[2],
        num_warps=1,
        enable_fp_fusion=False,
    )
    return query
