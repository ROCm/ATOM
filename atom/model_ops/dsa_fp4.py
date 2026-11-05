# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSA Q/K publication in OPUS gfx950's MXFP4 layout.

No weight quantization: Q is the BF16 projection, K the BF16 pre-LayerNorm
projection. Negative K slots never suppress a query (DCP ownership).
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op


@triton.jit
def _rope(
    x,
    Positions,
    Cos,
    Sin,
    token,
    CS_LD: tl.constexpr,
    MAX_POS: tl.constexpr,
    ROPE: tl.constexpr,
    NEOX: tl.constexpr,
):
    d = tl.arange(0, 128)
    if ROPE:
        half: tl.constexpr = ROPE // 2
        partner = d ^ (half if NEOX else 1)
        other = tl.gather(x, tl.broadcast_to(partner[None, :], x.shape), 1)
        pair = d % half if NEOX else d // 2
        lead = d < half if NEOX else d % 2 == 0
        pos = tl.minimum(tl.maximum(tl.load(Positions + token), 0), MAX_POS - 1)
        cos = tl.load(Cos + pos * CS_LD + pair, d < ROPE, 1).to(tl.float32)
        sin = tl.load(Sin + pos * CS_LD + pair, d < ROPE, 0).to(tl.float32)
        rotated = tl.where(lead[None, :], x * cos - other * sin, x * cos + other * sin)
        x = tl.where(d[None, :] < ROPE, rotated.to(tl.bfloat16).to(tl.float32), x)
    return x


@triton.jit
def _publish(
    Q,
    K,
    NW,
    NB,
    Slots,
    Cache,
    CacheScale,
    QOut,
    QScale,
    Positions,
    Cos,
    Sin,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KS: tl.constexpr,
    SLOT_COUNT: tl.constexpr,
    HEADS: tl.constexpr,
    EPS: tl.constexpr,
    ROPE: tl.constexpr,
    NEOX: tl.constexpr,
    CS_LD: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    d = tl.arange(0, 128)
    b = tl.arange(0, 64)
    g = tl.arange(0, 4)
    if task < HEADS:
        q = tl.load(Q + token * QS + task * QH + d).to(tl.float32)[None, :]
        q = _rope(q, Positions, Cos, Sin, token, CS_LD, MAX_POS, ROPE, NEOX)
        packed, scales = _mxfp4_quant_op(q, 128, 1, 32, SCALING_MODE=1, USE_ASM=True)
        tl.store(QOut + (token * HEADS + task) * 64 + b, packed.reshape(64))
        # OPUS's query scale [T,2,32,H/16], including H=32's two bytes/lane.
        at = (
            (g % 2 * 32 + task % 32) * (HEADS // 16)
            + g // 2 * (HEADS // 32)
            + task // 32
        )
        tl.store(QScale + token * HEADS * 4 + at, scales.reshape(4))
    else:
        slot = tl.load(Slots + token, token < SLOT_COUNT, -1).to(tl.int64)
        if slot >= 0:
            x = tl.load(K + token * KS + d).to(tl.float32)
            mean = tl.sum(x, 0) * (1.0 / 128)
            x = x - mean
            inv = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 128) + EPS)
            gamma = tl.load(NW + d).to(tl.float32)
            beta = tl.load(NB + d).to(tl.float32)
            if not ROPE:
                gamma = gamma.to(tl.bfloat16).to(tl.float32)
                beta = beta.to(tl.bfloat16).to(tl.float32)
                x = (x * inv) * gamma + beta
            else:
                x = tl.fma(x * inv, gamma, beta)
            x = x.to(tl.bfloat16).to(tl.float32)[None, :]
            x = _rope(x, Positions, Cos, Sin, token, CS_LD, MAX_POS, ROPE, NEOX)
            packed, scales = _mxfp4_quant_op(
                x, 128, 1, 32, SCALING_MODE=1, USE_ASM=True
            )
            group, row = slot // 64, slot % 64
            tl.store(
                Cache + group * 4096 + (b // 16 * 64 + row) * 16 + b % 16,
                packed.reshape(64),
            )
            at = g % 2 * 128 + row % 32 * 4 + g // 2 * 2 + row // 32
            tl.store(CacheScale + group * 256 + at, scales.reshape(4))


def publish(
    q,
    k,
    cache,
    cache_scale,
    slots,
    norm_weight,
    norm_bias,
    *,
    eps=1e-6,
    rope_dim=0,
    is_neox=False,
    positions=None,
    cos=None,
    sin=None,
):
    """Return packed Q and OPUS scales, publish normalized K in place.

    Physical slots number rows of consecutive 64-row groups. A block table
    may group several complete groups per logical block; no repack is needed.
    """
    rows, heads, dim = q.shape
    if dim != 128 or heads not in (32, 64) or rope_dim not in (0, 64):
        raise ValueError("OPUS DSA FP4 requires H=32/64, D=128, RoPE=0/64")
    if (
        cache.dtype != torch.uint8
        or tuple(cache.shape[1:]) != (4, 64, 16)
        or cache_scale.dtype != torch.uint8
        or tuple(cache_scale.shape) != (cache.shape[0], 2, 32, 4)
        or not cache.is_contiguous()
        or not cache_scale.is_contiguous()
    ):
        raise ValueError("expected OPUS uint8 cache [groups,4,64,16] / [groups,2,32,4]")
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or q.stride(-1) != 1
        or k.stride(-1) != 1
    ):
        raise ValueError("Q/K must be BF16 with contiguous dimensions")
    if slots.dtype != torch.int64 or k.shape != (rows, 128):
        raise ValueError("expected int64 physical slots and [rows,128] K")
    if rope_dim and (positions is None or cos is None or sin is None):
        raise ValueError("RoPE requires positions and BF16 cosine/sine tables")
    out = torch.empty((rows, heads, 64), dtype=torch.uint8, device=q.device)
    scales = torch.empty((rows, 2, 32, heads // 16), dtype=torch.uint8, device=q.device)
    if rows:
        _publish[(rows, heads + 1)](
            q,
            k,
            norm_weight,
            norm_bias,
            slots,
            cache,
            cache_scale,
            out,
            scales,
            positions,
            cos,
            sin,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            slots.numel(),
            heads,
            eps,
            rope_dim,
            is_neox,
            cos.stride(0) if rope_dim else 0,
            cos.shape[0] if rope_dim else 0,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, scales


def opus_logits(
    q,
    qs,
    cache,
    cache_scale,
    weights,
    block_tables,
    query_start_loc,
    row_ends,
    *,
    max_qlen,
    max_seq_len,
    page_size=64,
    weight_scale=1.0,
    out=None,
    out_offsets=None,
):
    """The PR 6952 API, imported only for the explicitly selected new layout."""
    from aiter.ops.opus.pa_mqa_logits_mxfp4 import (
        mqa_logits_mxfp4,
        mqa_logits_mxfp4_config,
    )

    cfg = mqa_logits_mxfp4_config(
        q.device,
        heads=q.shape[-2],
        page_size=64,
        pages_per_block=page_size // 64,
        max_rows=q.shape[0],
        max_seqs=query_start_loc.numel() - 1,
        max_qlen=max_qlen,
        max_seq_len=max_seq_len,
    )
    return mqa_logits_mxfp4(
        q,
        qs,
        cache,
        cache_scale,
        weights.contiguous(),
        block_tables,
        query_start_loc,
        row_ends,
        cfg=cfg,
        weight_scale=weight_scale,
        out=out,
        out_offsets=out_offsets,
    )
