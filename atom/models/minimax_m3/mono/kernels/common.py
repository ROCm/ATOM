# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's own device helpers: bf16 row loads, the per-token FP8 scale of
aiter's ``dynamic_per_token_scaled_quant``, and the all-reduce epilogue in the
order the original path's fused all-reduce + RMSNorm reduces
(``atom.mono.device`` holds the model-independent primitives)."""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from atom.models.minimax_m3.mono.config import (
    FP4_PAGE_BYTES,
    FP4_PAGE_TOKENS,
    FP4_SCALE_OFFSET,
    HEAD_DIM,
)
from atom.mono.device.mx import FP8_MAX, clamp_fp8
from atom.mono.device.ops import butterfly, fp8_pack4, hw_rcp, traced


def ld_bf16x4(r, k, cm=0):
    """4 bf16 from element ``k`` of buffer resource ``r``, as f32."""
    w = fx.Vector(
        bo.buffer_load(r, k // 2, vec_width=2, dtype=T.i32, cache_modifier=cm)
    )
    v = w.bitcast(fx.BFloat16).to(fx.Float32)
    return [v[j] for j in range(4)]


def pack_q_e4m3(v):
    """Four f32 -> one dword of four unit-scale E4M3 bytes, saturating at +-448:
    the aiter writer's ``q_out`` convention (K1's main-Q store over MXFP4 K / V)."""
    return fp8_pack4(*[clamp_fp8(x) for x in v])


def ld_bf16x12(r, word, cm=0):
    """12 bf16 from ``word`` (a 16 B then an 8 B load)."""
    a8 = fx.Vector(
        bo.buffer_load(r, word, vec_width=4, dtype=T.i32, cache_modifier=cm)
    ).bitcast(fx.BFloat16)
    a4 = fx.Vector(
        bo.buffer_load(r, word + 4, vec_width=2, dtype=T.i32, cache_modifier=cm)
    ).bitcast(fx.BFloat16)
    return [a8[e] for e in range(8)] + [a4[e] for e in range(4)]


def per_token_fp8_scale(amax):
    """A row's FP8 scale ``amax * (1 / FP8_MAX)`` and the multiplier its elements
    take, the scale's hardware reciprocal; an all-zero row (a pad row) quantizes
    to zeros, not 0 * inf."""
    x_scale = amax * (1.0 / FP8_MAX)
    return x_scale, (amax == 0.0).select(fx.Float32(0.0), hw_rcp(x_scale))


def ar_pack_sumsq(vals):
    """One logical thread of the custom all-reduce epilogue: its 8 elements'
    squares accumulated in order (hipcc contracts ``acc += v * v`` into fma)."""
    acc = fx.Float32(0.0)
    for v in vals:
        acc = fmath.fma(v, v, acc)
    return acc


@traced
def ar_block_sum(p0, p1, lane, wave, red):
    """Sum of the 768 per-pack partials of a 6144-wide bf16 row in the order the
    1-stage fused all-reduce + RMSNorm reduces them: 24 warps of 32, a 16..1
    butterfly per warp, then the 24 warp sums butterflied again in one warp.

    Logical warp ``3 * wave + lane // 32`` holds ``p0``; logical warp
    ``3 * wave + 2`` holds ``p1`` in lanes < 32 (``WAVES`` = 8). ``red`` needs
    24 words. Matching this order keeps rstd bit-exact with the original path.
    """
    return ar_block_sums([(p0, p1)], lane, wave, red)[0]


@traced
def ar_block_sums(parts, lane, wave, red):
    """``ar_block_sum`` of several rows ``parts = [(p0, p1), ...]`` behind one set
    of barriers; ``red`` needs 24 words per row."""
    bs = [
        (butterfly(p0, (16, 8, 4, 2, 1)), butterfly(p1, (16, 8, 4, 2, 1)))
        for p0, p1 in parts
    ]
    gpu.barrier()
    for k in range_constexpr(len(bs)):
        b0, b1 = bs[k]
        if lane % 32 == 0:
            fx.ptr_store(b0, red + (24 * k + wave * 3 + lane // 32))
        if lane == 0:
            fx.ptr_store(b1, red + (24 * k + wave * 3 + 2))
    gpu.barrier()
    li = lane % 32
    tots = [
        butterfly(
            (li < 24).select(
                fx.ptr_load(red + (24 * k + fx.min(li, 23))), fx.Float32(0.0)
            ),
            (16, 8, 4, 2, 1),
        )
        for k in range(len(parts))
    ]
    gpu.barrier()
    return tots


# aiter's fused writer (``mxfp4ScaleE8m0``): ceil(log2(amax / 6)) + 127 with amax
# floored at 6 * 2^-126, the scale of an all-zero group coming out as code 1
_FP4_RCP_MAX = 1.0 / 6.0
_FP4_AMAX_FLOOR = 6.0 * 2.0**-126


def mxfp4_scale_code(amax):
    """The e8m0 code of a 32-element group whose largest magnitude is ``amax``, bit
    for bit ``mxfp4ScaleE8m0`` of ``fused_qknorm_idxrqknorm.cu``."""
    bits = (
        fx.max(amax, fx.Float32(_FP4_AMAX_FLOOR)) * fx.Float32(_FP4_RCP_MAX)
    ).bitcast(fx.Int32)
    e = (bits >> 23) & 0xFF
    return (e == 255).select(e, e + ((bits & 0x7FFFFF) != 0).select(1, 0))


@traced
def mxfp4_row_store(r_page, slot, vals, l32, cm=0):
    """Token ``slot`` of a pool of 8704 B MXFP4 pages (``r_page``: its buffer
    resource): one 128-element row quantized as the aiter writer does, by the 32
    lanes of a half wave holding 4 consecutive elements ``vals`` each (already
    rounded to bf16, as an f32). Lane l writes bytes 2 l, 2 l + 1 of the row's 64
    data bytes; the first lane of each group of 8 writes its e8m0 byte."""
    am = fmath.absf(vals[0])
    for v in vals[1:]:
        am = fx.max(am, fmath.absf(v))
    am = butterfly(am, (4, 2, 1), fx.max)
    code = mxfp4_scale_code(am)
    scale = (code << 23).bitcast(fx.Float32)
    packed = fx.Int32(0)
    for pair in range_constexpr(2):
        packed = fx.Int32(
            rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32,
                as_ir_value(packed),
                as_ir_value(vals[2 * pair]),
                as_ir_value(vals[2 * pair + 1]),
                as_ir_value(scale),
                pair,
            )
        )
    page = slot // FP4_PAGE_TOKENS
    t = slot % FP4_PAGE_TOKENS
    base = page * FP4_PAGE_BYTES
    bo.buffer_store(
        fx.Int16(packed & 0xFFFF),
        r_page,
        base + t * (HEAD_DIM // 2) + l32 * 2,
        cache_modifier=cm,
        offset_is_bytes=True,
    )
    if l32 % 8 == 0:
        bo.buffer_store(
            fx.Int8(code & 0xFF),
            r_page,
            base + FP4_SCALE_OFFSET + t * (HEAD_DIM // 32) + l32 // 8,
            cache_modifier=cm,
            offset_is_bytes=True,
        )
