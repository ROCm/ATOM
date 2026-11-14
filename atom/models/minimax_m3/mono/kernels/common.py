# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's own device helpers: bf16 row loads, the per-token FP8 scale of
aiter's ``dynamic_per_token_scaled_quant``, and the all-reduce epilogue in the
order the original path's fused all-reduce + RMSNorm reduces
(``atom.mono.device`` holds the model-independent primitives)."""

from __future__ import annotations

import struct

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from atom.mono.device.mx import FP8_MAX
from atom.mono.device.ops import butterfly, hw_rcp, traced


def ld_bf16x4(r, k, cm=0):
    """4 bf16 from element ``k`` of buffer resource ``r``, as f32."""
    w = fx.Vector(
        bo.buffer_load(r, k // 2, vec_width=2, dtype=T.i32, cache_modifier=cm)
    )
    v = w.bitcast(fx.BFloat16).to(fx.Float32)
    return [v[j] for j in range(4)]


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


# vLLM's ``_quant4`` (``norm_rope_nvfp4_insert``): the scale of a 16-element group is
# e4m3(min(amax * RN(1/6) / gs, 448)), the elements are quantized as x / (gs * that
# scale as decoded) by the RNE e2m1 convert at scale 1
_NVFP4_RCP6 = struct.unpack("f", struct.pack("I", 0x3E2AAAAB))[0]
NVFP4_PAGE_BYTES = 9216
NVFP4_SCALES_AT = 128 * 64


@traced
def nvfp4_quant16(vals, gs):
    """(scale code, packed codes) of a 128-element row's 16-element group held by 4
    lanes (``vals``: each lane's 4 elements, already rounded to bf16, as f32), as
    vLLM's ``_quant4`` does: ``gs`` the layer's global scale (1.0 for the index
    cache). The code is the e4m3 byte of min(amax * RN(1/6) / gs, 448); the elements
    are x / (gs * that scale as decoded) rounded RNE to e2m1 at scale 1; ``packed``
    holds a lane's four codes (low nibble first) in its low 16 bits."""
    am = fmath.absf(vals[0])
    for v in vals[1:]:
        am = fx.max(am, fmath.absf(v))
    am = butterfly(am, (2, 1), fx.max)
    sf = fx.min(
        (am * fx.Float32(_NVFP4_RCP6)) * (fx.Float32(1.0) / gs), fx.Float32(448.0)
    )
    code = fx.Int32(
        rocdl.cvt_pk_fp8_f32(
            T.i32, as_ir_value(sf), as_ir_value(sf), as_ir_value(fx.Int32(0)), False
        )
    )
    sv = fx.Vector(
        rocdl.cvt_pk_f32_fp8(
            res=fx.Vector.make_type(2, fx.Float32),
            src=as_ir_value(code),
            word_sel=False,
        )
    )[0]
    quant = (sv != 0.0).select(fx.Float32(1.0) / (sv * gs), fx.Float32(0.0))
    one = as_ir_value(fx.Float32(1.0))
    packed = fx.Int32(0)
    for pair in range_constexpr(2):
        packed = fx.Int32(
            rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32,
                as_ir_value(packed),
                as_ir_value(vals[2 * pair] * quant),
                as_ir_value(vals[2 * pair + 1] * quant),
                one,
                pair,
            )
        )
    return code, packed


@traced
def nvfp4_row_store(r_page, slot, vals, l32, cm=0):
    """Token ``slot`` of a pool of single-buffer nvfp4 index pages (9216 B: 128 x 64 B
    e2m1, low nibble the even dim, then 128 x 8 B e4m3 scales, one per 16 dims; no
    global scale): one 128-element row quantized as vLLM's writer does, by the 32
    lanes of a half wave holding 4 consecutive elements ``vals`` each (already
    rounded to bf16, as an f32). Lane l writes bytes 2 l, 2 l + 1 of the row's 64
    data bytes; the first lane of each group of 4 writes its scale byte."""
    code, packed = nvfp4_quant16(vals, fx.Float32(1.0))
    page = slot // 128
    t = slot % 128
    base = page * NVFP4_PAGE_BYTES
    bo.buffer_store(
        fx.Int16(packed & 0xFFFF),
        r_page,
        base + t * 64 + l32 * 2,
        cache_modifier=cm,
        offset_is_bytes=True,
    )
    if l32 % 4 == 0:
        bo.buffer_store(
            fx.Int8(code & 0xFF),
            r_page,
            base + NVFP4_SCALES_AT + t * 8 + l32 // 4,
            cache_modifier=cm,
            offset_is_bytes=True,
        )


# The K / V page (128 tokens, 2 * Hkv * 9216 B; ``[K data | K scale | V data | V scale]``,
# a head's data 8192 B, its scales 1024 B): K data t * 64 + d // 2 (low nibble the even
# dim), K scale ((((t // 32) * 4 + j // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + j % 2
# (r = t % 32, n = 4 * (r // 8) + r % 4, j = d // 16), V data ((t // 8) * 128 + d) * 4 +
# (t % 8) // 2 with nibble t % 2, V scale j * 128 + t. Hkv = 1 (TP4).
KV_PAGE_BYTES = 2 * 9216
KV_V_DATA_AT = 9216
KV_V_SCALE_AT = 9216 + 8192


def nvfp4_k_scale_at(t, j):
    """Byte of K scale (token ``t``, group ``j``) inside the K scale region."""
    r = t % 32
    n = 4 * (r // 8) + r % 4
    return ((((t // 32) * 4 + j // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + j % 2


@traced
def nvfp4_k_store(r_page, slot, vals, l32, gs, cm=0):
    """Token ``slot`` of a pool of single-buffer nvfp4 K/V pages, its K row (128
    elements held 4 a lane by a half wave, bf16-rounded f32): the e2m1 data in token
    order and the e4m3 scales in the lane-permuted order, quantized by ``nvfp4_quant16``
    with the layer's global k scale ``gs``."""
    code, packed = nvfp4_quant16(vals, gs)
    base = (slot // 128) * KV_PAGE_BYTES
    t = slot % 128
    bo.buffer_store(
        fx.Int16(packed & 0xFFFF),
        r_page,
        base + t * 64 + l32 * 2,
        cache_modifier=cm,
        offset_is_bytes=True,
    )
    if l32 % 4 == 0:
        bo.buffer_store(
            fx.Int8(code & 0xFF),
            r_page,
            base + 8192 + nvfp4_k_scale_at(t, l32 // 4),
            cache_modifier=cm,
            offset_is_bytes=True,
        )


@traced
def nvfp4_v_store(r_page, cache_addr, slot, vals, l32, gs, cm=0):
    """``nvfp4_k_store`` for V: the scale byte of group j at ``j * 128 + t``; a data
    byte holds tokens 2 b and 2 b + 1 of one dim, so each nibble is merged into its
    dword with two atomics (the other tokens of the octet write theirs concurrently).
    ``cache_addr``: the cache's base address (Int64) for the atomics."""
    code, packed = nvfp4_quant16(vals, gs)
    page = slot // 128
    t = slot % 128
    base = page * KV_PAGE_BYTES
    if l32 % 4 == 0:
        bo.buffer_store(
            fx.Int8(code & 0xFF),
            r_page,
            base + KV_V_SCALE_AT + (l32 // 4) * 128 + t,
            cache_modifier=cm,
            offset_is_bytes=True,
        )
    shift = ((t % 8) // 2) * 8 + (t % 2) * 4
    for i in range_constexpr(4):
        d = l32 * 4 + i
        dw = fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
            cache_addr + fx.Int64(base + KV_V_DATA_AT + ((t // 8) * 128 + d) * 4),
        )
        nib = (packed >> (4 * i)) & 0xF
        fx.llvm.atomic_and(
            dw, fx.Int32(-1) ^ (fx.Int32(0xF) << shift), syncscope="agent"
        )
        fx.llvm.atomic_or(dw, nib << shift, syncscope="agent")
