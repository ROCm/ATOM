# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device primitives of the layer kernel beyond the shared ``atom.mono.device.ops``:
buffer / LDS element access, natural exp, wave-wide unsigned max and FP8 / MX conversions.
"""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.mono.device.ops import hw_exp2, uniform

NEG = -1.0e30
LOG2E = 1.4426950408889634


def uniform_f32(value):
    return uniform(fx.Float32(value).bitcast(fx.Int32)).bitcast(fx.Float32)


def write_lane_i32(value, lane, vector):
    """Write one i32 into ``lane`` of a wave-distributed value."""

    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.writelane.i32",
            [as_ir_value(fx.Int32(item)) for item in (value, lane, vector)],
            [],
            [],
        )
    )


def exp(value):
    return hw_exp2(fx.Float32(value) * LOG2E)


def wave_umax(value):
    return fx.Int32(fx.coop.warp_reduce(fx.Uint32(value), fx.ReductionOp.MAX, width=64))


def ld_f32(r, i):
    """One f32 from buffer resource ``r`` at element ``i``."""

    return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))


def ld_bf16(r, i):
    """One bf16 from buffer resource ``r`` at element ``i``, widened to f32."""

    return fx.Float32(fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16)))


def lds_ld(ptr, i):
    return fx.ptr_load(ptr + i)


def lds_st(ptr, i, v):
    fx.ptr_store(v, ptr + i)


def fp8_roundtrip(lhs, rhs):
    """Round an f32 pair through E4M3FN and return the f32 pair."""

    word = rocdl.cvt_pk_fp8_f32(T.i32, lhs, rhs, fx.Int32(0), False)
    pair_type = fx.Vector.make_type(2, fx.Float32)
    pair = fx.Vector(rocdl.cvt_pk_f32_fp8(res=pair_type, src=word, word_sel=False))
    return pair[0], pair[1]


def fp8_to_bf16x8(word0, word1):
    """Convert two dwords of eight FP8 values to a BF16 vector."""

    one = as_ir_value(fx.Float32(1.0))
    parts = []
    for word in (word0, word1):
        for half in range_constexpr(2):
            pair = fx.Vector(
                rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16), as_ir_value(word), one, bool(half)
                )
            )
            parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def mxfp8_to_bf16x8(word0, word1, scale):
    """Convert two dwords of eight FP8 values with one E8M0 block scale to BF16."""

    scale = as_ir_value(scale)
    parts = []
    for word in (word0, word1):
        for half in range_constexpr(2):
            pair = fx.Vector(
                rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16), as_ir_value(word), scale, bool(half)
                )
            )
            parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def mxfp4_to_bf16x8(word, scale):
    """Convert one packed dword of eight scaled E2M1 values to BF16."""

    parts = []
    for select in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp4(
                T.vec(2, T.bf16),
                as_ir_value(word),
                as_ir_value(scale),
                select,
            )
        )
        parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)
