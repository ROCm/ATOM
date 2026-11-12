# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 / MX arithmetic the mono kernels share: the E4M3 range, E8M0 scale
codes, the byte of a scale in aiter's ``shuffle_scale`` layout, an FP4 weight
tile's operand loads and the scaled MFMA that takes them.

An E8M0 code is a biased exponent; ``pow2(code)`` is the scale it stands for.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.mono.device.ops import CM_NT

FP8_MAX = 448.0
# the f8f6f4 MFMA's operand formats (cbsz / blgp)
FP8, FP4 = 0, 4
UNIT_SCALE = 127  # the E8M0 code of 1.0


def clamp_fp8(v):
    """``v`` clamped to the E4M3 range."""
    return fx.max(fx.min(v, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))


def code_ceil(v):
    """Biased exponent of the smallest power of two >= ``v`` (v >= 0, finite),
    clamped to [1, 254]: the codes a scale may take."""
    bits = v.bitcast(fx.Int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )
    return fx.max(fx.min(e, fx.Int32(254)), fx.Int32(1))


def pow2(code):
    """The fp32 power of two a biased exponent stands for."""
    return (code << 23).bitcast(fx.Float32)


def rcp_pow2(code):
    """1 / ``pow2(code)``, exact (code in [1, 253]): x times it is x / scale
    bit for bit, a multiply where the division is a dozen instructions."""
    return ((254 - code) << 23).bitcast(fx.Float32)


def scale_index(row, col, cols):
    """Byte of e8m0 scale (row, col) in aiter's ``shuffle_scale`` layout:
    ``view(rows/32, 2, 16, cols/8, 2, 4).permute(0, 3, 5, 2, 4, 1)``."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * (cols // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a


def fp4_tile_load(r_w, base, rg, k, st, at):
    """A lane's 16 B of an FP4 weight's (16, 16)-shuffled 16-row group ``rg``
    (K = ``k``) at 128-k step ``st``: dword ``base`` + the group's 2 k + 256 a
    step + ``at`` (``lane * 4``: 32 fp4, K block lane / 16 of row lane % 16).
    Nontemporal: a weight a step reads once."""
    return fx.Vector(
        bo.buffer_load(
            r_w,
            base + rg * (2 * k) + st * 256 + at,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=CM_NT,
        )
    )


def mfma_scaled(a, b, c, sa, sb, a_fmt=FP8, b_fmt=FP8, sa_byte=0, sb_byte=0):
    """One 16x16x128 scaled MFMA (gfx950 f8f6f4): A ``a`` (``a_fmt``) with E8M0
    ``sa`` (its byte ``sa_byte``), B ``b`` (``b_fmt``) with E8M0 ``sb`` (its
    byte ``sb_byte``)."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, c, a_fmt, b_fmt, sa_byte, sa, sb_byte, sb]
        )
    )


FP4_INV_MAX = (
    1.0 / 6.0
)  # f32(1/6) = 0x3E2AAAAB: the constant aiter's HIP / FlyDSL RoundUp multiply by


def fp4_scale(amax):
    """(E8M0 code, forward scale 2^(code - 127)) of an MXFP4 32-block of absolute maximum ``amax``:
    aiter's RoundUp, ceil_pow2(amax * f32(1/6)) (``mxfp4_gemm_common._e8m0_roundup``, csrc
    mx_quant_utils.h ``fp_f32_to_e8m0_scale<RoundUp, FP4_E2M1>``); see ``code_ceil`` for the clamp.
    """
    e = code_ceil(amax * fx.Float32(FP4_INV_MAX))
    return e, pow2(e)


def fp4_pack2(old, a, b, scale, sel):
    """``old`` with byte ``sel`` (0..3, compile time) := the e2m1 nibbles of ``a`` (low) and ``b``
    (high) divided by ``scale`` (a power of two, the forward scale), RNE, saturating at 6
    (``v_cvt_scalef32_pk_fp4_f32``; bit-exact with aiter's quantization, hip/fp4_mfma_probe.cpp).
    """
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32,
            as_ir_value(old),
            as_ir_value(a),
            as_ir_value(b),
            as_ir_value(scale),
            sel,
        )
    )


def fp4x8_bf16(word, code):
    """The eight E2M1 values of dword ``word`` (low nibble first) times the E8M0
    scale ``code`` (its low byte) as a v8bf16 MFMA operand: exact (a product of a
    power of two and a 2-bit-mantissa value), by four ``v_cvt_scalef32_pk_bf16_fp4``
    (aiter's a16w4 kernels' up-conversion)."""
    scale = as_ir_value(((code & 0xFF) << 23).bitcast(fx.Float32))
    out = []
    for sel in range_constexpr(4):
        pair = rocdl.cvt_scalef32_pk_bf16_fp4(
            T.vec(2, T.bf16), as_ir_value(word), scale, sel
        )
        out.append(fx.Vector(pair).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(out, fx.Int32).bitcast(fx.BFloat16)
