# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""MXFP4 (E2M1 values, one ceil-E8M0 scale a 32-value group) as the FP4 index plane and its query hold it: aiter's
_mxfp4_quant_op(SCALING_MODE=1) bit for bit -- _mxfp4_ceil_scale_from_amax, then _mxfp4_pack_bits (round to nearest
even, saturating at 6, the sign kept for -0), two values a byte (the even one in the low nibble)."""
from __future__ import annotations


import flydsl.expr as fx

F32, I32 = fx.Float32, fx.Int32
_DENORM_INT = ((127 - 1) + (23 - 1) + 1) << 23   # _mxfp4_pack_bits' denorm_mask_int


def ceil_scale(amax):
    """(E8M0 code, the f32 quant scale 2^-(code - 127)) of a group's amax: 2^ceil(log2(amax / 6)), an all-zero group
    2^-126."""
    ratio = fx.max(amax, F32(6.0 * 2.0 ** -126)) * F32(1.0 / 6.0)
    code = (ratio.bitcast(I32) + I32(0x7FFFFF)) >> 23   # (ratio > 0: no sign bit)
    return code, ((I32(254) - code) << 23).bitcast(F32)


def e2m1(q):
    """f32 q (already times the quant scale) -> its 4-bit E2M1 code (bit 3 the sign)."""
    u = q.bitcast(I32)
    s = u & I32(-2147483648)
    a = u ^ s
    af = a.bitcast(F32)
    sat = af >= F32(6.0)
    den = af < F32(1.0)
    dx = ((af + I32(_DENORM_INT).bitcast(F32)).bitcast(I32) - I32(_DENORM_INT)) & 255
    nx = a + I32(((1 - 127) << 23) + (1 << 21) - 1)
    nx = ((nx + ((a >> 22) & 1)) >> 22) & 255
    v = sat.select(I32(7), den.select(dx, nx))
    return v | ((s >> 28) & 8)


def pack4(vals, qs):
    """4 f32 (consecutive dims: the even ones low nibbles) times qs -> 2 bytes (an i32's low 16 bits)."""
    c = [e2m1(v * qs) for v in vals]
    return c[0] | (c[1] << 4) | (c[2] << 8) | (c[3] << 12)


def wmma4(a, sa, b, sb, c):
    """The scaled 16x16x128 WMMA with A and B both MXFP4 (8 dwords a lane each, the low 8 of the 16-dword operands;
    sa / sb: the lane's row's 4 E8M0 codes, one a 32-K group)."""
    from flydsl.expr import rocdl
    from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx

    def v16(x):
        return fx.Vector.from_elements([x[e] for e in range(8)] + [I32(0)] * 8, I32)

    r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(hx.vty(8, F32.ir_type), hx._v(v16(a)), hx._v(v16(b)), hx._v(c),
                                              hx._v(I32(sa)), hx._v(I32(sb)), fmtA=4, fmtB=4)
    return fx.Vector(r, (8,), F32)


def ld8(addr, l, load):
    """A lane's 8 dwords of a 64-byte MXFP4 row (128 values) as the WMMA reads them: half h = l // 16 holds values
    32 h .. 32 h + 31 and 64 + 32 h .. 64 + 32 h + 31 (bytes 16 h + 32 c, c = 0, 1: the order the MoE's shuffled FP4
    weights load in, moe.lda)."""
    p = [load(addr + (l // 16) * 16 + cc * 32) for cc in range(2)]
    return [p[cc][e] for cc in range(2) for e in range(4)]
