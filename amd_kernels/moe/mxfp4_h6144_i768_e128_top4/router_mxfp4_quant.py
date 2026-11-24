# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Helpers of the fused router + input quant kernels (``router_mxfp4_quant_line_layout``,
``router_mxfp4_mxfp8_quant_line_layout``, ``router_mxfp4_mxfp8_quant_row_scales``): the bf16 amax, the e2m1 converter,
the Quark-even e8m0 of a bf16 amax, the lane swaps, and the K split per row count."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .a16w4_utils import _raw
from .common.loaders import _as_f32


def _pk_max_u16(a, b):
    """v_pk_max_u16, per 16-bit half: on sign-cleared bf16 patterns the unsigned order is the value order."""
    va = _llvm.BitcastOp(T.i16x2, _raw(fx.Int32(a))).result
    vb = _llvm.BitcastOp(T.i16x2, _raw(fx.Int32(b))).result
    return fx.Int32(_llvm.BitcastOp(T.i32, _llvm.intr_umax(va, vb)).result)

def _amax_bf16(dwords):
    """max |x| over the 2 * len(dwords) bf16 values, as a 16-bit bf16 pattern in an i32."""
    m = None
    for d in dwords:
        a = fx.Int32(d) & fx.Int32(0x7FFF7FFF)
        m = a if m is None else _pk_max_u16(m, a)
    lo, hi = m & fx.Int32(0xFFFF), m >> 16
    return (lo > hi).select(lo, hi)

def _i32_max(a, b):
    return (a > b).select(a, b)

def _swap16(v):
    """(even-row value, odd-row value) of this lane's 16-lane row pair (klane q, q ^ 1)."""
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _raw(fx.Int32(v))
    res = fx.rocdl.permlane16_swap(pair_ty, raw, raw, False, False)
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(T.i32, res, [1])
    )

def _swap32(v):
    """(low-half value, high-half value): the values of klanes 0/1 and 2/3."""
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _raw(fx.Int32(v))
    res = fx.rocdl.permlane32_swap(pair_ty, raw, raw, False, False)
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(T.i32, res, [1])
    )


def _cvt_pk_fp4_bf16(old, dword, scale_f32, sel):
    """v_cvt_scalef32_pk_fp4_bf16: the two bf16 of ``dword`` (low first) / scale -> 2 e2m1 in byte ``sel``, the f32
    form's codes (a bf16 is exact in f32) without unpacking the pair."""
    pair = _llvm.BitcastOp(T.bf16x2, fx.Int32(dword).ir_value()).result
    return fx.Int32(
        _llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.cvt.scalef32.pk.fp4.bf16",
            [
                fx.Int32(old).ir_value(),
                pair,
                scale_f32.ir_value(),
                fx.Int32(sel).ir_value(),
            ],
            [],
            [],
        )
    )


def _bf16_to_f32(v):
    """8 x bf16 (2 per dword) -> 8 x f32 by a shift / a mask per value; ``.to(fx.Float32)`` takes 16 VALU per 8 (a
    second v_cvt_pk_bf16_f32 of every source value plus the shift)."""
    d = v.bitcast(fx.Int32)
    return Vec.from_elements(
        [
            _as_f32(x)
            for i in range(4)
            for x in (fx.Int32(d[i]) << 16, fx.Int32(d[i]) & fx.Int32(-65536))
        ],
        fx.Float32,
    )


def _e8m0_e2m1_even_bf16(amax_bits):
    """Quark even scale of a BF16 amax: carry at mantissa 0x60, byte >= 1."""
    t = (fx.Int32(amax_bits) - fx.Int32(0xE0)) >> 7
    return _i32_max(t, fx.Int32(1))


# (max rows, K split) that fill whole CU rounds: router + top-k -2..-18 us vs a neighbour split at 2049-21760 rows
# (above the largest DBO microbatch only a BF16 call with sequence parallelism off comes: the 10881-16384 split)
K_SPLITS = ((2048, 16), (2688, 12), (4096, 8), (5376, 6), (8192, 4), (10880, 3), (16384, 2), (21760, 3))
K_SPLIT_ABOVE = 2


def router_k_split(m: int) -> int:
    for bound, ks in K_SPLITS:
        if m <= bound:
            return ks
    return K_SPLIT_ABOVE
