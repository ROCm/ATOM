# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bit-exact scalar helpers shared by the TileRT ports (bf16 halves and RNE as TileRT's code does them)."""

import flydsl.expr as fx

Int32, F32 = fx.Int32, fx.Float32


def bf_lo(w):
    """f32 of a packed bf16 pair word's low half."""
    return (Int32(w) << 16).bitcast(F32)


def bf_hi(w):
    """f32 of a packed bf16 pair word's high half."""
    return (Int32(w) & Int32(-65536)).bitcast(F32)


def bf_rne_hi(x):
    """f32 -> its bf16 round-to-nearest-even in the high half of an i32 (low half garbage, no NaN fixup)."""
    b = F32(x).bitcast(Int32)
    return b + ((b >> 16) & 1) + 0x7FFF


def bf_round(x):
    """f32 rounded to bf16 (RNE), as f32."""
    return (bf_rne_hi(x) & Int32(-65536)).bitcast(F32)


def pack_bf2(lo, hi):
    """Two f32 -> one word (bf16(lo) | bf16(hi) << 16), RNE."""
    return (bf_rne_hi(hi) & Int32(-65536)) | ((bf_rne_hi(lo) >> 16) & 0xFFFF)


# --- shared with qkv_kg / mega_tail ---
from flydsl._mlir.dialects import llvm
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from atom.mono.device.ops import hw_exp2

EPS6 = 0x358637BD  # 1e-6
LOG2E, LOG2E_LO = 0x3FB8AA3B, 0x32A5705F
NLOG2E, NLOG2E_LO = 0xBFB8AA3B, 0xB2A5705F


def f32c(bits):
    """The f32 with these IEEE bits."""
    return Int32(bits).bitcast(F32)


def bperm(v, src_lane):
    """f32 v of lane src_lane (ds_bpermute)."""
    w = rocdl.ds_bpermute(T.i32, Int32(src_lane) * 4, F32(v).bitcast(Int32).ir_value())
    return Int32(w).bitcast(F32)


def _call(name, ret, *xs):
    return llvm.call_intrinsic(ret, name, [x.ir_value() for x in xs], [], [])


def exp_ocml(x, neg):
    """OCML expf of (neg ? -x : x) as TileRT's code has it: t = x*(+-log2e) split with its fma error
    and the low log2e word, v_exp_f32 of the fraction, ldexp by the rounded part, then the
    overflow / underflow clamps."""
    hi, lo = (f32c(NLOG2E), f32c(NLOG2E_LO)) if neg else (f32c(LOG2E), f32c(LOG2E_LO))
    t = x * hi
    err = fmath.fma(x, hi, -t)
    n = F32(_call("llvm.rint.f32", T.f32, t))
    err = fmath.fma(x, lo, err)
    f = (t - n) + err
    e = F32(_call("llvm.ldexp.f32.i32", T.f32, hw_exp2(f), n.to(Int32)))
    if neg:  # exp(-x): x > 103.28 -> 0, x < -88.72 -> inf
        e = (x <= f32c(0x42CE8ED0)).select(e, F32(0.0))
        return (x >= f32c(0xC2B17218)).select(e, f32c(0x7F800000))
    e = (x >= f32c(0xC2CE8ED0)).select(e, F32(0.0))
    return (x <= f32c(0x42B17218)).select(e, f32c(0x7F800000))


def log1p_ocml(e):
    """OCML log1pf exactly as hipcc emits it in TileRT's router (mega_afm_tail L4313-L4432), one
    statement an instruction (v<n>: the register it lives in there): 1 + e as a double-float, frexp
    with the 2/3 split, (m - 1) / (m + 1) in double-float (rcp + two correction rounds), the odd
    polynomial, + k ln2 in double-float; inf -> inf, |e| < 2^-24 -> e."""
    one, m1 = F32(1.0), F32(-1.0)
    v16 = one + e
    v17 = v16 + m1
    v20 = v17 - v16
    v20 = one + v20
    v17 = e - v17
    v21 = F32(_call("llvm.amdgcn.frexp.mant.f32", T.f32, v16))
    v17 = v17 + v20
    v20i = Int32(_call("llvm.amdgcn.frexp.exp.i32.f32", T.i32, v16))
    k = v20i - (f32c(0x3F2AAAAB) > v21).select(Int32(1), Int32(0))
    nk = Int32(0) - k
    v16 = F32(_call("llvm.ldexp.f32.i32", T.f32, v16, nk))
    v17 = F32(_call("llvm.ldexp.f32.i32", T.f32, v17, nk))
    v20 = v16 + m1
    v22 = v16 + one
    v21 = one + v20
    v23 = v22 + m1
    v21 = v16 - v21
    v16 = v16 - v23
    v16 = v17 + v16
    v27 = v22 + v16
    v29 = F32(_call("llvm.amdgcn.rcp.f32", T.f32, v27))
    v21 = v17 + v21
    v17 = v22 - v27
    v28 = v16 + v17
    v17 = v20 + v21
    v31 = v17 * v29
    v16 = v20 - v17
    v20 = v27 * v31
    v22 = fmath.fma(v31, v27, -v20)
    v22 = fmath.fma(v31, v28, v22)
    v30 = v21 + v16
    v16 = v20 + v22
    v21 = v17 - v16
    v24, v25 = v16 - v20, v17 - v21
    v23 = v16
    v16, v17 = v24 - v22, v25 - v23
    v17 = v30 + v17
    v16 = v16 + v17
    v17 = v21 + v16
    v30 = v29 * v17
    v20 = v27 * v30
    v22 = fmath.fma(v30, v27, -v20)
    v22 = fmath.fma(v30, v28, v22)
    v21 = v21 - v17
    v27 = v16 + v21
    v16 = v20 + v22
    v21 = v17 - v16
    v24, v25 = v16 - v20, v17 - v21
    v23 = v16
    v16, v17 = v24 - v22, v25 - v23
    v17 = v27 + v17
    v16 = v16 + v17
    v16 = v21 + v16
    v17 = v29 * v16
    v16 = v31 + v30
    v20 = v16 - v31
    v21 = v30 - v20
    v20 = k.to(F32)
    v22 = v21 + v17
    v21 = v16 + v22
    v17 = v21 * v21
    v23 = fmath.fma(f32c(0x3E9B6DAC), v17, f32c(0x3ECC95A3))
    v23 = fmath.fma(v17, v23, f32c(0x3F2AAADA))
    v17 = v21 * v17
    ln2 = f32c(0x3F317218)
    v26, v27 = v20 * ln2, v21 * F32(2.0)
    v25 = v17 * v23
    v17 = fmath.fma(v20, ln2, -v26)
    v24 = fmath.fma(v20, f32c(0xB102E308), v17)
    v28, v29 = v26 + v24, v27 + v25
    v20, v21, v17 = v21, v29, v27
    v16, v17 = v20 - v16, v21 - v17
    v23 = v25
    v16, v17 = v22 - v16, v23 - v17
    v25 = v28
    v16 = v16 + v16
    v27 = v16 + v17
    v16, v17 = v28 - v26, v29 - v27
    v20, v21 = v28 + v26, v29 + v27
    v26, v17 = v27, v21
    v30, v31 = v24 + v16, v25 + v17
    v34 = v21
    v32 = v31 - v28
    v35 = v31
    v36, v37 = v29, v32
    v22, v23 = v24 - v16, v25 - v17
    v34, v35 = v34 - v36, v35 - v37
    v27, v25, v17 = v28, v21, v32
    v26, v27 = v26 - v34, v27 - v35
    v16, v17 = v24 - v16, v25 - v17
    v23 = v31
    v16, v17 = v16 + v26, v17 + v27
    v20, v21 = v16 + v17, v17 + v16
    v24 = v31 + v20
    v17 = v24
    v28, v29 = v16 - v22, v17 - v23
    v27 = v20
    v16 = v16 - v28
    v20, v21 = v26 - v28, v27 - v29
    v16 = v22 - v16
    v16 = v20 + v16
    v16 = v16 + v21
    v16 = v24 + v16
    inf = f32c(0x7F800000)
    r = (e != inf).select(v16, inf)
    return (abs(e) < f32c(0x33800000)).select(e, r)
