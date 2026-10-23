# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Common AMD expression primitives for fused model-layer kernels."""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

ROW_TILE = 32

SCALE_BM = 128
FP8_MAX = 448.0


def _hardware_f32(name, value):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(value).ir_value()], [], [])
    )


def sqrt(value):
    """``v_sqrt_f32``: Triton's ``tl.sqrt`` on gfx950."""
    return _hardware_f32("llvm.amdgcn.sqrt.f32", value)


# ln 2 as LLVM's f32 log lowering splits it: a head (0x3f317217) and its tail (0x3377d1cf)
LN2_HI = 0.6931471228599548
LN2_LO = 5.769998878690785e-08


def log(value):
    """Triton's ``tl.log`` on gfx950 (LLVM's f32 log, for normal inputs): ``v_log_f32`` (log2), then ln 2
    in extended precision, ``fma(hi, y, fma(lo, y, fma(y, hi, -y * hi)))``."""
    y = _hardware_f32("llvm.amdgcn.log.f32", value)
    e = fmath.fma(y, fx.Float32(LN2_HI), -(y * LN2_HI))
    e = fmath.fma(fx.Float32(LN2_LO), y, e)
    return fmath.fma(fx.Float32(LN2_HI), y, e)


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
