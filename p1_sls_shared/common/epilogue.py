# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""LDS vector access, bf16 / e4m3 packing and the nnan max of the MoE GEMM epilogues."""

import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from .loaders import _lds_ptr_t

_NUM_XCDS = 8


def _lds_ptr(addr_i32):
    return _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(addr_i32))


def _lds_load_vec(addr_i32, n):
    ty = _ir.VectorType.get([n], _T.i32)
    return _llvm.LoadOp(ty, _lds_ptr(addr_i32), alignment=4 * n).result


def _lds_store_vec(vec, addr_i32, n):
    _llvm.StoreOp(vec, _lds_ptr(addr_i32), alignment=4 * n)


def _bf16x2(a, b):
    """Two f32 -> one i32 of two bf16 (RNE, v_cvt_pk_bf16_f32: the same rounding as ``arith.truncf``)."""
    return Vec.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]


def _cvt_pk_fp8(old, a, b, scale_f32, hi: bool):
    """v_cvt_scalef32_pk_fp8_f32: (a, b) / scale -> two e4m3 into the low (``hi`` False) or high 16 bits of ``old``
    (the intrinsic works on <2 x i16>; i32 in and out here)."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    old_v = _llvm.bitcast(v2i16, fx.as_ir_value(old))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            old_v,
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(scale_f32),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(_T.i32, res))


def _maxf_nn(a, b):
    """v_max_f32 with nnan: without the flag LLVM canonicalizes both inputs first (``v_max x, x, x``, maxnum must
    quiet sNaNs), 2 extra VALU per max."""
    fm = _ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def _v2i32(a, b):
    return Vec.from_elements([a, b], fx.Int32).ir_value()


_STORE_CPOL = 0x2
