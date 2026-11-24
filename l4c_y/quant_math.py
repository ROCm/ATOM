# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The bf16-row MXFP8 scale rule and layouts; copied unchanged from the norm family."""
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr.typing import T

from .norm_math import H, max_dpp, maxnum, minnum

SCALE_K1 = H // 256  # AITER scale layout: 256-column blocks per row
TINY = 0x00800000  # fp32 tiny, the amax floor
HWREG_MODE_FP16_OVFL = 1 | (23 << 6)  # hwreg(HW_REG_MODE, 23, 1)


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _set_fp16_ovfl():
    """MODE.fp16_ovfl = 1, which Triton sets before its fp8 conversions."""
    _llvm.call_intrinsic(
        None,
        "llvm.amdgcn.s.setreg",
        [fx.Int32(HWREG_MODE_FP16_OVFL).ir_value(), fx.Int32(1).ir_value()],
        [],
        [],
    )


def _cvt_pk_fp8(old, a, b, hi: bool):
    """v_cvt_scalef32_pk_fp8_f32 (scale 1): a, b -> 2 e4m3 in the low / high half."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def _e8m0(y):
    """(scale byte as fp32, 2^(127 - byte)) of a 32-column group whose 4 lanes hold
    ``y``: Triton's amax / log2 / ceil / clamp sequence."""
    a = fx.math.absf(y[0])
    for v in y[1:]:
        a = maxnum(a, fx.math.absf(v))
    a = max_dpp(a, 0x4E)  # quad_perm [2, 3, 0, 1]
    a = maxnum(max_dpp(a, 0xB1), _f32(TINY))  # quad_perm [1, 0, 3, 2]
    d = a / fx.Float32(448.0)
    small = d < _f32(TINY)
    lg = fx.Float32(
        _rocdl_ops.log(T.f32, fx.as_ir_value(small.select(d * fx.Float32(2.0**32), d)))
    ) - small.select(fx.Float32(32.0), fx.Float32(0.0))
    sb = minnum(
        maxnum(fx.math.ceil(lg) + fx.Float32(127.0), fx.Float32(0.0)),
        fx.Float32(254.0),
    )
    rescale = fx.Float32(_rocdl_ops.exp2(T.f32, fx.as_ir_value(fx.Float32(127.0) - sb)))
    return sb, rescale


def _codes(y, rescale):
    """8 e4m3 codes of y * rescale as 2 dwords (column order, low byte first)."""
    q = [v * rescale for v in y]
    lo = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[0], q[1], False), q[2], q[3], True)
    hi = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[4], q[5], False), q[6], q[7], True)
    return [lo, hi]


def _scale_off(m, g):
    """Byte of (row, group) in AITER's shuffled scale layout."""
    return (
        ((((m // 32) * SCALE_K1 + g // 8) * 4 + g % 4) * 16 + m % 16) * 2 + (g % 8) // 4
    ) * 2 + (m % 32) // 16


def _code_off(m, p):
    """Byte of (row, 16-column piece) in the (16, 16) code layout."""
    return (m // 16) * (16 * H) + (p // 4) * 1024 + (p % 4) * 256 + (m % 16) * 16
