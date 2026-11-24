# SPDX-License-Identifier: Apache-2.0
"""ATOM's MXFP8 group rule (atom/model_ops/blockscale_kernels/quantization.py fp8_groups) for the lanes layout of
amd-kernels' allreduce_gemma_bf16_mxfp8_h6144 (4 lanes x 8 values = one 32-column group), and v_rsq_f32."""
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T

from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144.norm_math import max_dpp, maxnum, minnum
from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144.quant_math import _cvt_pk_fp8

INV448 = 1.0 / 448.0  # Triton folds `amax * (1.0 / FP8_MAX)` to this fp32 constant


def rsq(x):
    """v_rsq_f32 (Triton's tl.rsqrt on gfx950 for normal inputs)."""
    return fx.Float32(_llvm.call_intrinsic(T.f32, "llvm.amdgcn.rsq.f32", [fx.as_ir_value(fx.Float32(x))], [], []))


def atom_e8m0_codes(y):
    """(E8M0 code as i32, 2 dwords of 8 e4m3 codes) of a 32-column group whose 4 lanes hold ``y`` (8 fp32 each):
    amax = max(|y|, 1e-10); code = ceil_pow2(amax * fl32(1/448)); q = clamp(y * 2^(127 - code), +-448) -> e4m3."""
    a = fx.math.absf(y[0])
    for v in y[1:]:
        a = maxnum(a, fx.math.absf(v))
    a = max_dpp(a, 0x4E)
    a = maxnum(max_dpp(a, 0xB1), fx.Float32(1e-10))
    bits = (a * fx.Float32(INV448)).bitcast(fx.Int32)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    e = fx.Int32(254) - code
    rec = ((e > 0).select(e, fx.Int32(0)) << 23).bitcast(fx.Float32)
    q = [minnum(maxnum(v * rec, fx.Float32(-448.0)), fx.Float32(448.0)) for v in y]
    lo = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[0], q[1], False), q[2], q[3], True)
    hi = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[4], q[5], False), q[6], q[7], True)
    return code, [lo, hi]
