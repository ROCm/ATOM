# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared pieces of the Gemma add + RMSNorm kernels: bf16 packing and the sum of
squares in the reduction order of Triton's ``tl.sum`` (gfx950, wave64)."""

import flydsl.expr as fx
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr.typing import T

H = 6144

# kernel arguments preloaded into SGPRs (as Triton does), not s_load'ed first
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def bf16x8(v):
    """8 fp32 of 4 dwords of bf16 (low half first)."""
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (v_cvt_pk_bf16_f32, RNE), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _dpp_f(v, ctrl, row_mask=0xF, old=None):
    """v_mov_b32_dpp of an fp32; source lanes outside the row read 0 (bound_ctrl),
    rows ``row_mask`` leaves out keep ``old``."""
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(_bits(old))
    r = _rocdl_ops.update_dpp(
        T.i32, o, fx.as_ir_value(_bits(v)), ctrl, row_mask, 0xF, True
    )
    return fx.Int32(r).bitcast(fx.Float32)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def max_dpp(a, ctrl):
    """maxnum(a, DPP lane of a): the group amax steps (quad_perm)."""
    b = fx.Int32(
        _rocdl_ops.update_dpp(
            T.i32,
            _llvm.mlir_undef(T.i32),
            fx.as_ir_value(_bits(a)),
            ctrl,
            0xF,
            0xF,
            True,
        )
    ).bitcast(fx.Float32)
    return maxnum(a, b)


def swap_xor1(words):
    """Each i32 of ``words`` from lane l ^ 1 (quad_perm [1, 0, 3, 2])."""
    return [
        fx.Int32(
            _rocdl_ops.update_dpp(
                T.i32, _llvm.mlir_undef(T.i32), fx.as_ir_value(w), 0xB1, 0xF, 0xF, True
            )
        )
        for w in words
    ]


def seq_sum_sq(vals):
    """((v0^2 + v1^2) + v2^2) + ...: each square rounded, no FMA (Triton's per-thread
    accumulation)."""
    sq = [v * v for v in vals]
    acc = sq[0] + sq[1]
    for q in sq[2:]:
        acc = acc + q
    return acc


def wave_sum_lane63(acc):
    """Triton's wave reduction: row_shr 8/4/2/1 adds (lane 15 of each row holds the
    row sum), row_bcast 15 into rows 1 and 3, row_bcast 31; the result is valid in
    lane 63: (R0 + R1) + (R2 + R3)."""
    for sh in (8, 4, 2, 1):
        acc = acc + _dpp_f(acc, 0x110 + sh)
    acc = _dpp_f(acc, 0x142, row_mask=0xA, old=acc) + acc
    return acc + _dpp_f(acc, 0x143)


def rsqrt_rn(ss, eps):
    """1 / v_sqrt_f32(ss / 6144 + eps), both divisions correctly rounded (Triton's
    ``1.0 / tl.sqrt(var + eps)`` with ``var = sum / n_cols``)."""
    var = ss / fx.Float32(float(H))
    sq = fx.Float32(_rocdl_ops.sqrt(T.f32, fx.as_ir_value(var + eps)))
    return fx.Float32(1.0) / sq


def xcd_row(bid, m):
    """(row, live) of workgroup ``bid``: workgroup b runs on XCD b % 8, and XCD x
    takes rows [x * (m // 8) + min(x, m % 8), ...), a contiguous run. Grid:
    ``xcd_grid(m)``."""
    x, k = bid % 8, bid // 8
    q8, r8 = m // 8, m % 8
    extra = x < r8
    row = x * q8 + extra.select(x, r8) + k
    return row, k < q8 + extra.select(fx.Int32(1), fx.Int32(0))


def xcd_grid(m):
    return (m + 7) // 8 * 8

def _block_sum(w):
    """Triton's fold of 16 wave sums (lane 0 of xor 8, 4, 2, 1), waves 12-15 = +0."""
    return (((w[0] + w[8]) + w[4]) + ((w[2] + w[10]) + w[6])) + (
        ((w[1] + w[9]) + w[5]) + ((w[3] + w[11]) + w[7])
    )
