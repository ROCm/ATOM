# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Helpers the I 768 GEMM1s and the router / quant kernels import (gfx950): the row-stationary FP4 MFMA class, the
e2m1 / e8m0 converters, the SwiGLU-OAI epilogue, the per-XCD claim helpers and ``interleave_w13_scales``."""

import flydsl.expr as fx
import torch
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from ..common.epilogue import _maxf_nn
from ..common.loaders import _as_f32, _bits, _permlane16_swap
from ..common.tile_map import CLAIM_COUNTERS
from ..mxfp8.gemm1_prefill_h6144_i3072 import SWIGLU_ALPHA, SWIGLU_LIMIT, Mfma16x16x128Fp8, _fmed3
from ..common.mfma_order import row_stationary_order


class Mfma16x16x128Fp4(Mfma16x16x128Fp8):
    """e2m1 x e2m1 16x16x128 scaled MFMAs (``cbsz:4 blgp:4``, 4-VGPR operands): two per tile pair and 128-B step, on
    half h of the fragments (``a[i][h]``) with scale byte (row half, h); every pair's half 0 goes first, then half 1, so
    a pair's two dependent accumulates are a whole quadrant of MFMAs apart."""

    def call(
        self,
        a,
        b,
        c,
        sa,
        sb,
        gu,
        k2,
        interleave=None,
        zero_acc=False,
        late=None,
        late_start=4,
        sb_index=None,
    ):
        if sb_index is None:
            sb_index = lambda j, gu: (j, gu)  # noqa: E731
        thunks = list(interleave) if interleave else []
        assert not late
        order = self._order()
        seq = [(h, i, j) for h in range(2) for (i, j) in order]
        n_mfma = len(seq)
        by_slot: dict[int, list] = {}
        for t, th in enumerate(thunks):
            by_slot.setdefault((t * n_mfma) // len(thunks), []).append(th)
        for m, (h, i, j) in enumerate(seq):
            acc = None if (zero_acc and h == 0) else c[self.idx(i, j)]
            sbj, sbb = sb_index(j, gu)
            c[self.idx(i, j)] = self._mfma4(
                a[i][h], b[j][h], acc, sa[i // 2], sb[sbj], i % 2, sbb, h
            )
            for th in by_slot.get(m, ()):
                th()
        for m in range(n_mfma, n_mfma + len(thunks) + 1):
            for th in by_slot.get(m, ()):
                th()
        return c

    def _mfma4(self, a_op, b_op, acc, sa_v, sb_v, ia, jb, h):
        a_op, b_op = b_op, a_op
        sa_v, sb_v = sb_v, sa_v
        ia, jb = jb, ia
        src2 = "$0" if acc is not None else "0"
        asm = (
            f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
            f"op_sel:[{ia},{jb},0] op_sel_hi:[{h},{h},0] cbsz:4 blgp:4"
        )
        ops = [
            fx.as_ir_value(a_op),
            fx.as_ir_value(b_op),
            fx.as_ir_value(sa_v),
            fx.as_ir_value(sb_v),
        ]
        cons = "=a,v,v,v,v"
        if acc is not None:
            ops.append(fx.as_ir_value(acc))
            cons += ",0"
        return _llvm.inline_asm(self.res_ty, ops, asm, cons, has_side_effects=True)


class MfmaRowStationaryFp4(Mfma16x16x128Fp4):
    """``Mfma16x16x128Fp4`` issuing each half's quadrant row by row."""

    def _order(self):
        return row_stationary_order(self.n_tiles_a, self.n_tiles_b)


def _cvt_pk_fp4(old, a, b, scale_f32, sel):
    """v_cvt_scalef32_pk_fp4_f32: two f32 / scale -> 2 e2m1 into byte ``sel``."""
    return fx.Int32(
        _llvm.call_intrinsic(
            _T.i32,
            "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
            [
                fx.as_ir_value(old),
                fx.as_ir_value(a),
                fx.as_ir_value(b),
                fx.as_ir_value(scale_f32),
                fx.as_ir_value(fx.Int32(sel)),
            ],
            [],
            [],
        )
    )


def _fmin_nn(a, b):
    """v_min_f32 with nnan: the select-based clamp lowers to minnum on a canonicalized input (one more ``v_max x, x``
    per value); an MFMA accumulator holds no signalling NaN, so the value is the same."""
    fm = _ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def _swiglu_oai_x4(gs, us):
    """``reference.swiglu_oai`` on four values: g clamped above (``_fmin_nn``), u both sides, t = (g * alpha) *
    (-log2 e) (one folded multiply changes the last bit), sigmoid = rcp(1 + exp2(t)), g * sig * (u + 1); the multiplies
    and adds as v_pk_mul_f32 / v_pk_add_f32 on two pairs, the two pairs interleaved in program order so a v_exp / v_rcp
    sits next to the other pair's work, not right after the packed op that produces its input (gfx950 puts a wait state
    between them)."""
    lim = fx.Float32(SWIGLU_LIMIT)
    nlim = fx.Float32(-SWIGLU_LIMIT)
    g = [
        Vec.from_elements(
            [_fmin_nn(gs[2 * q], lim), _fmin_nn(gs[2 * q + 1], lim)], fx.Float32
        )
        for q in range(2)
    ]
    u = [
        Vec.from_elements(
            [_fmed3(us[2 * q], nlim, lim), _fmed3(us[2 * q + 1], nlim, lim)], fx.Float32
        )
        for q in range(2)
    ]
    t = [g[q] * fx.Float32(SWIGLU_ALPHA) for q in range(2)]
    t = [t[q] * fx.Float32(-1.4426950408889634) for q in range(2)]
    e = [
        fx.Float32(_rocdl.exp2(_T.f32, fx.as_ir_value(fx.Float32(t[q][k]))))
        for k in range(2)
        for q in range(2)
    ]  # order e00 e10 e01 e11
    d = [
        Vec.from_elements([e[q], e[2 + q]], fx.Float32) + fx.Float32(1.0)
        for q in range(2)
    ]
    r = [
        fx.Float32(_rocdl.rcp(_T.f32, fx.as_ir_value(fx.Float32(d[q][k]))))
        for k in range(2)
        for q in range(2)
    ]
    u1 = [u[q] + fx.Float32(1.0) for q in range(2)]
    gs_ = [g[q] * Vec.from_elements([r[q], r[2 + q]], fx.Float32) for q in range(2)]
    h = [gs_[q] * u1[q] for q in range(2)]
    return [fx.Float32(h[q][k]) for q in range(2) for k in range(2)]


def _permlane32_swap(d_a, d_b):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl.permlane32_swap(
        pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False
    )
    return fx.Int32(_llvm.extractvalue(_T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(_T.i32, res, [1])
    )


def _e8m0_even_e2m1(amax):
    """Quark FP4 even scale of a nonnegative f32 amax, exponent byte >= 1: exponent(amax + half an e2m1 mantissa ulp,
    0x200000) - 2, the subtraction folded into the bias (a signed shift for tiny inputs)."""
    t = (_bits(amax) - fx.Int32(0xE00000)) >> 23
    return (t > fx.Int32(1)).select(t, fx.Int32(1))


def _e8m0_pair(amax_a, amax_b):
    """The e8m0 of two row tiles (ti 2 tp, 2 tp + 1) from their per-lane amaxes: a permlane16 swap of a with b puts rows
    (0, 1) / (2, 3) of each side by side, a permlane32 swap of that with a copy gives a in rows 0 / 2 and b in
    rows 1 / 3, one e8m0 per lane covers both tiles, and a permlane16 swap of it with a copy broadcasts each tile's
    to all rows. Per tile the same max over the same 4 lanes (l, l ^ 16, l ^ 32, l ^ 48) as two shuffle_xor steps
    (max is exact and order-free)."""
    x, y = _permlane16_swap(_bits(amax_a), _bits(amax_b))
    m = _maxf_nn(_as_f32(x), _as_f32(y))
    x, y = _permlane32_swap(_bits(m), _bits(m))
    e = _e8m0_even_e2m1(_maxf_nn(_as_f32(x), _as_f32(y)))
    return _permlane16_swap(e, e)


N_XCDS = 8
# the per-XCD claim counters: the tile map table's last entries
assert CLAIM_COUNTERS >= N_XCDS


def _cu_count() -> int:
    import torch

    return torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).multi_processor_count


def _buffer_atomic_add(rsrc, voffset_bytes, value):
    """``old = rsrc[voffset]; rsrc[voffset] += value`` (i32, per lane; lanes with an out-of-range offset do nothing and
    return 0)."""
    return fx.Int32(
        _llvm.call_intrinsic(
            _T.i32,
            "llvm.amdgcn.raw.ptr.buffer.atomic.add.i32",
            [
                fx.as_ir_value(fx.Int32(value)),
                fx.as_ir_value(rsrc),
                fx.as_ir_value(fx.Int32(voffset_bytes)),
                fx.as_ir_value(fx.Int32(0)),
                fx.as_ir_value(fx.Int32(0)),
            ],
            [],
            [],
        )
    )


def interleave_w13_scales(w13_scale_shuffled: torch.Tensor) -> torch.Tensor:
    """Aiter's e8m0_shuffle'd W13 scales [E, 2I, H/32] (gate rows, then up rows) -> the gate/up-interleaved shuffle
    GEMM1 reads (the MXFP8 chain's layout)."""
    from aiter.ops.shuffle import shuffle_scale

    e, n, k32 = w13_scale_shuffled.shape
    t = w13_scale_shuffled.reshape(e * n // 32, k32 // 8, 4, 16, 2, 2)
    raw = t.permute(0, 5, 3, 1, 4, 2).reshape(e, n, k32)  # row-major again
    return shuffle_scale(
        raw.reshape(e * n, k32), e, is_guinterleave=True, gate_up=True
    ).view(torch.uint8)
