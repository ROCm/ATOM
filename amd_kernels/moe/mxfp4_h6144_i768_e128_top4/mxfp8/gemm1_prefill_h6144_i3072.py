# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Helpers the entry's GEMMs import (gfx950): the 16x16x128 FP8 MFMA class and its operand packing, the scale
loader ``_G2SCM``, the SwiGLU-OAI constants, ``v_med3`` / ``v_max3`` helpers and the pipeline step ``BLOCK_K``."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from ..common.loaders import G2SLoaderAsm, _asm_void, _uniform_i32

SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0

BLOCK_K = 128  # K per pipeline step (128 B per row)


def _fmed3(a, b, c):
    """v_med3_f32 (rocdl.fmed3): the exact median of finite inputs, so max(min(u, lim), -lim) in one op."""
    return fx.Float32(
        _rocdl.fmed3(_T.f32, *[fx.as_ir_value(fx.Float32(v)) for v in (a, b, c)])
    )


def _fmax3_abs(a, b, c):
    """Max of 3 absolute values by v_max3_f32 with |.| source modifiers (asm: the MLIR bridge lacks amdgcn.fmax3)."""
    return fx.Float32(
        _llvm.inline_asm(
            _T.f32,
            [fx.as_ir_value(fx.Float32(v)) for v in (a, b, c)],
            "v_max3_f32 $0, |$1|, |$2|, |$3|",
            "=v,v,v,v",
            has_side_effects=False,
        )
    )


# The MFMAs are inline asm, so LLVM's hazard recognizer misses the XDL write -> v_accvgpr_read hazard and may copy a
# final accumulator out of the AGPRs before its last MFMA wrote it (BLOCK_M 256 with VGPR spills lost the last
# K-step). The last step hands each group's accumulators to the epilogue through ``Mfma16x16x128Fp8.hold`` only after
# the next group's 16 MFMAs (the last group after these wait states: 24 cover a 16x16x128 fp8 MFMA).
_MFMA_RESULT_WAIT = "s_nop 7\n s_nop 7\n s_nop 7"


def _pack8(h0, h1):
    """Two 4 x i32 halves -> one 8 x i32 MFMA operand: K [16 klane, +16) and K [64 + 16 klane, +16)."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements(
        [a[k] for k in range(4)] + [b[k] for k in range(4)], fx.Int32
    )


def _pack6(h0, h1):
    """MXFP6 even step: c1 (16 B) + the low 8 B of c2 (``S2RLoader128B`` fp6), the lane's 32 e2m3 -> 6 VGPRs."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements([a[k] for k in range(4)] + [b[0], b[1]], fx.Int32)


def _pack6_odd(h0, h1):
    """MXFP6 odd step: the high 8 B of c2 + c3 (16 B)."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements([a[0], a[1]] + [b[k] for k in range(4)], fx.Int32)


class Mfma16x16x128Fp8:
    """fp8 x fp8 16x16x128 scaled MFMAs over an n_tiles_a x n_tiles_b quadrant, accumulators pinned in AGPRs.

    One MFMA per tile pair per 128-K step; the scale byte is (row half / gate-up, step parity). ``fp6``: e2m3 x e2m3
    (``cbsz:2 blgp:2``, 6-VGPR operands) on ``S2RLoader128B`` fragments (``fp6_paired``: the paired-step layout's
    pieces, which differ by step parity; else head + tail)."""

    def __init__(self, n_tiles_a, n_tiles_b, fp6=False, fp6_paired=False):
        assert n_tiles_a % 2 == 0
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.fp6 = fp6
        self.paired = fp6_paired
        self.res_ty = Vec.make_type(4, fx.Float32)

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _order(self):
        order = []
        j0s = list(range(0, self.n_tiles_b, 2))
        for n, i0 in enumerate(range(0, self.n_tiles_a, 2)):
            for j0 in reversed(j0s) if n % 2 else j0s:
                order += [(i0 + di, j0 + dj) for di in range(2) for dj in range(2)]
        return order

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
        """The quadrant's MFMAs with the ``interleave`` / ``late`` (from MFMA ``late_start``) thunks spread over them.

        Args:
            a: ``a[i] = [half0, half1]`` (4 x i32 each); ``b[j]`` likewise.
            sa: ``sa[i // 2]``: the A scale dword of a 32-row group (byte = row half i % 2 + 2 k2).
            sb_index: ``(j, gu) -> (dword index into sb, byte)``; the default is the gate/up-interleaved W13
                (``sb[j]``, byte ``gu``); 32-row groups use ``(j // 2, j % 2)``.
        """
        if sb_index is None:
            sb_index = lambda j, gu: (j, gu)  # noqa: E731
        thunks = list(interleave) if interleave else []
        late_thunks = list(late) if late else []
        order = self._order()
        n_mfma = len(order)
        # thunk t goes right after MFMA (t * n_mfma) // n_thunks: with more thunks than MFMAs several share a slot,
        # so none trails the last MFMA
        by_slot = {}  # type: dict[int, list]
        for t, th in enumerate(thunks):
            by_slot.setdefault((t * n_mfma) // len(thunks), []).append(th)
        n_late_slots = max(n_mfma - late_start, 1)
        for t, th in enumerate(late_thunks):
            by_slot.setdefault(
                late_start + (t * n_late_slots) // len(late_thunks), []
            ).append(th)
        # fp6's paired-step layout: the pieces differ by step parity
        pack = _pack8
        if self.fp6:
            pack = _pack6_odd if self.paired and k2 else _pack6
        for m, (i, j) in enumerate(order):
            a_op = pack(a[i][0], a[i][1])
            b_op = pack(b[j][0], b[j][1])
            acc = None if zero_acc else c[self.idx(i, j)]
            sbj, sbb = sb_index(j, gu)
            c[self.idx(i, j)] = self._mfma_agpr(
                a_op, b_op, acc, sa[i // 2], sb[sbj], i % 2, sbb, k2
            )
            for th in by_slot.get(m, ()):
                th()
        for m in range(
            n_mfma, n_mfma + max(len(thunks), len(late_thunks)) + late_start + 1
        ):
            for th in by_slot.get(m, ()):
                th()
        return c

    def hold(self, c, wait=""):
        """``c`` through a side-effecting asm that ties each AGPR quad in place (``wait``: asm run first, e.g. wait
        states), so reads of the results cannot move above this point of the MFMA stream."""
        if wait:
            _llvm.inline_asm(None, [], wait, "", has_side_effects=True)
        return [
            _llvm.inline_asm(
                self.res_ty, [fx.as_ir_value(v)], "", "=a,0", has_side_effects=True
            )
            for v in c
        ]

    def _mfma_agpr(self, a_op, b_op, acc, sa_v, sb_v, ia, jb, k2):
        # (B, A) instead of (A, B): C^T = B^T A^T, so lane L holds C[row L % 16, cols 4 (L // 16) .. + 4]
        a_op, b_op = b_op, a_op
        sa_v, sb_v = sb_v, sa_v
        ia, jb = jb, ia
        opsel = f"op_sel:[{ia},{jb},0]"
        opsel_hi = f"op_sel_hi:[{k2},{k2},0]"
        src2 = "$0" if acc is not None else "0"
        fmt = 2 if self.fp6 else 0
        # LLVM adds no VALU write -> MFMA read wait states for asm: the fp6 operands must come straight from LDS
        # loads, as ``_pack6`` arranges
        asm = (
            f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
            f"{opsel} {opsel_hi} cbsz:{fmt} blgp:{fmt}"
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


class _G2SCM(G2SLoaderAsm):
    """``G2SLoaderAsm`` with a cache-policy suffix on the LDS loads (gfx950: sc0 / nt go before ``lds``)."""

    def __init__(self, *args, cm="", **kwargs):
        super().__init__(*args, **kwargs)
        self._CM = cm

    def _emit(self, lds_dst, k_offset, step):
        voff = self._voffset(step)
        soff = _uniform_i32(k_offset)
        stride = self._step_stride
        if step == 0:
            m0 = self._lds_base_sgpr(lds_dst)
            asm = (
                "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen"
                + self._CM
                + " lds"
            )
            _asm_void([m0, voff, self.rsrc, soff], asm, "s,v,s,s", "~{scc}")
        else:
            asm = (
                f"s_add_u32 m0, {stride}, m0\nbuffer_load_dwordx4 $0, $1, $2 offen"
                + self._CM
                + " lds"
            )
            _asm_void([voff, self.rsrc, soff], asm, "v,s,s", "~{scc}")
