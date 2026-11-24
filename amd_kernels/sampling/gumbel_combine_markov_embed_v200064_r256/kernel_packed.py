# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The tail of one DSpark Markov draft step, FlyDSL (gfx950): Gumbel combine + draft-id store + the next step's
``markov_w1`` row in the packed [m16, k32, k8, row16, k8] layout, one launch (``reference.py`` is the contract).

One wave per owned row. Lane l holds blocks l, l + 64, l + 128 and (l < 4) l + 192 of the 196 per-block partials;
three DPP wave reductions (first NaN block: min; maximum of the non-NaN values: max; first block equal to it: min)
give the winner in lane 63, read back with ``readlane``; then lane 0 loads its vocab id and stores it into
``draft_ids[row, step]`` and ``prev[local_rows[row]]``, and lane l < 32 copies eight bf16 features of the token's
``markov_w1`` row into the input layout of gemm/bf16_n200064_k256, using the owned-row number rather than the padded
batch row for its packed destination.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .reference import NB, RANK, V

BIG = 1 << 20
NEG_INF_BITS = -8388608  # 0xFF800000
ROW_B = RANK * 2  # bytes of a markov_w1 row
NK = (NB + 63) // 64  # partials per lane


def _dpp_keep(v, ctrl, row_mask=0xF):
    """v_mov_b32_dpp of an i32 with bound_ctrl off: lanes whose source is outside the row, and rows ``row_mask``
    leaves out, keep their own value (so min / max steps are idempotent there)."""
    x = fx.as_ir_value(fx.Int32(v))
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, x, x, ctrl, row_mask, 0xF, False))


def _imin(a, b):
    return (a < b).select(a, b)


def _fmax_bits(a, b):
    """maxnum of two fp32 given and returned as i32 bits (no NaN reaches it)."""
    fa, fb = fx.Int32(a).bitcast(fx.Float32), fx.Int32(b).bitcast(fx.Float32)
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(fa), fx.as_ir_value(fb)).result).bitcast(fx.Int32)


def _wave_reduce63(v, op):
    """op over the 64 lanes, read from lane 63 (row_shr 1/2/4/8, row_bcast 15 into rows 1 and 3, row_bcast 31 into
    rows 2 and 3)."""
    for sh in (1, 2, 4, 8):
        v = op(v, _dpp_keep(v, 0x110 + sh))
    v = op(v, _dpp_keep(v, 0x142, row_mask=0xA))
    v = op(v, _dpp_keep(v, 0x143, row_mask=0xC))
    return fx.Int32(fx.rocdl.readlane(T.i32, fx.as_ir_value(v), fx.as_ir_value(fx.Int32(63))))


def compile_gumbel_combine_markov_embed_packed():
    name = "gumbel_combine_markov_embed_packed_v200064_r256"

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        arg_max: fx.Int64,  # fp32 [n, NB]
        arg_argmax: fx.Int64,  # int64 [n, NB]
        arg_rows: fx.Int64,  # int64 [n]: each owned row's row in the padded batch
        arg_w1: fx.Int64,  # bf16 [V, RANK]
        arg_ids: fx.Int64,  # int64 [*, i32_ids_stride]: draft_ids[row, step]
        arg_prev: fx.Int64,  # int64 [R]
        arg_embed: fx.Int64,  # bf16 [R, RANK]
        i32_step: fx.Int32,
        i32_ids_stride: fx.Int32,
    ):
        lane, row = fx.thread_idx.x, fx.block_idx.x
        mr = buffer_ops.create_buffer_resource_from_addr(arg_max + fx.Int64(row) * (NB * 4), num_records_bytes=NB * 4)
        vb, blk, ok = [], [], []
        for k in range_constexpr(NK):
            b = lane + 64 * k
            raw = fx.Int32(buffer_ops.buffer_load(mr, b, vec_width=1, dtype=fx.Int32))
            inb = b < NB
            vb.append(inb.select(raw, fx.Int32(NEG_INF_BITS)))
            blk.append(b)
            ok.append(inb)
        isn = [((x & 0x7FFFFFFF) > 0x7F800000) for x in vb]
        nan_i = fx.Int32(BIG)
        mx = fx.Int32(NEG_INF_BITS)
        for k in range_constexpr(NK):
            nan_i = _imin(nan_i, isn[k].select(blk[k], fx.Int32(BIG)))
            mx = _fmax_bits(mx, isn[k].select(fx.Int32(NEG_INF_BITS), vb[k]))
        nan_w = _wave_reduce63(nan_i, _imin)
        mx_w = _wave_reduce63(mx, _fmax_bits).bitcast(fx.Float32)
        cand = fx.Int32(BIG)
        for k in range_constexpr(NK):
            eq = vb[k].bitcast(fx.Float32) == mx_w
            pick = isn[k].select(fx.Int32(BIG), ok[k].select(eq.select(blk[k], fx.Int32(BIG)), fx.Int32(BIG)))
            cand = _imin(cand, pick)
        cand_w = _wave_reduce63(cand, _imin)
        win = (nan_w < BIG).select(nan_w, cand_w)
        ar = buffer_ops.create_buffer_resource_from_addr(
            arg_argmax + fx.Int64(row) * (NB * 8), num_records_bytes=NB * 8
        )
        tok = Vec(buffer_ops.buffer_load(ar, win * 2, vec_width=2, dtype=fx.Int32))
        rr = buffer_ops.create_buffer_resource_from_addr(arg_rows + fx.Int64(row) * 8, num_records_bytes=8)
        lr = fx.Int32(buffer_ops.buffer_load(rr, 0, vec_width=1, dtype=fx.Int32))
        tok_lo = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(fx.Int32(tok[0]))))
        if lane == 0:
            idr = buffer_ops.create_buffer_resource_from_addr(
                arg_ids + (fx.Int64(row) * fx.Int64(i32_ids_stride) + fx.Int64(i32_step)) * 8, num_records_bytes=8
            )
            buffer_ops.buffer_store(tok, idr, 0)
            pvr = buffer_ops.create_buffer_resource_from_addr(arg_prev + fx.Int64(lr) * 8, num_records_bytes=8)
            buffer_ops.buffer_store(tok, pvr, 0)
        if lane < ROW_B // 16:
            wr = buffer_ops.create_buffer_resource_from_addr(arg_w1 + fx.Int64(tok_lo) * ROW_B, num_records_bytes=ROW_B)
            er = buffer_ops.create_buffer_resource_from_addr(arg_embed, num_records_bytes=(fx.Int64(row // 16) + 1) * 8192)
            w = Vec(buffer_ops.buffer_load(wr, lane * 4, vec_width=4, dtype=fx.Int32))
            packed_off = (row // 16) * 2048 + (lane // 4) * 256 + (lane % 4) * 64 + (row % 16) * 4
            buffer_ops.buffer_store(w, er, packed_off)

    @flyc.jit
    def launch_gumbel_combine_markov_embed_packed(
        arg_max: fx.Int64,
        arg_argmax: fx.Int64,
        arg_rows: fx.Int64,
        arg_w1: fx.Int64,
        arg_ids: fx.Int64,
        arg_prev: fx.Int64,
        arg_embed: fx.Int64,
        i32_n: fx.Int32,
        i32_step: fx.Int32,
        i32_ids_stride: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_max, arg_argmax, arg_rows, arg_w1, arg_ids, arg_prev, arg_embed, i32_step, i32_ids_stride).launch(
            grid=(fx.Int64(i32_n), 1, 1), block=(64, 1, 1), stream=stream
        )

    launch_gumbel_combine_markov_embed_packed.kernel_name = name
    return launch_gumbel_combine_markov_embed_packed
