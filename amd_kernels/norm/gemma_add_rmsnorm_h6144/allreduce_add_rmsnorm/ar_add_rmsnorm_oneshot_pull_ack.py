# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 all-reduce + residual add + Gemma RMSNorm, bf16 out, one-shot on registered graph inputs with read
acknowledgements.

Each wave announces its input ready, loads the 4 registered inputs in the row owner's order and acknowledges the reads
before the norm; it waits for all readers before it ends, so the producer can reuse its input. Both phases share one
generation word: 2e = ready, 2e + 1 = reads done. Hardware FP32 FTZ around the rank sum only."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..add_rmsnorm.add_rmsnorm import NW, _block_sum
from ..add_rmsnorm.common import (
    PRELOAD_HINTS,
    H,
    bf16x8,
    pk_bf16,
    rsqrt_rn,
    seq_sum_sq,
    wave_sum_lane63,
)
from .common import load_ptrs
from .common import (
    SYS,
    WORLD,
    _enter_fp32_ftz,
    _flag_store,
    _restore_fp32_denorm_mode,
    _spin,
    _sum4_hardware_ftz,
    sel4,
)

NT = 64 * NW  # 768 threads
ROWB = H * 2
MAX_ROWS = 1 << 13
# words 4..7 of the flags128 one-shot's 128-byte flag lines (the MXFP8 pull-ack uses 8..11), own call counts
from .ar_add_rmsnorm_oneshot_flags128_owner_order import WFLAG_OFF


def compile_ar_add_rmsnorm_oneshot_pull_ack():
    wave_flag_off = WFLAG_OFF
    name = "allreduce_oneshot_pull_ack_scoped_ftz_ranks4_gemma_add_rmsnorm_h6144"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_q: fx.Int32,  # ceil(rows / 4): row b's 2-stage chunk owner is b / q
        i32_me: fx.Int32,
        i32_sx: fx.Int32,  # partial row stride (elements, a multiple of 8)
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_x: fx.Int64,  # int64[4] registered contiguous input pointers
        arg_b0: fx.Int64,  # the ranks' buffers
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,  # residual [rows, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_out: fx.Int64,  # [rows, H] bf16
        arg_res: fx.Int64,  # residual out [rows, H] bf16
        arg_cnt: fx.Int64,  # int32[MAX_ROWS] call counts
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, rw = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        me = i32_me
        smem = fx.SharedAllocator().allocate(Shared).peek()
        part = smem.part.ptr
        bs = (arg_b0, arg_b1, arg_b2, arg_b3)
        mine = sel4(me, *bs)
        cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=MAX_ROWS * 4)
        e = fx.Int32(buffer_ops.buffer_load(cr, rw, vec_width=1, dtype=fx.Int32, is_scalar=True)) + 1
        off = tx * 4
        rowoff = fx.Int64(rw) * ROWB

        def load(base, cm=0):
            rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
            return Vec(buffer_ops.buffer_load(rs, off, vec_width=4, dtype=fx.Int32, cache_modifier=cm))

        xs = load_ptrs(arg_x)
        rv = load(arg_r + fx.Int64(rw) * fx.Int64(i32_sr * 2))
        wv = load(arg_w)
        pl = (me + 1 + lane) % WORLD
        wflag = fx.Int64(wave_flag_off) + fx.Int64((rw * NW + wave) * 128 + 16)
        if lane < WORLD - 1:
            _flag_store(sel4(pl, *bs) + wflag + fx.Int64(me * 4), e * 2)
            _spin(mine + wflag + fx.Int64(pl * 4), e * 2)
        owner = rw // i32_q
        ordered = [load(sel4((owner + t) % WORLD, *xs) + rowoff, SYS) for t in range_constexpr(WORLD)]
        ordered, mode = _enter_fp32_ftz(ordered)
        reduced, rv = _restore_fp32_denorm_mode([_sum4_hardware_ftz(ordered), rv], mode)
        # the rank sum consumed every source pack: acknowledge the reads now, their delivery overlaps the norm
        if lane < WORLD - 1:
            _flag_store(sel4(pl, *bs) + wflag + fx.Int64(me * 4), e * 2 + 1)
        xv = bf16x8(reduced)
        rvf = bf16x8(rv)
        wvf = bf16x8(wv)
        s = [xv[n] + rvf[n] for n in range_constexpr(8)]
        resr = buffer_ops.create_buffer_resource_from_addr(arg_res + rowoff, num_records_bytes=ROWB)
        res = [pk_bf16(s[2 * n], s[2 * n + 1]) for n in range_constexpr(4)]
        buffer_ops.buffer_store(Vec.from_elements(res, fx.Int32), resr, off)
        acc = wave_sum_lane63(seq_sum_sq(s))
        if lane == 63:
            part[wave] = acc
        fx.gpu.barrier()
        rstd = rsqrt_rn(_block_sum([part[n] for n in range_constexpr(NW)]), i32_eps.bitcast(fx.Float32))
        y = [(s[n] * rstd) * (wvf[n] + fx.Float32(1.0)) for n in range_constexpr(8)]
        out = [pk_bf16(y[2 * n], y[2 * n + 1]) for n in range_constexpr(4)]
        outr = buffer_ops.create_buffer_resource_from_addr(arg_out + rowoff, num_records_bytes=ROWB)
        buffer_ops.buffer_store(Vec.from_elements(out, fx.Int32), outr, off)
        # the producer may reuse its input only after all readers acknowledged
        if lane < WORLD - 1:
            _spin(mine + wflag + fx.Int64(pl * 4), e * 2 + 1)
        buffer_ops.buffer_store(e, cr, rw, mask=tx == 0)

    @flyc.jit
    def launch_ar_add_rmsnorm_oneshot_pull_ack(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,
        i32_me: fx.Int32,
        i32_sx: fx.Int32,
        i32_sr: fx.Int32,
        arg_x: fx.Int64,
        arg_b0: fx.Int64,
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_out: fx.Int64,
        arg_res: fx.Int64,
        arg_cnt: fx.Int64,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_q,
            i32_me,
            i32_sx,
            i32_sr,
            arg_x,
            arg_b0,
            arg_b1,
            arg_b2,
            arg_b3,
            arg_r,
            arg_w,
            arg_out,
            arg_res,
            arg_cnt,
            i32_eps,
        ).launch(grid=(fx.Int64(i32_rows), 1, 1), block=(NT, 1, 1), stream=stream)

    launch_ar_add_rmsnorm_oneshot_pull_ack.kernel_name = name
    launch_ar_add_rmsnorm_oneshot_pull_ack.compile_hints = PRELOAD_HINTS
    return launch_ar_add_rmsnorm_oneshot_pull_ack
