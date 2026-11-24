# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 all-reduce + residual add + Gemma RMSNorm, bf16 out, one-shot exchange for the fewest rows, eager or in graphs.

A workgroup owns one row on every rank. Each rank stages its partial row write-through in its own uncached buffer (two
copies by call parity); every wave signals the same wave of the 3 peers that its 512 columns are staged and waits for
theirs (lanes 0-2, no workgroup barrier), reads the peers' copies and sums the 4 partials in the two-stage order of the
row's owner chunk (rank ``b // ceil(rows / 4)`` first), hardware FP32 FTZ scoped to the rank sum, one bf16 rounding;
then ``add_rmsnorm.add_rmsnorm``'s norm. Each partial crosses all 3 links in one hop (twice the two-stage link bytes),
which pays only at the fewest rows. No peer input is read in place: no registration."""

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
ONESHOT_ROWS = 32  # the staging buffer's rows per parity
MAX_ROWS = 1 << 13  # the call-count array (shared with the other variants' rows)
STAGE_OFF = 0  # [2, ONESHOT_ROWS, H] bf16 staged partials
# signal word (b * NW + v) * WORLD + s: wave v of rank s staged its part of row b
WFLAG_OFF = STAGE_OFF + 2 * ONESHOT_ROWS * ROWB
WFLAG_BYTES = ONESHOT_ROWS * NW * WORLD * 4
ONESHOT_BUF_BYTES = WFLAG_OFF + (WFLAG_BYTES + 4095) // 4096 * 4096


def compile_ar_add_rmsnorm_oneshot():
    capacity, wave_flag_off = ONESHOT_ROWS, WFLAG_OFF
    name = "allreduce_oneshot_scoped_ftz_ranks4_gemma_add_rmsnorm_h6144"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_q: fx.Int32,  # ceil(rows / 4): row b's 2-stage chunk owner is b / q
        i32_me: fx.Int32,
        i32_sx: fx.Int32,  # partial row stride (elements, a multiple of 8)
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_x: fx.Int64,  # [rows, H] bf16, this rank's partial
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
        mycopy = fx.Int64(STAGE_OFF) + (fx.Int64(e & 1) * capacity + fx.Int64(rw)) * ROWB

        def load(base, cm=0):
            rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
            return Vec(buffer_ops.buffer_load(rs, off, vec_width=4, dtype=fx.Int32, cache_modifier=cm))

        local = load(arg_x + fx.Int64(rw) * fx.Int64(i32_sx * 2))
        stg = buffer_ops.create_buffer_resource_from_addr(mine + mycopy, num_records_bytes=ROWB)
        buffer_ops.buffer_store(local, stg, off, cache_modifier=SYS)
        rv = load(arg_r + fx.Int64(rw) * fx.Int64(i32_sr * 2))
        wv = load(arg_w)
        # the wave's own stores retire, then its lanes 0-2 signal peer me + 1 + lane for this wave's part and wait for
        # the same wave of that peer; the other lanes wait at the branch's join (no workgroup barrier)
        fx.rocdl.s_waitcnt(vmcnt=0)
        pl = (me + 1 + lane) % WORLD
        wflag = fx.Int64(wave_flag_off) + fx.Int64(((rw * NW + wave) * WORLD) * 4)
        if lane < WORLD - 1:
            _flag_store(sel4(pl, *bs) + wflag + fx.Int64(me * 4), e)
            _spin(mine + wflag + fx.Int64(pl * 4), e)
        peer = [load(sel4((me + 1 + t) % WORLD, *bs) + mycopy, SYS) for t in range_constexpr(WORLD - 1)]
        # packs by rank me + k -> in the order c, c + 1, c + 2, c + 3
        d = (rw // i32_q - me + WORLD) % WORLD
        allv, mode = _enter_fp32_ftz([local] + peer)
        byk = allv
        ordered = [
            Vec.from_elements(
                [
                    sel4(
                        d,
                        fx.Int32(byk[i % WORLD][n]),
                        fx.Int32(byk[(1 + i) % WORLD][n]),
                        fx.Int32(byk[(2 + i) % WORLD][n]),
                        fx.Int32(byk[(3 + i) % WORLD][n]),
                    )
                    for n in range_constexpr(4)
                ],
                fx.Int32,
            )
            for i in range_constexpr(WORLD)
        ]
        # the residual pack crosses the restore too (an inline asm needs 2+ outputs to return a struct): nothing of the
        # norm runs before the mode is restored
        reduced, rv = _restore_fp32_denorm_mode([_sum4_hardware_ftz(ordered), rv], mode)
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
        buffer_ops.buffer_store(e, cr, rw, mask=tx == 0)

    @flyc.jit
    def launch_ar_add_rmsnorm_oneshot(
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

    launch_ar_add_rmsnorm_oneshot.kernel_name = name
    launch_ar_add_rmsnorm_oneshot.compile_hints = PRELOAD_HINTS
    return launch_ar_add_rmsnorm_oneshot
