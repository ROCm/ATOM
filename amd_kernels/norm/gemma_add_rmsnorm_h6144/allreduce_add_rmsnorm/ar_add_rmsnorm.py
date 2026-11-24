# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 all-reduce + residual add + Gemma RMSNorm, bf16 out, on registered graph inputs, rows % 4 == 0 (``ArNormComm``).

AITER's two-stage all-reduce bits (``common``), then ``add_rmsnorm.add_rmsnorm``'s (thread t holds columns
8 t .. 8 t + 7, 12 waves). A workgroup reduces one row of chunk ``me`` or pulls one reduced row from its owner, then
norms it."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..add_rmsnorm.add_rmsnorm import _block_sum
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
    MAX_ROWS,
    RED_OFF,
    SYS,
    WORLD,
    arrive,
    load_ptrs,
    publish,
    sel4,
    sum4,
    unrotate,
    wait_pullers,
    wait_reduced,
)

NW = H // 8 // 64  # 12 waves of 8 columns per lane
NT = 64 * NW
ROWB = H * 2


def compile_ar_add_rmsnorm():
    name = "allreduce_ranks4_gemma_add_rmsnorm_h6144"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_q: fx.Int32,  # rows per chunk (rows / 4); grid = rows
        i32_me: fx.Int32,
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_xs: fx.Int64,  # int64[4]: the ranks' inputs [rows, H] bf16
        arg_b0: fx.Int64,  # the ranks' flag + reduced-row buffers
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
        tx, b = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        me, q = i32_me, i32_q
        # workgroups 0 .. q - 1 (dispatched first) reduce chunk me, the others pull chunks me + 1 .. me + 3;
        # row = c q + j
        reducer = b < q
        puller = b >= q
        k = b - q
        one, zero = fx.Int32(1), fx.Int32(0)
        i = (k >= q).select(one, zero) + (k >= 2 * q).select(one, zero)
        c = reducer.select(me, (me + 1 + i) % WORLD)
        j = reducer.select(b, k - i * q)  # the row's slot in its reducer's buffer
        rw = c * q + j
        smem = fx.SharedAllocator().allocate(Shared).peek()
        part = smem.part.ptr
        bs = (arg_b0, arg_b1, arg_b2, arg_b3)
        mine = sel4(me, *bs)
        cr = buffer_ops.create_buffer_resource_from_addr(
            arg_cnt, num_records_bytes=MAX_ROWS * 4
        )
        e = (
            fx.Int32(
                buffer_ops.buffer_load(
                    cr, rw, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            + 1
        )
        xs = load_ptrs(arg_xs)
        off = tx * 4
        rowoff = fx.Int64(rw) * ROWB

        def row(base, cm, m):
            rs = buffer_ops.create_buffer_resource_from_addr(
                base, num_records_bytes=ROWB
            )
            return Vec(
                buffer_ops.buffer_load(
                    rs, off, vec_width=4, dtype=fx.Int32, mask=m, cache_modifier=cm
                )
            )

        arrive(*bs, c, me, rw, e, puller)
        wait_pullers(mine, me, rw, e, reducer)
        # wave v issues its peer loads in the order me + 1 + (t + v) % 3: each CU's requests spread over the 3 links
        # (the sum keeps the order me, me + 1, ..)
        rot = wave % (WORLD - 1)
        peer = [
            row(
                sel4((me + 1 + (t + rot) % (WORLD - 1)) % WORLD, *xs) + rowoff,
                SYS,
                reducer,
            )
            for t in range_constexpr(WORLD - 1)
        ]
        own = row(sel4(me, *xs) + rowoff, 0, reducer)
        rr = buffer_ops.create_buffer_resource_from_addr(
            arg_r + fx.Int64(rw) * fx.Int64(i32_sr * 2), num_records_bytes=ROWB
        )
        rv = bf16x8(Vec(buffer_ops.buffer_load(rr, off, vec_width=4, dtype=fx.Int32)))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=ROWB)
        wv = bf16x8(Vec(buffer_ops.buffer_load(wr, off, vec_width=4, dtype=fx.Int32)))
        reduced = sum4([own] + unrotate(rot, peer))
        slot = buffer_ops.create_buffer_resource_from_addr(
            mine + RED_OFF + fx.Int64(j) * ROWB, num_records_bytes=ROWB
        )
        buffer_ops.buffer_store(reduced, slot, off, mask=reducer, cache_modifier=SYS)
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        publish(*bs, me, rw, e, reducer)
        # no vmcnt wait after publishing: the reducer's peer flag stores are acknowledged only behind the link traffic
        x0 = fx.Int32(reduced[0])
        x1 = fx.Int32(reduced[1])
        x2 = fx.Int32(reduced[2])
        x3 = fx.Int32(reduced[3])
        if puller:
            wait_reduced(mine, rw, e)
            pulled = row(sel4(c, *bs) + RED_OFF + fx.Int64(j) * ROWB, SYS, puller)
            x0 = fx.Int32(pulled[0])
            x1 = fx.Int32(pulled[1])
            x2 = fx.Int32(pulled[2])
            x3 = fx.Int32(pulled[3])
        x = bf16x8(Vec.from_elements([x0, x1, x2, x3], fx.Int32))
        s = [x[i] + rv[i] for i in range_constexpr(8)]
        res = [pk_bf16(s[2 * i], s[2 * i + 1]) for i in range_constexpr(4)]
        acc = wave_sum_lane63(seq_sum_sq(s))
        if lane == 63:
            part[wave] = acc
        resr = buffer_ops.create_buffer_resource_from_addr(
            arg_res + rowoff, num_records_bytes=ROWB
        )
        buffer_ops.buffer_store(Vec.from_elements(res, fx.Int32), resr, off)
        fx.gpu.barrier()
        rstd = rsqrt_rn(
            _block_sum([part[i] for i in range_constexpr(NW)]),
            i32_eps.bitcast(fx.Float32),
        )
        y = [(s[i] * rstd) * (wv[i] + fx.Float32(1.0)) for i in range_constexpr(8)]
        out = [pk_bf16(y[2 * i], y[2 * i + 1]) for i in range_constexpr(4)]
        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out + rowoff, num_records_bytes=ROWB
        )
        buffer_ops.buffer_store(Vec.from_elements(out, fx.Int32), outr, off)
        buffer_ops.buffer_store(e, cr, rw, mask=tx == 0)

    @flyc.jit
    def launch(
        i32_q: fx.Int32,
        i32_me: fx.Int32,
        i32_sr: fx.Int32,
        arg_xs: fx.Int64,
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
            i32_sr,
            arg_xs,
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
        ).launch(grid=(fx.Int64(i32_q * WORLD), 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
