# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``ar_add_rmsnorm_stream_pull`` with one handshake per workgroup batch (registered inputs).

The registered inputs are complete at kernel start, so a puller signals its batch's first row once; the reducer reduces
all rows, drains their stores and publishes once. Pullers wait for that publication before they read rows or end."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..add_rmsnorm.add_rmsnorm import NW, _block_sum
from ..add_rmsnorm.common import (
    PRELOAD_HINTS,
    bf16x8,
    pk_bf16,
    rsqrt_rn,
    seq_sum_sq,
    wave_sum_lane63,
    xcd_grid,
    xcd_row,
)
from .common import (
    load_ptrs,
    SYS,
    WORLD,
    arrive,
    sel4,
    sum4,
    unrotate,
    wait_pullers,
)

from .ar_add_rmsnorm_stream import (  # the staged kernel's buffers, flags and sub-batches
    MAX_ROWS, NT, RED_OFF, ROWB, publish_stream, sub_rows, wait_reduced_stream,
)


def compile_ar_add_rmsnorm_batch_pull(rpw=8, hoist_weight=False):
    """``hoist_weight``: load the Gemma weight once after the exchange and reuse it for every norm row."""
    R = rpw
    S = sub_rows(R)
    name = f"allreduce_batch_pull{'_hoist_weight' if hoist_weight else ''}_ranks4_gemma_add_rmsnorm_h6144_r{R}"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,  # rows per chunk ceil(rows / 4); grid = 4 xcd_grid(ceil(q / R))
        i32_me: fx.Int32,
        i32_sx: fx.Int32,  # unused (registered inputs are contiguous); the staged kernel's argument list
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_x: fx.Int64,  # int64[4]: the ranks' registered inputs [rows, H] bf16 (contiguous)
        arg_b0: fx.Int64,  # the ranks' flag + reduced-row + staging buffers
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
        # the workgroup's rows: c q + j0 + r, r < R
        qg = (q + (R - 1)) // R
        qp = (qg + 7) // 8 * 8
        one, zero = fx.Int32(1), fx.Int32(0)
        grp = (b >= qp).select(one, zero) + (b >= 2 * qp).select(one, zero) + (b >= 3 * qp).select(one, zero)
        jg, inq = xcd_row(b - grp * qp, qg)
        c = (me + grp) % WORLD
        j0 = jg * R  # the first row's slot in its reducer's buffer
        rw0 = c * q + j0
        live = inq & (rw0 < i32_rows)
        reducer = (grp == 0) & live
        puller = (grp != 0) & live
        if live:
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            bs = (arg_b0, arg_b1, arg_b2, arg_b3)
            mine = sel4(me, *bs)
            cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=MAX_ROWS * 4)
            off = tx * 4
            always = fx.Boolean(True)

            def row(base, cm, m):
                rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
                return Vec(
                    buffer_ops.buffer_load(rs, off, vec_width=4, dtype=fx.Int32, mask=m, cache_modifier=cm)
                )

            def store_row(v, base, m, cm):
                rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
                buffer_ops.buffer_store(v, rs, off, mask=m, cache_modifier=cm)

            xs = load_ptrs(arg_x)

            def xrow(rw, p=None):
                return sel4(me if p is None else p, *xs) + fx.Int64(rw) * ROWB

            def count():
                """The batch first row's paired generation, advanced once on all ranks."""
                return (
                    fx.Int32(buffer_ops.buffer_load(cr, rw0, vec_width=1, dtype=fx.Int32, is_scalar=True))
                    + 1
                )

            def batch_ok():
                return (j0 < q) & (rw0 < i32_rows)

            rot = wave % (WORLD - 1)
            n_sub = fx.Int32(R // S)
            if puller:
                arrive(*bs, c, me, rw0, count(), batch_ok())
            if reducer:
                wait_pullers(mine, me, rw0, count(), batch_ok())
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                        ri = j * S + r
                        rw = rw0 + ri
                        ok = (j0 + ri < q) & (rw < i32_rows)
                        loc = row(xrow(rw), 0, ok)
                        peer = [
                            row(
                                xrow(rw, (me + 1 + (t + rot) % (WORLD - 1)) % WORLD),
                                SYS,
                                ok,
                            )
                            for t in range_constexpr(WORLD - 1)
                        ]
                        reduced = sum4([loc] + unrotate(rot, peer))
                        store_row(reduced, mine + RED_OFF + fx.Int64(j0 + ri) * ROWB, ok, SYS)
                # one publish after the batch: a publish per sub-batch put its remote flag stores before the next poll
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                publish_stream(*bs, me, rw0, count(), batch_ok())
            # the publication acknowledges that every registered input was consumed: the producer may then reuse it
            if puller:
                wait_reduced_stream(mine, rw0, count())
            wv_once = bf16x8(row(arg_w, 0, always)) if hoist_weight else None
            for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                    ri = j * S + r
                    rw = rw0 + ri
                    ok = (j0 + ri < q) & (rw < i32_rows)
                    xv = bf16x8(row(sel4(c, *bs) + RED_OFF + fx.Int64(j0 + ri) * ROWB, SYS, ok))
                    rv = bf16x8(row(arg_r + fx.Int64(rw) * fx.Int64(i32_sr * 2), 0, ok))
                    wv = wv_once if hoist_weight else bf16x8(row(arg_w, 0, always))
                    s = [xv[n] + rv[n] for n in range_constexpr(8)]
                    acc = wave_sum_lane63(seq_sum_sq(s))
                    if lane == 63:
                        part[wave] = acc
                    res = [pk_bf16(s[2 * n], s[2 * n + 1]) for n in range_constexpr(4)]
                    store_row(Vec.from_elements(res, fx.Int32), arg_res + fx.Int64(rw) * ROWB, ok, 0)
                    fx.gpu.barrier()
                    rstd = rsqrt_rn(_block_sum([part[n] for n in range_constexpr(NW)]), i32_eps.bitcast(fx.Float32))
                    y = [(s[n] * rstd) * (wv[n] + fx.Float32(1.0)) for n in range_constexpr(8)]
                    out = [pk_bf16(y[2 * n], y[2 * n + 1]) for n in range_constexpr(4)]
                    store_row(Vec.from_elements(out, fx.Int32), arg_out + fx.Int64(rw) * ROWB, ok, 0)
                    fx.gpu.barrier()
            # advance only the batch's first-row count (its one flag); the other counts stay paired with their flag
            # words, so the sub-batch kernels interleave on the same communicator
            buffer_ops.buffer_store(count(), cr, rw0, mask=(tx == 0) & batch_ok())

    @flyc.jit
    def launch_ar_add_rmsnorm_batch_pull(
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
            i32_rows,
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
        ).launch(
            grid=(fx.Int64(xcd_grid((i32_q + (R - 1)) // R) * WORLD), 1, 1),
            block=(NT, 1, 1),
            stream=stream,
        )

    launch_ar_add_rmsnorm_batch_pull.kernel_name = name
    launch_ar_add_rmsnorm_batch_pull.compile_hints = PRELOAD_HINTS
    return launch_ar_add_rmsnorm_batch_pull
