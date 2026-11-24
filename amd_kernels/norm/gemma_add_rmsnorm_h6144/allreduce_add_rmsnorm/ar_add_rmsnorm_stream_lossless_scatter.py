# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 all-reduce + residual add + Gemma RMSNorm, bf16 out, rows streamed on staged inputs (``ar_add_rmsnorm_stream``'s
grid and handshake), both exchanges in the lossless 12-bit code of ``lossless_bf16_row``: pullers stage their partials
coded (raw waves in a separate full-row IPC arena, ``arg_input_raw``); the reducer decodes the 3 peers, keeps its own
partial in registers and sums the 4 in rank order (``common.sum4``); the all-gather moves the reduced rows coded (raw
waves in ``arg_raw``). Every reduced and raw store drains before the publication; the host keeps the grid resident and
the reducers are dispatched first.

Two LDS banks of the 3 peer headers alternate by row: the next row's pre-decode barrier protects the previous bank's
readers, and the last reduction barrier precedes the norm's reuse of the header LDS. Either raw arena is reused only
after the peers' next arrivals acknowledged the reads."""

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
    SYS,
    WORLD,
    arrive,
    sel4,
    sum4,
    unrotate,
    wait_pullers,
)

from .ar_add_rmsnorm_stream import (
    NT, ROWB, MAX_ROWS, RED_OFF, STG_OFF, sub_rows, publish_stream, wait_reduced_stream,
)
from .common import load_ptrs
from .lossless_bf16_row import store32, decode32

def compile_ar_add_rmsnorm_stream_lossless_scatter(rpw=32):
    R = rpw
    S = sub_rows(R)
    name = f"allreduce_lossless12_scatter_stream_ranks4_gemma_add_rmsnorm_h6144_r{R}"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]
        hdr: fx.Array[fx.Int32, 576]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,  # rows per chunk ceil(rows / 4); grid = 4 xcd_grid(ceil(q / R))
        i32_me: fx.Int32,
        i32_sx: fx.Int32,  # partial row stride (elements, a multiple of 8)
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_x: fx.Int64,  # [rows, H] bf16, this rank's partial
        arg_b0: fx.Int64,  # the ranks' flag + reduced-row + staging buffers
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,  # residual [rows, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_out: fx.Int64,  # [rows, H] bf16
        arg_res: fx.Int64,  # residual out [rows, H] bf16
        arg_cnt: fx.Int64,  # int32[MAX_ROWS] call counts
        arg_raw: fx.Int64,  # int64[WORLD]: disjoint raw-wave arenas
        arg_input_raw: fx.Int64,  # 4 pointers: full-row raw partial arenas
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
            hdr = smem.hdr.ptr
            bs = (arg_b0, arg_b1, arg_b2, arg_b3)
            mine = sel4(me, *bs)
            rawptrs = load_ptrs(arg_raw)
            mine_raw = sel4(me, *rawptrs)
            input_raws = load_ptrs(arg_input_raw)
            mine_input_raw = sel4(me, *input_raws)
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

            def xrow(rw):
                return arg_x + fx.Int64(rw) * fx.Int64(i32_sx * 2)

            def count(j):
                """Sub-batch j's call count + 1 (its handshake's first row)."""
                return (
                    fx.Int32(buffer_ops.buffer_load(cr, rw0 + j * S, vec_width=1, dtype=fx.Int32, is_scalar=True))
                    + 1
                )

            def sub_ok(j):
                return (j0 + j * S < q) & (rw0 + j * S < i32_rows)

            rot = wave % (WORLD - 1)
            n_sub = fx.Int32(R // S)
            if puller:
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                        rw = rw0 + j * S + r
                        ok = (j0 + j * S + r < q) & (rw < i32_rows)
                        store32(row(xrow(rw), 0, ok), mine + STG_OFF + fx.Int64(rw) * ROWB,
                                mine_input_raw + fx.Int64(rw) * ROWB, tx, ok)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    arrive(*bs, c, me, rw0 + j * S, count(j), sub_ok(j))
            if reducer:
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    wait_pullers(mine, me, rw0 + j * S, count(j), sub_ok(j))
                    for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                        ri = j * S + r
                        rw = rw0 + ri
                        ok = (j0 + ri < q) & (rw < i32_rows)
                        loc = row(xrow(rw), 0, ok)
                        # the 3 peer headers into LDS bank ri % 2: the next row's pre-decode barrier protects its reads
                        for t in range_constexpr(3):
                            src = sel4((me + 1 + t) % WORLD, *bs) + STG_OFF + fx.Int64(rw) * ROWB
                            hres = buffer_ops.create_buffer_resource_from_addr(src, num_records_bytes=ROWB)
                            hv = Vec(buffer_ops.buffer_load(hres, 2304 + 4 * lane, vec_width=4,
                                dtype=fx.Int32, mask=ok & (wave == t) & (lane < 24), cache_modifier=SYS))
                            if (wave == t) & (lane < 24):
                                for u in range_constexpr(4):
                                    hdr[288 * (ri % 2) + 96 * t + 4 * lane + u] = fx.Int32(hv[u])
                        fx.gpu.barrier()
                        peer = [decode32(
                            sel4((me + 1 + (t + rot) % 3) % WORLD, *bs) + STG_OFF + fx.Int64(rw) * ROWB,
                            sel4((me + 1 + (t + rot) % 3) % WORLD, *input_raws) + fx.Int64(rw) * ROWB,
                            tx, ok,
                            Vec.from_elements([fx.Int32(hdr[288 * (ri % 2) + 96 * ((t + rot) % 3) + 8 * wave + u])
                                for u in range_constexpr(8)], fx.Int32)) for t in range_constexpr(3)]
                        reduced = sum4([loc] + unrotate(rot, peer))
                        store32(reduced, mine + RED_OFF + fx.Int64(j0 + ri) * ROWB,
                                mine_raw + fx.Int64(j0 + ri) * ROWB, tx, ok)
                # one publish after the batch: a publish per sub-batch put its remote flag stores before the next poll
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    publish_stream(*bs, me, rw0 + j * S, count(j), sub_ok(j))
            # every rank reads the owner's rows (local for the reducer), a puller each sub-batch once it is published
            if puller:
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    if sub_ok(j):
                        wait_reduced_stream(mine, rw0 + j * S, count(j))
            for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                    ri = j * S + r
                    rw = rw0 + ri
                    ok = (j0 + ri < q) & (rw < i32_rows)
                    coded = sel4(c, *bs) + RED_OFF + fx.Int64(j0 + ri) * ROWB
                    raw = sel4(c, *rawptrs) + fx.Int64(j0 + ri) * ROWB
                    rs = buffer_ops.create_buffer_resource_from_addr(coded, num_records_bytes=ROWB)
                    hv = Vec(buffer_ops.buffer_load(rs, 2304 + 4 * lane, vec_width=4,
                        dtype=fx.Int32, mask=ok & (wave == 0) & (lane < 24), cache_modifier=SYS))
                    if (wave == 0) & (lane < 24):
                        for u in range_constexpr(4):
                            hdr[4 * lane + u] = fx.Int32(hv[u])
                    fx.gpu.barrier()
                    hw = Vec.from_elements([fx.Int32(hdr[8 * wave + u]) for u in range_constexpr(8)], fx.Int32)
                    xv = bf16x8(decode32(coded, raw, tx, ok, hw))
                    rv = bf16x8(row(arg_r + fx.Int64(rw) * fx.Int64(i32_sr * 2), 0, ok))
                    wv = bf16x8(row(arg_w, 0, always))
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
            for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                buffer_ops.buffer_store(count(j), cr, rw0 + j * S, mask=(tx == 0) & sub_ok(j))

    @flyc.jit
    def launch_ar_add_rmsnorm_stream_lossless_scatter(
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
        arg_raw: fx.Int64,  # int64[WORLD]: disjoint raw-wave arenas
        arg_input_raw: fx.Int64,  # 4 pointers: full-row raw partial arenas
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
            arg_raw,
            arg_input_raw,
            i32_eps,
        ).launch(
            grid=(fx.Int64(xcd_grid((i32_q + (R - 1)) // R) * WORLD), 1, 1),
            block=(NT, 1, 1),
            stream=stream,
        )

    launch_ar_add_rmsnorm_stream_lossless_scatter.kernel_name = name
    launch_ar_add_rmsnorm_stream_lossless_scatter.compile_hints = PRELOAD_HINTS
    return launch_ar_add_rmsnorm_stream_lossless_scatter
