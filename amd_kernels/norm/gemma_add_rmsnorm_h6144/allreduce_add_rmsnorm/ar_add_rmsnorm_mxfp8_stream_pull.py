# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``ar_add_rmsnorm_mxfp8_stream`` in registered graphs without the staging copy (as ``ar_add_rmsnorm_stream_pull`` for
bf16): pullers signal every sub-batch at kernel start and the reducer reads the 3 peers' registered inputs in place with
system-coherent loads. A puller waits for the publication of all its rows, so a rank's input is consumed when its launch
completes. ``hoist_weight``: the Gemma weight is loaded once after the exchange."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..add_rmsnorm.add_rmsnorm_mxfp8 import (
    _block_sum,
    _codes,
    _e8m0,
    _set_fp16_ovfl,
)
from ..add_rmsnorm.common import (
    PRELOAD_HINTS,
    H,
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
from .ar_add_rmsnorm_stream import (
    MAX_ROWS,
    RED_OFF,
    publish_stream,
    sub_rows,
    wait_reduced_stream,
)

from .ar_add_rmsnorm_mxfp8_stream import _store_rowmajor, _store_tiled  # the staged kernel's epilogues

NW = 8
NT = 64 * NW
ROWB = H * 2


def compile_ar_add_rmsnorm_mxfp8_stream_pull(rpw=8, tiled_q=True, hoist_weight=False):
    R = rpw
    S = sub_rows(R)
    layout = "tiled" if tiled_q else "rowmajor"
    name = f"allreduce_stream_pull{'_hoist_weight' if hoist_weight else ''}_ranks4_gemma_add_rmsnorm_mxfp8_{layout}_h6144_r{R}"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,  # rows per chunk ceil(rows / 4); grid = 4 xcd_grid(ceil(q / R))
        i32_me: fx.Int32,
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        i32_sx: fx.Int32,  # unused (registered inputs are contiguous); the staged kernel's argument list
        arg_x: fx.Int64,  # int64[4]: the ranks' registered inputs [rows, H] bf16 (contiguous)
        arg_b0: fx.Int64,  # the ranks' flag + reduced-row + staging buffers
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,  # residual [rows, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_res: fx.Int64,  # residual out [rows, H] bf16
        arg_q: fx.Int64,  # codes e4m3, (16, 16) tiles of 16 rows
        arg_s: fx.Int64,  # scales u8, AITER layout
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        arg_cnt: fx.Int64,  # int32[MAX_ROWS] call counts
        i32_eps: fx.Int32,  # fp32 bits
    ):
        _set_fp16_ovfl()
        tx, b = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        me, q = i32_me, i32_q
        # the workgroup's rows: c q + j0 + r, r < R
        qg = (q + (R - 1)) // R
        qp = (qg + 7) // 8 * 8
        one, zero = fx.Int32(1), fx.Int32(0)
        grp = (
            (b >= qp).select(one, zero)
            + (b >= 2 * qp).select(one, zero)
            + (b >= 3 * qp).select(one, zero)
        )
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
            cr = buffer_ops.create_buffer_resource_from_addr(
                arg_cnt, num_records_bytes=MAX_ROWS * 4
            )
            has1 = tx < 256  # second chunk (columns 4096 + 8 t) inside the row
            offs = [tx * 4, tx * 4 + 2048]
            always = fx.Boolean(True)
            def row(base, cm, m):
                rs = buffer_ops.create_buffer_resource_from_addr(
                    base, num_records_bytes=ROWB
                )
                return [
                    Vec(
                        buffer_ops.buffer_load(
                            rs,
                            offs[h],
                            vec_width=4,
                            dtype=fx.Int32,
                            mask=m if h == 0 else m & has1,
                            cache_modifier=cm,
                        )
                    )
                    for h in range_constexpr(2)
                ]

            def store_row(vs, base, m, cm):
                rs = buffer_ops.create_buffer_resource_from_addr(
                    base, num_records_bytes=ROWB
                )
                for h in range_constexpr(2):
                    buffer_ops.buffer_store(
                        vs[h],
                        rs,
                        offs[h],
                        mask=m if h == 0 else m & has1,
                        cache_modifier=cm,
                    )

            def count(j):
                """Sub-batch j's call count + 1 (its handshake's first row)."""
                return (
                    fx.Int32(
                        buffer_ops.buffer_load(
                            cr, rw0 + j * S, vec_width=1, dtype=fx.Int32, is_scalar=True
                        )
                    )
                    + 1
                )

            def sub_ok(j):
                return (j0 + j * S < q) & (rw0 + j * S < i32_rows)

            rot = wave % (WORLD - 1)
            n_sub = fx.Int32(R // S)
            xs = load_ptrs(arg_x)
            if puller:
                # the registered input is complete at kernel start: signal every sub-batch now
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    arrive(*bs, c, me, rw0 + j * S, count(j), sub_ok(j))
            if reducer:
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    ej = count(j)
                    wait_pullers(mine, me, rw0 + j * S, ej, sub_ok(j))
                    for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                        ri = j * S + r
                        rw = rw0 + ri
                        ok = (j0 + ri < q) & (rw < i32_rows)
                        loc = row(sel4(me, *xs) + fx.Int64(rw) * ROWB, 0, ok)
                        peer = [row(sel4((me + 1 + (t + rot) % (WORLD - 1)) % WORLD, *xs)
                                    + fx.Int64(rw) * ROWB, SYS, ok)
                                for t in range_constexpr(WORLD - 1)]
                        reduced = [sum4([loc[h]] + unrotate(rot, [peer[t][h]
                                   for t in range_constexpr(WORLD - 1)]))
                                   for h in range_constexpr(2)]
                        store_row(reduced, mine + RED_OFF + fx.Int64(j0 + ri) * ROWB, ok, SYS)
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
            wv_once = [bf16x8(v) for v in row(arg_w, 0, always)] if hoist_weight else None
            for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                    ri = j * S + r
                    rw = rw0 + ri
                    ok = (j0 + ri < q) & (rw < i32_rows)
                    xv = row(sel4(c, *bs) + RED_OFF + fx.Int64(j0 + ri) * ROWB, SYS, ok)
                    rv = row(arg_r + fx.Int64(rw) * fx.Int64(i32_sr * 2), 0, ok)
                    wvf = wv_once if hoist_weight else [bf16x8(v) for v in row(arg_w, 0, always)]
                    vals = [[x + r for x, r in zip(bf16x8(xv[h]), bf16x8(rv[h]))]
                            for h in range_constexpr(2)]
                    acc = wave_sum_lane63(seq_sum_sq(vals[0] + vals[1]))
                    if lane == 63:
                        part[wave] = acc
                    store_row([Vec.from_elements([pk_bf16(vals[h][2*n], vals[h][2*n+1])
                               for n in range_constexpr(4)], fx.Int32)
                               for h in range_constexpr(2)], arg_res + fx.Int64(rw) * ROWB, ok, 0)
                    fx.gpu.barrier()
                    qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=i64_q_bytes)
                    sr = buffer_ops.create_buffer_resource_from_addr(arg_s, num_records_bytes=i64_s_bytes)
                    rstd = rsqrt_rn(
                        _block_sum([part[n] for n in range_constexpr(NW)]),
                        i32_eps.bitcast(fx.Float32),
                    )
                    codes, sbs = [], []
                    for h in range_constexpr(2):
                        y = [
                            (vals[h][n] * rstd) * (wvf[h][n] + fx.Float32(1.0))
                            for n in range_constexpr(8)
                        ]
                        sb, rescale = _e8m0(y)
                        codes.append(_codes(y, rescale))
                        sbs.append(fx.Int32(sb.to(fx.Int32)))
                    (_store_tiled if tiled_q else _store_rowmajor)(
                        tx, rw, codes, sbs, has1, ok, qr, sr
                    )
                    fx.gpu.barrier()
            for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                buffer_ops.buffer_store(
                    count(j), cr, rw0 + j * S, mask=(tx == 0) & sub_ok(j)
                )

    @flyc.jit
    def launch_ar_add_rmsnorm_mxfp8_stream_pull(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,
        i32_me: fx.Int32,
        i32_sr: fx.Int32,
        i32_sx: fx.Int32,
        arg_x: fx.Int64,
        arg_b0: fx.Int64,
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_res: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        arg_cnt: fx.Int64,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_rows,
            i32_q,
            i32_me,
            i32_sr,
            i32_sx,
            arg_x,
            arg_b0,
            arg_b1,
            arg_b2,
            arg_b3,
            arg_r,
            arg_w,
            arg_res,
            arg_q,
            arg_s,
            i64_q_bytes,
            i64_s_bytes,
            arg_cnt,
            i32_eps,
        ).launch(
            grid=(fx.Int64(xcd_grid((i32_q + (R - 1)) // R) * WORLD), 1, 1),
            block=(NT, 1, 1),
            stream=stream,
        )

    launch_ar_add_rmsnorm_mxfp8_stream_pull.kernel_name = name
    launch_ar_add_rmsnorm_mxfp8_stream_pull.compile_hints = PRELOAD_HINTS
    return launch_ar_add_rmsnorm_mxfp8_stream_pull
