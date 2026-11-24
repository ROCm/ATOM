# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MoE combine + TP4 all-reduce + residual add + Gemma RMSNorm + MXFP8 quant (tiled codes), the all-gather as the
lossless 12-bit code (``lossless_bf16``), one row per workgroup, eager or in graphs.

``slots4_ar_add_rmsnorm_mxfp8`` with per-wave hand-offs and the software rank flush; only the reduced rows the pullers
read are coded. Raw (exception) waves use RED slot j + 128 (q <= 64 at rows <= 256). The reducer writes coded and raw
slots after all three pullers arrived, drains its stores and publishes; a puller finishes every read before its next
call can arrive, so no slot is reused early. Every workgroup of the <= 256-row grid is resident, reducers first.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..add_rmsnorm.add_rmsnorm_mxfp8 import (
    NT as CODE_NT,
)
from ..add_rmsnorm.add_rmsnorm_mxfp8 import (
    _block_sum,
    _code_off,
    _codes,
    _e8m0,
    _scale_off,
    _set_fp16_ovfl,
)
from ..add_rmsnorm.common import (
    PRELOAD_HINTS,
    bf16x8,
    pk_bf16,
    rsqrt_rn,
    seq_sum_sq,
    swap_xor1,
    wave_sum_lane63,
    xcd_grid,
    xcd_row,
)
from .common import (
    MAX_ROWS,
    RED_OFF,
    SYS,
    WORLD,
    sel4,
    sum4,
    unrotate,
)
from .lossless_bf16 import enc_store, fetch, finish
from .slots4_ar_add_rmsnorm_mxfp8 import (
    NT, NW, ROWB, SLOTS, STG_OFF, arrive_wave, local_sum, publish_wave, wait_pullers_wave, wait_reduced_wave,
)


def compile_slots4_ar_add_rmsnorm_mxfp8_lossless():
    """The one variant: one row per 8-wave workgroup."""
    R = 1
    name = "sum_slots4_shared_allreduce_lossless12_ranks4_gemma_add_rmsnorm_mxfp8_tiled_h6144"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW * R]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,  # rows per chunk ceil(rows / 4); grid = 4 xcd_grid(ceil(q / R))
        i32_me: fx.Int32,
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_slots: fx.Int64,  # [rows, 4, H] bf16, this rank's weighted routed slots
        arg_sh: fx.Int64,  # [rows, H] bf16, this rank's shared-expert output
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
            e = (
                fx.Int32(
                    buffer_ops.buffer_load(
                        cr, rw0, vec_width=1, dtype=fx.Int32, is_scalar=True
                    )
                )
                + 1
            )
            has1 = tx < 256  # second chunk (columns 4096 + 8 t) inside the row
            offs = [tx * 4, tx * 4 + 2048]
            always = fx.Boolean(True)
            rws = [rw0 + r for r in range_constexpr(R)]
            oks = [always] + [
                (j0 + r < q) & (rw0 + r < i32_rows) for r in range_constexpr(1, R)
            ]
            rowoffs = [fx.Int64(rws[r]) * ROWB for r in range_constexpr(R)]

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

            def local_row(r):
                sbase = arg_slots + fx.Int64(rws[r]) * (SLOTS * ROWB)
                sl = [row(sbase + k * ROWB, 0, oks[r]) for k in range_constexpr(SLOTS)]
                sh = row(arg_sh + rowoffs[r], 0, oks[r])
                return [
                    local_sum([sl[k][h] for k in range_constexpr(SLOTS)], sh[h])
                    for h in range_constexpr(2)
                ]

            local = [local_row(r) for r in range_constexpr(R)]
            store_row(local[0], mine + STG_OFF + rowoffs[0], puller, SYS)
            fx.rocdl.s_waitcnt(vmcnt=0)
            arrive_wave(*bs, c, me, rw0, wave, e, puller)
            wait_pullers_wave(mine, me, rw0, wave, e, reducer)
            # wave v loads the peers in the order me + 1 + (t + v) % 3: each CU's requests spread over the three links
            # (the sum keeps the rank order)
            rot = wave % (WORLD - 1)
            peer = [[
                row(
                    sel4((me + 1 + (t + rot) % (WORLD - 1)) % WORLD, *bs)
                    + STG_OFF + rowoffs[0], SYS, reducer,
                )
                for t in range_constexpr(WORLD - 1)
            ]]
            rv = [
                row(arg_r + fx.Int64(rws[r]) * fx.Int64(i32_sr * 2), 0, oks[r])
                for r in range_constexpr(R)
            ]
            wv = row(arg_w, 0, always)
            reduced = [
                [
                    sum4(
                        [local[r][h]]
                        + unrotate(
                            rot, [peer[r][t][h] for t in range_constexpr(WORLD - 1)]
                        )
                    )
                    for h in range_constexpr(2)
                ]
                for r in range_constexpr(R)
            ]
            if reducer:
                for h in range_constexpr(2):
                    mask = reducer if h == 0 else reducer & has1
                    enc_store(
                        reduced[0][h], mine + RED_OFF + fx.Int64(j0) * ROWB,
                        mine + RED_OFF + fx.Int64(j0 + 128) * ROWB, tx + 512 * h, mask,
                    )
            fx.rocdl.s_waitcnt(vmcnt=0)
            publish_wave(*bs, me, rw0, wave, e, reducer)
            # no vmcnt wait after publishing: the peers acknowledge its flag stores only behind the bulk traffic

            xw = [fx.Int32(reduced[0][h][i]) for h in range_constexpr(2) for i in range_constexpr(4)]
            if puller:
                wait_reduced_wave(mine, rw0, wave, e)
                base = sel4(c, *bs) + RED_OFF + fx.Int64(j0) * ROWB
                gdata = [
                    fetch(base, tx + 512 * h, always if h == 0 else has1)
                    for h in range_constexpr(2)
                ]
                xw = []
                for h in range_constexpr(2):
                    got = finish(
                        gdata[h], base + 128 * ROWB, tx + 512 * h,
                        always if h == 0 else has1,
                    )
                    xw += [fx.Int32(got[i]) for i in range_constexpr(4)]
            wvf = [bf16x8(wv[h]) for h in range_constexpr(2)]
            s = [
                [
                    [
                        xv + rf
                        for xv, rf in zip(
                            bf16x8(
                                Vec.from_elements(
                                    xw[(r * 2 + h) * 4 : (r * 2 + h) * 4 + 4], fx.Int32
                                )
                            ),
                            bf16x8(rv[r][h]),
                        )
                    ]
                    for h in range_constexpr(2)
                ]
                for r in range_constexpr(R)
            ]
            acc = [
                wave_sum_lane63(seq_sum_sq(s[r][0] + s[r][1])) for r in range_constexpr(R)
            ]
            for r in range_constexpr(R):
                if lane == 63:
                    part[r * NW + wave] = acc[r]
            for r in range_constexpr(R):
                store_row(
                    [
                        Vec.from_elements(
                            [
                                pk_bf16(s[r][h][2 * n], s[r][h][2 * n + 1])
                                for n in range_constexpr(4)
                            ],
                            fx.Int32,
                        )
                        for h in range_constexpr(2)
                    ],
                    arg_res + rowoffs[r],
                    oks[r],
                    0,
                )
            fx.gpu.barrier()
            odd = (lane & 1) == 1
            p = tx // 2 + odd.select(fx.Int32(256), fx.Int32(0))  # 16-column piece
            qr = buffer_ops.create_buffer_resource_from_addr(
                arg_q, num_records_bytes=i64_q_bytes
            )
            q4 = lane % 4
            g = tx // 4 + (q4 == 1).select(fx.Int32(128), fx.Int32(0))
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_s, num_records_bytes=i64_s_bytes
            )
            for r in range_constexpr(R):
                rstd = rsqrt_rn(
                    _block_sum([part[r * NW + n] for n in range_constexpr(NW)]),
                    i32_eps.bitcast(fx.Float32),
                )
                codes, sbs = [], []
                for h in range_constexpr(2):
                    y = [
                        (s[r][h][n] * rstd) * (wvf[h][n] + fx.Float32(1.0))
                        for n in range_constexpr(8)
                    ]
                    sb, rescale = _e8m0(y)
                    codes.append(_codes(y, rescale))
                    sbs.append(fx.Int32(sb.to(fx.Int32)))
                # lane pairs: the even lane stores chunk 0's 16 columns, the odd chunk 1's
                recv = swap_xor1(
                    [odd.select(codes[0][n], codes[1][n]) for n in range_constexpr(2)]
                )
                piece = Vec.from_elements(
                    [
                        odd.select(recv[0], codes[0][0]),
                        odd.select(recv[1], codes[0][1]),
                        odd.select(codes[1][0], recv[0]),
                        odd.select(codes[1][1], recv[1]),
                    ],
                    fx.Int32,
                )
                buffer_ops.buffer_store(
                    piece,
                    qr,
                    _code_off(rws[r], p) // 4,
                    mask=(((lane & 1) == 0) | has1) & oks[r],
                    cache_modifier=CODE_NT,
                )
                # one lane per 32-column group stores its scale byte
                sb = (q4 == 1).select(sbs[1], sbs[0])
                buffer_ops.buffer_store(
                    sb.to(fx.Int8),
                    sr,
                    _scale_off(rws[r], g),
                    mask=((q4 == 0) | ((q4 == 1) & has1)) & oks[r],
                    offset_is_bytes=True,
                )
            buffer_ops.buffer_store(e, cr, rw0, mask=tx == 0)

    @flyc.jit
    def launch_slots4_ar_add_rmsnorm_mxfp8_lossless(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,
        i32_me: fx.Int32,
        i32_sr: fx.Int32,
        arg_slots: fx.Int64,
        arg_sh: fx.Int64,
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
            arg_slots,
            arg_sh,
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

    launch_slots4_ar_add_rmsnorm_mxfp8_lossless.kernel_name = name
    launch_slots4_ar_add_rmsnorm_mxfp8_lossless.compile_hints = PRELOAD_HINTS
    return launch_slots4_ar_add_rmsnorm_mxfp8_lossless
