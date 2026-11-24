# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 all-reduce + residual add + Gemma RMSNorm + MXFP8 quant (tiled codes), register two-stage, eager or in graphs
(staged inputs, no registration): one bf16 partial per rank, fixed rank order, ``add_rmsnorm_mxfp8``'s tiled norm and
quant, wave or workgroup hand-offs on ``ar_add_rmsnorm_register``'s flags."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
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
    H,
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
    SYS, WORLD, _enter_fp32_ftz, _restore_fp32_denorm_mode,
    _sum4_hardware_ftz, sel4, sum4, unrotate,
)
from .ar_add_rmsnorm_register import (
    MAX_ROWS, RED_OFF, STG_OFF,
    register_arrive as arrive_wave,
    register_wait_pullers as wait_pullers_wave,
    register_publish as publish_wave,
    register_wait_reduced as wait_reduced_wave,
)

NW = 8
NT = 64 * NW
ROWB = H * 2


def compile_ar_add_rmsnorm_mxfp8_register(wave_handoffs=False, scoped_ftz=False):
    """One row per workgroup; ``scoped_ftz``: the rank sum flushes denormals with the hardware FP32 mode instead of
    software masks (same bits)."""
    R = 1
    name = (
        "allreduce_register_"
        + ("scoped_ftz_" if scoped_ftz else "")
        + "ranks4_gemma_add_rmsnorm_mxfp8_tiled_h6144"
        + ("_wave_handoffs" if wave_handoffs else "")
    )

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW * R]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,
        i32_q: fx.Int32,  # rows per chunk ceil(rows / 4); grid = 4 xcd_grid(ceil(q / R))
        i32_me: fx.Int32,
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        i32_sx: fx.Int32,
        arg_x: fx.Int64,  # [rows, H] bf16 partial, stride i32_sx
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
        # group 0 (dispatched first) reduces chunk me, groups 1-3 pull chunks me + 1 .. me + 3, a group's XCDs on
        # contiguous row runs; the workgroup's rows: c q + j0 + r, r < R
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
                return row(arg_x + fx.Int64(rws[r]) * fx.Int64(i32_sx * 2), 0, oks[r])

            local = [local_row(r) for r in range_constexpr(R)]
            # a puller stages its partials (write-through) before it arrives
            for r in range_constexpr(R):
                store_row(local[r], mine + STG_OFF + rowoffs[r], puller & oks[r], SYS)
            fx.rocdl.s_waitcnt(vmcnt=0)
            if const_expr(wave_handoffs):
                arrive_wave(*bs, c, me, rw0, wave, e, puller)
                wait_pullers_wave(mine, me, rw0, wave, e, reducer)
            else:
                fx.gpu.barrier()
                arrive_wave(*bs, c, me, rw0, fx.Int32(0), e, puller & (wave == 0))
                wait_pullers_wave(mine, me, rw0, fx.Int32(0), e, reducer & (wave == 0))
                fx.gpu.barrier()
            # wave v issues its peer loads in the order me + 1 + (t + v) % 3: each
            # CU's requests spread over the three links (the sum keeps the order)
            rot = wave % (WORLD - 1)
            peer = [
                [
                    row(
                        sel4((me + 1 + (t + rot) % (WORLD - 1)) % WORLD, *bs)
                        + STG_OFF
                        + rowoffs[r],
                        SYS,
                        reducer & oks[r],
                    )
                    for t in range_constexpr(WORLD - 1)
                ]
                for r in range_constexpr(R)
            ]
            rv = [
                row(arg_r + fx.Int64(rws[r]) * fx.Int64(i32_sr * 2), 0, oks[r])
                for r in range_constexpr(R)
            ]
            wv = row(arg_w, 0, always)
            if const_expr(scoped_ftz):
                # only the rank sum runs with hardware FP32 FTZ (AITER's flush), the norm after it keeps denormals;
                # every operand and sum crosses the mode-change asm, one mode region per row (8 packs per asm)
                reduced = []
                for r in range_constexpr(R):
                    flat, mode = _enter_fp32_ftz(
                        [
                            pk
                            for h in range_constexpr(2)
                            for pk in [local[r][h]]
                            + unrotate(
                                rot, [peer[r][t][h] for t in range_constexpr(WORLD - 1)]
                            )
                        ]
                    )
                    reduced.append(
                        _restore_fp32_denorm_mode(
                            [_sum4_hardware_ftz(flat[0:4]), _sum4_hardware_ftz(flat[4:8])],
                            mode,
                        )
                    )
            else:
                # the software flush in this emission order: another order took 67 VGPRs (64 here), and above 64 a CU
                # holds 3 workgroups, too few for a 1024-workgroup grid whose ranks wait on each other
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
            for r in range_constexpr(R):
                store_row(
                    reduced[r],
                    mine + RED_OFF + fx.Int64(j0 + r) * ROWB,
                    reducer & oks[r],
                    SYS,
                )
            fx.rocdl.s_waitcnt(vmcnt=0)
            if const_expr(wave_handoffs):
                publish_wave(*bs, me, rw0, wave, e, reducer)
            else:
                fx.gpu.barrier()
                publish_wave(*bs, me, rw0, fx.Int32(0), e, reducer & (wave == 0))
            # the reducer never waits on vmcnt after publishing: its flag stores to the
            # peers are acknowledged only behind the bulk traffic on the links

            def pull_rows():
                pulled = [
                    row(
                        sel4(c, *bs) + RED_OFF + fx.Int64(j0 + r) * ROWB,
                        SYS,
                        puller & oks[r],
                    )
                    for r in range_constexpr(R)
                ]
                return [
                    fx.Int32(pulled[r][h][i])
                    for r in range_constexpr(R)
                    for h in range_constexpr(2)
                    for i in range_constexpr(4)
                ]

            xw = [
                fx.Int32(reduced[r][h][i])
                for r in range_constexpr(R)
                for h in range_constexpr(2)
                for i in range_constexpr(4)
            ]
            if puller:
                if const_expr(wave_handoffs):
                    wait_reduced_wave(mine, rw0, wave, e)
                else:
                    wait_reduced_wave(mine, rw0, fx.Int32(0), e)
                    fx.gpu.barrier()
                xw = pull_rows()
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
    def launch_ar_add_rmsnorm_mxfp8_register(
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

    launch_ar_add_rmsnorm_mxfp8_register.kernel_name = name
    launch_ar_add_rmsnorm_mxfp8_register.compile_hints = PRELOAD_HINTS
    return launch_ar_add_rmsnorm_mxfp8_register
