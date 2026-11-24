# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MoE combine + TP4 one-shot all-reduce + residual add + Gemma RMSNorm + MXFP8 quant (tiled codes), the fewest rows,
eager or in graphs.

One workgroup per row: every rank stages its slot sum (``local_sum``) in one of two copies (call-count parity), signals
the peers per wave and reads their 3 copies, then sums the 4 partials in the two-stage kernels' chunk order. Only the
rank sum runs with hardware FP32 FTZ (``common._enter_fp32_ftz``: every operand crosses the mode-change asm, every sum
the restore), so the slot sums and the norm keep denormals.
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
)
from .common import (
    MAX_ROWS,
    SYS,
    WORLD,
    _enter_fp32_ftz,
    _flag_store,
    _restore_fp32_denorm_mode,
    _spin,
    _sum4_hardware_ftz,
    sel4,
)
from .slots4_ar_add_rmsnorm_mxfp8 import (
    NT,
    NW,
    ROWB,
    SLOTS,
    SLOTS_BUF_BYTES,
    local_sum,
)

ONESHOT_ROWS = 22
ONE_OFF = SLOTS_BUF_BYTES  # [2, ONESHOT_ROWS, H] bf16 partials (two copies by call-count parity), after the 2-stage's
# flag word (b * NW + v) * WORLD + s at WFLAG_OFF: wave v of rank s stored its part of row b (one 4 KB page keeps the
# stream kernels' region page-aligned)
WFLAG_OFF = ONE_OFF + 2 * ONESHOT_ROWS * ROWB
WFLAG_BYTES = 4096
assert ONESHOT_ROWS * NW * WORLD * 4 <= WFLAG_BYTES
ONESHOT_BUF_BYTES = WFLAG_OFF + WFLAG_BYTES


def compile_slots4_ar_add_rmsnorm_mxfp8_oneshot():
    capacity, wave_flag_off = ONESHOT_ROWS, WFLAG_OFF
    name = "sum_slots4_shared_allreduce_oneshot_scoped_ftz_ranks4_gemma_add_rmsnorm_mxfp8_tiled_h6144"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_rows: fx.Int32,  # grid = rows
        i32_q: fx.Int32,  # ceil(rows / 4): row b's 2-stage chunk owner is b / q
        i32_me: fx.Int32,
        i32_sr: fx.Int32,  # residual row stride (elements, a multiple of 8)
        arg_slots: fx.Int64,  # [rows, 4, H] bf16, this rank's weighted routed slots
        arg_sh: fx.Int64,  # [rows, H] bf16, this rank's shared-expert output
        arg_b0: fx.Int64,  # the ranks' buffers
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
        tx, rw = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        me = i32_me
        smem = fx.SharedAllocator().allocate(Shared).peek()
        part = smem.part.ptr
        bs = (arg_b0, arg_b1, arg_b2, arg_b3)
        mine = sel4(me, *bs)
        cr = buffer_ops.create_buffer_resource_from_addr(
            arg_cnt, num_records_bytes=MAX_ROWS * 4
        )
        e = (
            fx.Int32(
                buffer_ops.buffer_load(cr, rw, vec_width=1, dtype=fx.Int32, is_scalar=True)
            )
            + 1
        )
        has1 = tx < 256  # second chunk (columns 4096 + 8 t) inside the row
        offs = [tx * 4, tx * 4 + 2048]
        rowoff = fx.Int64(rw) * ROWB
        always = fx.Boolean(True)
        mycopy = fx.Int64(ONE_OFF) + (fx.Int64(e & 1) * capacity + fx.Int64(rw)) * ROWB

        def row(base, cm, m):
            rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
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

        sbase = arg_slots + fx.Int64(rw) * (SLOTS * ROWB)
        sl = [row(sbase + k * ROWB, 0, always) for k in range_constexpr(SLOTS)]
        sh = row(arg_sh + rowoff, 0, always)
        local = [
            local_sum([sl[k][h] for k in range_constexpr(SLOTS)], sh[h])
            for h in range_constexpr(2)
        ]
        stg = buffer_ops.create_buffer_resource_from_addr(
            mine + mycopy, num_records_bytes=ROWB
        )
        for h in range_constexpr(2):
            buffer_ops.buffer_store(
                local[h],
                stg,
                offs[h],
                mask=None if h == 0 else has1,
                cache_modifier=SYS,
            )
        # the wave's own stores retire, then its lanes 0-2 signal peer me + 1 + lane and wait for the same wave of that
        # peer; the other lanes wait at the branch's join (no workgroup barrier)
        fx.rocdl.s_waitcnt(vmcnt=0)
        pl = (me + 1 + lane) % WORLD
        wflag = fx.Int64(wave_flag_off) + fx.Int64(((rw * NW + wave) * WORLD) * 4)
        if lane < WORLD - 1:
            _flag_store(sel4(pl, *bs) + wflag + fx.Int64(me * 4), e)
            _spin(mine + wflag + fx.Int64(pl * 4), e)
        # fixed peer order: no per-wave rotation and unrotation at these few rows (the two-stage tiers rotate)
        peer = [
            row(
                sel4((me + 1 + t) % WORLD, *bs) + mycopy,
                SYS,
                always,
            )
            for t in range_constexpr(WORLD - 1)
        ]
        rv = row(arg_r + fx.Int64(rw) * fx.Int64(i32_sr * 2), 0, always)
        wv = row(arg_w, 0, always)
        # packs by rank me + k -> in the order c, c + 1, c + 2, c + 3
        d = (rw // i32_q - me + WORLD) % WORLD
        allv, mode = _enter_fp32_ftz(
            local + [peer[t][h] for t in range_constexpr(3) for h in range_constexpr(2)]
        )
        local = allv[:2]
        peer = [
            [allv[2 + t * 2 + h] for h in range_constexpr(2)]
            for t in range_constexpr(3)
        ]
        reduced = []
        for h in range_constexpr(2):
            byk = [local[h]] + [peer[t][h] for t in range_constexpr(WORLD - 1)]
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
            reduced.append(_sum4_hardware_ftz(ordered))
        reduced = _restore_fp32_denorm_mode(reduced, mode)
        xv = [bf16x8(reduced[h]) for h in range_constexpr(2)]
        rvf = [bf16x8(rv[h]) for h in range_constexpr(2)]
        wvf = [bf16x8(wv[h]) for h in range_constexpr(2)]
        s = [
            [xv[h][n] + rvf[h][n] for n in range_constexpr(8)] for h in range_constexpr(2)
        ]
        acc = wave_sum_lane63(seq_sum_sq(s[0] + s[1]))
        if lane == 63:
            part[wave] = acc
        resr = buffer_ops.create_buffer_resource_from_addr(
            arg_res + rowoff, num_records_bytes=ROWB
        )
        for h in range_constexpr(2):
            res = [pk_bf16(s[h][2 * n], s[h][2 * n + 1]) for n in range_constexpr(4)]
            buffer_ops.buffer_store(
                Vec.from_elements(res, fx.Int32),
                resr,
                offs[h],
                mask=None if h == 0 else has1,
            )
        fx.gpu.barrier()
        rstd = rsqrt_rn(
            _block_sum([part[n] for n in range_constexpr(NW)]),
            i32_eps.bitcast(fx.Float32),
        )
        codes, sbs = [], []
        for h in range_constexpr(2):
            y = [
                (s[h][n] * rstd) * (wvf[h][n] + fx.Float32(1.0))
                for n in range_constexpr(8)
            ]
            sb, rescale = _e8m0(y)
            codes.append(_codes(y, rescale))
            sbs.append(fx.Int32(sb.to(fx.Int32)))
        # lane pairs: the even lane stores chunk 0's 16 columns, the odd chunk 1's
        odd = (lane & 1) == 1
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
        pc = tx // 2 + odd.select(fx.Int32(256), fx.Int32(0))  # 16-column piece
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=i64_q_bytes
        )
        buffer_ops.buffer_store(
            piece,
            qr,
            _code_off(rw, pc) // 4,
            mask=((lane & 1) == 0) | has1,
            cache_modifier=CODE_NT,
        )
        # one lane per 32-column group stores its scale byte
        q4 = lane % 4
        g = tx // 4 + (q4 == 1).select(fx.Int32(128), fx.Int32(0))
        sb = (q4 == 1).select(sbs[1], sbs[0])
        sr = buffer_ops.create_buffer_resource_from_addr(
            arg_s, num_records_bytes=i64_s_bytes
        )
        buffer_ops.buffer_store(
            sb.to(fx.Int8),
            sr,
            _scale_off(rw, g),
            mask=(q4 == 0) | ((q4 == 1) & has1),
            offset_is_bytes=True,
        )
        buffer_ops.buffer_store(e, cr, rw, mask=tx == 0)

    @flyc.jit
    def launch_slots4_oneshot_scoped_ftz(
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
        ).launch(grid=(fx.Int64(i32_rows), 1, 1), block=(NT, 1, 1), stream=stream)

    launch_slots4_oneshot_scoped_ftz.kernel_name = name
    launch_slots4_oneshot_scoped_ftz.compile_hints = PRELOAD_HINTS
    return launch_slots4_oneshot_scoped_ftz
