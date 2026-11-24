# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MoE combine + TP4 all-reduce + residual add + Gemma RMSNorm + MXFP8 quant, streamed rows, both exchanges as the
lossless 12-bit code, eager or in graphs.

``slots4_ar_add_rmsnorm_mxfp8_stream_lossless`` plus a coded reduce-scatter: a puller writes its bf16 slot sum into its
staging row in the same format (raw waves into a second caller-owned arena, int64[WORLD] table, one ROWB row per call
row), and the reducer decodes the three peers' rows (headers through LDS, two banks) to the same bf16 bits, so every
sum and output is unchanged. Reuse rides on the stream protocol: the reducer reads a puller's staging and input-raw
rows before it publishes, the puller waits for that publication before its kernel ends, and the next call's stores
follow in stream order. Inactive decoded values (the second column piece of waves 4-7, masked rows) are zeroed before
they are summed.
"""

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
    load_ptrs,
    sel4,
    sum4,
    unrotate,
    wait_pullers,
)
from .slots4_ar_add_rmsnorm_mxfp8 import local_sum
from .slots4_ar_add_rmsnorm_mxfp8_stream import (
    NW, NT, ROWB, SLOTS, MAX_ROWS, stream_layout, sub_rows,
    publish_stream, wait_reduced_stream,
    _store_tiled, _store_rowmajor,
)
from .lossless_bf16_row import store32, decode32

# each rank's input raw arena: one row per call row of the largest call
INPUT_RAW_BYTES = MAX_ROWS * ROWB


def compile_slots4_ar_add_rmsnorm_mxfp8_stream_lossless_scatter(rpw=16, tiled_q=True):
    R = rpw
    S = sub_rows(R)
    _, red_off, stg_off, _ = stream_layout(MAX_ROWS)
    layout = "tiled" if tiled_q else "rowmajor"
    name = f"sum_slots4_shared_allreduce_lossless12_scatter_stream_ranks4_gemma_add_rmsnorm_mxfp8_{layout}_h6144_r{R}"
    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW]
        hdr: fx.Array[fx.Int32, 96]
        phdr: fx.Array[fx.Int32, 576]  # two banks x three peers x twelve wave headers

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
        arg_raw: fx.Int64,  # int64[WORLD]: each rank's raw arena (the stream_lossless kernel's RAW_BYTES)
        arg_input_raw: fx.Int64,  # int64[WORLD]: each rank's input raw arena (INPUT_RAW_BYTES)
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
            hdr = smem.hdr.ptr
            phdr = smem.phdr.ptr
            bs = (arg_b0, arg_b1, arg_b2, arg_b3)
            mine = sel4(me, *bs)
            rawptrs = load_ptrs(arg_raw)
            mine_raw = sel4(me, *rawptrs)
            input_raws = load_ptrs(arg_input_raw)
            mine_input_raw = sel4(me, *input_raws)
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
            if puller:
                # all operands die at each row
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                        ri = j * S + r
                        rw = rw0 + ri
                        ok = (j0 + ri < q) & (rw < i32_rows)
                        sl = [row(arg_slots + fx.Int64(rw) * (SLOTS * ROWB) + k * ROWB, 0, ok)
                              for k in range_constexpr(SLOTS)]
                        sh = row(arg_sh + fx.Int64(rw) * ROWB, 0, ok)
                        lv = [local_sum([sl[k][h] for k in range_constexpr(SLOTS)], sh[h])
                              for h in range_constexpr(2)]
                        for h in range_constexpr(2):
                            store32(lv[h], mine + stg_off + fx.Int64(rw) * ROWB,
                                    mine_input_raw + fx.Int64(rw) * ROWB, tx + 512 * h,
                                    ok if h == 0 else ok & has1)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    arrive(*bs, c, me, rw0 + j * S, count(j), sub_ok(j))
            if reducer:
                for j in range(fx.Int32(0), n_sub, fx.Int32(1)):
                    ej = count(j)
                    wait_pullers(mine, me, rw0 + j * S, ej, sub_ok(j))
                    for r in range(fx.Int32(0), fx.Int32(S), fx.Int32(1)):
                        ri = j * S + r
                        rw = rw0 + ri
                        ok = (j0 + ri < q) & (rw < i32_rows)
                        sl = [row(arg_slots + fx.Int64(rw) * (SLOTS * ROWB) + k * ROWB, 0, ok)
                              for k in range_constexpr(SLOTS)]
                        sh = row(arg_sh + fx.Int64(rw) * ROWB, 0, ok)
                        # waves 0-2 load peer t's twelve wave headers into this row's bank; the next row's barrier
                        # protects its reads
                        bank = 288 * (ri % 2)
                        for t in range_constexpr(WORLD - 1):
                            src = sel4((me + 1 + t) % WORLD, *bs) + stg_off + fx.Int64(rw) * ROWB
                            hres = buffer_ops.create_buffer_resource_from_addr(src, num_records_bytes=ROWB)
                            hv = Vec(buffer_ops.buffer_load(hres, 2304 + 4 * lane, vec_width=4, dtype=fx.Int32,
                                     mask=ok & (wave == t) & (lane < 24), cache_modifier=SYS))
                            if (wave == t) & (lane < 24):
                                for u in range_constexpr(4):
                                    phdr[bank + 96 * t + 4 * lane + u] = fx.Int32(hv[u])
                        fx.gpu.barrier()
                        peer = []
                        for t in range_constexpr(WORLD - 1):
                            pt = (t + rot) % (WORLD - 1)
                            p = (me + 1 + pt) % WORLD
                            pieces = []
                            for h in range_constexpr(2):
                                m = ok if h == 0 else ok & has1
                                vwave = wave + 8 * h
                                safe = (vwave < 12).select(vwave, fx.Int32(0))
                                hw = Vec.from_elements([fx.Int32(phdr[bank + 96 * pt + 8 * safe + u])
                                                        for u in range_constexpr(8)], fx.Int32)
                                d = decode32(sel4(p, *bs) + stg_off + fx.Int64(rw) * ROWB,
                                             sel4(p, *input_raws) + fx.Int64(rw) * ROWB, tx + 512 * h, m, hw)
                                pieces.append(Vec.from_elements([m.select(fx.Int32(d[u]), fx.Int32(0))
                                                                 for u in range_constexpr(4)], fx.Int32))
                            peer.append(pieces)
                        loc = [local_sum([sl[k][h] for k in range_constexpr(SLOTS)], sh[h])
                               for h in range_constexpr(2)]
                        reduced = [sum4([loc[h]] + unrotate(rot, [peer[t][h]
                                   for t in range_constexpr(WORLD - 1)]))
                                   for h in range_constexpr(2)]
                        for h in range_constexpr(2):
                            store32(reduced[h],mine+red_off+fx.Int64(j0+ri)*ROWB,
                                    mine_raw+fx.Int64(j0+ri)*ROWB,tx+512*h,ok if h==0 else ok & has1)
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
                    coded=sel4(c,*bs)+red_off+fx.Int64(j0+ri)*ROWB
                    raw=sel4(c,*rawptrs)+fx.Int64(j0+ri)*ROWB
                    rs=buffer_ops.create_buffer_resource_from_addr(coded,num_records_bytes=ROWB)
                    hv=Vec(buffer_ops.buffer_load(rs,2304+4*lane,vec_width=4,dtype=fx.Int32,
                                                  mask=ok & (wave==0) & (lane<24),cache_modifier=SYS))
                    if (wave==0) & (lane<24):
                        for u in range_constexpr(4):hdr[4*lane+u]=fx.Int32(hv[u])
                    fx.gpu.barrier()
                    xv=[]
                    for h in range_constexpr(2):
                        vwave=wave+8*h
                        safe=(vwave<12).select(vwave,fx.Int32(0))
                        hw=Vec.from_elements([fx.Int32(hdr[8*safe+u]) for u in range_constexpr(8)],fx.Int32)
                        decoded=decode32(coded,raw,tx+512*h,ok if h==0 else ok & has1,hw)
                        xv.append(Vec.from_elements([(ok if h==0 else ok & has1).select(fx.Int32(decoded[u]),fx.Int32(0)) for u in range_constexpr(4)],fx.Int32))
                    rv = row(arg_r + fx.Int64(rw) * fx.Int64(i32_sr * 2), 0, ok)
                    wv = row(arg_w, 0, always)
                    wvf = [bf16x8(wv[h]) for h in range_constexpr(2)]
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
    def launch_slots4_ar_stream_lossless_scatter(
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
        arg_raw: fx.Int64,
        arg_input_raw: fx.Int64,
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
            arg_raw,
            arg_input_raw,
            i32_eps,
        ).launch(
            grid=(fx.Int64(xcd_grid((i32_q + (R - 1)) // R) * WORLD), 1, 1),
            block=(NT, 1, 1),
            stream=stream,
        )

    launch_slots4_ar_stream_lossless_scatter.kernel_name = name
    launch_slots4_ar_stream_lossless_scatter.compile_hints = PRELOAD_HINTS
    return launch_slots4_ar_stream_lossless_scatter
