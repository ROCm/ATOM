# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K4 stages: the split-K sparse attention, the partitions' merge and
o_proj with the attention all-reduce (``kernels.stages.attention`` on wave32
and WMMA; the mailboxes and their contents are the gfx950 stages').

A split task's 256 keys are 16 page-16s, page w on wave w for QK (A = the task's
16 q heads, B = the page's keys, two 16x16x64 FP8 WMMAs over the head dims). PV
turns the tiling around: wave w takes dims 16 (w % 8) .. + 16 over key half w // 8
(A = P of the 16 heads, B = V's dims, k = keys), so the partials are two halves,
not one a wave. Both K and V page-16 layouts put a lane's 16 operand bytes in a
row: no transposes.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.minimax_m3.mono.config import HEAD_DIM, HIDDEN, MAX_TOKENS, PAGE16
from atom.models.minimax_m3.mono.gfx1250.layout import (
    N_O,
    N_SPLIT,
    O_K,
    O_ROW,
    O_ROWS,
    SPLIT_KEYS,
    H,
)
from atom.models.minimax_m3.mono.kernels.common import ld_bf16x4, per_token_fp8_scale
from atom.models.minimax_m3.mono.kernels.pre_attn import N_HEAD_TASKS as K1_HEAD_TASKS
from atom.models.minimax_m3.mono.kernels.stages.shared import NEG, PAGE_BYTES
from atom.mono.device.gfx1250 import (
    CM_DEV,
    CM_NT,
    WAVE,
    WAVES,
    acc_row_sum,
    butterfly,
    publish,
    rows_to_lds_async,
    shuffled_rows,
    wave_max,
    wmma_fp8_k64,
    wmma_fp8_k128,
    xshfl,
)
from atom.mono.device.mx import FP8_MAX
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    rsrc,
    traced,
    uniform,
)
from atom.mono.device.ranks import sum_partials
from atom.mono.device.sync import preg
from atom.mono.plan.arith import pick

PAGES = SPLIT_KEYS // PAGE16  # a split task's page-16s: one a wave
DT = HEAD_DIM // 16  # PV dim tiles
KH = WAVES // DT  # PV key halves
HALF_KEYS = SPLIT_KEYS // KH
assert PAGES == WAVES and KH == 2 and HALF_KEYS % 64 == 0


def _cat(vs):
    return fx.Vector.from_elements([v[e] for v in vs for e in range(4)], fx.Int32)


@traced
def split_defs(k4ctx):
    """``kernels.stages.attention.split_defs`` on gfx1250 (see the module)."""
    CM_K1 = k4ctx["CM_K1"]
    G = k4ctx["G"]
    acquire = k4ctx["acquire"]
    blk = k4ctx["blk"]
    fuse_k1 = k4ctx["fuse_k1"]
    g4 = k4ctx["g4"]
    hdone_mb = k4ctx["hdone_mb"]
    hst = k4ctx["hst"]
    k1_poll = k4ctx["k1_poll"]
    k_cache = k4ctx["k_cache"]
    k_scale = k4ctx["k_scale"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    mb_put = k4ctx["mb_put"]
    mb_put_bf = k4ctx["mb_put_bf"]
    mr = k4ctx["mr"]
    opart = k4ctx["opart"]
    p8 = k4ctx["p8"]
    q = k4ctx["q"]
    q8 = k4ctx["q8"]
    task_blocks = k4ctx["select_blocks" if k4ctx["index_topk"] else "reuse_selection"]
    sm_scale = k4ctx["sm_scale"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    v_cache = k4ctx["v_cache"]
    v_scale = k4ctx["v_scale"]
    wave = k4ctx["wave"]
    v4i = fx.Vector.make_type(4, fx.Int32)

    def kv_task(i):
        """K1 head task i of the K / V ones every token's split waits on."""
        return fx.min(i // 2, tokens - 1) * K1_HEAD_TASKS + H + i % 2

    def stage_split():
        r_k, r_v = rsrc(k_cache), rsrc(v_cache)
        r_ks, r_vs = rsrc(k_scale), rsrc(v_scale)
        for ts in range(start("split"), N_SPLIT * tokens, G):
            ts = fx.Int32(ts)
            tok = ts // N_SPLIT
            t = ts % N_SPLIT
            task_blocks(tok, t)
            stamp(17)
            if const_expr(fuse_k1):
                # this token's q heads (lanes < H) and every token's new K / V
                if wave == 0:
                    k1_poll(
                        [
                            (
                                hdone_mb,
                                (lane < H).select(
                                    tok * K1_HEAD_TASKS + lane, kv_task(lane - H)
                                ),
                                1,
                            ),
                            (hdone_mb, kv_task(lane + WAVE - H), 1),
                        ]
                    )
                gpu.barrier()
                acquire()
            n_ctx = uniform(fx.ptr_load(blk + 16))
            pages = [uniform(fx.ptr_load(blk + j)) for j in range(PAGES)]
            page = pages[0]
            for j in range_constexpr(1, PAGES):
                page = (wave == j).select(pages[j], page)
            qv = ld_bf16x4(rsrc(q), tok * O_K + tid * 4, CM_K1)
            # QK B operand: key l16 of this wave's page, dims 64 c + 16 g4 (+ 32)
            kw = []
            for c in range_constexpr(HEAD_DIM // 64):
                kw.append(
                    _cat(
                        [
                            fx.Vector(
                                bo.buffer_load(
                                    r_k,
                                    (
                                        page * PAGE_BYTES
                                        + (4 * c + 2 * h + g4) * 256
                                        + l16 * 16
                                    )
                                    // 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                    cache_modifier=CM_K1,
                                )
                            )
                            for h in range(2)
                        ]
                    )
                )
            # PV B operand: dim 16 (wave % 8) + l16 of pages 8 kh + 4 j + g4 (+ 2)
            dim = 16 * (wave % DT) + l16
            kh = wave // DT
            vw = []
            for j in range_constexpr(HALF_KEYS // 64):
                parts = []
                for h in range_constexpr(2):
                    pj = (HALF_KEYS // 16) * kh + 4 * j + 2 * h + g4
                    vpg = pages[0]
                    for p_ in range_constexpr(1, PAGES):
                        vpg = (pj == p_).select(pages[p_], vpg)
                    w4 = fx.Vector(
                        bo.buffer_load(
                            r_v,
                            (vpg * PAGE_BYTES + dim * PAGE16) // 4,
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=CM_K1,
                        )
                    )
                    # keys past the context carry P = 0, but a NaN V byte would poison
                    nv = n_ctx - (t * SPLIT_KEYS + 16 * pj)
                    for e in range_constexpr(4):
                        m = nv - 4 * e
                        mask = (m >= 4).select(
                            fx.Int32(-1),
                            (m <= 0).select(
                                fx.Int32(0), (fx.Int32(1) << (fx.Int32(m) * 8)) - 1
                            ),
                        )
                        parts.append(w4[e] & mask)
                vw.append(fx.Vector.from_elements(parts, fx.Int32))
            ks = fx.Float32(
                bo.buffer_load(
                    r_ks,
                    page * PAGE16 + l16,
                    vec_width=1,
                    dtype=T.f32,
                    cache_modifier=CM_K1,
                )
            )
            vs = fx.Float32(
                bo.buffer_load(
                    r_vs,
                    page * PAGE16 + l16,
                    vec_width=1,
                    dtype=T.f32,
                    cache_modifier=CM_K1,
                )
            )
            fx.ptr_store(fp8_pack4(qv[0], qv[1], qv[2], qv[3]), q8 + tid)
            gpu.barrier()
            stamp(12)
            # QK: C[head][key], lane holds key l16, heads 8 g4 + e
            key = t * SPLIT_KEYS + wave * PAGE16 + l16
            valid = key < n_ctx
            c = fx.Vector.filled(8, 0.0, fx.Float32)
            for cc in range_constexpr(HEAD_DIM // 64):
                qa = _cat(
                    [
                        fx.Vector(
                            fx.ptr_load(
                                q8 + (l16 * (HEAD_DIM // 4) + 16 * cc + 8 * h + 4 * g4),
                                result_type=v4i,
                            )
                        )
                        for h in range(2)
                    ]
                )
                c = wmma_fp8_k64(qa, kw[cc], c)
            qk = sm_scale * ks
            sc = [valid.select(qk * c[e], fx.Float32(NEG)) for e in range(8)]
            vmax_w = wave_max(valid.select(vs, fx.Float32(0.0)))
            for e in range_constexpr(8):
                mh = butterfly(sc[e], (8, 4, 2, 1), fx.max)
                if l16 == 0:
                    fx.ptr_store(mh, hst + (wave * H + 8 * g4 + e))
            if lane == 0:
                fx.ptr_store(vmax_w, hst + (3 * WAVES * H + wave))
            gpu.barrier()
            stamp(13)
            vmax = fx.ptr_load(hst + 3 * WAVES * H)
            for w in range_constexpr(1, WAVES):
                vmax = fx.max(vmax, fx.ptr_load(hst + (3 * WAVES * H + w)))
            mxs = []
            for e in range_constexpr(8):
                mx = fx.ptr_load(hst + (8 * g4 + e))
                for w in range_constexpr(1, WAVES):
                    mx = fx.max(mx, fx.ptr_load(hst + (w * H + 8 * g4 + e)))
                mxs.append(mx)
            fscale = FP8_MAX * hw_rcp(vmax + 1e-8)
            pscale = vmax * (1.0 / FP8_MAX)
            vsel = valid.select(vs, fx.Float32(0.0)) * fscale
            for e in range_constexpr(8):
                head = 8 * g4 + e
                pj = hw_exp2((sc[e] - mxs[e]) * LOG2E)
                b8 = fp8_pack4(vsel * pj, 0.0, 0.0, 0.0) & 0xFF
                w8 = b8 | (xshfl(b8, 1) << 8)
                w8 = w8 | (xshfl(w8, 2) << 16)
                if l16 % 4 == 0:
                    fx.ptr_store(
                        w8, p8 + (head * (SPLIT_KEYS // 4) + (wave * PAGE16 + l16) // 4)
                    )
                ls = butterfly(pj, (8, 4, 2, 1))
                if l16 == 0:
                    fx.ptr_store(ls, hst + (WAVES * H + wave * H + head))
                    fx.ptr_store(mxs[e], hst + (2 * WAVES * H + head))
            gpu.barrier()
            stamp(14)
            # PV: C[head][dim], lane holds dim l16 of the tile, heads 8 g4 + e
            cv = fx.Vector.filled(8, 0.0, fx.Float32)
            for j in range_constexpr(HALF_KEYS // 64):
                k0 = HALF_KEYS * kh + 64 * j
                pa = _cat(
                    [
                        fx.Vector(
                            fx.ptr_load(
                                p8
                                + (
                                    l16 * (SPLIT_KEYS // 4)
                                    + (k0 + 32 * h + 16 * g4) // 4
                                ),
                                result_type=v4i,
                            )
                        )
                        for h in range(2)
                    ]
                )
                cv = wmma_fp8_k64(pa, vw[j], cv)
            for e in range_constexpr(8):
                fx.ptr_store(cv[e], opart + (kh * O_K + (8 * g4 + e) * HEAD_DIM + dim))
            gpu.barrier()
            stamp(15)
            e0 = tid * 4
            head = e0 // HEAD_DIM
            lsum = fx.ptr_load(hst + (WAVES * H + head))
            for w in range_constexpr(1, WAVES):
                lsum = lsum + fx.ptr_load(hst + (WAVES * H + w * H + head))
            inv_l = 1.0 / (lsum > 0.0).select(lsum, fx.Float32(1.0))
            ov = []
            for k in range_constexpr(4):
                acc = fx.ptr_load(opart + (e0 + k))
                for h in range_constexpr(1, KH):
                    acc = acc + fx.ptr_load(opart + (h * O_K + e0 + k))
                ov.append((pscale * acc) * inv_l)
            mb_put_bf(mr("sp_o"), ts * O_K + e0, ov)
            if tid < H:
                mb_put(
                    mr("sp_m"), ts * H + tid, fx.ptr_load(hst + (2 * WAVES * H + tid))
                )
                lt = fx.ptr_load(hst + (WAVES * H + tid))
                for w in range_constexpr(1, WAVES):
                    lt = lt + fx.ptr_load(hst + (WAVES * H + w * H + tid))
                mb_put(mr("sp_l"), ts * H + tid, lt)
            gpu.barrier()

    return {"stage_split": stage_split}


@traced
def merge_defs(k4ctx):
    """``kernels.stages.attention.merge_defs`` on gfx1250."""
    G = k4ctx["G"]
    block_max = k4ctx["block_max"]
    lane = k4ctx["lane"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    mr = k4ctx["mr"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    wave = k4ctx["wave"]

    def merge_token(k):
        hm = tid // (HEAD_DIM // 4)
        e0 = tid * 4
        sp0 = k * N_SPLIT
        if wave == 0:
            mb_poll(
                [
                    (mr("sp_l"), sp0 * H + lane + WAVE * i, 1)
                    for i in range(N_SPLIT * H // WAVE)
                ]
            )
        gpu.barrier()
        got = mb_poll(
            [(mr("sp_m"), (sp0 + i) * H + hm, 1) for i in range(N_SPLIT)]
            + [(mr("sp_l"), (sp0 + i) * H + hm, 1) for i in range(N_SPLIT)]
            + [(mr("sp_o"), ((sp0 + i) * O_K + e0) // 2, 2) for i in range(N_SPLIT)],
            batch=3 * N_SPLIT,
        )
        mls = got[: 2 * N_SPLIT]
        ovs = got[2 * N_SPLIT :]
        stamp(3)
        m_i = [mls[i][0].bitcast(fx.Float32) for i in range(N_SPLIT)]
        l_i = [mls[N_SPLIT + i][0].bitcast(fx.Float32) for i in range(N_SPLIT)]
        gm = m_i[0]
        for i in range_constexpr(1, N_SPLIT):
            gm = fx.max(gm, m_i[i])
        sl = [l_i[i] * hw_exp2((m_i[i] - gm) * LOG2E) for i in range(N_SPLIT)]
        gs = sl[0]
        for i in range_constexpr(1, N_SPLIT):
            gs = gs + sl[i]
        gs = (gs > 0.0).select(gs, fx.Float32(1.0))
        av = [fx.Float32(0.0) for _ in range(4)]
        for i in range_constexpr(N_SPLIT):
            wi = sl[i] / gs
            o01 = bf2_f32(ovs[i][0])
            o23 = bf2_f32(ovs[i][1])
            av = [
                av[0] + wi * o01[0],
                av[1] + wi * o01[1],
                av[2] + wi * o23[0],
                av[3] + wi * o23[1],
            ]
        a0, a1, a2, a3 = (bf16_round(x) for x in av)
        amax = block_max(
            fx.max(
                fx.max(fmath.absf(a0), fmath.absf(a1)),
                fx.max(fmath.absf(a2), fmath.absf(a3)),
            )
        )
        x_scale, inv = per_token_fp8_scale(amax)
        return fp8_pack4(a0 * inv, a1 * inv, a2 * inv, a3 * inv), x_scale

    def stage_merge():
        if const_expr(tokens > 1):
            for k in range(start("merge"), tokens, G):
                k = fx.Int32(k)
                word, x_scale = merge_token(k)
                bo.buffer_store(
                    word, rsrc(mb("attnq")), k * (O_K // 4) + tid, cache_modifier=CM_DEV
                )
                publish(mb_put, mr("attnq_s"), k, x_scale, tid == 0)

    return {"merge_token": merge_token, "stage_merge": stage_merge}


@traced
def o_defs(k4ctx):
    """``kernels.stages.attention.o_defs`` on gfx1250: a 16-row group's GEMV on 8
    waves, each two 16x16x128 FP8 WMMAs."""
    AG_RS = k4ctx["AG_RS"]
    CM_K1 = k4ctx["CM_K1"]
    G = k4ctx["G"]
    SY = k4ctx["SY"]
    W = k4ctx["W"]
    acquire = k4ctx["acquire"]
    fuse_k1 = k4ctx["fuse_k1"]
    x8_in = k4ctx["x8_in"]
    g4 = k4ctx["g4"]
    h_in = k4ctx["h_in"]
    h_mid = k4ctx["h_mid"]
    k1_poll = k4ctx["k1_poll"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put_bf = k4ctx["mb_put_bf"]
    merge_token = k4ctx["merge_token"]
    misc = k4ctx["misc"]
    mr = k4ctx["mr"]
    peer_addr = k4ctx["peer_addr"]
    push_rows = k4ctx["push_rows"]
    rank = k4ctx["rank"]
    rdone_mb = k4ctx["rdone_mb"]
    red = k4ctx["red"]
    s_o = k4ctx["s_o"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    sym = k4ctx["sym"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    w_o = k4ctx["w_o"]
    wave = k4ctx["wave"]
    xs = k4ctx["xs"]
    v4i = fx.Vector.make_type(4, fx.Int32)
    GW = WAVES // (O_ROWS // 16)  # waves a 16-row group
    O_CPW = O_K // 128 // GW  # 128-k chunks a wave

    def stage_o():
        r_wo = rsrc(w_o)
        for t in range(start("o"), N_O, G):
            t = fx.Int32(t)
            rg = t * (O_ROWS // 16) + wave // GW
            kq = wave % GW
            wts = [
                shuffled_rows(r_wo, rg, O_K, kq * O_CPW + cc, lane, 128, CM_NT)
                for cc in range(O_CPW)
            ]
            ws = fx.Float32(
                bo.buffer_load(
                    rsrc(s_o), t * O_ROWS + tid % O_ROWS, vec_width=1, dtype=T.f32
                )
            )
            if const_expr(tokens == 1):
                word, x_scale = merge_token(0)
                x_scales = [x_scale]
                fx.ptr_store(word, xs + tid)
            else:
                got = mb_poll([(mr("attnq_s"), k, 1) for k in range(tokens)])
                stamp(3)
                x_scales = [got[k][0].bitcast(fx.Float32) for k in range(tokens)]
                acquire()
                rows_to_lds_async(tid, mb("attnq"), tokens, O_K // 4, xs, O_ROW)
            gpu.barrier()
            col_tok = fx.min(l16, tokens - 1)
            row_w = 1 if tokens == 1 else O_ROW
            c = fx.Vector.filled(8, 0.0, fx.Float32)
            for cc in range_constexpr(O_CPW):
                kc = kq * O_CPW + cc
                xb = _cat(
                    [
                        fx.Vector(
                            fx.ptr_load(
                                xs + (col_tok * row_w + 32 * kc + 8 * j + 4 * g4),
                                result_type=v4i,
                            )
                        )
                        for j in range(4)
                    ]
                )
                c = wmma_fp8_k128(wts[cc], xb, c)
            fx.ptr_store(c, red + (wave * WAVE + lane) * 8)
            gpu.barrier()
            if tid < O_ROWS * tokens:
                tk = tid // O_ROWS
                grp = (tid % O_ROWS) // 16
                rr = tid % 16
                tot = acc_row_sum(red, rr, tk, [grp * GW + w for w in range(GW)])
                fx.ptr_store(bf16_round(tot * pick(x_scales, tk) * ws), misc + tid)
            gpu.barrier()
            push_rows("attn", t * O_ROWS, O_ROWS, t % W if AG_RS else None)
            gpu.barrier()
            if const_expr(AG_RS):  # noqa: SIM102
                if (t % W == rank) & (tid < O_ROWS // 2 * tokens):
                    ptk = tid // (O_ROWS // 2)
                    row = t * O_ROWS + (tid % (O_ROWS // 2)) * 2
                    own = preg(sym, SY["attn"], "attn")
                    t0, t1 = sum_partials(
                        mb_poll,
                        own,
                        lambda src, k=ptk, r=row: (
                            ((src * MAX_TOKENS + k) * HIDDEN + r) // 2
                        ),
                        W,
                    )
                    for w in range_constexpr(W):
                        mb_put_bf(
                            preg(peer_addr(w), SY["a_ag"], "a_ag"),
                            ptk * HIDDEN + row,
                            [t0, t1],
                        )
            if const_expr(not AG_RS):  # noqa: SIM102
                if tid < O_ROWS // 2 * tokens:
                    ptk = tid // (O_ROWS // 2)
                    row = t * O_ROWS + (tid % (O_ROWS // 2)) * 2
                    # x8_in: h_in is the previous launch's
                    if const_expr(fuse_k1 and not x8_in):
                        k1_poll([(rdone_mb, ptk, 1)])
                        acquire()
                    hr = fx.Int32(
                        bo.buffer_load(
                            rsrc(h_in),
                            (ptk * HIDDEN + row) // 2,
                            vec_width=1,
                            dtype=T.i32,
                            cache_modifier=CM_K1,
                        )
                    )
                    own = preg(sym, SY["attn"], "attn")
                    t0, t1 = sum_partials(
                        mb_poll,
                        own,
                        lambda src, k=ptk, r=row: (
                            ((src * MAX_TOKENS + k) * HIDDEN + r) // 2
                        ),
                        W,
                    )
                    r0, r1 = bf2_f32(hr)
                    a0 = bf16_round(t0) + r0
                    a1 = bf16_round(t1) + r1
                    mb_put_bf(mr("a"), ptk * HIDDEN + row, [t0, t1])
                    bo.buffer_store(
                        fx.Vector.from_elements([a0, a1], fx.Float32).to(fx.BFloat16),
                        rsrc(h_mid),
                        ptk * HIDDEN + row,
                    )

    return {"stage_o": stage_o}
