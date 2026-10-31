# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K4 stages: the expert-grouped MoE, S > 4
(``kernels.stages.moe_wide`` on wave32 and the a8w4 WMMA).

An up / gate task is 16 mid rows (two GUGU row groups) of one distinct expert
for every token; a pair of them is a 32-row MXFP8 block of the down stage's mid
rows. A down row group runs on 8 waves, wave q taking 128-k chunk q of every
distinct expert, so which wave adds a chunk, and in what order, follows from
the expert alone.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.act import swiglu_mul_batch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.minimax_m3.mono.config import HIDDEN, INTER, MOE_SLOTS, N_ROUTED, SHARED_EXPERT, TOP_K
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.moe_common import DN_GW
from atom.models.minimax_m3.mono.gfx1250.layout import (
    DN_ROWS,
    MID_ROW,
    MIDSC_ROW,
    N_DN,
    N_ROUTER,
    SCORE_COPIES,
    UG_PER_SLOT,
    UG_ROWS,
    XN8_ROW,
    XSC_ROW,
)
from atom.models.minimax_m3.mono.kernels.stages.shared import (
    S2_BYTES,
    S13_BYTES,
    W2_BYTES,
    W13_BYTES,
    W13_ROWS,
)
from atom.mono.device.gfx1250 import (
    CM_DEV,
    CM_NT,
    THREADS,
    WAVE,
    WAVES,
    acc_row_sum,
    ballot,
    lanes_below,
    n32k4_word,
    publish,
    shuffled_rows,
    wmma_a8w4,
    xshfl,
)
from atom.mono.device.ops import bf16_round, row_live, rows_to_lds, rsrc, traced, uniform
from atom.mono.plan.trace import enter_stage

ROUTED_WAVES = N_ROUTED // WAVE  # waves holding one routed expert a lane


@traced
def moe_wide_defs(k4ctx):
    """``kernels.stages.moe_wide.moe_wide_defs`` on gfx1250."""
    BASE = k4ctx["BASE"]
    DN_DEPTH = k4ctx["DN_DEPTH"]
    DN_KC = k4ctx["DN_KC"]
    DN_WITERS = k4ctx["DN_WITERS"]
    G = k4ctx["G"]
    MIDSC_ROW_WORDS = k4ctx["MIDSC_ROW_WORDS"]
    MID_ROWS = k4ctx["MID_ROWS"]
    MID_ROW_WORDS = k4ctx["MID_ROW_WORDS"]
    ROUTE_ITERS = k4ctx["ROUTE_ITERS"]
    UGW_ITERS = k4ctx["UGW_ITERS"]
    UG_CPW = k4ctx["UG_CPW"]
    UG_PAIRS = k4ctx["UG_PAIRS"]
    U_MAX = k4ctx["U_MAX"]
    W_MIDSC = k4ctx["W_MIDSC"]
    W_SIG = k4ctx["W_SIG"]
    W_UPP = k4ctx["W_UPP"]
    XN8_WORDS = k4ctx["XN8_WORDS"]
    XSC_LDS = k4ctx["XSC_LDS"]
    XSC_WORDS = k4ctx["XSC_WORDS"]
    acquire = k4ctx["acquire"]
    batch_ids = k4ctx["batch_ids"]
    bid = k4ctx["bid"]
    ffn_finish = k4ctx["ffn_finish"]
    fpool = k4ctx["fpool"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    lds_b8 = k4ctx["lds_b8"]
    lds_scale4 = k4ctx["lds_scale4"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    misc = k4ctx["misc"]
    mr = k4ctx["mr"]
    mx8_q = k4ctx["mx8_q"]
    mx8_scale = k4ctx["mx8_scale"]
    poll_scores = k4ctx["poll_scores"]
    red = k4ctx["red"]
    route = k4ctx["route"]
    route_top4 = k4ctx["route_top4"]
    rwt = k4ctx["rwt"]
    s13 = k4ctx["s13"]
    s2 = k4ctx["s2"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    swiglu_limit = k4ctx["swiglu_limit"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    uexp = k4ctx["uexp"]
    umask = k4ctx["umask"]
    urow = k4ctx["urow"]
    uwt = k4ctx["uwt"]
    w13 = k4ctx["w13"]
    w2 = k4ctx["w2"]
    wave = k4ctx["wave"]
    xs = k4ctx["xs"]
    SKIP = k4ctx["SKIP"]
    NU_AT = U_MAX + 1  # uexp / umask word: the count of distinct routed experts
    CNT_AT = U_MAX + 2  # umask words: each routed-expert wave's count

    def slot_of(k, e):
        s = fx.Int32(TOP_K)
        for j in range_constexpr(TOP_K):
            s = (fx.Int32(fx.ptr_load(route + (8 * k + j))) == e).select(fx.Int32(j), s)
        return s

    def build_experts():
        """uexp / umask[0 .. n_u) := the distinct routed experts in expert order
        and their token masks, [n_u] := the shared expert, uexp[NU_AT] := n_u."""
        m = fx.Int32(0)
        if tid < N_ROUTED:
            for w in range_constexpr(WAVES):
                m = m | fx.Int32(fx.ptr_load(xs + (W_UPP + w * N_ROUTED + tid)))
        bal = ballot(m != 0)
        below = lanes_below(bal, lane)
        if (tid < N_ROUTED) & (lane == 0):
            fx.ptr_store(fx.Int32(fx.ctpop(bal)), umask + (CNT_AT + wave))
        gpu.barrier()
        if tid < N_ROUTED:
            off = fx.Int32(0)
            tot = fx.Int32(0)
            for w in range_constexpr(ROUTED_WAVES):
                cnt = fx.Int32(fx.ptr_load(umask + (CNT_AT + w)))
                off = off + (w < wave).select(cnt, fx.Int32(0))
                tot = tot + cnt
            rank_ = off + below
            if m != 0:
                fx.ptr_store(fx.Int32(tid), uexp + rank_)
                fx.ptr_store(m, umask + rank_)
            if tid == N_ROUTED - 1:
                fx.ptr_store(tot, uexp + NU_AT)
                fx.ptr_store(fx.Int32(SHARED_EXPERT), uexp + tot)
                fx.ptr_store(fx.Int32((1 << tokens) - 1), umask + tot)
        gpu.barrier()

    def wug_load(t, has):
        """Up / gate weights and scales of task t = (distinct expert t / 48, mid
        rows 16 (t % 48) ..): [row group 2 cg + h][chunk]."""
        u = t // UG_PER_SLOT
        cg = t % UG_PER_SLOT
        e_u = fx.max(fx.min(uniform(fx.ptr_load(uexp + u)), SHARED_EXPERT), 0)
        r_w = rsrc(w13 + fx.Int64(e_u) * W13_BYTES, has.select(fx.Int32(W13_BYTES), fx.Int32(0)))
        r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
        wts, scs = [], []
        for h in range_constexpr(2):
            rg = 2 * cg + h
            wts.append([])
            scs.append([])
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts[h].append(shuffled_rows(r_w, rg, HIDDEN // 2, kc, lane, 64, CM_NT))
                scs[h].append(n32k4_word(r_s, e_u * W13_ROWS + 16 * rg + l16, kc, HIDDEN))
        return wts, scs

    def wug_y(wts, scs):
        """A task's up / gate GEMV for every token -> swiglu: thread (mid row r,
        token k)'s value."""
        col = fx.min(l16, tokens - 1)
        acc = [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(2)]
        sbs = [lds_scale4(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4) for i in range(UG_CPW)]
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            xb = lds_b8(col * XN8_ROW + kc * 32)
            for h in range_constexpr(2):
                acc[h] = wmma_a8w4(wts[h][i], xb, acc[h], scs[h][i], sbs[i])
        fx.ptr_store(acc[0], red + (wave * WAVE + lane) * 8)
        fx.ptr_store(acc[1], fpool + (W_UPP + (wave * WAVE + lane) * 8))
        gpu.barrier()
        r = tid % UG_ROWS
        k = fx.min(tid // UG_ROWS, tokens - 1)
        p = r % 8
        first = r < 8
        g = first.select(acc_row_sum(red, 2 * p, k), acc_row_sum(fpool + W_UPP, 2 * p, k))
        u = first.select(acc_row_sum(red, 2 * p + 1, k), acc_row_sum(fpool + W_UPP, 2 * p + 1, k))
        gpu.barrier()
        return swiglu_mul_batch([g], [u], fx.Float32(-swiglu_limit))[0]

    def fp8_lanes4(q):
        lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, xshfl(q, 1), fx.Int32(0), False))
        return lo | (xshfl(lo, 2) << 16)

    def wug_publish(pr, y_lo, y_hi):
        u = pr // UG_PAIRS
        blk_ = pr % UG_PAIRS
        r = tid % UG_ROWS
        k = fx.min(tid // UG_ROWS, tokens - 1)
        lo = bf16_round(y_lo)
        hi = bf16_round(y_hi)
        sc, inv = mx8_scale([lo, hi], (1, 2, 4, 8))
        words = [fp8_lanes4(mx8_q(v, inv)) for v in (lo, hi)]
        e_u = fx.Int32(fx.ptr_load(uexp + u))
        m_u = fx.Int32(fx.ptr_load(umask + u))
        row = k * MOE_SLOTS + slot_of(k, e_u)
        if (tid < UG_ROWS * tokens) & (((m_u >> k) & 1) != 0):
            if r % 4 == 0:
                for h in range_constexpr(2):
                    bo.buffer_store(
                        words[h], rsrc(mb("mid8p")), row * MID_ROW_WORDS + blk_ * 8 + h * 4 + r // 4, cache_modifier=CM_DEV
                    )
            if r == 0:
                bo.buffer_store(sc, rsrc(mb("midsp")), row * MIDSC_ROW_WORDS + blk_, cache_modifier=CM_DEV)
        publish(mb_put, mr("ugdone"), pr, fx.Int32(1), tid == 0)

    def down_tables():
        nu = uniform(fx.ptr_load(uexp + NU_AT))
        for i in range_constexpr((U_MAX * 16 + THREADS - 1) // THREADS):
            ix = tid + THREADS * i
            if ix < U_MAX * 16:
                xu = ix // 16
                xk = ix % 16
                xe = fx.Int32(fx.ptr_load(uexp + fx.min(xu, nu)))
                xm = fx.Int32(fx.ptr_load(umask + fx.min(xu, nu)))
                xt = fx.min(xk, tokens - 1)
                xs_ = slot_of(xt, xe)
                xh = (((xm >> xk) & 1) != 0) & (xu <= nu)
                fx.ptr_store(xh.select(xt * MOE_SLOTS + xs_, fx.Int32(MID_ROWS)), urow + ix)
                fx.ptr_store(xh.select(fx.ptr_load(rwt + (8 * xt + xs_)), fx.Float32(0.0)), uwt + ix)

    def stage_moe_wide():
        td = start("down")
        has_dn = td < N_DN
        td = fx.min(td, N_DN - 1)
        mine = bid % SCORE_COPIES
        wave_mask = xs + (W_UPP + wave * N_ROUTED)
        for i in range_constexpr(N_ROUTED // WAVE):
            fx.ptr_store(fx.Int32(0), wave_mask + (lane + WAVE * i))
        wave_toks = [fx.min(wave + WAVES * j, tokens - 1) for j in range(ROUTE_ITERS)]
        scored = [poll_scores(k, mine) for k in wave_toks]
        for j in range_constexpr(ROUTE_ITERS):
            if wave + WAVES * j < tokens:
                route_top4(
                    wave_toks[j],
                    scored[j][0],
                    scored[j][1],
                    fpool + W_SIG,
                    wave_mask,
                    row_live(batch_ids, wave_toks[j]).select(fx.Int32(1) << wave_toks[j], fx.Int32(0)),
                )
        stamp(2)
        gpu.barrier()
        build_experts()
        stamp(6)
        if (bid == BASE["down"]) & (tid < MOE_SLOTS * tokens):
            sk = tid // MOE_SLOTS
            sj = tid % MOE_SLOTS
            mb_put(mr("sel"), sk * 2 * MOE_SLOTS + sj, fx.ptr_load(route + (8 * sk + sj)))
            mb_put(mr("sel"), sk * 2 * MOE_SLOTS + MOE_SLOTS + sj, fx.ptr_load(rwt + (8 * sk + sj)))
        n_pairs = (uniform(fx.ptr_load(uexp + NU_AT)) + 1) * UG_PAIRS

        def task(it):
            pr = fx.Int32(bid + it // 2 * G)
            return fx.min(pr, n_pairs - 1) * 2 + it % 2, pr < n_pairs

        t0, h0 = task(0)
        cur = wug_load(t0, h0)
        down_tables()
        stamp(7)
        n_rt = N_ROUTER * tokens
        if wave == 0:
            mb_poll([(mr("xndone"), fx.min(lane + WAVE * c, n_rt - 1), 1) for c in range((n_rt + WAVE - 1) // WAVE)])
        gpu.barrier()
        acquire()
        for base_w, n_row, lds_w, lds_row in (
            (mb("xn8p"), XN8_WORDS, 0, XN8_ROW),
            (mb("xscp"), XSC_WORDS, XSC_LDS, XSC_ROW),
        ):
            rows_to_lds(tid, base_w, tokens, n_row, xs + lds_w, lds_row)
        gpu.barrier()
        y_lo = fx.Float32(0.0)
        for it in range_constexpr(0 if "wug" in SKIP else UGW_ITERS):
            t, has_t = task(it)
            if const_expr(it + 1 < UGW_ITERS):
                tn, hn = task(it + 1)
                nxt = wug_load(tn, hn)
            if has_t:
                y = wug_y(cur[0], cur[1])
                if const_expr(it % 2 == 0):
                    y_lo = y
                elif const_expr("wpub" not in SKIP):
                    wug_publish(t // 2, y_lo, y)
            if const_expr(it + 1 < UGW_ITERS):
                cur = nxt
        stamp(8)
        if const_expr("wdown" not in SKIP):  # noqa: SIM102
            if has_dn:
                wide_down(td)
                stamp(11)

    def wide_down(td):
        """Row group td * 2 + wave // 8's down GEMV over every distinct expert
        (B column l % 16 = token l % 16's mid for it, the zero row where the token
        is not routed there), route-weighted per column -> ffn_finish."""
        enter_stage("down")
        nu = uniform(fx.ptr_load(uexp + NU_AT))
        rg_d = td * 2 + wave // DN_GW
        q = wave % DN_GW

        def unit_load(i):
            """(weights, scale, unit, ok) of this wave's chunk q of distinct expert i."""
            ok = (q < DN_KC) & (fx.Int32(i) <= nu)
            du = fx.Int32(min(i, U_MAX - 1))
            dkc = fx.min(q, DN_KC - 1)
            de = fx.max(fx.min(uniform(fx.ptr_load(uexp + du)), SHARED_EXPERT), 0)
            dwv = shuffled_rows(
                rsrc(w2 + fx.Int64(de) * W2_BYTES, ok.select(fx.Int32(W2_BYTES), fx.Int32(0))),
                rg_d,
                INTER // 2,
                dkc,
                lane,
                64,
                CM_NT,
            )
            dsc = n32k4_word(
                rsrc(s2, ok.select(fx.Int32(S2_BYTES), fx.Int32(0))), de * HIDDEN + 16 * rg_d + l16, dkc, INTER
            )
            return dwv, dsc, du, dkc, ok

        def with_rows(u):
            dwv, dsc, du, dkc, ok = u
            drow = fx.Int32(fx.ptr_load(urow + (du * 16 + l16)))
            dw = ok.select(fx.ptr_load(uwt + (du * 16 + l16)), fx.Float32(0.0))
            return dwv, dsc, drow, dkc, dw, ok

        acc = [fx.Float32(0.0) for _ in range(8)]
        inflight = [unit_load(i) for i in range(min(DN_DEPTH, DN_WITERS))]
        n_ug = (nu + 1) * UG_PAIRS
        mb_poll(
            [
                (mr("ugdone"), fx.min(tid + THREADS * i, n_ug - 1), 1)
                for i in range((U_MAX * UG_PAIRS + THREADS - 1) // THREADS)
            ]
        )
        gpu.barrier()
        acquire()
        for base_w, n_row, lds_w, lds_row in (
            (mb("mid8p"), MID_ROW_WORDS, 0, MID_ROW),
            (mb("midsp"), MIDSC_ROW_WORDS, W_MIDSC, MIDSC_ROW),
        ):
            rows_to_lds(tid, base_w, MID_ROWS, n_row, xs + lds_w, lds_row)
        if tid < MID_ROW_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (MID_ROWS * MID_ROW + tid))
        if tid < MIDSC_ROW_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (W_MIDSC + MID_ROWS * MIDSC_ROW + tid))
        gpu.barrier()
        stamp(10)
        for i in range_constexpr(DN_WITERS):
            if const_expr(i == 0):
                inflight = [with_rows(u) for u in inflight]
            dwv, dsc, drow, dkc, dw, ok = inflight.pop(0)
            if const_expr(i + DN_DEPTH < DN_WITERS):
                inflight.append(with_rows(unit_load(i + DN_DEPTH)))
            dcu = wmma_a8w4(
                dwv,
                lds_b8(drow * MID_ROW + dkc * 32),
                fx.Vector.filled(8, 0.0, fx.Float32),
                dsc,
                lds_scale4(W_MIDSC + drow * MIDSC_ROW + dkc * 4),
            )
            acc = [acc[e] + ok.select(dcu[e] * dw, fx.Float32(0.0)) for e in range(8)]
        fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * WAVE + lane) * 8)
        gpu.barrier()
        if tid < DN_ROWS * tokens:
            k = tid // DN_ROWS
            grp = (tid % DN_ROWS) // 16
            rr = tid % 16
            tot = acc_row_sum(red, rr, k, [grp * DN_GW + w for w in range(DN_GW)])
            fx.ptr_store(bf16_round(tot), misc + tid)
        gpu.barrier()
        ffn_finish(td)

    return {
        "slot_of": slot_of,
        "build_experts": build_experts,
        "wug_load": wug_load,
        "wug_y": wug_y,
        "fp8_lanes4": fp8_lanes4,
        "wug_publish": wug_publish,
        "down_tables": down_tables,
        "stage_moe_wide": stage_moe_wide,
        "wide_down": wide_down,
    }
