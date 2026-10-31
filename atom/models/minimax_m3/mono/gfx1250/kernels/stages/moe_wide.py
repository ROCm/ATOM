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

from atom.models.minimax_m3.mono.config import (
    HIDDEN,
    INTER,
    MOE_SLOTS,
    N_ROUTED,
    SHARED_EXPERT,
    TOP_K,
)
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.moe_common import DN_GW
from atom.models.minimax_m3.mono.gfx1250.layout import (
    ACC_WORDS,
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
    async_to_lds,
    ballot,
    lanes_below,
    lds_addr,
    lds_reads_done,
    n32k4_word,
    publish,
    release_stores,
    rows_to_lds_async,
    shuffled_rows,
    wait_async,
    wmma_a8w4,
    xshfl,
)
from atom.mono.device.ops import (
    batched_rounds,
    bf16_round,
    ld_dev,
    row_live,
    rsrc,
    traced,
    uniform,
)
from atom.mono.plan.trace import enter_stage

ROUTED_WAVES = N_ROUTED // WAVE  # waves holding one routed expert a lane
KQ = 4  # k quarters of an up / gate row group (``wide_ug``): WAVES // KQ row groups a block
QCH = HIDDEN // 128 // KQ  # 128-k chunks a k quarter
SLOT_WORDS = (WAVE * 32 + WAVE * 4) // 4  # a ring slot: a chunk's FP4 rows, its scales
TL_EXPERTS = 18  # timeline point (free on the map): the layer's distinct experts


@traced
def moe_wide_defs(k4ctx):
    """``kernels.stages.moe_wide.moe_wide_defs`` on gfx1250."""
    BASE = k4ctx["BASE"]
    DN_KC = k4ctx["DN_KC"]
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
    UGR = k4ctx["UGR"]
    UG_RING = k4ctx["UG_RING"]
    upart = k4ctx["upart"]
    ugring = k4ctx["ugring"]
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
        r_w = rsrc(
            w13 + fx.Int64(e_u) * W13_BYTES,
            has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
        )
        r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
        wts, scs = [], []
        for h in range_constexpr(2):
            rg = 2 * cg + h
            wts.append([])
            scs.append([])
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts[h].append(shuffled_rows(r_w, rg, HIDDEN // 2, kc, lane, 64, CM_NT))
                scs[h].append(
                    n32k4_word(r_s, e_u * W13_ROWS + 16 * rg + l16, kc, HIDDEN)
                )
        return wts, scs

    def wug_y(wts, scs):
        """A task's up / gate GEMV for every token -> swiglu: thread (mid row r,
        token k)'s value."""
        col = fx.min(l16, tokens - 1)
        acc = [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(2)]
        sbs = [
            lds_scale4(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4)
            for i in range(UG_CPW)
        ]
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
        g = first.select(
            acc_row_sum(red, 2 * p, k), acc_row_sum(fpool + W_UPP, 2 * p, k)
        )
        u = first.select(
            acc_row_sum(red, 2 * p + 1, k), acc_row_sum(fpool + W_UPP, 2 * p + 1, k)
        )
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
                        words[h],
                        rsrc(mb("mid8p")),
                        row * MID_ROW_WORDS + blk_ * 8 + h * 4 + r // 4,
                        cache_modifier=CM_DEV,
                    )
            if r == 0:
                bo.buffer_store(
                    sc,
                    rsrc(mb("midsp")),
                    row * MIDSC_ROW_WORDS + blk_,
                    cache_modifier=CM_DEV,
                )
        publish(mb_put, mr("ugdone"), pr, fx.Int32(1), tid == 0)

    def down_tables():
        """Distinct expert u's (mid row, route weight) of column k at entry
        16 u + k: the zero row, 0 where k is not routed there, past the
        distinct experts (entry U_MAX too)."""
        nu = uniform(fx.ptr_load(uexp + NU_AT))
        for i in range_constexpr(((U_MAX + 1) * 16 + THREADS - 1) // THREADS):
            ix = tid + THREADS * i
            if ix < (U_MAX + 1) * 16:
                xu = ix // 16
                xk = ix % 16
                xe = fx.Int32(fx.ptr_load(uexp + fx.min(xu, nu)))
                xm = fx.Int32(fx.ptr_load(umask + fx.min(xu, nu)))
                xt = fx.min(xk, tokens - 1)
                xs_ = slot_of(xt, xe)
                xh = (((xm >> xk) & 1) != 0) & (xu <= nu)
                fx.ptr_store(
                    xh.select(xt * MOE_SLOTS + xs_, fx.Int32(MID_ROWS)), urow + ix
                )
                fx.ptr_store(
                    xh.select(fx.ptr_load(rwt + (8 * xt + xs_)), fx.Float32(0.0)),
                    uwt + ix,
                )

    def wide_ug(n_pairs):
        """The up / gate tasks as streams: round r is block pr = bid + G r (32
        mid rows of one distinct expert, 4 GUGU row groups); wave w takes row
        group w // KQ of it over k quarter w % KQ (QCH 128-k chunks). A wave's
        rounds are one chunk stream through its UG_RING-slot LDS ring, filled by
        async copies (ASYNCcnt: the copies cannot be sunk to their use, so the
        ring's depth is real) UG_RING chunks ahead across round ends. A
        past-the-end round re-reads round 0's chunks (L2 hits) so every wait
        count stays a constant. -> the stream's state, its first chunks out."""
        q = wave // KQ
        kq = wave % KQ
        col = fx.min(l16, tokens - 1)
        ring = lds_addr(ugring)
        rounds = []
        for r in range_constexpr(UGR):
            pr = fx.Int32(bid + r * G)
            has = pr < n_pairs
            prc = fx.min(pr, n_pairs - 1)
            u = prc // UG_PAIRS
            e_u = fx.max(fx.min(uniform(fx.ptr_load(uexp + u)), SHARED_EXPERT), 0)
            g = 4 * (prc % UG_PAIRS) + q
            wbase = fx.Int64(w13) + fx.Int64(e_u) * W13_BYTES
            rows = e_u * W13_ROWS + 16 * g + l16
            rounds.append((has, prc, u, e_u, g, wbase, rows))

        def issue(f):
            r, c = divmod(f, QCH)
            has, _, _, _, g, wbase, rows = rounds[r]
            _, _, _, _, g0, wbase0, rows0 = rounds[0]
            g = has.select(g, g0)
            wb = has.select(wbase, wbase0)
            rr = has.select(rows, rows0)
            kc = kq * QCH + c
            slot = ring + ((wave * UG_RING + f % UG_RING) * SLOT_WORDS) * 4
            for piece in range_constexpr(2):
                async_to_lds(
                    wb
                    + fx.Int64((g * (HIDDEN // 64) + 2 * kc + piece) * 512 + lane * 16),
                    slot + lane * 32 + 16 * piece,
                    cm=CM_NT,
                )
            async_to_lds(
                fx.Int64(s13)
                + fx.Int64((rr // 32) * HIDDEN + kc * 128 + (rr % 32) * 4),
                slot + WAVE * 32 + lane * 4,
                b128=False,
                cm=CM_NT,
            )

        F = UGR * QCH
        assert UG_RING <= QCH
        if rounds[0][0]:
            for f in range_constexpr(UG_RING):
                issue(f)
        return rounds, F, issue, col, n_pairs

    def reduce_round(r, rounds):
        """Round r's block: the quarters summed in order, swiglu, the down stage's
        MXFP8 rows (thread (token tid / 32, block mid row lane): a wave is one
        token's 32 rows, the MXFP8 block; mid row m is pair m % 8 of row group
        m // 8), as ``wug_publish`` writes them."""
        has, prc, u, e_u, _, _, _ = rounds[r]
        k = tid // WAVE
        m = lane
        live_k = k < tokens
        kk = fx.min(k, tokens - 1)
        a_lane, a_slot = 16 * ((2 * (m % 8)) // 8) + kk, (2 * (m % 8)) % 8
        part = upart + (r % 2) * ACC_WORDS
        gs = fx.Float32(0.0)
        us = fx.Float32(0.0)
        for j in range_constexpr(KQ):
            slot0 = part + ((((m // 8) * KQ + j) * WAVE + a_lane) * 8 + a_slot)
            gs = gs + fx.ptr_load(slot0)
            us = us + fx.ptr_load(slot0 + 1)
        y = bf16_round(swiglu_mul_batch([gs], [us], fx.Float32(-swiglu_limit))[0])
        sc, inv = mx8_scale([y], (1, 2, 4, 8, 16))
        word = fp8_lanes4(mx8_q(y, inv))
        m_u = fx.Int32(fx.ptr_load(umask + u))
        row = kk * MOE_SLOTS + slot_of(kk, e_u)
        blk_ = prc % UG_PAIRS
        if has & live_k & (((m_u >> kk) & 1) != 0):
            if m % 4 == 0:
                bo.buffer_store(
                    word,
                    rsrc(mb("mid8p")),
                    row * MID_ROW_WORDS + blk_ * 8 + m // 4,
                    cache_modifier=CM_DEV,
                )
            if m == 0:
                bo.buffer_store(
                    sc,
                    rsrc(mb("midsp")),
                    row * MIDSC_ROW_WORDS + blk_,
                    cache_modifier=CM_DEV,
                )

    def wide_ug_run(state):
        """The rounds in order, a past-the-end one (and those after it) skipped:
        a chunk's wait counts the next round's copies only if it is live."""
        rounds, F, issue, col, n_pairs = state
        v8i = fx.Vector.make_type(8, fx.Int32)

        def ug_round(r):
            has_n = rounds[r + 1][0] if r + 1 < UGR else None
            acc = fx.Vector.filled(8, 0.0, fx.Float32)
            for c in range_constexpr(QCH):
                f = r * QCH + c
                n_same = min(UG_RING - 1, QCH - 1 - c)
                n_next = min(UG_RING - 1, F - 1 - f) - n_same
                if const_expr(n_next > 0):
                    if has_n:
                        wait_async(3 * (n_same + n_next))
                    else:
                        wait_async(3 * n_same)
                else:
                    wait_async(3 * n_same)
                sw = (wave * UG_RING + f % UG_RING) * SLOT_WORDS
                wv = fx.Vector(fx.ptr_load(ugring + (sw + lane * 8), result_type=v8i))
                sa = fx.Int32(fx.ptr_load(ugring + (sw + WAVE * 8 + lane)))
                kc = (wave % KQ) * QCH + c
                acc = wmma_a8w4(
                    wv,
                    lds_b8(col * XN8_ROW + kc * 32),
                    acc,
                    sa,
                    lds_scale4(XSC_LDS + col * XSC_ROW + kc * 4),
                )
                if const_expr(f + UG_RING < F):
                    lds_reads_done()  # chunk f's slot, refilled next
                    if const_expr((f + UG_RING) // QCH == r):
                        issue(f + UG_RING)
                    else:
                        if has_n:
                            issue(f + UG_RING)
            fx.ptr_store(acc, upart + ((r % 2) * ACC_WORDS + (wave * WAVE + lane) * 8))
            gpu.barrier()
            reduce_round(r, rounds)

        for r in range_constexpr(UGR):
            if rounds[r][0]:
                ug_round(r)
        stamp(9)
        release_stores()
        gpu.barrier()
        if tid < UGR:
            pr = bid + tid * G
            if pr < n_pairs:
                mb_put(mr("ugdone"), pr, fx.Int32(1))

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
                    row_live(batch_ids, wave_toks[j]).select(
                        fx.Int32(1) << wave_toks[j], fx.Int32(0)
                    ),
                )
        stamp(2)
        gpu.barrier()
        build_experts()
        stamp(6)
        # timeline point TL_EXPERTS: a count, not a time
        if const_expr(k4ctx["timeline"]):  # noqa: SIM102
            if tid == 0:
                fx.ptr_store(
                    fx.Int64(uniform(fx.ptr_load(uexp + NU_AT)) + 1),
                    k4ctx["tls"] + TL_EXPERTS,
                )
        if (bid == BASE["down"]) & (tid < MOE_SLOTS * tokens):
            sk = tid // MOE_SLOTS
            sj = tid % MOE_SLOTS
            mb_put(
                mr("sel"), sk * 2 * MOE_SLOTS + sj, fx.ptr_load(route + (8 * sk + sj))
            )
            mb_put(
                mr("sel"),
                sk * 2 * MOE_SLOTS + MOE_SLOTS + sj,
                fx.ptr_load(rwt + (8 * sk + sj)),
            )
        n_pairs = (uniform(fx.ptr_load(uexp + NU_AT)) + 1) * UG_PAIRS

        def task(it):
            pr = fx.Int32(bid + it // 2 * G)
            return fx.min(pr, n_pairs - 1) * 2 + it % 2, pr < n_pairs

        if const_expr("ugold" in SKIP):
            t0, h0 = task(0)
            cur = wug_load(t0, h0)
        else:
            ug_state = wide_ug(n_pairs)  # the first chunks' loads out
        down_tables()
        stamp(7)
        n_rt = N_ROUTER * tokens
        if wave == 0:
            mb_poll(
                [
                    (mr("xndone"), fx.min(lane + WAVE * c, n_rt - 1), 1)
                    for c in range((n_rt + WAVE - 1) // WAVE)
                ]
            )
        gpu.barrier()
        acquire()
        for base_w, n_row, lds_w, lds_row in (
            (mb("xn8p"), XN8_WORDS, 0, XN8_ROW),
            (mb("xscp"), XSC_WORDS, XSC_LDS, XSC_ROW),
        ):
            rows_to_lds_async(tid, base_w, tokens, n_row, xs + lds_w, lds_row)
        gpu.barrier()
        if const_expr("ugold" in SKIP):
            y_lo = fx.Float32(0.0)
            for it in range_constexpr(UGW_ITERS):
                t, has_t = task(it)
                if const_expr(it + 1 < UGW_ITERS):
                    tn, hn = task(it + 1)
                    nxt = wug_load(tn, hn)
                if has_t:
                    y = wug_y(cur[0], cur[1])
                    if const_expr(it % 2 == 0):
                        y_lo = y
                    else:
                        wug_publish(t // 2, y_lo, y)
                if const_expr(it + 1 < UGW_ITERS):
                    cur = nxt
        else:
            wide_ug_run(ug_state)
        stamp(8)
        if const_expr("wdown" not in SKIP):  # noqa: SIM102
            if has_dn:
                wide_down(td)
                stamp(11)

    def wide_down(td):
        """Row group td * 2 + wave // 8's down GEMV over every distinct expert
        (B column l % 16 = token l % 16's mid for it, the zero row where the token
        is not routed there), route-weighted per column -> ffn_finish. The row
        group's (expert, 128-k chunk) units are dealt round robin to its 8 waves
        and streamed through their up / gate rings (free by now), the first
        before the mids are in; a unit past the distinct experts re-reads unit
        0's chunk (L2 hits) so the wait counts stay constant. A unit's expert
        comes from a lane of ``eids`` (no LDS round trip before its copy), and
        its mid row lookup and B operand are read 2 and 1 units ahead."""
        enter_stage("down")
        nu = uniform(fx.ptr_load(uexp + NU_AT))
        rg_d = td * 2 + wave // DN_GW
        q = wave % DN_GW
        ring = lds_addr(ugring)
        NJ = DN_KC * U_MAX
        NT = (NJ + DN_GW - 1) // DN_GW

        def unit(t):
            j = q + DN_GW * t
            i = j // DN_KC
            ok = (j < NJ) & (i <= nu)
            return ok, ok.select(i, fx.Int32(0)), ok.select(j % DN_KC, q % DN_KC)

        # lane l of eids[c]: unit 32 c + l's expert, past this wave's units too
        eids = []
        for c in range_constexpr((NT + UG_RING + WAVE - 1) // WAVE):
            _, i_l, _ = unit(lane + WAVE * c)
            eids.append(
                fx.max(fx.min(fx.Int32(fx.ptr_load(uexp + i_l)), SHARED_EXPERT), 0)
            )

        def issue(t):
            _, _, kc = unit(t)
            if const_expr(isinstance(t, int)):
                src = eids[t // WAVE]
            else:
                src = eids[-1]
                for c in range_constexpr(len(eids) - 2, -1, -1):
                    src = (t < WAVE * (c + 1)).select(eids[c], src)
            de = fx.Int32(rocdl.readlane(T.i32, src, t % WAVE))
            wb = fx.Int64(w2) + fx.Int64(de) * W2_BYTES
            slot = ring + ((wave * UG_RING + t % UG_RING) * SLOT_WORDS) * 4
            for piece in range_constexpr(2):
                async_to_lds(
                    wb
                    + fx.Int64(
                        (rg_d * (INTER // 64) + 2 * kc + piece) * 512 + lane * 16
                    ),
                    slot + lane * 32 + 16 * piece,
                    cm=CM_NT,
                )
            rr = de * HIDDEN + 16 * rg_d + l16
            async_to_lds(
                fx.Int64(s2) + fx.Int64((rr // 32) * INTER + kc * 128 + (rr % 32) * 4),
                slot + WAVE * 32 + lane * 4,
                b128=False,
                cm=CM_NT,
            )

        def lookup(t):
            j = q + DN_GW * t
            e = fx.min(j // DN_KC, U_MAX) * 16 + l16
            return fx.Int32(fx.ptr_load(urow + e)), fx.ptr_load(uwt + e), j % DN_KC

        def operand(lk):
            drow, w, kc = lk
            bsc = fx.Int32(fx.ptr_load(xs + (W_MIDSC + drow * MIDSC_ROW + kc)))
            return lds_b8(drow * MID_ROW + kc * 32), bsc, w

        for t in range_constexpr(UG_RING):
            issue(t)
        n_ug = (nu + 1) * UG_PAIRS
        mb_poll(
            [
                (mr("ugdone"), fx.min(tid + THREADS * i, n_ug - 1), 1)
                for i in range((U_MAX * UG_PAIRS + THREADS - 1) // THREADS)
            ]
        )
        gpu.barrier()
        acquire()
        # the mid rows: async copies, all out before the scales' loads
        P16 = MID_ROW_WORDS // 4
        xs_b = lds_addr(xs)
        for i in range_constexpr((MID_ROWS * P16 + THREADS - 1) // THREADS):
            v = tid + THREADS * i
            if v < MID_ROWS * P16:
                async_to_lds(
                    fx.Int64(mb("mid8p")) + fx.Int64(v * 16),
                    xs_b + (v // P16 * MID_ROW + v % P16 * 4) * 4,
                    cm=CM_DEV,
                )
        # the mid scales packed as WMMA scale words: word kc of a row, byte j
        # the code of its 32-block 4 kc + j
        NPK = MIDSC_ROW_WORDS // 4
        batched_rounds(
            tid,
            MID_ROWS * NPK,
            lambda v: fx.Vector(ld_dev(mb("midsp"), v * 4, 4)),
            lambda v, c: fx.ptr_store(
                (c[0] & 0xFF)
                | ((c[1] & 0xFF) << 8)
                | ((c[2] & 0xFF) << 16)
                | (c[3] << 24),
                xs + (W_MIDSC + v // NPK * MIDSC_ROW + v % NPK),
            ),
        )
        if tid < MID_ROW_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (MID_ROWS * MID_ROW + tid))
        if tid < NPK:
            fx.ptr_store(fx.Int32(0), xs + (W_MIDSC + MID_ROWS * MIDSC_ROW + tid))
        wait_async(0)
        gpu.barrier()
        stamp(10)
        v8i = fx.Vector.make_type(8, fx.Int32)
        # this wave's live units, a prefix: q + 8 t < 6 (nu + 1)
        n_w = (DN_KC * (nu + 1) - q + DN_GW - 1) // DN_GW

        def ring_read(t):
            sw = (wave * UG_RING + t % UG_RING) * SLOT_WORDS
            return (
                fx.Vector(fx.ptr_load(ugring + (sw + lane * 8), result_type=v8i)),
                fx.Int32(fx.ptr_load(ugring + (sw + WAVE * 8 + lane))),
            )

        # iteration t: unit t + UG_RING's copy out into unit t's slot, read by
        # iteration t - 1 (past the live units too, a stand-in chunk, so every
        # wait is the same); unit t + 1's B operand read, unit t + 2's lookup,
        # unit t + 1's copy waited for and read; unit t's WMMA on what iteration
        # t - 1 read
        op = operand(lookup(0))
        lk = lookup(1)
        wait_async(3 * (UG_RING - 1))
        a, sa = ring_read(0)
        acc = [fx.Float32(0.0) for _ in range(8)]
        init = acc + [a, sa, op[0], op[1], op[2], lk[0], lk[1], lk[2]]
        fin = init
        for t_, st in range(0, n_w, 1, init=init):
            t = fx.Int32(t_)
            acc = [fx.Float32(v) for v in st[0:8]]
            a, sa = fx.Vector(st[8]), fx.Int32(st[9])
            b, bsc, dw = fx.Vector(st[10]), fx.Int32(st[11]), fx.Float32(st[12])
            lk = (fx.Int32(st[13]), fx.Float32(st[14]), fx.Int32(st[15]))
            lds_reads_done()
            issue(t + UG_RING)
            op_n = operand(lk)
            lk_n = lookup(t + 2)
            wait_async(3 * (UG_RING - 1))
            a_n, sa_n = ring_read(t + 1)
            dcu = wmma_a8w4(a, b, fx.Vector.filled(8, 0.0, fx.Float32), sa, bsc)
            acc = [acc[e] + dcu[e] * dw for e in range(8)]
            fin = yield acc + [
                a_n,
                sa_n,
                op_n[0],
                op_n[1],
                op_n[2],
                lk_n[0],
                lk_n[1],
                lk_n[2],
            ]
        acc = [fx.Float32(v) for v in fin[0:8]]
        stamp(19)
        fx.ptr_store(
            fx.Vector.from_elements(acc, fx.Float32), red + (wave * WAVE + lane) * 8
        )
        gpu.barrier()
        if tid < DN_ROWS * tokens:
            k = tid // DN_ROWS
            grp = (tid % DN_ROWS) // 16
            rr = tid % 16
            tot = acc_row_sum(red, rr, k, [grp * DN_GW + w for w in range(DN_GW)])
            fx.ptr_store(bf16_round(tot), misc + tid)
        gpu.barrier()
        stamp(20)
        ffn_finish(td)
        wait_async(0)  # the stand-in copies, before the CTA's LDS goes

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
