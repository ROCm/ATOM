# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 stages: the expert-grouped MoE (S > 4)."""

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
    TOKEN_TILE,
    token_mask_i32,
    token_tiles,
)
from atom.models.minimax_m3.mono.kernels.stages.shared import (
    S2_BYTES,
    S13_BYTES,
    W2_BYTES,
    W2_SCALE_COLS,
    W13_BYTES,
    W13_ROWS,
    W13_SCALE_COLS,
)
from atom.models.minimax_m3.mono.layout import (
    DN_ROWS,
    MID_ROW,
    MIDSC_ROW,
    N_DN,
    N_ROUTER,
    SCORE_COPIES,
    SH_TASKS,
    UG_PER_SLOT,
    UG_ROWS,
    XN8_ROW,
    XSC_ROW,
)
from atom.mono.device.mx import FP4, FP8, fp4_tile_load, mfma_scaled
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    ballot,
    bf16_round,
    fp8_pack4,
    lanes_below,
    popcount,
    row_live,
    row_sum,
    rows_to_lds,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import THREADS, WAVES
from atom.mono.plan.trace import enter_stage


@traced
def _down_expert_batches(n_exp, depth, load, with_rows, consume, acc):
    per_batch = depth // 2
    for rnd in range(0, (n_exp + per_batch - 1) // per_batch, 1):
        first = rnd * per_batch
        inflight = [
            with_rows(load(first + i // 2, i % 2))
            for i in range_constexpr(depth)
        ]
        for i in range_constexpr(depth):
            acc = consume(acc, inflight[i])
    return acc


@traced
def moe_wide_defs(k4ctx):
    """The functions of the expert-grouped MoE (S > 4)."""
    BASE = k4ctx["BASE"]
    DN_DEPTH = k4ctx["DN_DEPTH"]
    DN_KC = k4ctx["DN_KC"]
    DN_WITERS = k4ctx["DN_WITERS"]
    G = k4ctx["G"]
    MIDSC_ROW_WORDS = k4ctx["MIDSC_ROW_WORDS"]
    MID_ROWS = k4ctx["MID_ROWS"]
    MID_ROW_WORDS = k4ctx["MID_ROW_WORDS"]
    MULTI_TILE = k4ctx["MULTI_TILE"]
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
    e8m0_load = k4ctx["e8m0_load"]
    ffn_finish = k4ctx["ffn_finish"]
    fpool = k4ctx["fpool"]
    g4 = k4ctx["g4"]
    group_scale_byte = k4ctx["group_scale_byte"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    lds_b8 = k4ctx["lds_b8"]
    lds_i32 = k4ctx["lds_i32"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    misc = k4ctx["misc"]
    mr = k4ctx["mr"]
    mx8_q = k4ctx["mx8_q"]
    mx8_scale = k4ctx["mx8_scale"]
    shared_fp8 = k4ctx.get("shared_fp8", False)
    sh_compute = k4ctx.get("sh_compute")
    sh_load = k4ctx.get("sh_load")
    down_load_shared = k4ctx.get("down_load_shared")
    red = k4ctx["red"]
    route = k4ctx["route"]
    route_top4 = k4ctx["route_top4"]
    rwt = k4ctx["rwt"]
    s13 = k4ctx["s13"]
    s2 = k4ctx["s2"]
    scores_copy = k4ctx["scores_copy"]
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

    def slot_of(k, e):
        """Token k's slot routed to expert e (TOP_K: the shared expert)."""
        s = fx.Int32(TOP_K)
        for j in range_constexpr(TOP_K):
            s = (fx.Int32(fx.ptr_load(route + (8 * k + j))) == e).select(fx.Int32(j), s)
        return s

    def build_experts():
        """uexp / umask[0 .. n_u) := the tokens' distinct routed experts in
        expert order and their token masks, [n_u] := the shared expert (every
        token), uexp[U_MAX + 1] := n_u. The waves' masks must be in (synced)."""
        m = fx.Int32(0)
        if tid < N_ROUTED:
            for w in range_constexpr(WAVES):
                m = m | fx.Int32(fx.ptr_load(xs + (W_UPP + w * N_ROUTED + tid)))
        bal = ballot(m != 0)
        below = lanes_below(bal, lane)
        if tid == 0:
            fx.ptr_store(popcount(bal), umask + (U_MAX + 1))
        gpu.barrier()
        if tid < N_ROUTED:
            off = (wave == 0).select(
                fx.Int32(0), fx.Int32(fx.ptr_load(umask + (U_MAX + 1)))
            )
            rank_ = off + below
            if m != 0:
                fx.ptr_store(fx.Int32(tid), uexp + rank_)
                fx.ptr_store(m, umask + rank_)
            if tid == N_ROUTED - 1:
                nu_ = off + popcount(bal)
                fx.ptr_store(nu_, uexp + (U_MAX + 1))
                if const_expr(not shared_fp8):
                    fx.ptr_store(fx.Int32(SHARED_EXPERT), uexp + nu_)
                    fx.ptr_store(fx.Int32(token_mask_i32(tokens)), umask + nu_)
                else:
                    fx.ptr_store(fx.Int32(0), uexp + nu_)
                    fx.ptr_store(fx.Int32(0), umask + nu_)
        gpu.barrier()

    def wug_load(t, has, scales=True):
        """Up / gate weights (and ``scales``: the scale dwords, which hold the
        pair's both halves) of task t = (distinct expert t / 48, row group
        t % 48); without the task, reads go through zero-sized buffers."""
        u = t // UG_PER_SLOT
        cg = t % UG_PER_SLOT
        e_u = fx.min(uniform(fx.ptr_load(uexp + u)), SHARED_EXPERT)
        r_w = rsrc(
            w13 + fx.Int64(e_u) * W13_BYTES,
            has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
        )
        r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
        wts = []
        scs = []
        for gu in range_constexpr(2):
            rg = gu * UG_PER_SLOT + cg
            wts.append([])
            scs.append([])
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts[gu].append(
                    fp4_tile_load(
                        r_w,
                        0,
                        rg,
                        HIDDEN,
                        kc,
                        lane * 4,
                        cache_modifier=0 if MULTI_TILE else CM_NT,
                    )
                )
                if const_expr(scales and i % 2 == 0):
                    # the dword holding this lane's scales of chunks kc, kc + 1
                    # of both 16-row halves of the 32-row block (bytes 2 d + a)
                    scs[gu].append(
                        fx.Int32(
                            bo.buffer_load(
                                r_s,
                                group_scale_byte(e_u, W13_ROWS, rg, kc, W13_SCALE_COLS)
                                // 4,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                    )
        return wts, scs

    def wug_y(wts, scs, half, n):
        """A task's up / gate GEMV for every token (B column l % 16 = token
        l % 16) -> swiglu: thread (row r, token k)'s mid value. ``half``: the
        task's 16-row half of its 32-row scale block (compile time)."""
        col = fx.min(l16, n - 1)
        cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
        cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
        sbs = [
            lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
            for i in range(UG_CPW)
        ]
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            xb = lds_b8(col * XN8_ROW + kc * 32)
            sel = (i % 2) * 2 + half
            cg_ = mfma_scaled(
                wts[0][i], xb, cg_, scs[0][i // 2], sbs[i], a_fmt=FP4, sa_byte=sel
            )
            cu_ = mfma_scaled(
                wts[1][i], xb, cu_, scs[1][i // 2], sbs[i], a_fmt=FP4, sa_byte=sel
            )
        fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
        fx.ptr_store(cu_, fpool + (W_UPP + (wave * 64 + lane) * 4))
        gpu.barrier()
        r = tid % UG_ROWS
        k = fx.min(tid // UG_ROWS, n - 1)
        gsum = row_sum(red, r, k)
        usum = row_sum(fpool + W_UPP, r, k)
        gpu.barrier()  # red and the up partials are free for the next task
        return swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]

    def fp8_lanes4(q):
        """Lane r (r % 4 == 0): the E4M3 dword of lanes r .. r + 3's values."""
        lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, xshfl(q, 1), fx.Int32(0), False))
        return lo | (xshfl(lo, 2) << 16)

    def wug_publish(pr, y_lo, y_hi, t0, n):
        """Task pair pr = (distinct expert pr / UG_PAIRS, 32-row block
        pr % UG_PAIRS): its mid rows (thread (r, k) holds rows r and 16 + r of
        the block) as the down stage's MXFP8 -- the rounding its bf16 staging
        did, one 32-block a 16-lane group -- then the pair's flag."""
        u = pr // UG_PAIRS
        blk_ = pr % UG_PAIRS
        r = tid % UG_ROWS
        local_k = fx.min(tid // UG_ROWS, n - 1)
        k = t0 + local_k
        lo = bf16_round(y_lo)
        hi = bf16_round(y_hi)
        sc, inv = mx8_scale([lo, hi], (1, 2, 4, 8))
        words = [fp8_lanes4(mx8_q(v, inv)) for v in (lo, hi)]
        e_u = fx.Int32(fx.ptr_load(uexp + u))
        m_u = fx.Int32(fx.ptr_load(umask + u))
        row = k * MOE_SLOTS + slot_of(k, e_u)
        if (tid < UG_ROWS * n) & (((m_u >> k) & 1) != 0):
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

    def down_tables(t0, n):
        """urow / uwt := per (distinct expert, B column) the down stage's mid row
        (the zero row where the column's token is not routed there) and route
        weight. Routing only: built while the first up / gate weights land."""
        nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
        n_exp = nu + (0 if shared_fp8 else 1)
        for i in range_constexpr((U_MAX * 16 + THREADS - 1) // THREADS):
            ix = tid + THREADS * i
            if ix < U_MAX * 16:
                xu = ix // 16
                xk = ix % 16
                xe = fx.Int32(fx.ptr_load(uexp + fx.min(xu, nu)))
                xm = fx.Int32(fx.ptr_load(umask + fx.min(xu, nu)))
                local_t = fx.min(xk, n - 1)
                xt = t0 + local_t
                xs_ = slot_of(xt, xe)
                xh = (xk < n) & (((xm >> xt) & 1) != 0) & (xu < n_exp)
                fx.ptr_store(
                    xh.select(local_t * MOE_SLOTS + xs_, fx.Int32(n * MOE_SLOTS)),
                    urow + ix,
                )
                fx.ptr_store(
                    xh.select(fx.ptr_load(rwt + (8 * xt + xs_)), fx.Float32(0.0)),
                    uwt + ix,
                )

    def stage_moe_wide():
        td = start("down")  # this CTA's down task (the split tasks come first)
        has_dn = td < N_DN
        td = fx.min(td, N_DN - 1)
        if const_expr(shared_fp8):
            has_sh = start("shared") < SH_TASKS
            sh_j = fx.min(start("shared"), SH_TASKS - 1)
            sh_w = sh_load(sh_j, has_sh)
        # every token's routing: wave w routes tokens w, w + WAVES, .., their
        # scores polled in one batch
        mine = bid % SCORE_COPIES
        # wave w's expert -> mask of the tokens it routed there, in the up
        # partials' words (free until the first up / gate task)
        wave_mask = xs + (W_UPP + wave * N_ROUTED)
        fx.ptr_store(fx.Int32(0), wave_mask + lane)
        fx.ptr_store(fx.Int32(0), wave_mask + (lane + 64))
        wave_toks = [fx.min(wave + WAVES * j, tokens - 1) for j in range(ROUTE_ITERS)]
        sg = mb_poll(
            [
                (scores_copy(mine, k), (lane + 64 * h) * 2, 2)
                for k in wave_toks
                for h in range(2)
            ]
        )
        for j in range_constexpr(ROUTE_ITERS):
            if wave + WAVES * j < tokens:
                route_top4(
                    wave_toks[j],
                    sg[2 * j][0],
                    sg[2 * j][1].bitcast(fx.Float32),
                    sg[2 * j + 1][0],
                    sg[2 * j + 1][1].bitcast(fx.Float32),
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
        if (bid == BASE["down"]) & (tid < MOE_SLOTS * tokens):
            sk = tid // MOE_SLOTS
            sj = tid % MOE_SLOTS
            mb_put(
                mr("sel"),
                sk * 2 * MOE_SLOTS + sj,
                fx.ptr_load(route + (8 * sk + sj)),
            )
            mb_put(
                mr("sel"),
                sk * 2 * MOE_SLOTS + MOE_SLOTS + sj,
                fx.ptr_load(rwt + (8 * sk + sj)),
            )
        n_pairs = (
            uniform(fx.ptr_load(uexp + (U_MAX + 1))) + (0 if shared_fp8 else 1)
        ) * UG_PAIRS

        def task(it):
            """Iteration it's task: pair bid + (it / 2) G's half it % 2."""
            pr = fx.Int32(bid + it // 2 * G)
            return fx.min(pr, fx.max(n_pairs, 1) - 1) * 2 + it % 2, pr < n_pairs

        # Every router slice is global before a task stages one token tile.
        n_rt = N_ROUTER * tokens
        if wave == 0:
            mb_poll(
                [
                    (mr("xndone"), fx.min(lane + 64 * c, n_rt - 1), 1)
                    for c in range((n_rt + 63) // 64)
                ]
            )
        gpu.barrier()
        acquire()

        if const_expr(not MULTI_TILE):
            t0, h0 = task(0)
            cur = wug_load(t0, h0)
            down_tables(0, tokens)
            for base_w, n_row, lds_w, lds_row in (
                (mb("xn8p"), XN8_WORDS, 0, XN8_ROW),
                (mb("xscp"), XSC_WORDS, XSC_LDS, XSC_ROW),
            ):
                rows_to_lds(tid, base_w, tokens, n_row, xs + lds_w, lds_row)
            gpu.barrier()
            if const_expr(shared_fp8):  # noqa: SIM102
                if has_sh:
                    sh_compute(sh_j, sh_w[0], sh_w[1], 0, tokens)
            stamp(7)
            y_lo = fx.Float32(0.0)
            for it in range_constexpr(UGW_ITERS):
                t, has_t = task(it)
                if const_expr(it + 1 < UGW_ITERS):
                    tn, hn = task(it + 1)
                    nxt = wug_load(tn, hn, scales=(it + 1) % 2 == 0)
                if const_expr(it % 2 == 0):
                    pair_scs = cur[1]
                if has_t:
                    y = wug_y(cur[0], pair_scs, it % 2, tokens)
                    if const_expr(it % 2 == 0):
                        y_lo = y
                    else:
                        wug_publish(t // 2, y_lo, y, 0, tokens)
                        publish(mb_put, mr("ugdone"), t // 2, fx.Int32(1), tid == 0)
                if const_expr(it + 1 < UGW_ITERS):
                    cur = nxt
        else:
            def load_x_tile(t0, n):
                rows_to_lds(
                    tid,
                    mb("xn8p") + fx.Int64(t0 * XN8_WORDS * 4),
                    n,
                    XN8_WORDS,
                    xs,
                    XN8_ROW,
                )
                rows_to_lds(
                    tid,
                    mb("xscp") + fx.Int64(t0 * XSC_WORDS * 4),
                    n,
                    XSC_WORDS,
                    xs + XSC_LDS,
                    XSC_ROW,
                )
                gpu.barrier()

            if const_expr(shared_fp8):  # noqa: SIM102
                if has_sh:
                    for tile0, tile_n in token_tiles(tokens):
                        load_x_tile(tile0, tile_n)
                        sh_compute(sh_j, sh_w[0], sh_w[1], tile0, tile_n)
            stamp(7)
            tiles = token_tiles(tokens)
            for tile_ix, (tile0, tile_n) in enumerate(tiles):
                load_x_tile(tile0, tile_n)
                for it in range_constexpr(0, UGW_ITERS, 2):
                    lo, has_pair = task(it)
                    y_lo = fx.Float32(0.0)
                    y_hi = fx.Float32(0.0)
                    if has_pair:
                        low = wug_load(lo, has_pair)
                        y_lo = wug_y(low[0], low[1], 0, tile_n)
                        hi, _ = task(it + 1)
                        high = wug_load(hi, has_pair, scales=False)
                        y_hi = wug_y(high[0], low[1], 1, tile_n)
                        wug_publish(lo // 2, y_lo, y_hi, tile0, tile_n)
                    if const_expr(tile_ix + 1 == len(tiles)):
                        publish(
                            mb_put,
                            mr("ugdone"),
                            lo // 2,
                            fx.Int32(1),
                            has_pair & (tid == 0),
                        )
        stamp(8)
        if has_dn:
            (wide_down_tiled if MULTI_TILE else wide_down_legacy)(td)
            stamp(11)

    def wide_down_legacy(td):
        """The original one-tile down path for S <= TOKEN_TILE."""
        enter_stage("down")
        nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
        n_exp = nu + (0 if shared_fp8 else 1)
        rg_d = td * 2 + wave // 4
        q = wave % 4

        def unit_load(i):
            dkc = q + 4 * (i % 2)
            ok = (dkc < DN_KC) & (fx.Int32(i // 2) < n_exp)
            du = fx.Int32(min(i // 2, U_MAX - 1))
            dkc = fx.min(dkc, DN_KC - 1)
            de = fx.min(uniform(fx.ptr_load(uexp + du)), SHARED_EXPERT)
            dwv = fp4_tile_load(
                rsrc(
                    w2 + fx.Int64(de) * W2_BYTES,
                    ok.select(fx.Int32(W2_BYTES), fx.Int32(0)),
                ),
                0,
                rg_d,
                INTER,
                dkc,
                lane * 4,
            )
            dsc = e8m0_load(
                rsrc(s2, ok.select(fx.Int32(S2_BYTES), fx.Int32(0))),
                group_scale_byte(de, HIDDEN, rg_d, dkc, W2_SCALE_COLS),
            )
            return dwv, dsc, du, dkc, ok

        def with_rows(u):
            dwv, dsc, du, dkc, ok = u
            drow = fx.Int32(fx.ptr_load(urow + (du * TOKEN_TILE + l16)))
            dw = ok.select(
                fx.ptr_load(uwt + (du * TOKEN_TILE + l16)), fx.Float32(0.0)
            )
            return dwv, dsc, drow, dkc, dw, ok

        acc = [fx.Float32(0.0) for _ in range(4)]
        inflight = [unit_load(i) for i in range(min(DN_DEPTH, DN_WITERS))]
        n_ug = n_exp * UG_PAIRS
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
        if const_expr(shared_fp8):
            for k in range_constexpr(tokens):
                mids = mb_poll(
                    [
                        (
                            mr("mid"),
                            (k * MOE_SLOTS + TOP_K) * INTER
                            + fx.min(tid * 8 + i * 2, INTER - 2),
                            2,
                        )
                        for i in range(4)
                    ]
                )
                if tid < INTER // 8:
                    f = [v.bitcast(fx.Float32) for pair in mids for v in pair]
                    sc, inv = mx8_scale(f, (1, 2))
                    qv = [mx8_q(v, inv) for v in f]
                    row = k * MOE_SLOTS + TOP_K
                    fx.ptr_store(fp8_pack4(*qv[:4]), xs + (row * MID_ROW + tid * 2))
                    fx.ptr_store(
                        fp8_pack4(*qv[4:]), xs + (row * MID_ROW + tid * 2 + 1)
                    )
                    if tid % 4 == 0:
                        fx.ptr_store(
                            sc, xs + (W_MIDSC + row * MIDSC_ROW + tid // 4)
                        )
        if tid < MID_ROW_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (MID_ROWS * MID_ROW + tid))
        if tid < MIDSC_ROW_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (W_MIDSC + MID_ROWS * MIDSC_ROW + tid))
        gpu.barrier()
        stamp(10)
        if const_expr(shared_fp8):
            sh_units = down_load_shared(td, fx.Int32(1))
        for i in range_constexpr(DN_WITERS):
            if const_expr(i == 0):
                inflight = [with_rows(u) for u in inflight]
            dwv, dsc, drow, dkc, dw, ok = inflight.pop(0)
            if const_expr(i + DN_DEPTH < DN_WITERS):
                inflight.append(with_rows(unit_load(i + DN_DEPTH)))
            dcu = mfma_scaled(
                a=dwv,
                sa=dsc,
                b=lds_b8(drow * MID_ROW + dkc * 32),
                sb=lds_i32(W_MIDSC + drow * MIDSC_ROW + dkc * 4 + g4),
                c=fx.Vector.filled(4, 0.0, fx.Float32),
                a_fmt=FP4,
            )
            acc = [
                acc[e] + ok.select(dcu[e] * dw, fx.Float32(0.0))
                for e in range(4)
            ]
        if const_expr(shared_fp8):
            col_tok = fx.min(l16, tokens - 1)
            for wv, sc, kc, wgt in sh_units:
                cu = mfma_scaled(
                    a=wv,
                    sa=sc,
                    b=lds_b8((col_tok * MOE_SLOTS + TOP_K) * MID_ROW + kc * 32),
                    sb=lds_i32(
                        W_MIDSC
                        + (col_tok * MOE_SLOTS + TOP_K) * MIDSC_ROW
                        + kc * 4
                        + g4
                    ),
                    c=fx.Vector.filled(4, 0.0, fx.Float32),
                    a_fmt=FP8,
                )
                acc = [acc[e] + cu[e] * wgt for e in range(4)]
        fx.ptr_store(
            fx.Vector.from_elements(acc, fx.Float32),
            red + (wave * 64 + lane) * 4,
        )
        gpu.barrier()
        if tid < DN_ROWS * tokens:
            k = tid // DN_ROWS
            grp = (tid % DN_ROWS) // 16
            rr = tid % 16
            fx.ptr_store(
                bf16_round(row_sum(red, rr, k, [grp * 4 + w for w in range(4)])),
                misc + tid,
            )
        gpu.barrier()
        ffn_finish(td, 0, tokens)

    def wide_down_tiled(td):
        enter_stage("down")
        """Row group td's down GEMV over every distinct expert (B column l % 16 =
        token l % 16's mid for it, the zero row where the token is not routed
        there), route-weighted per column -> ffn_finish."""
        nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
        n_exp = nu + (0 if shared_fp8 else 1)
        rg_d = td * 2 + wave // 4
        # wave q of the row group takes chunks q and q + 4 of every expert, unit i
        # being expert i // 2 for all 4: which wave adds a chunk, and in what
        # order, follows from the expert alone, not its rank in the step's union,
        # so another row's experts cannot regroup this row's sums
        q = wave % 4

        def unit_load(du, half):
            """This wave's weights and scale for one expert/chunk unit."""
            dkc = q + 4 * half
            ok = (dkc < DN_KC) & (du < n_exp)
            # clamped into the table: a unit past the last reads a real or zero
            # row, never past urow (a garbage row is garbage B, and 0 * NaN)
            du = fx.min(du, U_MAX - 1)
            dkc = fx.min(dkc, DN_KC - 1)
            de = fx.min(uniform(fx.ptr_load(uexp + du)), SHARED_EXPERT)
            dwv = fp4_tile_load(
                rsrc(
                    w2 + fx.Int64(de) * W2_BYTES,
                    ok.select(fx.Int32(W2_BYTES), fx.Int32(0)),
                ),
                0,
                rg_d,
                INTER,
                dkc,
                lane * 4,
                cache_modifier=0,
            )
            dsc = e8m0_load(
                rsrc(s2, ok.select(fx.Int32(S2_BYTES), fx.Int32(0))),
                group_scale_byte(de, HIDDEN, rg_d, dkc, W2_SCALE_COLS),
            )
            return dwv, dsc, du, dkc, ok

        def with_rows(u):
            """A loaded unit plus its B row and route weight (the tables are in),
            read ahead so the unit's B loads wait on one LDS trip."""
            dwv, dsc, du, dkc, ok = u
            drow = fx.Int32(fx.ptr_load(urow + (du * 16 + l16)))
            dw = ok.select(fx.ptr_load(uwt + (du * 16 + l16)), fx.Float32(0.0))
            return dwv, dsc, drow, dkc, dw, ok

        def consume(acc, u):
            dwv, dsc, drow, dkc, dw, ok = u
            dcu = mfma_scaled(
                a=dwv,
                sa=dsc,
                b=lds_b8(drow * MID_ROW + dkc * 32),
                sb=lds_i32(W_MIDSC + drow * MIDSC_ROW + dkc * 4 + g4),
                c=fx.Vector.filled(4, 0.0, fx.Float32),
                a_fmt=FP4,
            )
            return [
                acc[e] + ok.select(dcu[e] * dw, fx.Float32(0.0))
                for e in range(4)
            ]

        # Every up / gate task pair is complete before token tiles reuse the pool.
        n_ug = n_exp * UG_PAIRS
        mb_poll(
            [
                (mr("ugdone"), fx.min(tid + THREADS * i, n_ug - 1), 1)
                for i in range((U_MAX * UG_PAIRS + THREADS - 1) // THREADS)
            ]
        )
        gpu.barrier()
        acquire()
        for tile0, tile_n in token_tiles(tokens):
            tile_mid_rows = tile_n * MOE_SLOTS
            down_tables(tile0, tile_n)
            gpu.barrier()
            rows_to_lds(
                tid,
                mb("mid8p") + fx.Int64(tile0 * MOE_SLOTS * MID_ROW_WORDS * 4),
                tile_mid_rows,
                MID_ROW_WORDS,
                xs,
                MID_ROW,
            )
            rows_to_lds(
                tid,
                mb("midsp") + fx.Int64(tile0 * MOE_SLOTS * MIDSC_ROW_WORDS * 4),
                tile_mid_rows,
                MIDSC_ROW_WORDS,
                xs + W_MIDSC,
                MIDSC_ROW,
            )
            if const_expr(shared_fp8):
                for local_k in range_constexpr(tile_n):
                    k = tile0 + local_k
                    mids = mb_poll(
                        [
                            (
                                mr("mid"),
                                (k * MOE_SLOTS + TOP_K) * INTER
                                + fx.min(tid * 8 + i * 2, INTER - 2),
                                2,
                            )
                            for i in range(4)
                        ]
                    )
                    if tid < INTER // 8:
                        f = [v.bitcast(fx.Float32) for pair in mids for v in pair]
                        sc, inv = mx8_scale(f, (1, 2))
                        qv = [mx8_q(v, inv) for v in f]
                        row = local_k * MOE_SLOTS + TOP_K
                        fx.ptr_store(
                            fp8_pack4(*qv[:4]), xs + (row * MID_ROW + tid * 2)
                        )
                        fx.ptr_store(
                            fp8_pack4(*qv[4:]),
                            xs + (row * MID_ROW + tid * 2 + 1),
                        )
                        if tid % 4 == 0:
                            fx.ptr_store(
                                sc, xs + (W_MIDSC + row * MIDSC_ROW + tid // 4)
                            )
            if tid < MID_ROW_WORDS:
                fx.ptr_store(fx.Int32(0), xs + (tile_mid_rows * MID_ROW + tid))
            if tid < MIDSC_ROW_WORDS:
                fx.ptr_store(
                    fx.Int32(0), xs + (W_MIDSC + tile_mid_rows * MIDSC_ROW + tid)
                )
            gpu.barrier()
            stamp(10)
            acc = [fx.Float32(0.0) for _ in range(4)]
            if const_expr(shared_fp8):
                sh_units = down_load_shared(td, fx.Int32(1))
            acc = _down_expert_batches(
                n_exp, DN_DEPTH, unit_load, with_rows, consume, acc
            )
            if const_expr(shared_fp8):
                col_tok = fx.min(l16, tile_n - 1)
                for wv, sc, kc, wgt in sh_units:
                    cu = mfma_scaled(
                        a=wv,
                        sa=sc,
                        b=lds_b8(
                            (col_tok * MOE_SLOTS + TOP_K) * MID_ROW + kc * 32
                        ),
                        sb=lds_i32(
                            W_MIDSC
                            + (col_tok * MOE_SLOTS + TOP_K) * MIDSC_ROW
                            + kc * 4
                            + g4
                        ),
                        c=fx.Vector.filled(4, 0.0, fx.Float32),
                        a_fmt=FP8,
                    )
                    acc = [acc[e] + cu[e] * wgt for e in range(4)]
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32),
                red + (wave * 64 + lane) * 4,
            )
            gpu.barrier()
            if tid < DN_ROWS * tile_n:
                local_k = tid // DN_ROWS
                grp = (tid % DN_ROWS) // 16
                rr = tid % 16
                tot = row_sum(red, rr, local_k, [grp * 4 + w for w in range(4)])
                fx.ptr_store(bf16_round(tot), misc + tid)
            gpu.barrier()
            ffn_finish(td, tile0, tile_n)

    return {
        "slot_of": slot_of,
        "build_experts": build_experts,
        "wug_load": wug_load,
        "wug_y": wug_y,
        "fp8_lanes4": fp8_lanes4,
        "wug_publish": wug_publish,
        "down_tables": down_tables,
        "stage_moe_wide": stage_moe_wide,
        "wide_down": wide_down_tiled if MULTI_TILE else wide_down_legacy,
    }
