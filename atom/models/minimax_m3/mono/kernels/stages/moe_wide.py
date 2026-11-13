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
)
from atom.models.minimax_m3.mono.kernels.stages.shared import (
    S2_BYTES,
    S2S_BYTES,
    S13_BYTES,
    S13S_BYTES,
    W2_BYTES,
    W2_SCALE_COLS,
    W2S_BYTES,
    W13_BYTES,
    W13_ROWS,
    W13_SCALE_COLS,
    W13S_BYTES,
)
from atom.models.minimax_m3.mono.layout import (
    DN_ROWS,
    MID4_ROW,
    MID4_WORDS,
    MID_ROW,
    MIDB_ROW,
    MIDSC_ROW,
    N_DN,
    N_ROUTER,
    SCORE_COPIES,
    UG_PER_SLOT,
    UG_ROWS,
    XN4_WORDS,
    XN8_ROW,
    XSC_ROW,
)
from atom.mono.device.mx import (
    FP4,
    FP8,
    fp4_pack2,
    fp4_scale,
    fp4_tile_load,
    fp4x8_bf16,
    mfma_scaled,
)
from atom.mono.device.ops import (
    CM_DEV,
    ballot,
    bf16_pair,
    bf16_round,
    butterfly,
    lanes_below,
    ld_dev,
    mfma_bf16,
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
def moe_wide_defs(k4ctx):
    """The functions of the expert-grouped MoE (S > 4)."""
    A16W = k4ctx["A16W"]
    A4W = k4ctx["A4W"]
    DN4 = k4ctx["DN4"]
    BASE = k4ctx["BASE"]
    DN_DEPTH = k4ctx["DN_DEPTH"]
    DN_ROWS_AHEAD = k4ctx["DN_ROWS_AHEAD"]
    DN_KC = k4ctx["DN_KC"]
    DN_WITERS = k4ctx["DN_WITERS"]
    G = k4ctx["G"]
    MIDSC_ROW_WORDS = k4ctx["MIDSC_ROW_WORDS"]
    MID_ROWS = k4ctx["MID_ROWS"]
    MID_ROW_WORDS = k4ctx["MID_ROW_WORDS"]
    ROUTE_ITERS = k4ctx["ROUTE_ITERS"]
    UGW_ITERS = k4ctx["UGW_ITERS"]
    UG_SHARED_UNITS = k4ctx["UG_SHARED_UNITS"]
    UG_CPW = k4ctx["UG_CPW"]
    DNL = k4ctx["DNL"]
    MIDB_ROWS = k4ctx["MIDB_ROWS"]
    SH8 = k4ctx["shared_mxfp8"]
    SH_FIRST = SH8 and k4ctx["sh_first"] != 0
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
    fp8_scale_load = k4ctx["fp8_scale_load"]
    fp8_tile_load = k4ctx["fp8_tile_load"]
    s2s = k4ctx["s2s"]
    s13s = k4ctx["s13s"]
    w2s = k4ctx["w2s"]
    w13s = k4ctx["w13s"]
    bid = k4ctx["bid"]
    e8m0_load = k4ctx["e8m0_load"]
    ffn_finish = k4ctx["ffn_finish"]
    fpool = k4ctx["fpool"]
    g4 = k4ctx["g4"]
    group_scale_byte = k4ctx["group_scale_byte"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    lds_b8 = k4ctx["lds_b8"]
    lds_b4 = k4ctx["lds_b4"]
    lds_bf8 = k4ctx["lds_bf8"]
    lds_i32 = k4ctx["lds_i32"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    misc = k4ctx["misc"]
    mr = k4ctx["mr"]
    prod_round = k4ctx["prod_round"]
    mx8_q = k4ctx["mx8_q"]
    mx8_scale = k4ctx["mx8_scale"]
    red = k4ctx["red"]
    route = k4ctx["route"]
    route_top4 = k4ctx["route_top4"]
    rwt = k4ctx["rwt"]
    s13 = k4ctx["s13"]
    s2 = k4ctx["s2"]
    scores_read = k4ctx["scores_read"]
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
    wrsrc = k4ctx["wrsrc"]
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
                fx.ptr_store(fx.Int32(SHARED_EXPERT), uexp + nu_)
                fx.ptr_store(fx.Int32((1 << tokens) - 1), umask + nu_)
        gpu.barrier()

    def wug_load(t, has, scales=True):
        """Up / gate weights (and ``scales``: the scale dwords, which hold the
        pair's both halves) of task t = (distinct expert t / 48, row group
        t % 48); without the task, reads go through zero-sized buffers."""
        u = t // UG_PER_SLOT
        cg = t % UG_PER_SLOT
        e_u = fx.min(uniform(fx.ptr_load(uexp + u)), SHARED_EXPERT)
        r_w = wrsrc(
            "moeu",
            w13 + fx.Int64(e_u) * W13_BYTES,
            has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
        )
        r_s = wrsrc("moeu", s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
        wts = []
        scs = []
        for gu in range_constexpr(2):
            rg = gu * UG_PER_SLOT + cg
            wts.append([])
            scs.append([])
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts[gu].append(fp4_tile_load(r_w, 0, rg, HIDDEN, kc, lane * 4))
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

    def wsh_load(cg, has):
        """The shared expert's up / gate tiles of 16-row group ``cg`` (MXFP8, one scale
        a tile): wts[gu][chunk], scs[gu][chunk]; without the task, zero-sized reads."""
        r_w = wrsrc("shu", w13s, has.select(fx.Int32(W13S_BYTES), fx.Int32(0)))
        r_s = wrsrc("shu", s13s, has.select(fx.Int32(S13S_BYTES), fx.Int32(0)))
        wts, scs = [], []
        for gu in range_constexpr(2):
            row = (gu * UG_PER_SLOT + cg) * 16 + l16
            kcs = [wave * UG_CPW + i for i in range(UG_CPW)]
            wts.append([fp8_tile_load(r_w, row, HIDDEN, kc) for kc in kcs])
            scs.append([fp8_scale_load(r_s, row, HIDDEN, kc) for kc in kcs])
        return wts, scs

    def wug_y(wts, scs, half, fp8=False, xb=None, xq=None):
        return wug_reduce(*wug_mma(wts, scs, half, fp8, xb, xq), rnd=fp8)

    def wug_mma(wts, scs, half, fp8=False, xb=None, xq=None):
        """A task's up / gate GEMV for every token (B column l % 16 = token
        l % 16) -> swiglu: thread (row r, token k)'s mid value. ``half``: the
        task's 16-row half of its 32-row scale block (compile time). ``xb`` (a16w4):
        this lane's bf16 xn operands, ``xb[chunk][d]`` (``stage_moe_wide``)."""
        if const_expr(xq is not None):
            # a4w4: this lane's MXFP4 xn operand and code of chunk i (``xq[i]``), fp4 x fp4
            cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
            cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
            for i in range_constexpr(UG_CPW):
                sel = (i % 2) * 2 + half
                cg_ = mfma_scaled(
                    wts[0][i], xq[i][0], cg_, scs[0][i // 2], xq[i][1],
                    a_fmt=FP4, b_fmt=FP4, sa_byte=sel,
                )  # fmt: skip
                cu_ = mfma_scaled(
                    wts[1][i], xq[i][0], cu_, scs[1][i // 2], xq[i][1],
                    a_fmt=FP4, b_fmt=FP4, sa_byte=sel,
                )  # fmt: skip
            return cg_, cu_
        if const_expr(xb is not None):
            cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
            cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
            for i in range_constexpr(UG_CPW):
                sh8 = ((i % 2) * 2 + half) * 8
                code_g = scs[0][i // 2] >> sh8
                code_u = scs[1][i // 2] >> sh8
                for d in range_constexpr(4):
                    cg_ = mfma_bf16(fp4x8_bf16(wts[0][i][d], code_g), xb[i][d], cg_)
                    cu_ = mfma_bf16(fp4x8_bf16(wts[1][i][d], code_u), xb[i][d], cu_)
            return cg_, cu_
        col = fx.min(l16, tokens - 1)
        cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
        cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
        sbs = [
            lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
            for i in range(UG_CPW)
        ]
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            xb = lds_b8(col * XN8_ROW + kc * 32)
            if const_expr(fp8):  # the shared expert's tiles: a scale each
                cg_ = mfma_scaled(wts[0][i], xb, cg_, scs[0][i], sbs[i], a_fmt=FP8)
                cu_ = mfma_scaled(wts[1][i], xb, cu_, scs[1][i], sbs[i], a_fmt=FP8)
                continue
            sel = (i % 2) * 2 + half
            cg_ = mfma_scaled(
                wts[0][i], xb, cg_, scs[0][i // 2], sbs[i], a_fmt=FP4, sa_byte=sel
            )
            cu_ = mfma_scaled(
                wts[1][i], xb, cu_, scs[1][i // 2], sbs[i], a_fmt=FP4, sa_byte=sel
            )
        return cg_, cu_

    def wug_reduce(cg_, cu_, rnd=False):
        """The task's two accumulators -> swiglu: thread (row r, token k)'s mid value.
        ``rnd`` (the shared expert's MXFP8 GEMVs): its gate / up are bf16 in production.
        """
        fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
        fx.ptr_store(cu_, fpool + (W_UPP + (wave * 64 + lane) * 4))
        gpu.barrier()
        r = tid % UG_ROWS
        k = fx.min(tid // UG_ROWS, tokens - 1)
        gsum = row_sum(red, r, k)
        usum = row_sum(fpool + W_UPP, r, k)
        if const_expr(rnd and prod_round):
            gsum = bf16_round(gsum)
            usum = bf16_round(usum)
        gpu.barrier()  # red and the up partials are free for the next task
        return swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]

    def fp8_lanes4(q):
        """Lane r (r % 4 == 0): the E4M3 dword of lanes r .. r + 3's values."""
        lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, xshfl(q, 1), fx.Int32(0), False))
        return lo | (xshfl(lo, 2) << 16)

    def wug_publish16(pr, y_lo, y_hi):
        """a16w4: routed task pair pr's mid rows as bf16 plain words (``mid16p``, row
        k TOP_K + slot), then the pair's flag. Thread (r, k) holds rows r and 16 + r
        of the 32-row block; an even r stores its pair with r + 1's."""
        u = pr // UG_PAIRS
        blk_ = pr % UG_PAIRS
        r = tid % UG_ROWS
        k = fx.min(tid // UG_ROWS, tokens - 1)
        lo = bf16_round(y_lo)
        hi = bf16_round(y_hi)
        nb_lo = xshfl(lo, 1)
        nb_hi = xshfl(hi, 1)
        e_u = fx.Int32(fx.ptr_load(uexp + u))
        m_u = fx.Int32(fx.ptr_load(umask + u))
        row = k * TOP_K + slot_of(k, e_u)
        if (tid < UG_ROWS * tokens) & (((m_u >> k) & 1) != 0):  # noqa: SIM102
            if r % 2 == 0:
                for h, (a_, b_) in enumerate(((lo, nb_lo), (hi, nb_hi))):
                    bo.buffer_store(
                        bf16_pair(a_, b_).bitcast(fx.Int32),
                        rsrc(mb("mid16p")),
                        row * (INTER // 2) + blk_ * 16 + h * 8 + r // 2,
                        cache_modifier=CM_DEV,
                    )
        publish(mb_put, mr("ugdone"), pr, fx.Int32(1), tid == 0)

    def wug_publish4(pr, y_lo, y_hi):
        """a4w4: routed task pair pr's mid rows as MXFP4 plain words (``mid4p``, row k TOP_K + slot:
        a 32-row block is 4 words, rows 0..7 / 8..15 / 16..23 / 24..31, low nibble = lower row) and
        its E8M0 code (``mids4p``), then the pair's flag. Thread (r, k) holds rows r and 16 + r of
        the block: the 16 lanes of a token share the block's amax (as ``wug_publish``), an even
        lane packs its neighbour's value beside its own, 4 even lanes' bytes make a word.
        """
        u = pr // UG_PAIRS
        blk_ = pr % UG_PAIRS
        r = tid % UG_ROWS
        k = fx.min(tid // UG_ROWS, tokens - 1)
        lo = bf16_round(y_lo)
        hi = bf16_round(y_hi)
        amax = butterfly(fx.max(fx.max(lo, -lo), fx.max(hi, -hi)), (1, 2, 4, 8), fx.max)
        e4, sc4 = fp4_scale(amax)
        words = []
        for v in (lo, hi):
            b = fp4_pack2(fx.Int32(0), v, xshfl(v, 1), sc4, 0) & 0xFF
            b = b | (xshfl(b, 2) << 8)
            words.append(b | (xshfl(b, 4) << 16))
        e_u = fx.Int32(fx.ptr_load(uexp + u))
        m_u = fx.Int32(fx.ptr_load(umask + u))
        row = k * TOP_K + slot_of(k, e_u)
        if (tid < UG_ROWS * tokens) & (((m_u >> k) & 1) != 0):
            if r % 8 == 0:
                for h in range_constexpr(2):
                    bo.buffer_store(
                        words[h],
                        rsrc(mb("mid4p")),
                        row * MID4_WORDS + blk_ * 4 + h * 2 + r // 8,
                        cache_modifier=CM_DEV,
                    )
            if r == 0:
                bo.buffer_store(
                    e4,
                    rsrc(mb("mids4p")),
                    row * (INTER // 32) + blk_,
                    cache_modifier=CM_DEV,
                )
        publish(mb_put, mr("ugdone"), pr, fx.Int32(1), tid == 0)

    def wug_publish(pr, y_lo, y_hi, shared_row=False):
        """Task pair pr = (distinct expert pr / UG_PAIRS, 32-row block
        pr % UG_PAIRS): its mid rows (thread (r, k) holds rows r and 16 + r of
        the block) as the down stage's MXFP8 -- the rounding its bf16 staging
        did, one 32-block a 16-lane group -- then the pair's flag. ``shared_row``
        (a16w4): the shared expert's mid row of token k is row k."""
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
        if const_expr(shared_row):
            row = k
        else:
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
        """urow / uwt := per (distinct expert, B column) the down stage's mid row
        (the zero row where the column's token is not routed there) and route
        weight. Routing only: built while the first up / gate weights land."""
        nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
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
                if const_expr(A16W or A4W):
                    # the routed experts' bf16 (a4w4: MXFP4) rows (k TOP_K + slot, the zero row last)
                    # and the shared expert's MXFP8 rows (k, its own zero row)
                    is_sh = xu == nu
                    hit = is_sh.select(xt, xt * TOP_K + xs_)
                    miss = is_sh.select(fx.Int32(tokens), fx.Int32(MIDB_ROWS))
                    fx.ptr_store(xh.select(hit, miss), urow + ix)
                else:
                    fx.ptr_store(
                        xh.select(xt * MOE_SLOTS + xs_, fx.Int32(MID_ROWS)), urow + ix
                    )
                fx.ptr_store(
                    xh.select(fx.ptr_load(rwt + (8 * xt + xs_)), fx.Float32(0.0)),
                    uwt + ix,
                )

    def stage_moe_wide():
        td = start("down")  # this CTA's down task (the split tasks come first)
        has_dn = td < N_DN
        td = fx.min(td, N_DN - 1)
        if const_expr(SH_FIRST):
            # the shared pair needs no routing: its first half's tiles go out now,
            # while the router's scores are still on their way, and its GEMVs are
            # the first thing a host does once the xn is in (``sh_first``)
            jsh = fx.Int32(bid) - (G - UG_PAIRS)
            has_s = jsh >= 0
            jsh = fx.min(jsh, UG_PAIRS - 1)
            sh_cur = wsh_load(2 * jsh, has_s)
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
                (scores_read(mine, k), (lane + 64 * h) * 2, 2)
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
        n_routed = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
        # the MXFP8 shared expert is not a task of the grouped loop (its tiles are fp8)
        n_pairs = (n_routed + (0 if SH8 else 1)) * UG_PAIRS

        # the MXFP8 shared expert's pairs (UG_PAIRS, each ``UG_SHARED_UNITS`` routed pairs' time) run
        # on the last UG_PAIRS CTAs after their routed pairs: those carry that many pairs
        # fewer, so every CTA ends within a pair of the others. Rounds (a round is
        # one pair a CTA): the first ``full`` are every CTA's (pair round * G +
        # bid); the last ``UG_SHARED_UNITS`` are the other G - UG_PAIRS CTAs' (the routed
        # pairs left, in order). Not SH8: pair round * G + bid throughout
        SH_HOST0 = G - UG_PAIRS
        if const_expr(SH8):
            rounds = (n_pairs + UG_SHARED_UNITS * UG_PAIRS + (G - 1)) // G
            full = fx.max(rounds - UG_SHARED_UNITS, 0)

        def task(it):
            """Iteration it's task: its pair's half it % 2."""
            if const_expr(SH8):
                rd = it // 2
                tail = full * G + (fx.Int32(rd) - full) * SH_HOST0 + bid
                pr = (fx.Int32(rd) < full).select(
                    fx.Int32(rd * G) + bid, (bid < SH_HOST0).select(tail, n_pairs)
                )
            else:
                pr = fx.Int32(bid + it // 2 * G)
            return fx.min(pr, n_pairs - 1) * 2 + it % 2, pr < n_pairs

        t0, h0 = task(0)
        cur = wug_load(t0, h0)  # task 0: a lo half, with the pair's scales
        down_tables()
        stamp(7)
        # the first task's weights are in flight: now the xn
        # every token's MXFP8 xn and scales -> the pool, once every router task
        # flagged its slice (plain words, 16 B a load)
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
        xq = None
        if const_expr(A4W):
            # a4w4: this lane's MXFP4 xn operand and code of each of its wave's chunks for good (a
            # lane's column is token l % 16, its wave's k eighth, 16 B a 128-k chunk), read once for
            # every task; only the shared expert's CTAs hold MXFP8 rows
            colx = fx.min(l16, tokens - 1)
            xq = [
                (
                    fx.Vector(
                        ld_dev(
                            mb("xn4p"),
                            colx * XN4_WORDS + (wave * UG_CPW + i) * 16 + g4 * 4,
                            4,
                        )
                    ),
                    fx.Int32(
                        ld_dev(
                            mb("xs4p"),
                            colx * XSC_WORDS + (wave * UG_CPW + i) * 4 + g4,
                            1,
                        )
                    ),
                )
                for i in range(UG_CPW)
            ]
            xb = None
            if bid >= G - UG_PAIRS:
                for base_w, n_row, lds_w, lds_row in (
                    (mb("xn8p"), XN8_WORDS, 0, XN8_ROW),
                    (mb("xscp"), XSC_WORDS, XSC_LDS, XSC_ROW),
                ):
                    rows_to_lds(tid, base_w, tokens, n_row, xs + lds_w, lds_row)
        elif const_expr(A16W):
            # a16w4: this lane's bf16 xn operands for good (a lane's column is token
            # l % 16, its wave's k eighth, four 16 B pieces a 128-k chunk), read once
            # for every task; only the shared expert's CTAs hold MXFP8 rows
            colx = fx.min(l16, tokens - 1)
            xb = [
                [
                    fx.Vector(
                        ld_dev(
                            mb("xnbp"),
                            colx * (HIDDEN // 2)
                            + (wave * UG_CPW + i) * 64
                            + g4 * 16
                            + d * 4,
                            4,
                        )
                    ).bitcast(fx.BFloat16)
                    for d in range(4)
                ]
                for i in range(UG_CPW)
            ]
            if bid >= G - UG_PAIRS:
                for base_w, n_row, lds_w, lds_row in (
                    (mb("xn8p"), XN8_WORDS, 0, XN8_ROW),
                    (mb("xscp"), XSC_WORDS, XSC_LDS, XSC_ROW),
                ):
                    rows_to_lds(tid, base_w, tokens, n_row, xs + lds_w, lds_row)
        else:
            xb = None
            for base_w, n_row, lds_w, lds_row in (
                (mb("xn8p"), XN8_WORDS, 0, XN8_ROW),
                (mb("xscp"), XSC_WORDS, XSC_LDS, XSC_ROW),
            ):
                rows_to_lds(tid, base_w, tokens, n_row, xs + lds_w, lds_row)
        gpu.barrier()
        if const_expr(SH_FIRST):  # noqa: SIM102
            if has_s:
                # the second half's tiles follow the first's MFMAs
                cg0, cu0 = wug_mma(sh_cur[0], sh_cur[1], 0, True)
                sh_nxt = wsh_load(2 * jsh + 1, has_s)
                y_lo = wug_reduce(cg0, cu0, rnd=True)
                y_hi = wug_y(sh_nxt[0], sh_nxt[1], 1, True)
                wug_publish(
                    n_routed * UG_PAIRS + jsh, y_lo, y_hi, shared_row=A16W or A4W
                )
        y_lo = fx.Float32(0.0)  # the pair's first half, until the second's
        for it in range_constexpr(UGW_ITERS):
            t, has_t = task(it)
            if const_expr(it + 1 < UGW_ITERS):
                tn, hn = task(it + 1)
                # a hi half reads no scales: its lo half's dwords hold them
                nxt = wug_load(tn, hn, scales=(it + 1) % 2 == 0)
            if const_expr(it % 2 == 0):
                pair_scs = cur[1]
            if has_t:
                y = wug_y(
                    cur[0], pair_scs, it % 2, xb=xb, xq=xq
                )  # a pair: row groups cg, cg + 1
                if const_expr(it % 2 == 0):
                    y_lo = y
                elif const_expr(A4W):
                    wug_publish4(t // 2, y_lo, y)
                elif const_expr(A16W):
                    wug_publish16(t // 2, y_lo, y)
                else:
                    wug_publish(t // 2, y_lo, y)
            if const_expr(it + 1 < UGW_ITERS):
                cur = nxt
        if const_expr(SH8 and not SH_FIRST):
            # the shared expert's pair j (task pair n_routed * UG_PAIRS + j of the
            # step's numbering) on CTA SH_HOST0 + j: both halves' tiles in flight
            # before the first GEMV, the pair's flag after the second
            jsh = fx.Int32(bid) - SH_HOST0
            has_s = jsh >= 0
            jsh = fx.min(jsh, UG_PAIRS - 1)
            sh_cur = wsh_load(2 * jsh, has_s)
            sh_nxt = wsh_load(2 * jsh + 1, has_s)
            if has_s:
                y_lo = wug_y(sh_cur[0], sh_cur[1], 0, True)
                y_hi = wug_y(sh_nxt[0], sh_nxt[1], 1, True)
                wug_publish(
                    n_routed * UG_PAIRS + jsh, y_lo, y_hi, shared_row=A16W or A4W
                )
        stamp(8)
        if has_dn:
            if const_expr(A4W):
                wide_down4(td)
            elif const_expr(A16W):
                wide_down16(td)
            else:
                wide_down(td)
            stamp(11)

    def wide_down(td):
        enter_stage("down")
        """Row group td's down GEMV over every distinct expert (B column l % 16 =
        token l % 16's mid for it, the zero row where the token is not routed
        there), route-weighted per column -> ffn_finish."""
        nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
        rg_d = td * 2 + wave // 4
        # wave q of the row group takes chunks q and q + 4 of every expert (units
        # 2 e, 2 e + 1; only chunk q where q + 4 is past the last: unit e): which wave
        # adds a chunk, and in what order, follows from the expert alone, not its
        # rank in the step's union, so another row's experts cannot regroup this
        # row's sums
        q = uniform(wave % 4)

        def unit_load(i):
            """(weights, scale, unit) of this wave's i-th (expert, 128-k chunk)
            unit, zero-sized reads past the last."""
            # waves with one chunk an expert (q >= DN_KC - 4) take one expert a unit,
            # not every other unit empty: all their units' weights are real loads
            # in flight, their sums unchanged (the same units in the same order)
            one = q >= DN_KC - 4
            e_i = one.select(fx.Int32(i), fx.Int32(i // 2))
            dkc = one.select(q, q + 4 * (i % 2))
            ok = e_i < nu if SH8 else e_i <= nu
            # clamped into the table: a unit past the last reads a real or zero
            # row, never past urow (a garbage row is garbage B, and 0 * NaN)
            du = fx.min(e_i, U_MAX - 1)
            dkc = fx.min(dkc, DN_KC - 1)
            de = fx.min(uniform(fx.ptr_load(uexp + du)), SHARED_EXPERT)
            dwv = fp4_tile_load(
                wrsrc(
                    "moed",
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
                wrsrc("moed", s2, ok.select(fx.Int32(S2_BYTES), fx.Int32(0))),
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

        if const_expr(SH8):
            # the shared expert's two units of this wave (chunks q, q + 4: MXFP8
            # tiles), their weights in flight with the pipeline's
            sh_units = []
            for h in range_constexpr(2):
                skc = q + 4 * h
                sok = skc < DN_KC
                skc = fx.min(skc, DN_KC - 1)
                srow = rg_d * 16 + l16
                sr_w = wrsrc("shd", w2s, sok.select(fx.Int32(W2S_BYTES), fx.Int32(0)))
                sr_s = wrsrc("shd", s2s, sok.select(fx.Int32(S2S_BYTES), fx.Int32(0)))
                sh_units.append(
                    (
                        fp8_tile_load(sr_w, srow, INTER, skc),
                        fp8_scale_load(sr_s, srow, INTER, skc),
                        skc,
                        sok,
                    )
                )
        # software pipeline: the first units' weights go out before the mids are
        # in; unit i + DN_DEPTH's before unit i's MFMA
        acc = [fx.Float32(0.0) for _ in range(4)]
        inflight = [unit_load(i) for i in range(min(DN_DEPTH, DN_WITERS))]
        ahead = []  # the next units, their rows read (the rest of the pipeline is raw)
        # every up / gate task pair's flag, then the MXFP8 mid rows and their
        # scales, the pool's layout (plain words, 16 B a load)
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
            if const_expr(i == 0):  # the tables are in: the early units' rows
                ahead = [with_rows(u) for u in inflight[:DN_ROWS_AHEAD]]
                inflight = inflight[DN_ROWS_AHEAD:]
            dwv, dsc, drow, dkc, dw, ok = ahead.pop(0)
            if const_expr(i + DN_DEPTH < DN_WITERS):
                inflight.append(unit_load(i + DN_DEPTH))
            if const_expr(len(inflight) > 0):
                ahead.append(with_rows(inflight.pop(0)))
            dcu = mfma_scaled(
                a=dwv,
                sa=dsc,
                b=lds_b8(drow * MID_ROW + dkc * 32),
                sb=lds_i32(W_MIDSC + drow * MIDSC_ROW + dkc * 4 + g4),
                c=fx.Vector.filled(4, 0.0, fx.Float32),
                a_fmt=FP4,
            )
            # a unit past the last (or a wave's missing second chunk) adds
            # nothing, whatever its MFMA gave
            acc = [acc[e] + ok.select(dcu[e] * dw, fx.Float32(0.0)) for e in range(4)]
        if const_expr(SH8):
            # the shared expert's units: its mid rows are the table's entry nu
            srow_b = fx.Int32(fx.ptr_load(urow + (nu * 16 + l16)))
            sw = fx.ptr_load(uwt + (nu * 16 + l16))
            shacc = [fx.Float32(0.0) for _ in range(4)]
            for swv, ssc, skc, sok in sh_units:
                scu = mfma_scaled(
                    a=swv,
                    sa=ssc,
                    b=lds_b8(srow_b * MID_ROW + skc * 32),
                    sb=lds_i32(W_MIDSC + srow_b * MIDSC_ROW + skc * 4 + g4),
                    c=fx.Vector.filled(4, 0.0, fx.Float32),
                    a_fmt=FP8,
                )
                if const_expr(
                    prod_round
                ):  # kept apart: production rounds each of the two
                    shacc = [
                        shacc[e] + sok.select(scu[e] * sw, fx.Float32(0.0))
                        for e in range(4)
                    ]
                else:
                    acc = [
                        acc[e] + sok.select(scu[e] * sw, fx.Float32(0.0))
                        for e in range(4)
                    ]
        a0, a1, a2, a3 = acc
        fx.ptr_store(
            fx.Vector.from_elements([a0, a1, a2, a3], fx.Float32),
            red + (wave * 64 + lane) * 4,
        )
        if const_expr(SH8 and prod_round):
            # (the up partials' words: free in the down stage)
            fx.ptr_store(
                fx.Vector.from_elements(shacc, fx.Float32),
                fpool + (W_UPP + (wave * 64 + lane) * 4),
            )
        gpu.barrier()
        if tid < DN_ROWS * tokens:
            k = tid // DN_ROWS
            grp = (tid % DN_ROWS) // 16
            rr = tid % 16
            if const_expr(SH8 and prod_round):
                grp_waves = [grp * 4 + w for w in range(4)]
                tot = bf16_round(row_sum(red, rr, k, grp_waves)) + bf16_round(
                    row_sum(fpool + W_UPP, rr, k, grp_waves)
                )
            else:
                tot = row_sum(red, rr, k, [grp * 4 + w for w in range(4)])
            fx.ptr_store(bf16_round(tot), misc + tid)
        gpu.barrier()
        ffn_finish(td)

    def wide_down16(td):
        enter_stage("down")
        """a16w4: row group td's down GEMV over every distinct expert, B column l % 16 =
        token l % 16's bf16 mid for it (the zero row where the token is not routed
        there). Wave q of the row group takes the experts q, q + 4, .. of the step's
        union whole (its 6 chunks chained into one accumulator), so production's
        bf16(acc * w) per (expert, token) is the wave's own; the waves' sums, the
        shared expert's MXFP8 GEMV (rounded to bf16) and the final bf16 add follow."""
        nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
        rg_d = td * 2 + wave // 4
        q = uniform(wave % 4)
        n_ew = (U_MAX - 1 + 3) // 4  # routed experts a wave takes, at most
        dn_w = n_ew * DN_KC

        def unit_load(i):
            """(weights, scale) of this wave's i-th (expert, 128-k chunk) unit,
            zero-sized reads past its last expert."""
            du = q + 4 * (i // DN_KC)
            ok = du < nu
            de = fx.min(
                uniform(fx.ptr_load(uexp + fx.min(du, U_MAX - 1))), SHARED_EXPERT
            )
            dwv = fp4_tile_load(
                wrsrc(
                    "moed",
                    w2 + fx.Int64(de) * W2_BYTES,
                    ok.select(fx.Int32(W2_BYTES), fx.Int32(0)),
                ),
                0,
                rg_d,
                INTER,
                i % DN_KC,
                lane * 4,
            )
            dsc = e8m0_load(
                wrsrc("moed", s2, ok.select(fx.Int32(S2_BYTES), fx.Int32(0))),
                group_scale_byte(de, HIDDEN, rg_d, i % DN_KC, W2_SCALE_COLS),
            )
            return dwv, dsc

        sh_units = []
        for h in range_constexpr(2):
            skc = q + 4 * h
            sok = skc < DN_KC
            skc = fx.min(skc, DN_KC - 1)
            srow = rg_d * 16 + l16
            sr_w = wrsrc("shd", w2s, sok.select(fx.Int32(W2S_BYTES), fx.Int32(0)))
            sr_s = wrsrc("shd", s2s, sok.select(fx.Int32(S2S_BYTES), fx.Int32(0)))
            sh_units.append(
                (
                    fp8_tile_load(sr_w, srow, INTER, skc),
                    fp8_scale_load(sr_s, srow, INTER, skc),
                    skc,
                    sok,
                )
            )
        inflight = [unit_load(i) for i in range(min(DN_DEPTH, dn_w))]
        # every up / gate task pair's flag, then the bf16 mid rows of the routed experts,
        # the shared expert's MXFP8 rows and scales (plain words, 16 B a load)
        n_ug = (nu + 1) * UG_PAIRS
        mb_poll(
            [
                (mr("ugdone"), fx.min(tid + THREADS * i, n_ug - 1), 1)
                for i in range((U_MAX * UG_PAIRS + THREADS - 1) // THREADS)
            ]
        )
        gpu.barrier()
        acquire()
        rows_to_lds(tid, mb("mid16p"), MIDB_ROWS, INTER // 2, xs, MIDB_ROW)
        rows_to_lds(tid, mb("mid8p"), tokens, MID_ROW_WORDS, xs + DNL["sh8"], MID_ROW)
        rows_to_lds(
            tid, mb("midsp"), tokens, MIDSC_ROW_WORDS, xs + DNL["sh8sc"], MIDSC_ROW
        )
        if tid < INTER // 2:
            fx.ptr_store(fx.Int32(0), xs + (MIDB_ROWS * MIDB_ROW + tid))
        if tid < MID_ROW_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (DNL["sh8"] + tokens * MID_ROW + tid))
        if tid < MIDSC_ROW_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (DNL["sh8sc"] + tokens * MIDSC_ROW + tid))
        gpu.barrier()
        stamp(10)
        tot = [fx.Float32(0.0) for _ in range(4)]
        for i in range_constexpr(dn_w):
            dwv, dsc = inflight.pop(0)
            if const_expr(i + DN_DEPTH < dn_w):
                inflight.append(unit_load(i + DN_DEPTH))
            kc = i % DN_KC
            if const_expr(kc == 0):
                du = q + 4 * (i // DN_KC)
                ok = du < nu
                duc = fx.min(du, U_MAX - 1)
                drow = fx.Int32(fx.ptr_load(urow + (duc * 16 + l16)))
                dw = ok.select(fx.ptr_load(uwt + (duc * 16 + l16)), fx.Float32(0.0))
                cacc = fx.Vector.filled(4, 0.0, fx.Float32)
            for d in range_constexpr(4):
                cacc = mfma_bf16(
                    fp4x8_bf16(dwv[d], dsc),
                    lds_bf8(drow * MIDB_ROW + kc * 64 + g4 * 16 + d * 4),
                    cacc,
                )
            if const_expr(kc == DN_KC - 1):
                if const_expr(prod_round):
                    # production's gemm2 epilogue: bf16(acc * topk weight) per (expert, token)
                    r_ = [bf16_round(cacc[e] * dw) for e in range(4)]
                else:
                    r_ = [cacc[e] * dw for e in range(4)]
                tot = [tot[e] + ok.select(r_[e], fx.Float32(0.0)) for e in range(4)]
        # the shared expert's units: its mid rows are the table's entry nu
        srow_b = fx.Int32(fx.ptr_load(urow + (nu * 16 + l16)))
        sw = fx.ptr_load(uwt + (nu * 16 + l16))
        shacc = [fx.Float32(0.0) for _ in range(4)]
        for swv, ssc, skc, sok in sh_units:
            scu = mfma_scaled(
                a=swv,
                sa=ssc,
                b=lds_b8(DNL["sh8"] + srow_b * MID_ROW + skc * 32),
                sb=lds_i32(DNL["sh8sc"] + srow_b * MIDSC_ROW + skc * 4 + g4),
                c=fx.Vector.filled(4, 0.0, fx.Float32),
                a_fmt=FP8,
            )
            shacc = [
                shacc[e] + sok.select(scu[e] * sw, fx.Float32(0.0)) for e in range(4)
            ]
        fx.ptr_store(
            fx.Vector.from_elements(tot, fx.Float32), red + (wave * 64 + lane) * 4
        )
        fx.ptr_store(
            fx.Vector.from_elements(shacc, fx.Float32),
            fpool + (DNL["shp"] + (wave * 64 + lane) * 4),
        )
        gpu.barrier()
        if tid < DN_ROWS * tokens:
            k = tid // DN_ROWS
            grp = (tid % DN_ROWS) // 16
            rr = tid % 16
            waves = [grp * 4 + w for w in range(4)]
            routed = row_sum(red, rr, k, waves)
            shared = row_sum(fpool + DNL["shp"], rr, k, waves)
            if const_expr(prod_round):
                tot_k = bf16_round(routed) + bf16_round(shared)
            else:
                tot_k = routed + shared
            fx.ptr_store(bf16_round(tot_k), misc + tid)
        gpu.barrier()
        ffn_finish(td)

    def wide_down4(td):
        enter_stage("down")
        """a4w4: row group td's down GEMV over every distinct expert, B column l % 16 = token l % 16's
        MXFP4 mid for it (the zero row where the token is not routed there), fp4 x fp4. As
        ``wide_down16``: wave q of the row group takes the experts q, q + 4, .. whole (its 6 chunks
        chained into one accumulator), so production's bf16(acc * w) per (expert, token) is the
        wave's own; then the waves' sums, the shared expert's MXFP8 GEMV (rounded to bf16) and the
        final bf16 add."""
        nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
        rg_d = td * 2 + wave // 4
        q = uniform(wave % 4)
        n_ew = (U_MAX - 1 + 3) // 4  # routed experts a wave takes, at most
        dn_w = n_ew * DN_KC

        def unit_load(i):
            """(weights, scale) of this wave's i-th (expert, 128-k chunk) unit,
            zero-sized reads past its last expert."""
            du = q + 4 * (i // DN_KC)
            ok = du < nu
            de = fx.min(
                uniform(fx.ptr_load(uexp + fx.min(du, U_MAX - 1))), SHARED_EXPERT
            )
            dwv = fp4_tile_load(
                wrsrc(
                    "moed",
                    w2 + fx.Int64(de) * W2_BYTES,
                    ok.select(fx.Int32(W2_BYTES), fx.Int32(0)),
                ),
                0,
                rg_d,
                INTER,
                i % DN_KC,
                lane * 4,
            )
            dsc = e8m0_load(
                wrsrc("moed", s2, ok.select(fx.Int32(S2_BYTES), fx.Int32(0))),
                group_scale_byte(de, HIDDEN, rg_d, i % DN_KC, W2_SCALE_COLS),
            )
            return dwv, dsc

        sh_units = []
        for h in range_constexpr(2):
            skc = q + 4 * h
            sok = skc < DN_KC
            skc = fx.min(skc, DN_KC - 1)
            srow = rg_d * 16 + l16
            sr_w = wrsrc("shd", w2s, sok.select(fx.Int32(W2S_BYTES), fx.Int32(0)))
            sr_s = wrsrc("shd", s2s, sok.select(fx.Int32(S2S_BYTES), fx.Int32(0)))
            sh_units.append(
                (
                    fp8_tile_load(sr_w, srow, INTER, skc),
                    fp8_scale_load(sr_s, srow, INTER, skc),
                    skc,
                    sok,
                )
            )
        inflight = [unit_load(i) for i in range(min(DN_DEPTH, dn_w))]
        # every up / gate task pair's flag, then the MXFP4 mid rows and codes of the routed experts, the
        # shared expert's MXFP8 rows and codes (plain words, 16 B a load)
        n_ug = (nu + 1) * UG_PAIRS
        mb_poll(
            [
                (mr("ugdone"), fx.min(tid + THREADS * i, n_ug - 1), 1)
                for i in range((U_MAX * UG_PAIRS + THREADS - 1) // THREADS)
            ]
        )
        gpu.barrier()
        acquire()
        rows_to_lds(tid, mb("mid4p"), MIDB_ROWS, MID4_WORDS, xs, MID4_ROW)
        rows_to_lds(
            tid, mb("mids4p"), MIDB_ROWS, INTER // 32, xs + DN4["sc4"], MIDSC_ROW
        )
        rows_to_lds(tid, mb("mid8p"), tokens, MID_ROW_WORDS, xs + DN4["sh8"], MID_ROW)
        rows_to_lds(
            tid, mb("midsp"), tokens, MIDSC_ROW_WORDS, xs + DN4["sh8sc"], MIDSC_ROW
        )
        if tid < MID4_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (MIDB_ROWS * MID4_ROW + tid))
        if tid < INTER // 32:
            fx.ptr_store(fx.Int32(0), xs + (DN4["sc4"] + MIDB_ROWS * MIDSC_ROW + tid))
        if tid < MID_ROW_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (DN4["sh8"] + tokens * MID_ROW + tid))
        if tid < MIDSC_ROW_WORDS:
            fx.ptr_store(fx.Int32(0), xs + (DN4["sh8sc"] + tokens * MIDSC_ROW + tid))
        gpu.barrier()
        stamp(10)
        tot = [fx.Float32(0.0) for _ in range(4)]
        for i in range_constexpr(dn_w):
            dwv, dsc = inflight.pop(0)
            if const_expr(i + DN_DEPTH < dn_w):
                inflight.append(unit_load(i + DN_DEPTH))
            kc = i % DN_KC
            if const_expr(kc == 0):
                du = q + 4 * (i // DN_KC)
                ok = du < nu
                duc = fx.min(du, U_MAX - 1)
                drow = fx.Int32(fx.ptr_load(urow + (duc * 16 + l16)))
                dw = ok.select(fx.ptr_load(uwt + (duc * 16 + l16)), fx.Float32(0.0))
                cacc = fx.Vector.filled(4, 0.0, fx.Float32)
            cacc = mfma_scaled(
                a=dwv,
                sa=dsc,
                b=lds_b4(drow * MID4_ROW + kc * 16),
                sb=lds_i32(DN4["sc4"] + drow * MIDSC_ROW + kc * 4 + g4),
                c=cacc,
                a_fmt=FP4,
                b_fmt=FP4,
            )
            if const_expr(kc == DN_KC - 1):
                if const_expr(prod_round):
                    # production's gemm2 epilogue: bf16(acc * topk weight) per (expert, token)
                    r_ = [bf16_round(cacc[e] * dw) for e in range(4)]
                else:
                    r_ = [cacc[e] * dw for e in range(4)]
                tot = [tot[e] + ok.select(r_[e], fx.Float32(0.0)) for e in range(4)]
        # the shared expert's units: its mid rows are the table's entry nu
        srow_b = fx.Int32(fx.ptr_load(urow + (nu * 16 + l16)))
        sw = fx.ptr_load(uwt + (nu * 16 + l16))
        shacc = [fx.Float32(0.0) for _ in range(4)]
        for swv, ssc, skc, sok in sh_units:
            scu = mfma_scaled(
                a=swv,
                sa=ssc,
                b=lds_b8(DN4["sh8"] + srow_b * MID_ROW + skc * 32),
                sb=lds_i32(DN4["sh8sc"] + srow_b * MIDSC_ROW + skc * 4 + g4),
                c=fx.Vector.filled(4, 0.0, fx.Float32),
                a_fmt=FP8,
            )
            shacc = [
                shacc[e] + sok.select(scu[e] * sw, fx.Float32(0.0)) for e in range(4)
            ]
        fx.ptr_store(
            fx.Vector.from_elements(tot, fx.Float32), red + (wave * 64 + lane) * 4
        )
        fx.ptr_store(
            fx.Vector.from_elements(shacc, fx.Float32),
            fpool + (DN4["shp"] + (wave * 64 + lane) * 4),
        )
        gpu.barrier()
        if tid < DN_ROWS * tokens:
            k = tid // DN_ROWS
            grp = (tid % DN_ROWS) // 16
            rr = tid % 16
            waves = [grp * 4 + w for w in range(4)]
            routed = row_sum(red, rr, k, waves)
            shared = row_sum(fpool + DN4["shp"], rr, k, waves)
            if const_expr(prod_round):
                tot_k = bf16_round(routed) + bf16_round(shared)
            else:
                tot_k = routed + shared
            fx.ptr_store(bf16_round(tot_k), misc + tid)
        gpu.barrier()
        ffn_finish(td)

    return {
        "wide_down4": wide_down4,
        "wug_publish4": wug_publish4,
        "slot_of": slot_of,
        "build_experts": build_experts,
        "wug_load": wug_load,
        "wug_y": wug_y,
        "wug_mma": wug_mma,
        "wug_reduce": wug_reduce,
        "fp8_lanes4": fp8_lanes4,
        "wug_publish": wug_publish,
        "wug_publish16": wug_publish16,
        "down_tables": down_tables,
        "stage_moe_wide": stage_moe_wide,
        "wide_down": wide_down,
        "wide_down16": wide_down16,
    }
