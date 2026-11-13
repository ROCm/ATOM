# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The indexer's scores (FULL layers; REINDEX the same over the candidate table's compacted columns).

    stage_score_any: a task = (row r with context > 512, SUB chunks of 128 columns: a wave's SUB key tiles in flight); the row's FP8 query and head weights
    staged in LDS, a wave a 16-column key tile: its keys' FP8 values (the plane's 8-row tiles: dims 16j..16j+15 of tile
    row tt at 128 j + 16 tt) and fp32 scale (at 1024 + 4 tt), acc = WMMA_fp8(q[r, heads mi*16..], keys) (scale codes
    1.0: the query scale is folded into iw_out), total += relu(acc_i * kscale) * w[r, head] over the lane's 8 heads,
    both head tiles, then + the lane 16 apart (the native gfx1250 reduction order); score -> ilog[r][col], -inf
    past the row's context. Layer 20 (iprod) also writes the 8-row block maxima -> ibmax.
    Columns map to plane rows through the request's block table: page = btab[q][col // rpp], flat = page rpp + col % rpp.
"""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels import index_plan as IP
from atom.models.deepseek_v41.mono.gfx1250.kernels import topk as TK
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import CTAS, T

F32, I32, I64 = fx.Float32, fx.Int32, fx.Int64
HEADS, DIM = D.INDEX_HEADS, D.INDEX_DIM
CHUNK, TOPK, SUB = IP.CHUNK, IP.TOPK, IP.SUB   # (a wave: SUB 16-column key tiles a task, their loads in flight together)
TCOLS = SUB * CHUNK
ONE4 = 0x7F7F7F7F                 # four E8M0 codes of 1.0
FLT_MAX = 3.4028234663852886e38


def i64(x):
    return fx.Int64(x)


# LDS (bytes from the stage's base): the task's row's query and head weights (room for 8 rows), the per-row chunk counts
O_Q = 512                        # (the first 512 B unused: kept so the layout matches the validated build)
O_W = O_Q + 8 * HEADS * DIM
SCORE_LDS = O_W + 8 * HEADS * 4


O_SCNT = SCORE_LDS               # per-row chunk counts [TR]
O_SRB = O_SCNT + 64 * 4          # per-row context, bound and batch slot [TR] each
SCORE_RB = 4                     # tasks (rounds) a batch; <= 8 (the O_Q / O_W slots)
O_PF = O_SRB + 3 * 64 * 4        # a batch's key rows [RB][SUB][256] int64


@ASTRewriter.transform
def stage_score_any(c, isrx):
    """(see the module doc) isrx traced: FULL (context = the bound) and REINDEX (context = rx_context, the columns
    lifted through icand) with one code instance."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    TR = c["TR"]
    SUB_ = IP.score_sub(TR, CTAS)
    TCOLS_ = SUB_ * CHUNK
    ocnt = sm + O_SCNT
    tc = fx.min(t, I32(TR - 1))
    bdt = hx.gload(c["ibound"] + i64(tc) * 4, I32)
    cxt = fx.max(bdt, I32(0))
    if isrx:
        cxt = TK.rx_context(c, tc, bdt)
    qt = hx.gload(c["ibat"] + i64(tc) * 4, I32)
    osr = sm + O_SRB
    if t < TR:   # the task loop reads the row's context / bound / batch slot from LDS (no dependent global loads)
        hx.lstore(ocnt + t * 4, IP.score_tasks(cxt, TCOLS_))
        hx.lstore(osr + t * 4, cxt)
        hx.lstore(osr + (64 + t) * 4, bdt)
        hx.lstore(osr + (128 + t) * 4, qt)
    gpu.barrier()
    # the rows' inclusive chunk prefix in every wave's lanes (rows lane, lane + 32; TR <= 64): no serial LDS walks
    ln = hx.fresh(t) % 32
    c0 = hx.lload(ocnt + fx.min(ln, I32(TR - 1)) * 4, I32)
    c1 = hx.lload(ocnt + fx.min(ln + 32, I32(TR - 1)) * 4, I32)
    c0 = (ln < TR).select(c0, I32(0))
    c1 = (ln + 32 < TR).select(c1, I32(0))
    inc0 = TK.wave_incl_scan(c0, ln)
    tot0 = hx.lane_gather(inc0, I32(31))
    inc1 = TK.wave_incl_scan(c1, ln) + tot0
    nt = hx.lane_gather(inc1, I32(31))
    iprod = c["iprod"]
    # a batch of RB of this CTA's tasks (rounds) at a time: their query rows / head weights staged in LDS slots
    # 0..RB-1, every tile's icand then block-table load issued together (2 trips a batch, not 2 a round: a long context
    # gives a CTA many rounds), the key rows to LDS; then a runtime loop over the batch's tasks: keys, WMMA
    RB = SCORE_RB

    def tmap(ii):
        p0 = inc0 <= ii
        p1 = inc1 <= ii
        r = fx.min(TK._popc(TK._ballot(p0)) + TK._popc(TK._ballot(p1)), I32(TR - 1))
        ch = ii - TK.wave_max(fx.max(p0.select(inc0, I32(0)), p1.select(inc1, I32(0))))
        return r, ch, hx.lload(osr + r * 4, I32), hx.lload(osr + (64 + r) * 4, I32), fx.max(hx.lload(osr + (128 + r) * 4, I32), I32(0))

    nme = fx.max((nt - b + (CTAS - 1)) // CTAS, I32(0))
    def pages_to_lds(base_i):   # the batch's key rows: compacted column -> real row (REINDEX), -> page -> LDS
        pcols = []
        qs = []
        bds = []
        rr = []
        for k in range_constexpr(RB):
            r0_, ch0, cx0, bd0, q0 = tmap(fx.min(base_i + k * CTAS, nt - 1))
            for u in range_constexpr(SUB_):
                pcols.append(ch0 * TCOLS_ + u * CHUNK + w * 16 + l % 16)
                qs.append(q0)
                bds.append(bd0)
                rr.append(r0_)
        krows = pcols
        if isrx:   # compacted column -> its candidate block's real row
            blks = [hx.gload(c["icand"] + i64(rr[j] * 2048 + fx.min(pcols[j] // 8, I32(2047))) * 4, I32) for j in range(RB * SUB_)]
            krows = [fx.max(blks[j], I32(0)) * 8 + pcols[j] % 8 for j in range(RB * SUB_)]
        pages = [hx.gload(c["btab"] + (i64(qs[j]) * i64(c["bstride"]) + i64(IP.key_row(krows[j], bds[j]) // c["irpp"])) * 4, I32)
                 for j in range(RB * SUB_)]
        for j in range_constexpr(RB * SUB_):
            hx.lstore(sm + O_PF + (j * 256 + t) * 8, i64(pages[j]) * i64(c["irpp"]) + i64(krows[j] % c["irpp"]))

    # the first batch's page chain (icand -> block table: this step's tables, not this layer's outputs) before
    # the wait for the query (c["prewait"]: the INDEX_QUANT handoff), every CTA passes it
    if fx.const_expr("prewait" in c):
        if nme > 0:
            pages_to_lds(b)
        c["prewait"]()
    for gb in range((nme + (RB - 1)) // RB):
        base_i = b + gb * (RB * CTAS)
        for k in range_constexpr(RB):
            r0_, ch0, cx0, bd0, q0 = tmap(fx.min(base_i + k * CTAS, nt - 1))
            hx.lstore(sm + O_Q + k * (HEADS * DIM) + t * 16, hx.gload(c["iq_out"] + i64(r0_ * HEADS * DIM) + i64(t) * 16, I32, 4))
            if t < HEADS:
                hx.lstore(sm + O_W + k * (HEADS * 4) + t * 4, hx.gload(c["iw_out"] + i64(r0_ * HEADS + t) * 4, F32))
        if fx.const_expr("prewait" in c):
            if gb > 0:
                pages_to_lds(base_i)
        else:
            pages_to_lds(base_i)
        gpu.barrier()
        for k in range(fx.min(nme - gb * RB, I32(RB))):   # (this batch's tasks: a 1-round CTA runs one)
            ik = base_i + k * CTAS
            live = ik < nt   # (a short last batch: the clamped task recomputed, its stores skipped)
            r, ch, cx, bd, q = tmap(fx.min(ik, nt - 1))
            kbs = []
            kss = []
            cols = []
            for u in range_constexpr(SUB_):
                col = ch * TCOLS_ + u * CHUNK + w * 16 + l % 16
                flat = hx.lload(sm + O_PF + ((k * SUB_ + u) * 256 + t) * 8, I64)
                kbase = c["iplane"] + (flat // 8) * 1056
                tt = flat % 8
                kp = [hx.gload(kbase + i64(((l // 16) + 2 * cc) * 128) + tt * 16, I32, 4) for cc in range(4)]
                kbs.append(fx.Vector.from_elements([kp[cc][e] for cc in range(4) for e in range(4)], I32))
                kss.append(hx.gload(kbase + 1024 + tt * 4, F32))
                cols.append(col)
            qvs = []
            for mi in range_constexpr(2):
                qrow = sm + O_Q + k * (HEADS * DIM) + (mi * 16 + l % 16) * DIM
                qp = [hx.lload(qrow + (l // 16) * 16 + cc * 32, I32, 4) for cc in range(4)]
                qvs.append(fx.Vector.from_elements([qp[cc][e] for cc in range(4) for e in range(4)], I32))
            hwvs = [hx.lload(sm + O_W + k * (HEADS * 4) + (mi * 16 + (l // 16) * 8) * 4, F32, 8) for mi in range(2)]
            for u in range_constexpr(SUB_):
                col = cols[u]
                tot = F32(0.0)
                for mi in range_constexpr(2):
                    acc = hx.wmma_mx(qvs[mi], ONE4, kbs[u], ONE4, fx.Vector.filled(8, 0.0, F32))
                    for i_ in range_constexpr(8):
                        tot = tot + fx.max(acc[i_] * kss[u], F32(0.0)) * hwvs[mi][i_]
                sc = tot + tot.shuffle_xor(16, 32)
                v = (col < cx).select(sc, F32(float("-inf")))
                if live & (l < 16) & (col < ((cx + (CHUNK - 1)) // CHUNK) * CHUNK):   # (as before: the context's whole 128-column chunks)
                    c["stx"](c["ilog"] + i64(r * c["ilw"] + col) * 4, v)
                if iprod:   # layer 20: the 8-row block maxima (NaN ignored, capped finite, the newest block +inf)
                    m = (v != v).select(F32(float("-inf")), v)
                    for o_ in range_constexpr(3):
                        m = fx.max(m, m.shuffle_xor(1 << o_, 32))
                    m = fx.min(m, F32(FLT_MAX))
                    bk = col // 8
                    nblk = (bd + 7) // 8
                    m = (bk == nblk - 1).select(F32(float("inf")), m)
                    if live & (l < 16) & (l % 8 == 0) & (bk < nblk):
                        c["stx"](c["ibmax"] + i64(r * (c["ilw"] // 8) + bk) * 4, m)

        gpu.barrier()


GROUPED_MIN_BOUND = 8192   # index rows past which a c1 FULL layer scores its key tiles once for every row


@ASTRewriter.transform
def stage_score_grouped(c):
    """FULL layers at c1 (one request, its TR rows over the same keys): a task = one 128-column chunk of the request's
    longest row; its keys are read once and scored for every row (each row masked to its own context), not once a
    row -- the per-row tasks read the same key tiles TR times (at a 64K context ~50 MB a layer). Same scores, same
    stores (ilog; layer 20's ibmax) as stage_score_any; c["prewait"] (the INDEX_QUANT handoff) runs on every CTA."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    TR = c["TR"]
    osr = sm + O_SRB
    tc = fx.min(t, I32(TR - 1))
    bdt = hx.gload(c["ibound"] + i64(tc) * 4, I32)
    if t < TR:
        hx.lstore(osr + t * 4, fx.max(bdt, I32(0)))
        hx.lstore(osr + (64 + t) * 4, bdt)
    q = fx.max(hx.gload(c["ibat"], I32), I32(0))   # (every row is this request's)
    gpu.barrier()
    ln = hx.fresh(t) % 32
    cmax = TK.wave_max((ln < TR).select(hx.lload(osr + fx.min(ln, I32(TR - 1)) * 4, I32), I32(0)))
    bmax = TK.wave_max((ln < TR).select(hx.lload(osr + (64 + fx.min(ln, I32(TR - 1))) * 4, I32), I32(0)))
    nt = IP.score_tasks(cmax, I32(CHUNK))
    nme = fx.max((nt - b + (CTAS - 1)) // CTAS, I32(0))
    iprod = c["iprod"]

    def page_row(task):
        col = task * CHUNK + w * 16 + l % 16
        krow = IP.key_row(col, bmax)
        page = hx.gload(c["btab"] + (i64(q) * i64(c["bstride"]) + i64(krow // c["irpp"])) * 4, I32)
        return i64(page) * i64(c["irpp"]) + i64(krow % c["irpp"])

    flat0 = page_row(fx.min(b, fx.max(nt - 1, I32(0))))   # (the first task's page chain before the wait for the query)
    if fx.const_expr("prewait" in c):
        c["prewait"]()
    for k in range_constexpr(TR):   # every row's query and head weights -> LDS slots 0..TR-1
        hx.lstore(sm + O_Q + k * (HEADS * DIM) + t * 16, hx.gload(c["iq_out"] + i64(k * HEADS * DIM) + i64(t) * 16, I32, 4))
        if t < HEADS:
            hx.lstore(sm + O_W + k * (HEADS * 4) + t * 4, hx.gload(c["iw_out"] + i64(k * HEADS + t) * 4, F32))
    gpu.barrier()
    for i in range(nme):
        task = b + i * CTAS
        col = task * CHUNK + w * 16 + l % 16
        flat = (i == I32(0)).select(flat0, page_row(task))
        kbase = c["iplane"] + (flat // 8) * 1056
        tt = flat % 8
        kp = [hx.gload(kbase + i64(((l // 16) + 2 * cc) * 128) + tt * 16, I32, 4) for cc in range(4)]
        kb = fx.Vector.from_elements([kp[cc][e] for cc in range(4) for e in range(4)], I32)
        ks = hx.gload(kbase + 1024 + tt * 4, F32)
        for r in range_constexpr(TR):
            cx = hx.lload(osr + r * 4, I32)
            bd = hx.lload(osr + (64 + r) * 4, I32)
            tot = F32(0.0)
            for mi in range_constexpr(2):
                qrow = sm + O_Q + r * (HEADS * DIM) + (mi * 16 + l % 16) * DIM
                qp = [hx.lload(qrow + (l // 16) * 16 + cc * 32, I32, 4) for cc in range(4)]
                qv = fx.Vector.from_elements([qp[cc][e] for cc in range(4) for e in range(4)], I32)
                hw = hx.lload(sm + O_W + r * (HEADS * 4) + (mi * 16 + (l // 16) * 8) * 4, F32, 8)
                acc = hx.wmma_mx(qv, ONE4, kb, ONE4, fx.Vector.filled(8, 0.0, F32))
                for i_ in range_constexpr(8):
                    tot = tot + fx.max(acc[i_] * ks, F32(0.0)) * hw[i_]
            sc = tot + tot.shuffle_xor(16, 32)
            v = (col < cx).select(sc, F32(float("-inf")))
            if (cx > I32(TOPK)) & (l < 16) & (col < ((cx + (CHUNK - 1)) // CHUNK) * CHUNK):
                c["stx"](c["ilog"] + i64(r * c["ilw"] + col) * 4, v)
            if iprod:
                m = (v != v).select(F32(float("-inf")), v)
                for o_ in range_constexpr(3):
                    m = fx.max(m, m.shuffle_xor(1 << o_, 32))
                m = fx.min(m, F32(FLT_MAX))
                bk = col // 8
                nblk = (bd + 7) // 8
                m = (bk == nblk - 1).select(F32(float("inf")), m)
                if (cx > I32(TOPK)) & (l < 16) & (l % 8 == 0) & (bk < nblk):
                    c["stx"](c["ibmax"] + i64(r * (c["ilw"] // 8) + bk) * 4, m)
    gpu.barrier()
