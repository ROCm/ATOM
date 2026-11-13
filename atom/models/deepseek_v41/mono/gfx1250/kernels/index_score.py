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
    An FP4 index (c["idx4"]): keys flat * 64 (MXFP4, natural layout) with their 4 E8M0 codes at iscl + flat * 4, the
    query MXFP4 with its codes (iquant): one fp4 x fp4 scaled WMMA, total += relu(acc_i) * w (iw_out = w / 64).
"""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels import index_plan as IP
from atom.models.deepseek_v41.mono.gfx1250.kernels import mxfp4 as MX4
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


# LDS (bytes from the stage's base): four tasks' queries/head weights, up to
# three consecutive query rows per FULL task. REINDEX keeps one row per task.
QUERY_SLOTS = 12
O_Q = 512                        # (the first 512 B unused: kept so the layout matches the validated build)
O_W = O_Q + QUERY_SLOTS * HEADS * DIM
SCORE_LDS = O_W + QUERY_SLOTS * HEADS * 4


O_SCNT = SCORE_LDS               # per-row chunk counts [T]
O_SRB = O_SCNT + T * 4           # per-row context, bound and batch slot [T] each (row r: r, T + r, 2 T + r)
SCORE_RB = 4                     # tasks (rounds) a batch; <= 8 (the O_Q / O_W slots)
O_PF = O_SRB + 3 * T * 4         # a batch's key rows [RB][SUB][256] int64


@ASTRewriter.transform
def stage_score_general(c, isrx):
    """(see the module doc) isrx traced: FULL (context = the bound) and REINDEX (context = rx_context, the columns
    lifted through icand) with one code instance."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    TR = c["TR"]
    idx4 = c.get("idx4", False)
    GROUP = c.get("score_group", 1)
    assert GROUP in (1, 3) and SCORE_RB * GROUP <= QUERY_SLOTS
    if fx.const_expr(GROUP > 1):
        # The target's six-row verify blocks align to groups of three. A
        # REINDEX row has its own candidate list and cannot share these keys.
        qn = isrx.select(I32(1), I32(GROUP))
    else:
        qn = I32(1)
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
        task_cx = cxt
        if fx.const_expr(GROUP > 1):
            last = fx.min(tc + GROUP - 1, I32(TR - 1))
            bd_last = hx.gload(c["ibound"] + i64(last) * 4, I32)
            task_cx = isrx.select(cxt, fx.max(cxt, bd_last))
            tasks = (isrx | (tc % GROUP == 0)).select(IP.score_tasks(task_cx, TCOLS_), I32(0))
        else:
            tasks = IP.score_tasks(task_cx, TCOLS_)
        hx.lstore(ocnt + t * 4, tasks)
        hx.lstore(osr + t * 4, cxt)
        hx.lstore(osr + (T + t) * 4, bdt)
        hx.lstore(osr + (2 * T + t) * 4, qt)
    gpu.barrier()
    # the rows' inclusive chunk prefix in every wave's lanes (index_plan.task_prefix): no serial LDS walks
    ln = hx.fresh(t) % 32
    incs, nt = IP.task_prefix(lambda r_: hx.lload(ocnt + r_ * 4, I32), TR, ln, lambda v: TK.wave_incl_scan(v, ln),
                              lambda v: hx.lane_gather(v, I32(31)))
    iprod = c["iprod"]
    # a batch of RB of this CTA's tasks (rounds) at a time: their query rows / head weights staged in LDS slots
    # 0..RB-1, every tile's icand then block-table load issued together (2 trips a batch, not 2 a round: a long context
    # gives a CTA many rounds), the key rows to LDS; then a runtime loop over the batch's tasks: keys, WMMA
    RB = SCORE_RB

    def tmap(ii):
        r, ch = IP.task_of(incs, ii, TR, lambda p: TK._popc(TK._ballot(p)), TK.wave_max)
        return r, ch, hx.lload(osr + r * 4, I32), hx.lload(osr + (T + r) * 4, I32), fx.max(hx.lload(osr + (2 * T + r) * 4, I32), I32(0))

    nme = fx.max((nt - b + (CTAS - 1)) // CTAS, I32(0))
    for gb in range((nme + (RB - 1)) // RB):
        base_i = b + gb * (RB * CTAS)
        pcols = []
        krows = []
        qs = []
        bds = []
        rr = []
        for k in range_constexpr(RB):
            r0_, ch0, cx0, bd0, q0 = tmap(fx.min(base_i + k * CTAS, nt - 1))
            for qr in range(qn):
                rq = r0_ + qr
                slot = k * GROUP + qr
                if fx.const_expr(idx4):   # MXFP4 query (2 KB) then its 128 B of scale codes
                    if t < HEADS * DIM // 2 // 16:
                        hx.lstore(sm + O_Q + slot * (HEADS * DIM) + t * 16, hx.gload(c["iq_out"] + i64(rq * HEADS * DIM // 2) + i64(t) * 16, I32, 4))
                    if t < HEADS:
                        hx.lstore(sm + O_Q + slot * (HEADS * DIM) + HEADS * DIM // 2 + t * 4,
                                  hx.gload(c["iqs_out"] + i64((rq * HEADS + t) * 4), I32))
                else:
                    hx.lstore(sm + O_Q + slot * (HEADS * DIM) + t * 16, hx.gload(c["iq_out"] + i64(rq * HEADS * DIM) + i64(t) * 16, I32, 4))
                if t < HEADS:
                    hx.lstore(sm + O_W + slot * (HEADS * 4) + t * 4, hx.gload(c["iw_out"] + i64(rq * HEADS + t) * 4, F32))
            if fx.const_expr(GROUP > 1):
                bd_last = hx.lload(osr + (T + fx.min(r0_ + GROUP - 1, I32(TR - 1))) * 4, I32)
                bd0 = isrx.select(bd0, fx.max(bd0, bd_last))
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
                if fx.const_expr(idx4):
                    kbs.append(MX4.ld8(c["iplane"] + flat * 64, l, lambda a_: hx.gload(a_, I32, 4)))
                    kss.append(hx.gload(c["iscl"] + flat * 4, I32))
                else:
                    kbase = c["iplane"] + (flat // 8) * 1056
                    tt = flat % 8
                    kp = [hx.gload(kbase + i64(((l // 16) + 2 * cc) * 128) + tt * 16, I32, 4) for cc in range(4)]
                    kbs.append(fx.Vector.from_elements([kp[cc][e] for cc in range(4) for e in range(4)], I32))
                    kss.append(hx.gload(kbase + 1024 + tt * 4, F32))
                cols.append(col)
            for qr in range(qn):
                rq = r + qr
                slot = k * GROUP + qr
                cx = hx.lload(osr + rq * 4, I32)
                bd = hx.lload(osr + (T + rq) * 4, I32)
                qvs = []
                qss = []
                for mi in range_constexpr(2):
                    if fx.const_expr(idx4):
                        qrow = sm + O_Q + slot * (HEADS * DIM) + (mi * 16 + l % 16) * (DIM // 2)
                        qvs.append(MX4.ld8(qrow, l, lambda a_: hx.lload(a_, I32, 4)))
                        qss.append(hx.lload(sm + O_Q + slot * (HEADS * DIM) + HEADS * DIM // 2 + (mi * 16 + l % 16) * 4, I32))
                    else:
                        qrow = sm + O_Q + slot * (HEADS * DIM) + (mi * 16 + l % 16) * DIM
                        qp = [hx.lload(qrow + (l // 16) * 16 + cc * 32, I32, 4) for cc in range(4)]
                        qvs.append(fx.Vector.from_elements([qp[cc][e] for cc in range(4) for e in range(4)], I32))
                hwvs = [hx.lload(sm + O_W + slot * (HEADS * 4) + (mi * 16 + (l // 16) * 8) * 4, F32, 8) for mi in range(2)]
                for u in range_constexpr(SUB_):
                    col = cols[u]
                    tot = F32(0.0)
                    for mi in range_constexpr(2):
                        if fx.const_expr(idx4):
                            acc = MX4.wmma4(qvs[mi], qss[mi], kbs[u], kss[u], fx.Vector.filled(8, 0.0, F32))
                            for i_ in range_constexpr(8):
                                tot = tot + fx.max(acc[i_], F32(0.0)) * hwvs[mi][i_]
                        else:
                            acc = hx.wmma_mx(qvs[mi], ONE4, kbs[u], ONE4, fx.Vector.filled(8, 0.0, F32))
                            for i_ in range_constexpr(8):
                                tot = tot + fx.max(acc[i_] * kss[u], F32(0.0)) * hwvs[mi][i_]
                    sc = tot + tot.shuffle_xor(16, 32)
                    v = (col < cx).select(sc, F32(float("-inf")))
                    if live & (l < 16) & (col < ((cx + (CHUNK - 1)) // CHUNK) * CHUNK):   # (as before: the context's whole 128-column chunks)
                        c["stx"](c["ilog"] + i64(rq * c["ilw"] + col) * 4, v)
                    if iprod:   # layer 20: the 8-row block maxima (NaN ignored, capped finite, the newest block +inf)
                        m = (v != v).select(F32(float("-inf")), v)
                        for o_ in range_constexpr(3):
                            m = fx.max(m, m.shuffle_xor(1 << o_, 32))
                        m = fx.min(m, F32(FLT_MAX))
                        bk = col // 8
                        nblk = (bd + 7) // 8
                        m = (bk == nblk - 1).select(F32(float("inf")), m)
                        if live & (l < 16) & (l % 8 == 0) & (bk < nblk):
                            c["stx"](c["ibmax"] + i64(rq * (c["ilw"] // 8) + bk) * 4, m)

        gpu.barrier()


@ASTRewriter.transform
def stage_score_query_resident(c):
    """FULL FP4 GROUP3: one fixed query group per CTA, weighted key splits.

    Each nonempty group gets one CTA and a cumulative-floor share of the
    remaining CTAs. Four key-task page lookahead and SUB key fragments stay
    unchanged; query operands and weights are prepared once before that loop.
    REINDEX must use stage_score_general because candidate tables differ by row.
    """
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    TR, GROUP, RB = c["TR"], 3, SCORE_RB
    assert c.get("idx4", False) and TR % GROUP == 0
    SUB_ = IP.score_sub(TR, CTAS)
    TCOLS_ = SUB_ * CHUNK
    tc = fx.min(t, I32(TR - 1))
    bound = fx.max(hx.gload(c["ibound"] + i64(tc) * 4, I32), I32(0))
    batch = hx.gload(c["ibat"] + i64(tc) * 4, I32)
    osr, ocnt = sm + O_SRB, sm + O_SCNT
    if t < TR:
        last = fx.min(tc + GROUP - 1, I32(TR - 1))
        maximum = fx.max(bound, hx.gload(c["ibound"] + i64(last) * 4, I32))
        count = (tc % GROUP == 0).select(IP.score_tasks(maximum, TCOLS_), I32(0))
        hx.lstore(ocnt + t * 4, count)
        hx.lstore(osr + t * 4, bound)
        hx.lstore(osr + (T + t) * 4, batch)
    gpu.barrier()
    ln = hx.fresh(t) % 32
    task_inc, total = IP.task_prefix(lambda r: hx.lload(ocnt + r * 4, I32), TR, ln,
                                    lambda v: TK.wave_incl_scan(v, ln), lambda v: hx.lane_gather(v, I32(31)))
    active_inc, active = IP.task_prefix(lambda r: (hx.lload(ocnt + r * 4, I32) > 0).select(I32(1), I32(0)), TR, ln,
                                      lambda v: TK.wave_incl_scan(v, ln), lambda v: hx.lane_gather(v, I32(31)))
    if total > 0:
        allocations = []
        for g in range_constexpr(IP.row_groups(TR)):
            valid = ln + 32 * g < TR
            allocations.append(valid.select(active_inc[g] + (CTAS - active) * task_inc[g] // total, I32(CTAS)))
        row, split = IP.task_of(allocations, b, TR, lambda p: TK._popc(TK._ballot(p)), TK.wave_max)
        end = I32(0)
        for g in range_constexpr(IP.row_groups(TR)):
            allocation_end = hx.lane_gather(allocations[g], row % 32)
            end = (row // 32 == g).select(allocation_end, end)
        splits = end - b + split
        maximum = fx.max(hx.lload(osr + row * 4, I32), hx.lload(osr + (row + GROUP - 1) * 4, I32))
        tasks = IP.score_tasks(maximum, TCOLS_)
        batch = fx.max(hx.lload(osr + (T + row) * 4, I32), I32(0))
        nme = fx.max((tasks - split + splits - 1) // splits, I32(0))
        if nme > 0:
            for qr in range_constexpr(GROUP):
                rq = row + qr
                if t < HEADS * DIM // 2 // 16:
                    hx.lstore(sm + O_Q + qr * HEADS * DIM + t * 16,
                              hx.gload(c["iq_out"] + i64(rq * HEADS * DIM // 2) + i64(t) * 16, I32, 4))
                if t < HEADS:
                    hx.lstore(sm + O_Q + qr * HEADS * DIM + HEADS * DIM // 2 + t * 4,
                              hx.gload(c["iqs_out"] + i64((rq * HEADS + t) * 4), I32))
                    hx.lstore(sm + O_W + qr * HEADS * 4 + t * 4,
                              hx.gload(c["iw_out"] + i64(rq * HEADS + t) * 4, F32))
            gpu.barrier()
            qa, qb, sa, sb, wa, wb, bounds = [], [], [], [], [], [], []
            for qr in range_constexpr(GROUP):
                base = sm + O_Q + qr * HEADS * DIM
                qa.append(MX4.ld8(base + (l % 16) * (DIM // 2), l, lambda a: hx.lload(a, I32, 4)))
                qb.append(MX4.ld8(base + (16 + l % 16) * (DIM // 2), l, lambda a: hx.lload(a, I32, 4)))
                sa.append(hx.lload(base + HEADS * DIM // 2 + (l % 16) * 4, I32))
                sb.append(hx.lload(base + HEADS * DIM // 2 + (16 + l % 16) * 4, I32))
                wa.append(hx.lload(sm + O_W + qr * HEADS * 4 + (l // 16) * 8 * 4, F32, 8))
                wb.append(hx.lload(sm + O_W + qr * HEADS * 4 + (16 + (l // 16) * 8) * 4, F32, 8))
                bounds.append(hx.lload(osr + (row + qr) * 4, I32))
            for gb in range((nme + RB - 1) // RB):
                key_rows = []
                for k in range_constexpr(RB):
                    chunk = fx.min(split + (gb * RB + k) * splits, tasks - 1)
                    for u in range_constexpr(SUB_):
                        col = chunk * TCOLS_ + u * CHUNK + w * 16 + l % 16
                        key_rows.append(IP.key_row(col, maximum))
                pages = [hx.gload(c["btab"] + (i64(batch) * i64(c["bstride"]) + i64(key_rows[j] // c["irpp"])) * 4, I32)
                         for j in range(RB * SUB_)]
                for j in range_constexpr(RB * SUB_):
                    hx.lstore(sm + O_PF + (j * 256 + t) * 8, i64(pages[j]) * i64(c["irpp"]) + i64(key_rows[j] % c["irpp"]))
                gpu.barrier()
                for k in range(fx.min(nme - gb * RB, I32(RB))):
                    chunk = split + (gb * RB + k) * splits
                    keys, scales = [], []
                    for u in range_constexpr(SUB_):
                        flat = hx.lload(sm + O_PF + ((k * SUB_ + u) * 256 + t) * 8, I64)
                        keys.append(MX4.ld8(c["iplane"] + flat * 64, l, lambda a: hx.gload(a, I32, 4)))
                        scales.append(hx.gload(c["iscl"] + flat * 4, I32))
                    for qr in range_constexpr(GROUP):
                        cx = bounds[qr]
                        rq = row + qr
                        for u in range_constexpr(SUB_):
                            col = chunk * TCOLS_ + u * CHUNK + w * 16 + l % 16
                            acc0 = MX4.wmma4(qa[qr], sa[qr], keys[u], scales[u], fx.Vector.filled(8, 0.0, F32))
                            total_score = F32(0.0)
                            for i in range_constexpr(8):
                                total_score = total_score + fx.max(acc0[i], F32(0.0)) * wa[qr][i]
                            acc1 = MX4.wmma4(qb[qr], sb[qr], keys[u], scales[u], fx.Vector.filled(8, 0.0, F32))
                            for i in range_constexpr(8):
                                total_score = total_score + fx.max(acc1[i], F32(0.0)) * wb[qr][i]
                            score = total_score + total_score.shuffle_xor(16, 32)
                            value = (col < cx).select(score, F32(float("-inf")))
                            if (l < 16) & (col < ((cx + CHUNK - 1) // CHUNK) * CHUNK):
                                c["stx"](c["ilog"] + i64(rq * c["ilw"] + col) * 4, value)
                            if c["iprod"]:
                                m = (value != value).select(F32(float("-inf")), value)
                                for o in range_constexpr(3):
                                    m = fx.max(m, m.shuffle_xor(1 << o, 32))
                                m = fx.min(m, F32(FLT_MAX))
                                block = col // 8
                                m = (block == ((cx + 7) // 8 - 1)).select(F32(float("inf")), m)
                                if (l < 16) & (l % 8 == 0) & (block < ((cx + 7) // 8)):
                                    c["stx"](c["ibmax"] + i64(rq * (c["ilw"] // 8) + block) * 4, m)
                gpu.barrier()


@ASTRewriter.transform
def stage_score_any(c, isrx):
    if fx.const_expr(c.get("score_query_resident", False)):
        if isrx:
            stage_score_general(c, isrx)
        else:
            stage_score_query_resident(c)
    else:
        stage_score_general(c, isrx)
