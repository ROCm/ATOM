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


def fma(a, b, c):
    return F32(hx.intrin(F32.ir_type, "llvm.fma.f32", [a, b, c]))


# FP4 (index_fp4): a row's query slot in O_Q keeps its 4096 B: the 32 heads' 64 B E2M1 at +0, their E8M0 dwords at +2048
QS_OFF = 2048


def _slot_row(rows, s):
    """The row of LDS query slot s (traced): rows[s] (ints or traced rows)."""
    r = I32(rows[0]) if isinstance(rows[0], int) else rows[0]
    for k in range(1, len(rows)):
        r = (s == I32(k)).select(I32(rows[k]) if isinstance(rows[k], int) else rows[k], r)
    return r


@ASTRewriter.transform
def stage_q_slots(c, sm, t, rows):
    """The rows' queries (FP4: their E8M0 scales too) and head weights -> LDS slots 0 .. len(rows) - 1 (slot k: row
    rows[k]). Every load is issued first, then the LDS stores: one load round trip for all the slots, not one a slot and
    kind behind its thread guard (6 FP4 rows: 12 serial load -> wait -> store trips, 2.3 us cold and hot on the block-4
    fixture). A thread past the items re-stores the last one (clamped index: the same bytes)."""
    NS = len(rows)
    NCH = HEADS * DIM // 32 if c["ifp4"] else 256        # 16-B chunks a slot's values
    RSB = HEADS * DIM // 2 if c["ifp4"] else HEADS * DIM   # a row's value bytes in iq_out
    vs = []
    for j in range_constexpr((NS * NCH + 255) // 256):
        u = fx.min(t + I32(j * 256), I32(NS * NCH - 1))
        sl = u // NCH
        vs.append((sl, u % NCH, hx.gload(c["iq_out"] + i64(_slot_row(rows, sl)) * RSB + i64(u % NCH) * 16, I32, 4)))
    us = fx.min(t, I32(NS * HEADS - 1))
    su = us // HEADS
    hu = us % HEADS
    ru = _slot_row(rows, su)
    wv = hx.gload(c["iw_out"] + i64(ru * HEADS + hu) * 4, F32)
    if fx.const_expr(c["ifp4"]):
        sv = hx.gload(c["iqs_out"] + i64(ru * HEADS + hu) * 4, I32)
    for (sl, p, v) in vs:
        hx.lstore(sm + O_Q + sl * (HEADS * DIM) + p * 16, v)
    if fx.const_expr(c["ifp4"]):
        hx.lstore(sm + O_Q + su * (HEADS * DIM) + QS_OFF + hu * 4, sv)
    hx.lstore(sm + O_W + su * (HEADS * 4) + hu * 4, wv)


def fp4_key(c, flat, l):
    """FP4 natural plane: key row flat's B operand (lane l: K blocks l // 16 and l // 16 + 2, 16 B each) and its
    4 E8M0 codes (one dword)."""
    row = c["iplane"] + flat * 64
    k0 = hx.gload(row + i64((l // 16) * 16), I32, 4)
    k1 = hx.gload(row + i64(32 + (l // 16) * 16), I32, 4)
    z = I32(0)
    kv = fx.Vector.from_elements([k0[e] for e in range(4)] + [k1[e] for e in range(4)] + [z] * 8, I32)
    return kv, hx.gload(c["iscale"] + flat * 4, I32)


def fp4_score(sm, k, l, kv, ksc):
    """Row slot k's 32 heads against a lane's key tile: native gfx1250 order (pa_mqa_logits_mxfp4): even / odd fma
    chains of relu(acc) * w over heads mi*16 + 8 (l // 16) + i, then (even + odd) + the lane 16 apart."""
    se = F32(0.0)
    so = F32(0.0)
    for mi in range_constexpr(2):
        qrow = sm + O_Q + k * (HEADS * DIM) + (mi * 16 + l % 16) * 64
        q0 = hx.lload(qrow + (l // 16) * 16, I32, 4)
        q1 = hx.lload(qrow + 32 + (l // 16) * 16, I32, 4)
        z = I32(0)
        qv = fx.Vector.from_elements([q0[e] for e in range(4)] + [q1[e] for e in range(4)] + [z] * 8, I32)
        qsc = hx.lload(sm + O_Q + k * (HEADS * DIM) + QS_OFF + (mi * 16 + l % 16) * 4, I32)
        hw = hx.lload(sm + O_W + k * (HEADS * 4) + (mi * 16 + (l // 16) * 8) * 4, F32, 8)
        acc = hx.wmma_f4f4(qv, qsc, kv, ksc, fx.Vector.filled(8, 0.0, F32))
        for p_ in range_constexpr(4):
            se = fma(fx.max(acc[2 * p_], F32(0.0)), hw[2 * p_], se)
            so = fma(fx.max(acc[2 * p_ + 1], F32(0.0)), hw[2 * p_ + 1], so)
    sc = se + so
    return sc + sc.shuffle_xor(16, 32)



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
def rx_prefill(c):
    """(P2) A REINDEX layer's first page batch of this CTA (stage_score_any's pages_to_lds(b): compacted column ->
    icand -> real row -> block table -> O_PF, the same values) computed by waves 6 and 7 while waves 0..5 run iq / iw
    (no LDS there, no barrier): the scorer then skips that chain before its INDEX_QUANT wait. Wave-local: every row's
    context / bound / slot in the lanes (TR <= 32), the task map by ballots as tmap. 64 threads x RB*SUB*128/64 items:
    item q = (w - 6) * 32 + l + 64 i -> slot j = q // 128 (task k = j // SUB, chunk u = j % SUB), column w' 16 + c16 =
    q % 128, stored for threads w' 32 + c16 and + 16 (the lanes 16 apart share a column)."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    TR = c["TR"]
    assert TR <= 32
    SUB_ = IP.score_sub(TR, CTAS)
    TCOLS_ = SUB_ * CHUNK
    RB = SCORE_RB
    ln = l
    tc = fx.min(ln, I32(TR - 1))
    bdt = hx.gload(c["ibound"] + i64(tc) * 4, I32)
    cxt = TK.rx_context(c, tc, bdt)
    qt = fx.max(hx.gload(c["ibat"] + i64(tc) * 4, I32), I32(0))
    cnt = (ln < TR).select(IP.score_tasks(cxt, TCOLS_), I32(0))
    inc = TK.wave_incl_scan(cnt, ln)
    nt = hx.lane_gather(inc, I32(31))
    nme = fx.max((nt - b + (CTAS - 1)) // CTAS, I32(0))
    if nme > 0:
        NI = RB * SUB_ * 128 // 64
        NG = 4   # (items a group: their icand then block-table loads in flight together; the groups a runtime loop)
        for gi in range(NI // NG):
            rows, pcols, qs, bds, qq = [], [], [], [], []
            for i in range_constexpr(NG):
                q_ = (w - 6) * 32 + ln + 64 * (gi * NG + i)
                j = q_ // 128
                ii = fx.min(b + (j // SUB_) * CTAS, nt - 1)
                p0 = inc <= ii
                r = fx.min(TK._popc(TK._ballot(p0)), I32(TR - 1))
                ch = ii - TK.wave_max(p0.select(inc, I32(0)))
                rows.append(r)
                pcols.append(ch * TCOLS_ + (j % SUB_) * CHUNK + q_ % 128)
                qs.append(hx.lane_gather(qt, r))
                bds.append(hx.lane_gather(bdt, r))
                qq.append(q_)
            blks = [hx.gload(c["icand"] + i64(rows[i] * 2048 + fx.min(pcols[i] // 8, I32(2047))) * 4, I32) for i in range(NG)]
            krows = [fx.max(blks[i], I32(0)) * 8 + pcols[i] % 8 for i in range(NG)]
            pages = [hx.gload(c["btab"] + (i64(qs[i]) * i64(c["bstride"]) + i64(IP.key_row(krows[i], bds[i]) // c["irpp"])) * 4, I32)
                     for i in range(NG)]
            for i in range_constexpr(NG):
                j = qq[i] // 128
                cw = qq[i] % 128
                tt = (cw // 16) * 32 + cw % 16
                v = i64(pages[i]) * i64(c["irpp"]) + i64(krows[i] % c["irpp"])
                hx.lstore(sm + O_PF + (j * 256 + tt) * 8, v)
                hx.lstore(sm + O_PF + (j * 256 + tt + 16) * 8, v)


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
    # the wait for the query (c["prewait"]: the INDEX_QUANT handoff), every CTA passes it -- unless rx_prefill (P2) put
    # it in LDS already, during iq / iw (c["rxpre"]: true on a REINDEX layer whose prefill ran)
    if fx.const_expr("prewait" in c):
        if fx.const_expr("rxpre" in c):
            if (nme > 0) & (c["rxpre"] == (fx.Int32(0) != fx.Int32(0))):
                pages_to_lds(b)
        else:
            if nme > 0:
                pages_to_lds(b)
        c["prewait"]()
    for gb in range((nme + (RB - 1)) // RB):
        base_i = b + gb * (RB * CTAS)
        qrows = []
        for k in range_constexpr(RB):
            r0_, ch0, cx0, bd0, q0 = tmap(fx.min(base_i + k * CTAS, nt - 1))
            qrows.append(r0_)
        stage_q_slots(c, sm, t, qrows)
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
                if fx.const_expr(c["ifp4"]):
                    kv4, ks4 = fp4_key(c, flat, l)
                    kbs.append(kv4)
                    kss.append(ks4)
                else:
                    kbase = c["iplane"] + (flat // 8) * 1056
                    tt = flat % 8
                    kp = [hx.gload(kbase + i64(((l // 16) + 2 * cc) * 128) + tt * 16, I32, 4) for cc in range(4)]
                    kbs.append(fx.Vector.from_elements([kp[cc][e] for cc in range(4) for e in range(4)], I32))
                    kss.append(hx.gload(kbase + 1024 + tt * 4, F32))
                cols.append(col)
            qvs = []
            hwvs = []
            if fx.const_expr(not c["ifp4"]):
                for mi in range_constexpr(2):
                    qrow = sm + O_Q + k * (HEADS * DIM) + (mi * 16 + l % 16) * DIM
                    qp = [hx.lload(qrow + (l // 16) * 16 + cc * 32, I32, 4) for cc in range(4)]
                    qvs.append(fx.Vector.from_elements([qp[cc][e] for cc in range(4) for e in range(4)], I32))
                hwvs = [hx.lload(sm + O_W + k * (HEADS * 4) + (mi * 16 + (l // 16) * 8) * 4, F32, 8) for mi in range(2)]
            for u in range_constexpr(SUB_):
                col = cols[u]
                if fx.const_expr(c["ifp4"]):
                    sc = fp4_score(sm, k, l, kbs[u], kss[u])
                else:
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


def _grouped_key(c, flat, l):
    """(plain python: outside the AST rewriter) key row flat's B operand and scale for the grouped scorer."""
    if c["ifp4"]:
        return fp4_key(c, flat, l)
    kbase = c["iplane"] + (flat // 8) * 1056
    tt = flat % 8
    kp = [hx.gload(kbase + i64(((l // 16) + 2 * cc) * 128) + tt * 16, I32, 4) for cc in range(4)]
    return fx.Vector.from_elements([kp[cc][e] for cc in range(4) for e in range(4)], I32), hx.gload(kbase + 1024 + tt * 4, F32)


# the grouped scorer's code -> L2 just before it waits for its query (device.code_prefetch_near, bounded by the launch's
# recorded end; one CTA in 16 issues it: a cluster shares the L2): the entry's L2 code prefetch no longer holds by then
# (the weight stream evicts it), and its first score task paid ~3 us for its code (step probe, layer 2, 133K, 11-18:
# 7.31 -> 4.11 us every CTA, 4.04 one in 16; the I-cache prefetch alone: no change); 24 KB covers its loop. The per-row
# scorer (stage_score_any: the readers) showed no change on layer 24, so it has none
SCORE_L2_KB = 24
SCORE_L2_ONE_IN = 16
SCORE_PREFETCH = True     # the grouped scorer prefetches the next task's page row (False: the old loop)
GROUPED_MIN_BOUND = 8192   # index rows past which a c1 FULL layer scores its key tiles once for every row


@ASTRewriter.transform
def iq_wait(iqcnt, hcnt, t, iqonly, niq):
    """(R4) The scorer's wait for its query: the niq iquant CTAs' count (iqonly), else the INDEX_QUANT handoff."""
    if iqonly:
        if t == 0:
            hx.wait_count(iqcnt, niq, 4)
        gpu.barrier()
    else:
        hx.handoff(hcnt, t, True)


@ASTRewriter.transform
def handoff_unless(cnt, t, done):
    """(B) The INDEX_QUANT handoff here unless the layer took it already, right after iquant (step.py `early`)."""
    if done == (I32(0) != I32(0)):
        hx.handoff(cnt, t, True)


CROW0, NCROW = 192, T     # (B) the compressor rows' CTAs (compressor.CROW0 .. + T): no score tasks when `early`


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
    # (B) `early`: the tasks over the CTAs outside the compressor rows' (still writing this step's keys)
    early = c.get("early", I32(0) != I32(0))
    ncta = early.select(I32(CTAS - NCROW), I32(CTAS))
    vb = early.select((b >= CROW0).select(b - NCROW, b), b)
    busy = early & (b >= CROW0) & (b < CROW0 + NCROW)
    nme = busy.select(I32(0), fx.max((nt - vb + (ncta - 1)) // ncta, I32(0)))
    newc = cmax - 16          # (B) tasks reaching a column >= this read keys the compressor rows write this step
    iprod = c["iprod"]

    def page_row(task):
        col = task * CHUNK + w * 16 + l % 16
        krow = IP.key_row(col, bmax)
        page = hx.gload(c["btab"] + (i64(q) * i64(c["bstride"]) + i64(krow // c["irpp"])) * 4, I32)
        return i64(page) * i64(c["irpp"]) + i64(krow % c["irpp"])

    flat0 = page_row(fx.min(vb, fx.max(nt - 1, I32(0))))   # (the first task's page chain before the wait for the query)
    if fx.const_expr("codeend" in c):   # the score loop's code -> L2 while the CTA waits (see SCORE_L2_KB)
        pc0 = hx.getpc()   # (here: the prefetch sits in a branch the compiler may move)
        if b % SCORE_L2_ONE_IN == 0:
            hx.code_prefetch_near(t, b, c["codeend"], SCORE_L2_KB, pc=pc0)
    if fx.const_expr("prewait" in c):
        c["prewait"]()
    stage_q_slots(c, sm, t, list(range(TR)))   # every row's query and head weights -> LDS slots 0..TR-1
    gpu.barrier()
    key_of = lambda flat: _grouped_key(c, flat, l)

    def score_task(task, kb, ks):
        col = task * CHUNK + w * 16 + l % 16
        for r in range_constexpr(TR):
            cx = hx.lload(osr + r * 4, I32)
            bd = hx.lload(osr + (64 + r) * 4, I32)
            if fx.const_expr(c["ifp4"]):
                sc = fp4_score(sm, r, l, kb, ks)
            else:
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

    pf = c.get("score_pf", SCORE_PREFETCH)
    last = fx.max(vb + (nme - 1) * ncta, vb)
    fnext = flat0
    for i in range(nme):
        task = vb + i * ncta
        if early & ((task + 1) * CHUNK > newc):   # (B) this step's new keys: the compressor rows' flags
            hx.wait_flags(c["crflag"], NCROW, c["crtag"], w, l, 11)
            gpu.barrier()
        if fx.const_expr(pf):
            # the next task's page row (block table: this step's tables, not this layer's outputs) loaded before this
            # task's key tile, so the next round's chain is the key load alone, not block table -> key (2 VGPRs carried;
            # several tasks' key tiles in flight instead spilled the step kernel: 300K +150 us)
            flat = fnext
            fnext = page_row(fx.min(task + ncta, last))
        else:
            flat = (i == I32(0)).select(flat0, page_row(task))
        kb, ks = key_of(flat)
        score_task(task, kb, ks)
        if fx.const_expr("segcnt" in c):   # (segsel) the candidate layer: this task's 8192-column segment, one more
            if iprod:
                hx.wait_storecnt0()
                gpu.barrier()
                if t == 0:
                    hx.atomic_add(c["segcnt"] + i64(task * CHUNK // IP.SEG) * 4, 1)
    gpu.barrier()
