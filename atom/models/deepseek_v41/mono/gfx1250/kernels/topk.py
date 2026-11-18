# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The indexer's selection: exact top-k of up to 8192 fp32 values a CTA (aiter top_k_per_row_decode(stable=True): the
u32 order key of the bits, ties to the lowest index, winners ascending by index, -1 padded), and the selection tree over it
(stage_select_any: the FULL layers' top-512 -> isel / irow, the REINDEX layers' top-512 of the compacted candidates,
layer 20's top-2048 blocks -> icand)."""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels import index_plan as IP
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import T

I32, F32 = fx.Int32, fx.Float32
THREADS, WAVES = D.THREADS, D.WAVES
PER_THREAD = 32
LDS_KEYS = THREADS * PER_THREAD          # 8192
INT_MIN = -(2 ** 31)
HIST_WORDS = 256 + 32
TOPK = 512
# LDS (bytes, from the stage's base): keys, 3 histograms, the scan buffer
O_KEYS = 0
O_HIST = O_KEYS + LDS_KEYS * 4
O_BUF = O_HIST + 3 * HIST_WORDS * 4
LDS_BYTES = O_BUF + WAVES * 4


def i64(x):
    return fx.Int64(x)


def order_key(bits):
    return (bits < 0).select(~bits, bits | I32(INT_MIN))


def ult(a, b):
    return (a ^ I32(INT_MIN)) < (b ^ I32(INT_MIN))


def wave_incl_scan(v, lane):
    lane = hx.fresh(lane)
    for off in (1, 2, 4, 8, 16):
        moved = hx.lane_gather(v, fx.max(lane - off, I32(0)))
        v = v + (lane >= off).select(moved, I32(0))
    return v


def _ballot(p):
    return I32(hx.intrin(hx.i32(), "llvm.amdgcn.ballot.i32", [p]))


def _popc(m):
    return I32(hx.intrin(hx.i32(), "llvm.ctpop.i32", [m]))


def wave_max(v):
    for off in (1, 2, 4, 8, 16):
        v = fx.max(v, v.shuffle_xor(off, 32))
    return v


@ASTRewriter.transform
def block_excl_scan(v, lane, wave, buf):
    lane = hx.fresh(lane)
    wave = hx.fresh(wave)
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == 31:
        hx.lstore(buf + wave * 4, incl)
    gpu.barrier()
    before = I32(0)
    for w in range_constexpr(WAVES):
        cnt = hx.lload(buf + w * 4, I32)
        before = before + (w < wave).select(cnt, I32(0))
    return incl - v + before


O_OUT = LDS_BYTES                        # lds_topk's staged outputs (k <= 2048: (bits, i) pairs)
O_CNT = O_OUT + 2048 * 8                 # stage_select_any's per-row task counts [TR]
O_RBD = O_CNT + 64 * 4                   # stage_select_any's per-row bound, batch slot and input count [TR] each
LDS_BYTES2 = O_RBD + 3 * 64 * 4


# (v7) lds_topk's histogram copies, their merged bins, the block scan's wave totals (past LDS_BYTES2)
F_H8 = LDS_BYTES2
NCP = 16
F_MG = F_H8 + NCP * 256 * 4
F_SB = F_MG + 256 * 4
F_LDS = F_SB + 64


@ASTRewriter.transform
def lds_topk(n, k, load, emit, sm, tid, ST=None):
    hist8 = sm + F_H8
    merged = sm + F_MG
    sbuf = sm + F_SB
    """The top k (traced, <= 2048) of n (<= 8192) values: load(q, i) -> the 4 values i..i+3 (thread's quad q);
    emit(pos, bits, i) for pos < k (bits 0 / i -1 past min(n, k)), pos in position order (ties to the lower index).
    The thread's order keys in LDS (thread-contiguous runs); up to 4 radix passes of 8 bits, each key adding to copy lane % NCP
    of the pass's histogram (the scores' concentrated sign / exponent bytes made one shared histogram a chain of
    same-address atomics), bin t's copies merged (and zeroed) by thread t, every wave scanning the merged bins itself;
    the passes stop once the k-th key's bin holds exactly the keys still needed (the comparisons then use the resolved
    prefix, key & mask); the winners placed after a one-barrier block scan. Same winners and order
    as the 4-pass shared-histogram version it replaced (GPU check: bit-identical), 8192 -> 512 18.0 -> 10.4 us."""
    tid = hx.fresh(tid)
    lane, wave = tid % 32, tid // 32
    keys, hist, buf, outs = sm + O_KEYS, sm + O_HIST, sm + O_BUF, sm + O_OUT
    # thread-contiguous runs of nq quads (index order = (thread, slot) order, as the placement scan needs), nq = the
    # fewest that cover n: the passes below loop nq (not 8) quads, so a short row (n 1500: nq 2) costs 2 / 8 of the work
    nq = fx.min((fx.max(n, I32(1)) + (4 * THREADS - 1)) // (4 * THREADS), I32(PER_THREAD // 4))
    base = tid * nq * 4
    last = fx.max(n - 1, I32(0)) & I32(~3)
    for q in range_constexpr(PER_THREAD // 4):   # (all 8 loads issued: in flight together; quads past nq unused)
        quad = load(q, fx.min(base + 4 * q, last))
        if q < nq:
            hx.lstore(keys + (base + 4 * q) * 4, fx.Vector.from_elements([order_key(quad[e]) for e in range(4)], I32))
    for r in range_constexpr((2 * HIST_WORDS + THREADS - 1) // THREADS):
        cz0 = hx.fresh(tid) + THREADS * r
        if cz0 < 2 * HIST_WORDS:
            hx.lstore(hist + cz0 * 4, I32(0))
    for r in range_constexpr(NCP * 256 // THREADS):
        hx.lstore(hist8 + (hx.fresh(tid) + r * THREADS) * 4, I32(0))
    gpu.barrier()
    prefix = I32(0)
    mask = I32(0)
    need = I32(k)
    n = fx.max(n, I32(0))
    if n > k:
        # (v5, after hi's topk) 4 passes of 8 bits: a key adds to copy lane % 8 of the pass's 256-bin histogram (8 LDS
        # copies: the scores' concentrated sign / exponent bytes no longer make 32 lanes add to one word); every wave
        # sums the copies and scans the bins itself (no second barrier); the passes stop once the k-th key's bin holds
        # exactly the keys still needed (the comparisons below then use the resolved prefix, key & mask)
        lp = hx.fresh(tid) % 32
        hcopy = (lp % NCP) * (256 * 4)
        done = I32(0)
        for p in range(4):
            if done == I32(0):
                shift = 24 - 8 * p
                hs = hist8
                for dq in range(nq):
                    kp4 = hx.lload(keys + (base + 4 * dq) * 4, I32, 4)
                    for e in range_constexpr(4):
                        keyp = kp4[e]
                        if (base + 4 * dq + e < n) & ((keyp & mask) == prefix):
                            hx.latomic_add_i32(hs + hcopy + ((keyp >> shift) & 255) * 4, I32(1))
                gpu.barrier()
                tb_ = hx.fresh(tid)   # (thread t: bin t of the NCP copies -> merged, the copies zeroed for the next pass)
                sm_ = I32(0)
                for cp in range_constexpr(NCP):
                    sm_ = sm_ + hx.lload(hs + (cp * 256 + tb_) * 4, I32)
                    hx.lstore(hs + (cp * 256 + tb_) * 4, I32(0))
                hx.lstore(merged + tb_ * 4, sm_)
                gpu.barrier()
                v0 = hx.lload(merged + (248 - 8 * lp) * 4, I32, 4)
                v1 = hx.lload(merged + (252 - 8 * lp) * 4, I32, 4)
                cs = [v1[3], v1[2], v1[1], v1[0], v0[3], v0[2], v0[1], v0[0]]
                s4 = cs[0]
                for q in range_constexpr(1, 8):
                    s4 = s4 + cs[q]
                before = wave_incl_scan(s4, lp) - s4
                found = I32(-1)
                fcnt = I32(-1)
                for q in range_constexpr(8):
                    hit = (before < need) & (need <= before + cs[q])
                    found = hit.select(((255 - 8 * lp - q) << 16) | (need - before), found)
                    fcnt = hit.select(cs[q], fcnt)
                    before = before + cs[q]
                found = wave_max(found)
                fcnt = wave_max(fcnt)
                prefix = prefix | (((found >> 16) & 255) << shift)
                mask = mask | (I32(255) << shift)
                need = found & 0xFFFF
                done = (fcnt == need).select(I32(1), I32(0))
    take_all = n <= k
    # (v6) the winners from the registers (no LDS reload), one barrier for the block scan, an unrolled placement
    gt_bits = I32(0)
    eq_bits = I32(0)
    above = I32(0)
    ties = I32(0)
    for dg in range(nq):
        kg4 = hx.lload(keys + (base + 4 * dg) * 4, I32, 4)
        for e in range_constexpr(4):
            vd = base + 4 * dg + e < n
            gt = (vd & ult(prefix, kg4[e] & mask)).select(I32(1), I32(0))
            eq = (vd & ((kg4[e] & mask) == prefix)).select(I32(1), I32(0))
            gt_bits = gt_bits | (gt << (4 * dg + e))
            eq_bits = eq_bits | (eq << (4 * dg + e))
            above = above + gt
            ties = ties + eq
    pv = above | (ties << 16)
    incl = wave_incl_scan(pv, lane)
    if lane == 31:
        hx.lstore(sbuf + wave * 4, incl)
    gpu.barrier()
    bw = I32(0)
    for wv in range_constexpr(WAVES):
        cw = hx.lload(sbuf + wv * 4, I32)
        bw = bw + (wv < wave).select(cw, I32(0))
    packed = incl - pv + bw
    ab = packed & 0xFFFF
    tb = packed >> 16
    for dw in range(nq):
        kw4 = hx.lload(keys + (base + 4 * dw) * 4, I32, 4)
        for e in range_constexpr(4):
            dd = 4 * dw + e
            iw = base + dd
            gtw = (gt_bits >> dd) & 1
            tiew = (eq_bits >> dd) & 1
            win = (iw < n) & (take_all | (gtw != 0) | ((tiew != 0) & (tb < need)))
            at = take_all.select(iw, ab + fx.min(tb, need))
            if win:
                kd = kw4[e]
                hx.lstore(outs + at * 8, fx.Vector.from_elements([(kd < 0).select(kd & I32(0x7FFFFFFF), ~kd), iw], I32))
            ab = ab + gtw
            tb = tb + tiew
    n_out = fx.min(n, I32(k))
    for rp in range((I32(k) + (THREADS - 1)) // THREADS):   # k traced: runtime loops, one copy of the emit code
        posp = hx.fresh(tid) + THREADS * rp
        if (posp >= n_out) & (posp < k):
            hx.lstore(outs + posp * 8, fx.Vector.from_elements([I32(0), I32(-1)], I32))
    gpu.barrier()
    for re in range((I32(k) + (THREADS - 1)) // THREADS):
        pose = hx.fresh(tid) + THREADS * re
        if pose < k:
            ve = hx.lload(outs + pose * 8, I32, 2)
            emit(pose, ve[0], ve[1])
    gpu.barrier()


SEG, KSEL, KBLK, NCAND, NLMAX = IP.SEG, IP.KSEL, IP.KBLK, IP.NCAND, IP.NLMAX
_cdiv, _lists, _lists_rt, _list_inputs_rt, _list_count_rt = IP.cdiv, IP.lists, IP.lists_rt, IP.list_inputs_rt, IP.list_count_rt
MODE_SEL, MODE_RX, MODE_BLK = 0, 1, 2


@ASTRewriter.transform
def selection_levels(c, isrx):
    """Every wave reads the same row bounds, so every CTA executes the same handoffs."""
    bound = I32(0)
    for group in range_constexpr((c["TR"] + 31) // 32):
        row = fx.min(c["l"] + group * 32, I32(c["TR"] - 1))
        bound = fx.max(bound, hx.gload(c["ibound"] + i64(row) * 4, I32))
    bound = wave_max(bound)
    # REINDEX scores at most 2048 retained blocks x 8 rows.
    sel = isrx.select(I32(2), IP.tree_levels(bound, KSEL))
    blk = IP.tree_levels(_cdiv(bound, I32(8)), KBLK)
    return sel, blk


def rx_context(c, tok, bd):
    """A REINDEX token's compacted columns (index_plan.rx_context; its last kept block from icand)."""
    kept = IP.rx_kept(bd)
    last = hx.gload(c["icand"] + i64(tok * NCAND + fx.max(kept - 1, I32(0))) * 4, I32)
    return IP.rx_context(kept, last, bd)


@ASTRewriter.transform
def _n_of_rt(c, mode, tok, bd):
    nx = fx.max(bd, I32(0))
    if mode == MODE_RX:
        nx = rx_context(c, tok, bd)
    if mode == MODE_BLK:
        nx = _cdiv(fx.max(bd, I32(0)), I32(8))
    return nx


@ASTRewriter.transform
def _emit_any(c, mode, lev, final, tok, j, q, k, lstride, cur, pbase, pos, bits, ii):
    """lds_topk's emit of a selection task (the mode and the level traced): the token's final list -> icand (blk) or
    isel / irow (sel, rx: lifted through icand); else -> the level's list in lvl."""
    lsafe = c["lvl"]
    cl = hx.gload((lev == 0).select(lsafe, pbase + i64(fx.max(ii, I32(0))) * 8 + 4), I32)
    col = (lev == 0).select(j * SEG + ii, cl)
    col = (ii >= 0).select(col, I32(-1))
    cc = fx.max(col, I32(0))
    fslot = i64(final.select(tok, I32(T)) * k + pos) * 4
    if mode == MODE_BLK:
        c["stx"](c["icand"] + fslot, col)
    else:
        real = col
        if mode == MODE_RX:   # the lift: compacted column -> its candidate block's real row
            blk = hx.gload(c["icand"] + i64(tok * NCAND + fx.min(cc // 8, I32(NCAND - 1))) * 4, I32)
            real = (col >= 0).select(fx.max(blk, I32(0)) * 8 + cc % 8, I32(-1))
        rc = fx.max(real, I32(0))
        page = hx.gload(c["btab"] + (i64(q) * i64(c["bstride"]) + i64(rc // c["irpp"])) * 4, I32)
        prow = i64(page) * i64(c["cprow"]) + i64(c["imain"] + rc % c["irpp"])
        c["stx"](c["isel"] + fslot, real)
        c["stx"](c["irow"] + fslot, (real >= 0).select(fx.Int32(prow), I32(0)))
    if final.select(I32(0), I32(1)) != I32(0):   # a final list goes to isel / irow / icand only
        lslot = (i64(tok) * i64(lstride) + i64(j * k + pos)) * 8
        c["stx"](cur + lslot, fx.Vector.from_elements([(ii >= 0).select(bits, I32(0)), col], I32))


def blk_lvl_off(c):
    """Bytes from lvl to the block tree's lists: past the top-512 tree's NLMAX levels (buffers: ilvl holds both)."""
    return i64(NLMAX) * i64(T + 1) * i64(_cdiv(c["ilw"], I32(SEG)) * KSEL) * 8


@ASTRewriter.transform
def seg_wait(cnt, nchunk, s0, ns, t):
    """(segsel) Segments s0 .. s0 + ns - 1 of 8192 columns: each one's score tasks (64, the last ones fewer) counted."""
    if t == 0:
        for s in range(ns):
            sg = s0 + s
            hx.wait_count(cnt + i64(sg) * 4, fx.min(fx.max(nchunk - sg * (SEG // 128), I32(0)), I32(SEG // 128)), 4)
    gpu.barrier()


@ASTRewriter.transform
def stage_select_any(c, lev, nlev, mode):
    """Level lev (traced, < nlev) of a selection tree, mode (MODE_*, traced): MODE_SEL the top-512 of a FULL layer's
    scores (ilog, n = bound) -> isel / irow; MODE_RX the same over a REINDEX layer's compacted columns (n = rx_context),
    lifted to real rows through icand; MODE_BLK layer 20's top-2048 blocks of the block maxima (ibmax, n = blocks(bound))
    -> icand. A task = (token, list j) for j < lists(n, lev); the per-row task counts in LDS, tasks round-robin over the
    CTAs. Level 0 reads the values [t][j SEG + i]; level > 0 the lists of level lev - 1 (lvl[lev-1][t][list][k] (bits,
    ident) pairs, the fan lists j fan.. end to end). One instance of this code serves every pass of a layer."""
    t, b = c["t"], c["b"]
    isblk = mode == MODE_BLK
    k = isblk.select(I32(KBLK), I32(KSEL))
    fan = isblk.select(I32(SEG // KBLK), I32(SEG // KSEL))
    vals = isblk.select(c["ibmax"], c["ilog"])
    vstride = isblk.select(c["ilw"] // 8, c["ilw"])
    TR = c["TR"]
    ocnt = c["sm"] + O_CNT
    tc = fx.min(t, I32(TR - 1))
    bdt = hx.gload(c["ibound"] + i64(tc) * 4, I32)
    qt = hx.gload(c["ibat"] + i64(tc) * 4, I32)
    nnt = _n_of_rt(c, mode, tc, bdt)
    myc = _list_count_rt(nnt, lev, fan)   # thread t: row t's task count
    orb = c["sm"] + O_RBD
    if t < TR:   # row t's count, bound, batch slot, inputs: the task loop reads them from LDS (no dependent global loads)
        hx.lstore(ocnt + t * 4, myc)
        hx.lstore(orb + t * 4, bdt)
        hx.lstore(orb + (64 + t) * 4, qt)
        hx.lstore(orb + (128 + t) * 4, nnt)
    gpu.barrier()
    # the rows' inclusive task prefix (rows lane and lane + 32, TR <= 64) in every wave's lanes: no serial LDS walks
    ln = hx.fresh(t) % 32
    c0 = hx.lload(ocnt + fx.min(ln, I32(TR - 1)) * 4, I32)
    c1 = hx.lload(ocnt + fx.min(ln + 32, I32(TR - 1)) * 4, I32)
    c0 = (ln < TR).select(c0, I32(0))
    c1 = (ln + 32 < TR).select(c1, I32(0))
    inc0 = wave_incl_scan(c0, ln)
    tot0 = hx.lane_gather(inc0, I32(31))
    inc1 = wave_incl_scan(c1, ln) + tot0
    nt = hx.lane_gather(inc1, I32(31))
    bb = b
    if fx.const_expr("rev" in c):   # (traced: the block tree beside the top-512 tree takes its tasks from the last CTA down)
        bb = c["rev"].select(I32(255) - b, b)
    for i in range(bb, nt, 256):
        # the task's row: the rows whose inclusive prefix is <= i are before it; j = i - that prefix
        p0 = inc0 <= i
        p1 = inc1 <= i
        tok = fx.min(_popc(_ballot(p0)) + _popc(_ballot(p1)), I32(TR - 1))
        j = i - wave_max(fx.max(p0.select(inc0, I32(0)), p1.select(inc1, I32(0))))
        q = fx.max(hx.lload(orb + (64 + tok) * 4, I32), I32(0))
        nn = hx.lload(orb + (128 + tok) * 4, I32)
        n = _list_inputs_rt(nn, lev, j, k, fan)
        final = (lev == nlev - 1) | (_lists_rt(nn, lev, fan) <= 1)
        lstride = _cdiv(vstride, I32(SEG)) * k
        prev = c["lvl"] + i64(fx.max(lev - 1, I32(0))) * i64(T + 1) * i64(lstride) * 8
        cur = c["lvl"] + i64(lev) * i64(T + 1) * i64(lstride) * 8
        row = vals + i64(tok) * i64(vstride) * 4 + i64(j * SEG) * 4
        pbase = prev + (i64(tok) * i64(lstride) + i64(j * fan * k)) * 8
        l0 = lev == 0

        def load(qd, ii):
            v4 = hx.gload(l0.select(row + i64(ii) * 4, vals), I32, 4)
            v8 = hx.gload(l0.select(c["lvl"], pbase + i64(ii) * 8), I32, 8)
            return [l0.select(v4[e], v8[2 * e]) for e in range(4)]

        def emit(pos, bits, ii):
            _emit_any(c, mode, lev, final, tok, j, q, k, lstride, cur, pbase, pos, bits, ii)

        if fx.const_expr("segw" in c):   # (segsel) level 0: its segment's score tasks counted (no SCORE handoff)
            if c["segw"] & l0:
                seg_wait(c["segcnt"], c["segn"], isblk.select(j * 8, j), isblk.select(I32(8), I32(1)), t)
        lds_topk(n, k, load, emit, c["sm"], t)


# ---- (c1) the FULL / REINDEX layers' top-KSEL by a grouped multi-CTA radix select -----------------------------------
MC_ROWS = 8                       # row slots (TR <= 8)
MC_G = 32                         # CTAs a row once the step's longest row has > MC_G_MIN_N keys (TR * MC_G <= 256)
MC_G_SHORT = 8                    # CTAs a row otherwise
MC_G_MIN_N = 4096                 # (CTAs G r .. G r + G - 1 take row r; the rest skip to the stage's handoff)
MC_HIST = 4 * MC_ROWS * 256 * 4   # bytes a layer: [4 rounds][rows][256] i32
MC_CTR = MC_ROWS * 8 * 4          # bytes a layer: [rows] (above, ties, round 0..3 group counts, -, -) i32
MC_LAYER = MC_HIST + MC_CTR
O_MC = LDS_BYTES2                 # LDS: (prefix, mask, need, n) i32, then the CTA's 256-bin histogram
O_MCH = O_MC + 16


@ASTRewriter.transform
def select_mc(c, hb, ctr, isrx, G, anc=None, x0=None):
    """Every row's top KSEL of its n scores (FULL: n = bound, the ilog row; REINDEX: n = its compacted columns, lifted
    at the emit) -> isel / irow (the tree's final emit), by G CTAs a row (traced: MC_G or MC_G_SHORT): CTA g of row r takes keys
    [g CH, (g + 1) CH), CH = ceil(n / G); up to 4 radix rounds of 8 bits (done once the k-th key's bin is exactly the keys
    still needed): its LDS histogram -> global atomics into
    hb[round][r][256] (zeroed at the step's entry), the group's count ctr[r][2 + round] (every wave's s_wait_storecnt 0, no
    fences; the bins read by agent-scope loads), every CTA of
    the group scans the 256 bins alike; then the winners: keys above the k-th key's value take positions from
    ctr[r][0], ties from ctr[r][1] (< need). n <= k: all, position = index; past min(n, k): -1 (the group's CTA 0)."""
    t, b, sm, TR = c["t"], c["b"], c["sm"], c["TR"]
    lane = t % 32
    k = I32(KSEL)
    vals, vstride = c["ilog"], c["ilw"]
    mode = isrx.select(I32(MODE_RX), I32(MODE_SEL))
    if fx.const_expr(x0 is not None):   # (select_ma's fallback: its row groups from CTA x0)
        b = (b - x0 + I32(MA_CTAS)) % I32(MA_CTAS)
    r = fx.min(b // G, I32(TR - 1))
    g = b % G
    if b < TR * G:
        if t == 0:
            bd0 = hx.gload(c["ibound"] + i64(r) * 4, I32)
            bd = fx.max(bd0, I32(0))
            if isrx:
                bd = fx.max(rx_context(c, r, bd0), I32(0))
            hx.lstore(sm + O_MC, fx.Vector.from_elements([I32(0), I32(0), k, bd], I32))
        gpu.barrier()
        n = hx.lload(sm + O_MC + 12, I32)
        CH = (n + (G - 1)) // G
        i0 = g * CH
        i1 = fx.min(i0 + CH, n)
        rowp = vals + i64(r) * i64(vstride) * 4
        if n > k:
            for p in range(4):
                if hx.lload(sm + O_MC + 12, I32) >= 0:   # (bit 31: an earlier round's bin held exactly the keys
                    # still needed: every key of its prefix is in, the selection is final)
                    shift = 24 - 8 * p
                    hx.lstore(sm + O_MCH + t * 4, I32(0))
                    gpu.barrier()
                    st4 = hx.lload(sm + O_MC, I32, 4)
                    for i in range(i0 + t, i1, 256):
                        key = order_key(hx.gload(rowp + i64(i) * 4, I32))
                        if (key & st4[1]) == st4[0]:
                            hx.latomic_add_i32(sm + O_MCH + ((key >> shift) & 255) * 4, I32(1))
                    gpu.barrier()
                    v = hx.lload(sm + O_MCH + t * 4, I32)
                    if v != 0:
                        hx.atomic_add(hb + i64(((p * MC_ROWS + r) * 256 + t) * 4), v)
                    hx.wait_storecnt0()   # (every wave its bin atomics done: the count needs no release fence)
                    gpu.barrier()
                    if t == 0:
                        hx.atomic_add(ctr + i64((r * 8 + 2 + p) * 4), 1)
                        hx.wait_count(ctr + i64((r * 8 + 2 + p) * 4), G, 8)
                    gpu.barrier()
                    if t < 32:   # wave 0: the bin of the k-th key (lds_topk's scan, over the global bins)
                        # (agent-scope loads, no acquire fence: the bins are device-scope atomics, read once a layer, so
                        # loads at their scope see them -- as wait_count's see the count; the fence's cache invalidate
                        # every round cost the group's CTAs)
                        cs = [hx.gload(hb + i64(((p * MC_ROWS + r) * 256 + 255 - 8 * lane - q) * 4), I32, scope="agent") for q in range(8)]
                        s4 = cs[0]
                        for q in range_constexpr(1, 8):
                            s4 = s4 + cs[q]
                        need = st4[2]
                        before = wave_incl_scan(s4, lane) - s4
                        found = I32(-1)
                        done = I32(0)
                        for q in range_constexpr(8):
                            hit = (before < need) & (need <= before + cs[q])
                            found = hit.select(((255 - 8 * lane - q) << 16) | (need - before), found)
                            done = (hit & (need == before + cs[q])).select(I32(1), done)
                            before = before + cs[q]
                        found = wave_max(found)
                        done = wave_max(done)
                        if lane == 0:
                            hx.lstore(sm + O_MC, fx.Vector.from_elements(
                                [st4[0] | (((found >> 16) & 255) << shift), st4[1] | (I32(255) << shift), found & 0xFFFF,
                                 (done != 0).select(n | I32(INT_MIN), n)], I32))
                    gpu.barrier()
        st4 = hx.lload(sm + O_MC, I32, 4)
        q = fx.max(hx.gload(c["ibat"] + i64(r) * 4, I32), I32(0))
        for i in range(i0 + t, i1, 256):
            pos = I32(-1)
            if n <= k:
                pos = i
            else:
                key = order_key(hx.gload(rowp + i64(i) * 4, I32)) & st4[1]   # (the resolved prefix: rounds may end early)
                if ult(st4[0], key):
                    pos = hx.atomic_add(ctr + i64(r * 32), 1)
                if key == st4[0]:
                    tie = hx.atomic_add(ctr + i64(r * 32 + 4), 1)
                    pos = (tie < st4[2]).select(k - st4[2] + tie, I32(-1))
            if pos >= 0:
                _emit_any(c, mode, I32(0), I32(1) > I32(0), r, I32(0), q, k, I32(0), c["lvl"], c["lvl"], pos, I32(0), i)
        if g == 0:   # padding past min(n, k)
            for pp in range(fx.min(n, k) + t, k, 256):
                _emit_any(c, mode, I32(0), I32(1) > I32(0), r, I32(0), q, k, I32(0), c["lvl"], c["lvl"], pp, I32(0), I32(-1))
        if fx.const_expr(anc is not None):   # REINDEX: the 512-th score (its resolved prefix) -> select_ma's anchor
            if isrx & (g == 0) & (t == 0) & (n > k):
                hx.gstore(anc + i64(r) * 4, _ma_key_f32(st4[0]))


# ---- (l20) a row's top-k by G CTAs, positions in index order (a replaceable selection primitive) --------------------
OC_ROW = 64 * 8                   # select_ord's per-CTA (above, ties) counts: [rows][64 CTAs] int2, then [rows] group counters
OC_BYTES = MC_ROWS * OC_ROW + MC_ROWS * 4


@ASTRewriter.transform
def select_ord(c, hb, ctr, oc, r, g, G, n, k, vals, vstride, mode, q, ordered=True):
    """Row r's top k of its n values (vals + r vstride, f32 bits ordered by order_key) by the G CTAs of its group (this
    one: g), select_mc's radix rounds (hb [4 rounds][rows][256] / ctr [rows][8]: this instance's zeroed slot) to the k-th
    key, then the winners' positions in INDEX order (not select_mc's atomic arrival order): every CTA counts its chunk's
    keys above the k-th one and its ties, the group exchanges the counts (oc), and a key's position is the selected keys
    before it (the keys above + min(the ties, need) before it: ties taken in index order, deterministic). -> _emit_any
    (mode: MODE_SEL isel / irow, MODE_BLK icand -- ascending blocks, as rx_context's last kept block needs); past
    min(n, k): -1 (CTA g 0). The selection's interface: (r, g, G, n, k, vals, vstride, mode) -> the row's final list."""
    t, sm = c["t"], c["sm"]
    lane = t % 32
    wv = t // 32
    CH = (n + (G - 1)) // G
    i0 = g * CH
    i1 = fx.min(i0 + CH, n)
    rowp = vals + i64(r) * i64(vstride) * 4
    if t == 0:
        hx.lstore(sm + O_MC, fx.Vector.from_elements([I32(0), I32(0), I32(k), n], I32))
    gpu.barrier()
    if n > I32(k):
        for p in range(4):
            if hx.lload(sm + O_MC + 12, I32) >= 0:
                shift = 24 - 8 * p
                hx.lstore(sm + O_MCH + t * 4, I32(0))
                gpu.barrier()
                st4 = hx.lload(sm + O_MC, I32, 4)
                for i in range(i0 + t, i1, 256):
                    key = order_key(hx.gload(rowp + i64(i) * 4, I32))
                    if (key & st4[1]) == st4[0]:
                        hx.latomic_add_i32(sm + O_MCH + ((key >> shift) & 255) * 4, I32(1))
                gpu.barrier()
                v = hx.lload(sm + O_MCH + t * 4, I32)
                if v != 0:
                    hx.atomic_add(hb + i64(((p * MC_ROWS + r) * 256 + t) * 4), v)
                hx.wait_storecnt0()
                gpu.barrier()
                if t == 0:
                    hx.atomic_add(ctr + i64((r * 8 + 2 + p) * 4), 1)
                    hx.wait_count(ctr + i64((r * 8 + 2 + p) * 4), G, 8)
                gpu.barrier()
                if t < 32:
                    cs = [hx.gload(hb + i64(((p * MC_ROWS + r) * 256 + 255 - 8 * lane - qq) * 4), I32, scope="agent") for qq in range(8)]
                    s4 = cs[0]
                    for qq in range_constexpr(1, 8):
                        s4 = s4 + cs[qq]
                    need = st4[2]
                    before = wave_incl_scan(s4, lane) - s4
                    found = I32(-1)
                    done = I32(0)
                    for qq in range_constexpr(8):
                        hit = (before < need) & (need <= before + cs[qq])
                        found = hit.select(((255 - 8 * lane - qq) << 16) | (need - before), found)
                        done = (hit & (need == before + cs[qq])).select(I32(1), done)
                        before = before + cs[qq]
                    found = wave_max(found)
                    done = wave_max(done)
                    if lane == 0:
                        hx.lstore(sm + O_MC, fx.Vector.from_elements(
                            [st4[0] | (((found >> 16) & 255) << shift), st4[1] | (I32(255) << shift), found & 0xFFFF,
                             (done != 0).select(n | I32(INT_MIN), n)], I32))
                gpu.barrier()
    st4 = hx.lload(sm + O_MC, I32, 4)
    big = n > I32(k)
    need = st4[2]
    if fx.const_expr(not ordered):   # (select_mc's placement: positions in atomic arrival order, one coalesced pass)
        for i in range(i0 + t, i1, 256):
            pos = I32(-1)
            if big:
                key = order_key(hx.gload(rowp + i64(i) * 4, I32)) & st4[1]
                if ult(st4[0], key):
                    pos = hx.atomic_add(ctr + i64(r * 32), 1)
                if key == st4[0]:
                    tie = hx.atomic_add(ctr + i64(r * 32 + 4), 1)
                    pos = (tie < need).select(I32(k) - need + tie, I32(-1))
            else:
                pos = i
            if pos >= 0:
                _emit_any(c, I32(mode), I32(0), I32(1) > I32(0), r, I32(0), q, I32(k), I32(0), c["lvl"], c["lvl"], pos, I32(0), i)
    else:
        # thread t takes a contiguous slice of the CTA's chunk: its keys above the k-th one and its ties (packed: above << 16 |
        # ties, a chunk < 65536 keys), one block scan -> the thread's bases in the CTA, the CTA's totals -> the group's
        # exchange -> the CTA's bases in the row; then the thread walks its slice in index order and places its winners
        SL = (i1 - i0 + 255) // 256
        s0 = fx.min(i0 + t * SL, i1)
        s1 = fx.min(s0 + SL, i1)
        pk = I32(0)
        for i in range(s0, s1):
            key = order_key(hx.gload(rowp + i64(i) * 4, I32)) & st4[1]
            pk = pk + ult(st4[0], key).select(I32(1 << 16), I32(0)) + (key == st4[0]).select(I32(1), I32(0))
        ob = sm + O_MCH
        ex = block_excl_scan(pk, lane, wv, ob)
        if t == 255:
            hx.lstore(ob + 128, ex + pk)
        gpu.barrier()
        tot = hx.lload(ob + 128, I32)
        if t == 0:
            c["stx"](oc + i64((r * 64 + g) * 8), fx.Vector.from_elements([tot >> 16, tot & 0xFFFF], I32))
            hx.wait_storecnt0()
            hx.atomic_add(oc + i64(MC_ROWS * OC_ROW + r * 4), 1)
            hx.wait_count(oc + i64(MC_ROWS * OC_ROW + r * 4), G, 8)
        gpu.barrier()
        if t < 32:   # the group's counts before this CTA (G <= 64: lanes l, l + 32)
            pa = hx.gload(oc + i64((r * 64 + fx.min(lane, I32(G - 1))) * 8), fx.Int64, scope="agent")   # (above | ties << 32)
            pb = hx.gload(oc + i64((r * 64 + fx.min(lane + 32, I32(G - 1))) * 8), fx.Int64, scope="agent")
            a0 = (lane < g).select(I32(pa), I32(0)) + (lane + 32 < g).select(I32(pb), I32(0))
            t0 = (lane < g).select(I32(pa >> 32), I32(0)) + (lane + 32 < g).select(I32(pb >> 32), I32(0))
            for off in (1, 2, 4, 8, 16):
                a0 = a0 + a0.shuffle_xor(off, 32)
                t0 = t0 + t0.shuffle_xor(off, 32)
            if lane == 0:
                hx.lstore(ob + 136, fx.Vector.from_elements([a0, t0], I32))
        gpu.barrier()
        base = hx.lload(ob + 136, I32, 2)
        cA = base[0] + (ex >> 16)
        cT = base[1] + (ex & 0xFFFF)
        for i in range(s0, s1):
            pos = i
            if big:
                key = order_key(hx.gload(rowp + i64(i) * 4, I32)) & st4[1]
                fa = ult(st4[0], key)
                ft = key == st4[0]
                pos = (fa | (ft & (cT < need))).select(cA + fx.min(cT, need), I32(-1))
                cA = cA + fa.select(I32(1), I32(0))
                cT = cT + ft.select(I32(1), I32(0))
            if pos >= 0:
                _emit_any(c, I32(mode), I32(0), I32(1) > I32(0), r, I32(0), q, I32(k), I32(0), c["lvl"], c["lvl"], pos, I32(0), i)
    if g == 0:   # padding past min(n, k)
        for pp in range(fx.min(n, I32(k)) + t, I32(k), 256):
            _emit_any(c, I32(mode), I32(0), I32(1) > I32(0), r, I32(0), q, I32(k), I32(0), c["lvl"], c["lvl"], pp, I32(0), I32(-1))



# ---- a REINDEX row's first radix round from the previous step's 512-th score (the anchor a) ------------------------
# A row's 512-th score moves little between steps (|delta| p99 ~1, max 2.1 over 30K / 133K, repeated and natural text):
# round 0's digit is a fine linear bin of the score around the row's anchor -- bins 1..254 over [a - MA_S, a + MA_S),
# 0 below, 255 above (a monotone digit), counted in one group round (histogram -> device atomics -> the group count).
# The 512-th score's bin j: keys above it are the winners; of j's keys (~14, p99 ~48 at MA_S 2, all within 0.016 of
# the 512-th score) the first `need` in device-atomic order fill the rest -- an approximate top-512 (~99% of the exact
# set), no ranking pass; one device atomic a CTA gives both position bases. The group's CTA 0 writes the bin's centre
# as the next step's anchor. j = 0 / 255: select_mc as before (it writes the anchor then). The anchors are kept across
# steps (step.MA_KB); the first step of a request falls back.
MA_CTAS = 256


def _ma_mbcnt(m):
    """The lanes below this one in the wave32 mask m."""
    return I32(hx.intrin(hx.i32(), "llvm.amdgcn.mbcnt.lo", [I32(m), I32(0)]))


MA_S = 2.0
MA_LAYER = MC_ROWS * 256 * 4             # a REINDEX layer's MA slot (zeroed a step): the anchored histograms [MC_ROWS][256]
MA_SLOTS = 4                             # REINDEX layers (24 / 28 / 32 / 36): slot (Ly >> 2) & 3
MA_KANC = 0                              # kept across steps (step.MA_KB, MA_KEEP bytes a REINDEX layer): the rows' anchors
MA_KEEP = 64                             # [MC_ROWS] f32
O_MAL = O_MCH + 1024                     # LDS: the waves' (winners, j-bin) counts [8] x 2, + 64 the CTA's bases, + 128 the
# histogram's junk lanes
O_MAP = O_MAL + 4096                     # select_ma_prep's values (before the scorer: past the scorer's LDS) -- (n, lo,
# -, -), then a thread's (real, pool row) of its 2 keys
MA_X0 = 64                               # select_ma's row groups from CTA 64 (past the iquant CTAs: the prep in their wait)


def _ma_bin(x, lo, inv):
    f = (x.bitcast(F32) - lo) * inv + F32(1.0)
    return fx.min(fx.max(fx.min(fx.max(f, F32(0.0)), F32(255.0)).to(I32), I32(0)), I32(255))   # (NaN: an int clamp too)


def _ma_key_f32(key):
    """order_key's inverse: the f32 of an order key."""
    return (key < 0).select(key & I32(0x7FFFFFFF), ~key).bitcast(F32)


def _ma_lift(c, r, q, col):
    """A REINDEX pick's compacted column -> (real row, pool row) (_emit_any's MODE_RX lift)."""
    cc = fx.max(col, I32(0))
    blk = hx.gload(c["icand"] + i64(r * NCAND + fx.min(cc // 8, I32(NCAND - 1))) * 4, I32)
    real = fx.max(blk, I32(0)) * 8 + cc % 8
    page = hx.gload(c["btab"] + (i64(q) * i64(c["bstride"]) + i64(real // c["irpp"])) * 4, I32)
    return real, fx.Int32(i64(page) * i64(c["cprow"]) + i64(c["imain"] + real % c["irpp"]))


def _ma_put(c, r, pos, real, prow):
    fslot = i64(r * KSEL + pos) * 4
    c["stx"](c["isel"] + fslot, real)
    c["stx"](c["irow"] + fslot, prow)


@ASTRewriter.transform
def select_ma_prep(c, G, kp):
    """select_ma's score-independent part, before the scorer (the iquant CTAs' wait): the row's n and the anchor's bin
    origin, a thread's 2 key columns' real / pool rows -> LDS O_MAP (kept through the scorer)."""
    t, b, sm, TR = c["t"], c["b"], c["sm"], c["TR"]
    bx = (b - I32(MA_X0) + I32(MA_CTAS)) % I32(MA_CTAS)
    if bx < TR * G:
        r = bx // G
        g = bx % G
        bd0 = hx.gload(c["ibound"] + i64(r) * 4, I32)
        n = fx.max(rx_context(c, r, bd0), I32(0))
        ia = g * ((n + (G - 1)) // G) + t
        lo = hx.gload(kp + i64(MA_KANC) + i64(r) * 4, F32) - F32(MA_S)
        qb = fx.max(hx.gload(c["ibat"] + i64(r) * 4, I32), I32(0))
        ra, pa = _ma_lift(c, r, qb, ia)
        rb, pb = _ma_lift(c, r, qb, ia + 256)
        if t == 0:
            hx.lstore(sm + O_MAP, fx.Vector.from_elements([n, lo.bitcast(I32), I32(0), I32(0)], I32))
        hx.lstore(sm + O_MAP + 16 + t * 16, fx.Vector.from_elements([ra, pa, rb, pb], I32))
    gpu.barrier()   # (select_ma reads thread 0's n / lo)


@ASTRewriter.transform
def select_ma(c, hb, ctr, G, ma, kp):
    """A REINDEX layer's rows (module comment above), G CTAs a row as select_mc (n <= 16384: CH <= 512, 2 keys a thread
    in registers), the row groups from CTA MA_X0; select_ma_prep's values in LDS (before the scorer); ma: the layer's MA
    slot (histograms), kp: its kept slot (the anchors). Returns 1 when its picks are written, 0 when select_mc must run
    (uniform)."""
    t, b, sm, TR = c["t"], c["b"], c["sm"], c["TR"]
    lane = t % 32
    wave = t // 32
    k = I32(KSEL)
    bx = (b - I32(MA_X0) + I32(MA_CTAS)) % I32(MA_CTAS)
    r = fx.min(bx // G, I32(TR - 1))
    g = bx % G
    pv = hx.lload(sm + O_MAP, I32, 4)
    n = (bx < TR * G).select(fx.min(fx.max(pv[0], I32(0)), I32(NCAND * 8)), I32(0))
    lo = pv[1].bitcast(F32)
    CH = (n + (G - 1)) // G
    i1 = fx.min(g * CH + CH, n)
    ia = g * CH + t
    ib = ia + 256
    va = ia < i1
    vb = ib < i1
    done = I32(0)
    if (bx < TR * G) & (n > k):
        rowp = c["ilog"] + i64(r) * i64(c["ilw"]) * 4
        inv = F32(254.0 / (2.0 * MA_S))
        xa = hx.gload(rowp + i64(fx.max(fx.min(ia, n - 1), I32(0))) * 4, I32)
        xb = hx.gload(rowp + i64(fx.max(fx.min(ib, n - 1), I32(0))) * 4, I32)
        lad = ma + i64(r * 256) * 4
        hx.lstore(sm + O_MCH + t * 4, I32(0))
        gpu.barrier()
        ba = _ma_bin(xa, lo, inv)
        bb = _ma_bin(xb, lo, inv)
        junk = I32(O_MAL // 4 + 32) + lane
        hx.latomic_add_i32(sm + va.select(I32(O_MCH // 4) + ba, junk) * 4, I32(1))
        hx.latomic_add_i32(sm + vb.select(I32(O_MCH // 4) + bb, junk) * 4, I32(1))
        gpu.barrier()
        v = hx.lload(sm + O_MCH + t * 4, I32)
        if v != 0:
            hx.atomic_add(lad + i64(t) * 4, v)
        hx.wait_storecnt0()
        gpu.barrier()
        if t == 0:
            hx.atomic_add(ctr + i64((r * 8 + 6) * 4), 1)
            hx.wait_count(ctr + i64((r * 8 + 6) * 4), G, 8)
        gpu.barrier()
        if t < 32:   # wave 0: the bin of the 512-th key (select_mc's scan)
            cs = [hx.gload(lad + i64(255 - 8 * lane - qq) * 4, I32, scope="agent") for qq in range(8)]
            s4 = cs[0]
            for qq in range_constexpr(1, 8):
                s4 = s4 + cs[qq]
            before = wave_incl_scan(s4, lane) - s4
            found = I32(-1)
            mb = I32(-1)
            for qq in range_constexpr(8):
                hit = (before < k) & (k <= before + cs[qq])
                found = hit.select(((255 - 8 * lane - qq) << 16) | (k - before), found)
                mb = hit.select(cs[qq], mb)
                before = before + cs[qq]
            found = wave_max(found)
            mb = wave_max(mb)
            if lane == 0:
                hx.lstore(sm + O_MC, fx.Vector.from_elements([found >> 16, mb, found & 0xFFFF, n], I32))
        gpu.barrier()
        st4 = hx.lload(sm + O_MC, I32, 4)
        j = st4[0]
        need = st4[2]
        if (j >= 1) & (j <= 254):
            done = I32(1)
            lf = hx.lload(sm + O_MAP + 16 + t * 16, I32, 4)
            wa = va & (ba > j)
            wb = vb & (bb > j)
            ma_ = va & (ba == j)
            mb_ = vb & (bb == j)
            bw = [_ballot(wa), _ballot(wb), _ballot(ma_), _ballot(mb_)]
            if lane == 0:   # the wave's winners / j-bin counts -> LDS
                hx.lstore(sm + O_MAL + wave * 8, fx.Vector.from_elements(
                    [_popc(bw[0]) + _popc(bw[1]), _popc(bw[2]) + _popc(bw[3])], I32))
            gpu.barrier()
            wpre = I32(0)
            mpre = I32(0)
            wtot = I32(0)
            mtot = I32(0)
            for wv in range_constexpr(WAVES):
                pr = hx.lload(sm + O_MAL + wv * 8, I32, 2)
                wpre = wpre + (wave > wv).select(pr[0], I32(0))
                mpre = mpre + (wave > wv).select(pr[1], I32(0))
                wtot = wtot + pr[0]
                mtot = mtot + pr[1]
            if t == 0:   # winners | j-bin keys << 16 (<= 512 winners: no carry), one device atomic; the next anchor
                hx.lstore(sm + O_MAL + 64, hx.atomic_add(ctr + i64(r * 32), wtot | (mtot << 16)))
                if g == 0:
                    hx.gstore(kp + i64(MA_KANC) + i64(r) * 4, lo + (j.to(F32) - F32(0.5)) / inv)
            gpu.barrier()
            bs = hx.lload(sm + O_MAL + 64, I32)
            wb0 = bs & 0xFFFF
            mb0 = (bs >> 16) & 0xFFFF
            m0 = mb0 + mpre + _ma_mbcnt(bw[2])
            m1 = mb0 + mpre + _popc(bw[2]) + _ma_mbcnt(bw[3])
            if wa:
                _ma_put(c, r, wb0 + wpre + _ma_mbcnt(bw[0]), lf[0], lf[1])
            if wb:
                _ma_put(c, r, wb0 + wpre + _popc(bw[0]) + _ma_mbcnt(bw[1]), lf[2], lf[3])
            if ma_ & (m0 < need):
                _ma_put(c, r, k - need + m0, lf[0], lf[1])
            if mb_ & (m1 < need):
                _ma_put(c, r, k - need + m1, lf[2], lf[3])
        gpu.barrier()   # (select_mc rewrites O_MC when this returns 0)
    return done
