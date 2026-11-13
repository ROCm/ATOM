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
from atom.models.deepseek_v41.mono.gfx1250.kernels import stamps
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
O_CNT = O_OUT + 2048 * 8                 # stage_select_any's per-row task counts [T]
O_RBD = O_CNT + T * 4                    # stage_select_any's per-row bound, batch slot and input count [T] each
LDS_BYTES2 = O_RBD + 3 * T * 4


@ASTRewriter.transform
def lds_topk(n, k, load, emit, sm, tid):
    """The top k (traced, <= 2048) of n (<= 8192) values: load(q, i) -> the 4 values i..i+3 (thread's quad q);
    emit(pos, bits, i) for pos < k (bits 0 / i -1 past min(n, k)). A small code footprint (the step kernel is
    instruction-fetch bound): the thread's 32 order keys go to LDS (keys[tid*32 + d]), and the 4 radix passes, the
    above / tie counts, the winners' placement (staged in LDS outs) and the emit are runtime loops."""
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
    gpu.barrier()
    stamps.point(2)
    prefix = I32(0)
    mask = I32(0)
    need = I32(k)
    n = fx.max(n, I32(0))
    if n > k:
        for p in range(4):
            lp = hx.fresh(tid) % 32
            shift = 24 - 8 * p
            h = hist + (p % 3) * (HIST_WORDS * 4)
            dummy = 256 + lp
            run_bin = I32(256) + lp
            run = I32(0)
            for dq in range(nq):
                kp4 = hx.lload(keys + (base + 4 * dq) * 4, I32, 4)
                for e in range_constexpr(4):
                    keyp = kp4[e]
                    cntd = (base + 4 * dq + e < n) & ((keyp & mask) == prefix)
                    binp = cntd.select((keyp >> shift) & 255, dummy)
                    samep = binp == run_bin
                    hx.latomic_add_i32(h + samep.select(dummy, run_bin) * 4, samep.select(I32(0), run))
                    run = samep.select(run + 1, I32(1))
                    run_bin = binp
            hx.latomic_add_i32(h + run_bin * 4, run)
            gpu.barrier()
            if p < 2:
                for r in range_constexpr((HIST_WORDS + THREADS - 1) // THREADS):
                    cz = hx.fresh(tid) + r * THREADS
                    if cz < HIST_WORDS:
                        hx.lstore(hist + (((p + 2) % 3) * HIST_WORDS + cz) * 4, I32(0))
            cs = [hx.lload(h + (255 - 8 * lp - q) * 4, I32) for q in range(8)]
            s4 = cs[0]
            for q in range_constexpr(1, 8):
                s4 = s4 + cs[q]
            before = wave_incl_scan(s4, lp) - s4
            found = I32(-1)
            for q in range_constexpr(8):
                hit = (before < need) & (need <= before + cs[q])
                found = hit.select(((255 - 8 * lp - q) << 16) | (need - before), found)
                before = before + cs[q]
            found = wave_max(found)
            prefix = prefix | (((found >> 16) & 255) << shift)
            mask = mask | (I32(255) << shift)
            need = found & 0xFFFF
    stamps.point(3)
    take_all = n <= k
    gt_bits = I32(0)
    eq_bits = I32(0)
    above = I32(0)
    ties = I32(0)
    for dg in range(nq):
        kg4 = hx.lload(keys + (base + 4 * dg) * 4, I32, 4)
        for e in range_constexpr(4):
            vd = base + 4 * dg + e < n
            gt = (vd & ult(prefix, kg4[e])).select(I32(1), I32(0))
            eq = (vd & (kg4[e] == prefix)).select(I32(1), I32(0))
            gt_bits = gt_bits | (gt << (4 * dg + e))
            eq_bits = eq_bits | (eq << (4 * dg + e))
            above = above + gt
            ties = ties + eq
    packed = block_excl_scan(above | (ties << 16), lane, wave, buf)
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
    stamps.point(4)
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
    stamps.point(5)


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
        hx.lstore(orb + (T + t) * 4, qt)
        hx.lstore(orb + (2 * T + t) * 4, nnt)
    gpu.barrier()
    # the rows' inclusive task prefix in every wave's lanes (index_plan.task_prefix): no serial LDS walks
    ln = hx.fresh(t) % 32
    incs, nt = IP.task_prefix(lambda r_: hx.lload(ocnt + r_ * 4, I32), TR, ln, lambda v: wave_incl_scan(v, ln),
                              lambda v: hx.lane_gather(v, I32(31)))
    for i in range(b, nt, 256):
        # the task's row: the rows whose inclusive prefix is <= i are before it; j = i - that prefix
        tok, j = IP.task_of(incs, i, TR, lambda p: _popc(_ballot(p)), wave_max)
        q = fx.max(hx.lload(orb + (T + tok) * 4, I32), I32(0))
        nn = hx.lload(orb + (2 * T + tok) * 4, I32)
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

        lds_topk(n, k, load, emit, c["sm"], t)
