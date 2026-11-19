# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""Top-p sampling with the recipe's T > 0 verification (ATOM dspark_prob_sample: PROB_DRAFT + BLOCK_VERIFY) fused in.

Rows r = req * 6 + k (k = 0..4: the rows that verify draft x_k of the
request, k = 5: the bonus row), 5 CTAs a row, 4 ranks (vocab shards of VR columns), as sample_common.

sp [48][8] (16 + 16 B a row): invT f32, topp f32, seed i32, greedy i32 | invTd f32 (draft 1/T), hasq i32 (q of this request's
drafts kept by the kernel's prob drafter last step), qsrc i32 (its row in that step's qlog), seed_acc i32.
Forced acceptance (ATOM synthetic_acceptance_rates): a request's bonus row words 5..7 = SIM_MAGIC, nf, fr (below).
qlog [8][5][VR] f32: this rank's pre-temperature draft logits of the previous step (markov_prob writes them).
dids [8][5] i32 drafts being verified. prow [48][8] (32 B a row) scratch. Outputs: outtok [8][6],
verdict [3][8], anc [8] (written by CTA 0 after the last handoff).

Passes (sample_common's P0..P4, then):
 P0  + q row (k < 5, hasq): online (mq, zq) of d * invTd, dx = d(x_k) * invTd if x_k is in the CTA's columns
 P1  + second exchange unit (mq, zq, dx): global q statistics; qx = exp(dx - mq) / zq
 P3  + nucleus mass Zn = sum_{z >= th} exp(z - M), zx = z(x_k) if owned
 P4  + second exchange unit (Zn, zx): px = (zx >= th) exp(zx - M) / Zn; row (px, qx) -> prow; handoff
 P5  over the row's columns (k < 5, T > 0): P_k = min-clamped product of p/q over rows < k of the request (prow);
     rm_k = sum max(P_k p - q, 0); residual Gumbel-max candidate log(max(P_k p - q, 0)) + g (no q: p without x_k), with p's
     own draw as the fallback; per-CTA records -> handoff -> rank-row -> exchange -> prow; handoff
 D   (CTA 0, thread per request): greedy -> argmax match; else block test tau = max{k + 1 : u_k <= h_k},
     h_k = rm_{k+1} / (rm_{k+1} + 1 - P_{k+1}), h_4 = P_5 (no q: sequential u_k < p(x_k)); token at tau = the residual
     winner of row tau, or the bonus row's sample when tau = 5.
Noise: hash32 counters (as sample_common): u_k = hash32(r, 0x40000000, seed_acc), residual g = Gumbel(hash32(r, gid, seed_acc ^
0x2545F491)). The reference draws the same in distribution (Philox there); a host model reproduces them with these hashes."""
from __future__ import annotations


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64
from flydsl.compiler.ast_rewriter import ASTRewriter

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels import sample_common as S4
from atom.models.deepseek_v41.mono.gfx1250.kernels.sample_common import (ROWS, CPR, NB, R, F32, I32, LN2, c32, i64, expf, log2f, lshr, I64MIN, hash32, skey,
                          latomic_max64, handoff, suffix_scan, crossing, UNITS, HB, roff, zload, poll_unit, xchg_hist)

NREQ, ND = D.MAX_REQ, D.DSPARK_W
SPW = 8                      # sp words a row
SEED_RES = 0x2545F491
U_IDX = 0x40000000
# forced acceptance (ATOM --spec-decode-acceptance-length; ``inputs.sim_acc_words``): the request's bonus row (k = 5) sp
# words 5..7, unused by that row's own sampling (it verifies no draft), = SIM_MAGIC, nf i32, fr f32 -> its rows k < 5
# are argmax rows and D accepts draft p iff u_p < c_p, c_p = 1 (p < nf), fr (p = nf), 0 (p > nf)
SIM_MAGIC = 0x53494D41
NEG = -3e38
PRB = 2048                   # LDS: P1's row records (CPR x (m, s), then CPR x (mq, sq, dx, -)), past P0's q statistics
OUT_SLOT_ROW = 47            # (outsl) the sparam row whose word 7 holds the step's output slot (0 / 1): past every served row
OUT_SLOT_BYTES = 8 * (ND + 1) * 4     # outtok [8][6] i32 a slot (2 slots), then verdict [3][8] i32 a slot (2 slots)
VER_SLOT_BYTES = 3 * 8 * 4
RACE_OFF = 6144              # LDS: (keepall) the race's CTA winner (8 B) + 32 z(x_k) slots, before ZOFF
ZOFF = 8192                  # LDS: this CTA's z = l / T chunk [PT][256] f32, then the q row's d * invTd chunk (read once in P0)


def ln(x):
    return F32(LN2) * log2f(x)


def unif(row, idx, seed):
    return (F32(lshr(hash32(row, idx, seed), 9)) + F32(0.5)) * F32(1.0 / 8388608.0)   # (23 bits: u < 1 exactly; 24 bits + 0.5 rounded to 1.0 in fp32)


def gumbel(row, idx, seed):
    return F32(-LN2) * log2f(F32(-LN2) * log2f(unif(row, idx, seed)))


def pkey(score, gid):
    """(score, lowest id on ties) as one ordered i64."""
    return fx.Int64(skey(score)) * fx.Int64(1 << 32) + (fx.Int64(c32(0xFFFFFFFF) - gid) & fx.Int64(0xFFFFFFFF))


def kid(key):
    return c32(0xFFFFFFFF) - I32(key & fx.Int64(0xFFFFFFFF))


def kscore(key):
    """Back from the ordered i32 of pkey to f32."""
    s = I32(key >> 32)
    return ((s < 0).select(s ^ I32(0x7FFFFFFF), s)).bitcast(F32)


def _ctxmid(ctxp, i):   # (plain python: outside the AST rewriter)
    return None if ctxp is None else ctxp[i]


@ASTRewriter.transform
def xchg6(recv, rank, p, row, c, t, sm, ws, tag, mid=None):
    """3 n i32 words of this rank's row record (n tagged units of slot p; 6 words: 2) -> LDS HB + src * 12 n for src
    0..3 (_lw's stride)."""
    nu = len(ws) // 3
    if (c == 0) & (t == 0):
        for u_ in range_constexpr(nu):
            v = fx.Vector.from_elements([ws[3 * u_], ws[3 * u_ + 1], ws[3 * u_ + 2], I32(tag)], I32)
            for o in range_constexpr(4):
                if fx.const_expr(o != rank):
                    hx.gstore(recv[o] + roff(p, rank, row, u_), v, nt=True)
    if t == rank:
        for q in range_constexpr(3 * nu):
            hx.lstore(sm + HB + t * (12 * nu) + q * 4, ws[q])
    if fx.const_expr(mid is not None):   # other work (all threads) while the peers' words travel
        mid()
    if (t < 4 * nu) & ((t % 4) != rank):
        src = t % 4
        u = t // 4
        got = poll_unit(recv[rank] + roff(p, src, row, u), tag)
        for q in range_constexpr(3):
            hx.lstore(sm + HB + src * (12 * nu) + (u * 3 + q) * 4, got[q])
    gpu.barrier()


def _lw(sm, src, q, kind=F32, nw=6):
    return hx.lload(sm + HB + src * (4 * nw) + q * 4, kind)


# (row folds) a row's CPR = 32 CTA records, record q in lane q of every wave, folded by an xor butterfly: 5 shuffle steps
# instead of a 32-long serial chain a thread. Every lane ends with the same bits (each step adds / maxes a lane's value
# with its partner's, commutative), so every CTA of the row and every rank fold alike.
assert CPR == 32


def _bf_sum(x):
    for o in (16, 8, 4, 2, 1):
        x = x + x.shuffle_xor(o, 32)
    return x


def _bf_max(x):
    for o in (16, 8, 4, 2, 1):
        x = fx.max(x, x.shuffle_xor(o, 32))
    return x


def _bf_max64(x):
    for o in (16, 8, 4, 2, 1):
        lo = I32(x & fx.Int64(0xFFFFFFFF)).shuffle_xor(o, 32)
        hi = I32(x >> 32).shuffle_xor(o, 32)
        x = fx.max(x, (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF)))
    return x


@ASTRewriter.transform
def sample_once_p(lg, rec0, rec1, rec2, rec3, ids, cnt, t, b, sm, V, recv, rank, tag0, sp, qlog, dids, prow, outtok,
                  verdict, anc, qids=None, ctxp=None, com=None, outsl=None):
    t = hx.opaque_v(t)
    b = hx.opaque_s(b)
    row = b // CPR
    c = b % CPR
    act = b < ROWS * CPR
    row = fx.min(row, I32(ROWS - 1))
    req = row // (ND + 1)
    k = row % (ND + 1)
    spb = sp + i64(row) * (SPW * 4)
    invT = hx.gload(spb, F32)
    topp = hx.gload(spb + i64(4), F32)
    seed = hx.gload(spb + i64(8), I32)
    isd = k < ND                                   # a row that verifies draft x_k
    simr = hx.gload(sp + i64(req * (ND + 1) + ND) * (SPW * 4) + i64(20), I32) == I32(SIM_MAGIC)
    greedy = (hx.gload(spb + i64(12), I32) != I32(0)) | (simr & isd)   # forced acceptance: the correction is the argmax
    invTd = hx.gload(spb + i64(16), F32)
    hasq = hx.gload(spb + i64(20), I32) != I32(0)
    qsrc = fx.max(hx.gload(spb + i64(24), I32), I32(0))
    seed_acc = hx.gload(spb + i64(28), I32)
    prob = act & isd & ~greedy                     # T > 0 verification row
    # q is used only if the drafts being verified are the ones the prob drafter kept q for (qids, written with qlog);
    # a mismatch (host truncation, a native step in between, ...) -> the request is verified as without q (exact)
    useq = prob & hasq
    if fx.const_expr(qids is not None):
        for p_ in range_constexpr(ND):
            useq = useq & (hx.gload(dids + i64(req * ND + p_) * 4, I32) == hx.gload(qids + i64(qsrc * ND + p_) * 4, I32))
    xk = hx.gload(dids + i64(req * ND + fx.min(k, I32(ND - 1))) * 4, I32)
    CH = (V + CPR - 1) // CPR
    PT = (CH + 255) // 256
    c0 = c * CH
    base = lg + i64(row) * (V * 4)
    qbase = qlog + i64((qsrc * ND + fx.min(k, I32(ND - 1))) * V) * 4
    xl = xk - rank * V                             # x_k's local column (outside [0, V): another rank's)
    NCT = ROWS * CPR   # (the sampling CTAs alone: tail.target runs the draft context on the others meanwhile)
    # Does every row of the step keep every id (greedy, a forced-acceptance argmax row, or top_p >= 1)? sparam: the same
    # on every rank, a uniform branch (as anyprob). Then th = -inf: P1 / P2 (the top-p threshold) are skipped and P3's
    # draw needs nothing of this pass's exchange: P0 also draws it (the same Gumbel keys, below), X0 carries it, and P3,
    # its handoff and its exchange X3 are skipped.
    nucl = I32(0)
    for r_ in range_constexpr(ROWS):
        sr_ = sp + i64(r_) * (SPW * 4)
        bon_ = sp + i64((r_ // (ND + 1)) * (ND + 1) + ND) * (SPW * 4)
        sim_ = (hx.gload(bon_ + i64(20), I32) == I32(SIM_MAGIC)) & (I32(r_ % (ND + 1)) < I32(ND))
        keep_ = (hx.gload(sr_ + i64(12), I32) != I32(0)) | sim_ | (hx.gload(sr_ + i64(4), F32) >= F32(1.0))
        nucl = nucl + keep_.select(I32(0), I32(1))
    keepall = nucl == I32(0)
    if keepall:   # (the race's per-CTA maxima, before any thread's atomics: the barrier)
        if t == 0:
            hx.lstore(sm + RACE_OFF, I64MIN())
        if t < 32:
            hx.lstore(sm + RACE_OFF + 8 + t * 4, I32(0))
        gpu.barrier()
    # P0: (max, sum exp) of z = l / T, and of the q row's d * invTd; d(x_k)
    m = F32(-1e30)
    s = F32(0.0)
    mq = F32(-1e30)
    sq = F32(0.0)
    dx = F32(NEG)
    # the chunk's z (and the q row's d) read from global once, all loads in flight together, into LDS (each thread
    # its own elements: no barrier); every later pass reads LDS (passes over global memory were latency chains)
    QOFF = ZOFF + PT * 1024
    for kk in range_constexpr(PT):
        hx.lstore(sm + ZOFF + (kk * 256 + t) * 4, zload(base, c0, CH, V, t, kk, invT))
    if useq:
        for kk in range_constexpr(PT):
            hx.lstore(sm + QOFF + (kk * 256 + t) * 4, zload(qbase, c0, CH, V, t, kk, invTd))
    for kk in range(PT):
        z = hx.lload(sm + ZOFF + (kk * 256 + t) * 4, F32)
        mn = fx.max(m, z)
        s = s * expf(m - mn) + expf(z - mn)
        m = mn
        if useq:
            d = hx.lload(sm + QOFF + (kk * 256 + t) * 4, F32)
            mqn = fx.max(mq, d)
            sq = sq * expf(mq - mqn) + expf(d - mqn)
            mq = mqn
            j = c0 + t + kk * 256
            dx = (j == xl).select(d, dx)
    best0 = I64MIN()
    zx0 = F32(NEG)
    if keepall:
        # P3's draw with th = -inf, here: the same key a column (u from hash32(row, global id, the row's seed): the
        # same counter, row and seed words; greedy rows g = 0), every column (as z >= -inf in P3), z(x_k) as P3's
        for kk in range(PT):
            z = hx.lload(sm + ZOFF + (kk * 256 + t) * 4, F32)
            j = c0 + t + kk * 256
            u = (F32(lshr(hash32(row, j + rank * V, seed), 9)) + F32(0.5)) * F32(1.0 / 8388608.0)
            g = F32(-LN2) * log2f(F32(-LN2) * log2f(u))
            g = greedy.select(F32(0.0), g)
            pk = fx.Int64(skey(z + g)) * fx.Int64(1 << 32) + (fx.Int64(c32(0xFFFFFFFF) - (j + rank * V)) & fx.Int64(0xFFFFFFFF))
            best0 = fx.max(best0, pk)
            zx0 = (j == xl).select(z, zx0)
        latomic_max64(sm + RACE_OFF, best0)
        hx.latomic_max_u32(sm + RACE_OFF + 8 + (t % 32) * 4, skey(zx0) ^ I32(-2147483648))
    hx.lstore(sm + t * 8, m)
    hx.lstore(sm + t * 8 + 4, s)
    hx.lstore(sm + 2048 + t * 12, mq)
    hx.lstore(sm + 2048 + t * 12 + 4, sq)
    hx.lstore(sm + 2048 + t * 12 + 8, dx)
    gpu.barrier()
    for kr in range_constexpr(8):
        d2 = 128 >> kr
        if t < d2:
            m1 = hx.lload(sm + t * 8, F32)
            s1 = hx.lload(sm + t * 8 + 4, F32)
            m2 = hx.lload(sm + (t + d2) * 8, F32)
            s2 = hx.lload(sm + (t + d2) * 8 + 4, F32)
            mn = fx.max(m1, m2)
            hx.lstore(sm + t * 8, mn)
            hx.lstore(sm + t * 8 + 4, s1 * expf(m1 - mn) + s2 * expf(m2 - mn))
            q1 = hx.lload(sm + 2048 + t * 12, F32)
            r1 = hx.lload(sm + 2048 + t * 12 + 4, F32)
            q2 = hx.lload(sm + 2048 + (t + d2) * 12, F32)
            r2 = hx.lload(sm + 2048 + (t + d2) * 12 + 4, F32)
            qn = fx.max(q1, q2)
            hx.lstore(sm + 2048 + t * 12, qn)
            hx.lstore(sm + 2048 + t * 12 + 4, r1 * expf(q1 - qn) + r2 * expf(q2 - qn))
            hx.lstore(sm + 2048 + t * 12 + 8, fx.max(hx.lload(sm + 2048 + t * 12 + 8, F32),
                                                     hx.lload(sm + 2048 + (t + d2) * 12 + 8, F32)))
        gpu.barrier()
    if act & (t < 2):
        hx.gstore(rec0 + i64(row * CPR + c) * 8 + i64(t) * 4, hx.lload(sm + t * 4, F32))
    if act & (t < 3):   # q statistics after the (m, s) records: [ROWS*CPR][4] f32
        hx.gstore(rec0 + i64(ROWS * CPR * 8) + i64(row * CPR + c) * 16 + i64(t) * 4, hx.lload(sm + 2048 + t * 4, F32))
    if keepall & act & (t == 0):   # the race's CTA record, where P3 writes its own (rec3)
        zxs = I32(-2147483648)
        for q in range_constexpr(32):
            zxs = fx.max(zxs, hx.lload(sm + RACE_OFF + 8 + q * 4, I32) ^ I32(-2147483648))
        hx.gstore(rec3 + i64(row * CPR + c) * 8, hx.lload(sm + RACE_OFF, fx.Int64))
        hx.gstore(rec3 + i64(ROWS * CPR * 12) + i64(row * CPR + c) * 8 + i64(4), ((zxs < 0).select(zxs ^ I32(0x7FFFFFFF), zxs)).bitcast(F32))
    handoff(cnt, t, NCT)
    # P1: rank-row (m, s) and (mq, sq, dx) -> exchange -> global; histogram of z in [M - R, M)
    # the row's CPR records read once (lane q < CPR: record q) into LDS PRB, then every thread folds them in q order (the
    # same values / order as each thread's own global loads: ~200 loads a thread, 12.1 us, ktrace gapS)
    qb = rec0 + i64(ROWS * CPR * 8) + i64(row * CPR) * 16
    if t < CPR:
        hx.lstore(sm + PRB + t * 8, hx.gload(rec0 + i64(row * CPR + t) * 8, F32, 2))
        hx.lstore(sm + PRB + CPR * 8 + t * 16, hx.gload(qb + i64(t * 16), F32, 4))
    gpu.barrier()
    lq = t % 32
    mq_ = hx.lload(sm + PRB + lq * 8, F32)
    mr = _bf_max(mq_)
    sr = _bf_sum(hx.lload(sm + PRB + lq * 8 + 4, F32) * expf(mq_ - mr))
    qm_ = hx.lload(sm + PRB + CPR * 8 + lq * 16, F32)
    mqr = _bf_max(qm_)
    dxr = _bf_max(hx.lload(sm + PRB + CPR * 8 + lq * 16 + 8, F32))
    sqr = _bf_sum(hx.lload(sm + PRB + CPR * 8 + lq * 16 + 4, F32) * expf(qm_ - mqr))
    wb0 = I64MIN()
    zxr0 = F32(NEG)
    if keepall:   # the rank-row race winner and z(x_k) (P4's fold of the CTA records)
        wb0 = _bf_max64(hx.gload(rec3 + i64(row * CPR + lq) * 8, fx.Int64))
        zxr0 = _bf_max(hx.gload(rec3 + i64(ROWS * CPR * 12) + i64(row * CPR + lq) * 8 + i64(4), F32))
    xchg6(recv, rank, 0, row, c, t, sm, [mr.bitcast(I32), sr.bitcast(I32), mqr.bitcast(I32), sqr.bitcast(I32),
                                          dxr.bitcast(I32), zxr0.bitcast(I32), I32(wb0 & fx.Int64(0xFFFFFFFF)),
                                          I32(wb0 >> 32), I32(0)], tag0, mid=_ctxmid(ctxp, 0))
    M = F32(-1e30)
    MQ = F32(-1e30)
    DX = F32(NEG)
    GB0 = I64MIN()
    ZX0 = F32(NEG)
    for o in range_constexpr(4):
        M = fx.max(M, _lw(sm, o, 0, nw=9))
        MQ = fx.max(MQ, _lw(sm, o, 2, nw=9))
        DX = fx.max(DX, _lw(sm, o, 4, nw=9))
        ZX0 = fx.max(ZX0, _lw(sm, o, 5, nw=9))
        GB0 = fx.max(GB0, (fx.Int64(_lw(sm, o, 7, I32, nw=9)) << 32) | (fx.Int64(_lw(sm, o, 6, I32, nw=9)) & fx.Int64(0xFFFFFFFF)))
    Z = F32(0.0)
    ZQ = F32(0.0)
    for o in range_constexpr(4):
        Z = Z + _lw(sm, o, 1, nw=9) * expf(_lw(sm, o, 0, nw=9) - M)
        ZQ = ZQ + _lw(sm, o, 3, nw=9) * expf(_lw(sm, o, 2, nw=9) - MQ)
    qx = expf(DX - MQ) / ZQ
    gpu.barrier()
    hx.lstore(sm + t * 4, F32(0.0))
    gpu.barrier()
    lo1 = M - F32(R)
    # P1 / P2 find the top-p threshold th. A row that is greedy (or a forced-acceptance argmax row) or has top_p >= 1
    # keeps every id: th = -inf whatever they give (top_p 1: the target p Z is above the mass of [M - R, M) unless the
    # whole row lies within R of M -- then the two passes cut the lowest sub-bin, here nothing is cut, ATOM's top_p 1).
    # When every row of the step is such a row (sparam: the same on every rank, a uniform branch, as anyprob), the two
    # passes, their handoffs and their exchanges are skipped (T = 0, and T > 0 at the default top_p 1).
    th = F32(NEG)
    gb = GB0
    ZN = Z     # (every id kept: the nucleus mass is Z)
    ZX = ZX0
    if nucl > I32(0):
        sc1 = F32(NB / R)
        for kk in range(PT):
            z = hx.lload(sm + ZOFF + (kk * 256 + t) * 4, F32)
            if z >= lo1:
                bi = fx.min(I32((z - lo1) * sc1), I32(NB - 1))
                hx.latomic_add_f32(sm + bi * 4, expf(z - M))
        gpu.barrier()
        if act:
            hx.gstore(rec1 + (i64(row * CPR + c) * NB + i64(t)) * 4, hx.lload(sm + t * 4, F32))
        handoff(cnt + i64(4), t, NCT)
        # P2: crossing bin b1; sub-histogram inside it
        target = topp * Z
        hs = F32(0.0)
        for q in range_constexpr(CPR):
            hs = hs + hx.gload(rec1 + (i64(row * CPR + q) * NB + i64(t)) * 4, F32)
        hx.lstore(sm + t * 4, hs)
        gpu.barrier()
        xchg_hist(recv, rank, 1, row, c, t, sm, tag0 + 1, mid=_ctxmid(ctxp, 1))
        suffix_scan(sm, t)
        b1, rem1 = crossing(sm, t, target)
        hx.lstore(sm + t * 4, F32(0.0))
        gpu.barrier()
        lo2 = lo1 + F32(b1) * F32(R / NB)
        sc2 = F32(NB * NB / R)
        for kk in range(PT):
            z = hx.lload(sm + ZOFF + (kk * 256 + t) * 4, F32)
            if (b1 >= 0) & (z >= lo2):
                bi = fx.min(I32((z - lo2) * sc2), I32(NB - 1))
                if (z < lo2 + F32(R / NB)) | (b1 == NB - 1):
                    hx.latomic_add_f32(sm + bi * 4, expf(z - M))
        gpu.barrier()
        if act:
            hx.gstore(rec2 + (i64(row * CPR + c) * NB + i64(t)) * 4, hx.lload(sm + t * 4, F32))
        handoff(cnt + i64(8), t, NCT)
        # P3: threshold, Gumbel-max over the kept set; + the nucleus mass and z(x_k)
        hs = F32(0.0)
        for q in range_constexpr(CPR):
            hs = hs + hx.gload(rec2 + (i64(row * CPR + q) * NB + i64(t)) * 4, F32)
        hx.lstore(sm + t * 4, hs)
        gpu.barrier()
        xchg_hist(recv, rank, 2, row, c, t, sm, tag0 + 2, mid=_ctxmid(ctxp, 2))
        suffix_scan(sm, t)
        b2, rem2 = crossing(sm, t, rem1)
        th = (b1 < 0).select(F32(NEG), lo2 + F32(fx.max(b2, I32(0))) * F32(R / (NB * NB)))
        th = greedy.select(F32(NEG), th)
        if t == 0:
            hx.lstore(sm + 2048, I64MIN())
        hx.lstore(sm + 4096 + t * 4, F32(0.0))
        gpu.barrier()
        best = I64MIN()
        zn = F32(0.0)
        zx = F32(NEG)
        for kk in range(PT):
            z = hx.lload(sm + ZOFF + (kk * 256 + t) * 4, F32)
            j = c0 + t + kk * 256
            u = (F32(lshr(hash32(row, j + rank * V, seed), 9)) + F32(0.5)) * F32(1.0 / 8388608.0)   # (23 bits: u < 1 exactly; 24 bits + 0.5 rounded to 1.0 in fp32)
            g = F32(-LN2) * log2f(F32(-LN2) * log2f(u))
            g = greedy.select(F32(0.0), g)
            pk = fx.Int64(skey(z + g)) * fx.Int64(1 << 32) + (fx.Int64(c32(0xFFFFFFFF) - (j + rank * V)) & fx.Int64(0xFFFFFFFF))
            best = (z >= th).select(fx.max(best, pk), best)
            zn = zn + (z >= th).select(expf(z - M), F32(0.0))
            zx = (j == xl).select(z, zx)
        latomic_max64(sm + 2048, best)
        hx.latomic_add_f32(sm + 4096 + (t % 32) * 4, zn)
        hx.latomic_max_u32(sm + 4096 + 128 + (t % 32) * 4, skey(zx) ^ I32(-2147483648))
        gpu.barrier()
        if act & (t == 0):
            hx.gstore(rec3 + i64(row * CPR + c) * 8, hx.lload(sm + 2048, fx.Int64))
            hx.gstore(rec3 + i64(ROWS * CPR * 8) + i64(row * CPR + c) * 4, th)
            znc = F32(0.0)
            zxs = I32(-2147483648)
            for q in range_constexpr(32):
                znc = znc + hx.lload(sm + 4096 + q * 4, F32)
                zxs = fx.max(zxs, hx.lload(sm + 4096 + 128 + q * 4, I32) ^ I32(-2147483648))   # unsigned slot -> its skey
            hx.gstore(rec3 + i64(ROWS * CPR * 12) + i64(row * CPR + c) * 8, znc)
            hx.gstore(rec3 + i64(ROWS * CPR * 12) + i64(row * CPR + c) * 8 + i64(4), ((zxs < 0).select(zxs ^ I32(0x7FFFFFFF), zxs)).bitcast(F32))
        handoff(cnt + i64(12), t, NCT)
        # P4: rank-row winner, nucleus mass, z(x_k) -> exchange -> global (every CTA of a row, every rank)
        wb = _bf_max64(hx.gload(rec3 + i64(row * CPR + lq) * 8, fx.Int64))
        znr = _bf_sum(hx.gload(rec3 + i64(ROWS * CPR * 12) + i64(row * CPR + lq) * 8, F32))
        zxr = _bf_max(hx.gload(rec3 + i64(ROWS * CPR * 12) + i64(row * CPR + lq) * 8 + i64(4), F32))
        xchg6(recv, rank, 3, row, c, t, sm, [I32(wb & fx.Int64(0xFFFFFFFF)), I32(wb >> 32), znr.bitcast(I32), zxr.bitcast(I32),
                                              I32(0), I32(0)], tag0 + 3, mid=_ctxmid(ctxp, 3))
        gb = I64MIN()
        ZN = F32(0.0)
        ZX = F32(NEG)
        for o in range_constexpr(4):
            gb = fx.max(gb, (fx.Int64(_lw(sm, o, 1, I32)) << 32) | (fx.Int64(_lw(sm, o, 0, I32)) & fx.Int64(0xFFFFFFFF)))
            ZN = ZN + _lw(sm, o, 2)
            ZX = fx.max(ZX, _lw(sm, o, 3))
    th = greedy.select(F32(NEG), th)
    px = (ZX >= th).select(expf(ZX - M) / ZN, F32(0.0))
    if act & (c == 0) & (t == 0):
        hx.gstore(ids + i64(row) * 4, kid(gb))
        hx.gstore(prow + i64(row) * 32, px)
        hx.gstore(prow + i64(row) * 32 + i64(4), qx)
    handoff(cnt + i64(16), t, NCT)
    # a batch without T > 0 requests (sparam identical on every rank: a uniform branch) skips P5 and its two handoffs and
    # exchange; D then needs only the ids (P4's handoff)
    anyprob = I32(0) > I32(1)
    for r_ in range_constexpr(NREQ):
        anyprob = anyprob | (hx.gload(sp + i64(r_ * (ND + 1)) * (SPW * 4) + i64(12), I32) == I32(0))
    if anyprob:
        # P5: rm_k and the residual candidates of row k (T > 0 rows k < 5)
        lp = F32(0.0)
        for j5 in range_constexpr(ND):
            pj = hx.gload(prow + i64(req * (ND + 1) + j5) * 32, F32)
            qj = hx.gload(prow + i64(req * (ND + 1) + j5) * 32 + i64(4), F32)
            lr = (pj > F32(0.0)).select(ln(pj) - ln(qj), F32(NEG))
            lp = (I32(j5) < k).select(fx.min(lp + lr, F32(0.0)), lp)
        Pk = expf(lp)
        if t == 0:
            hx.lstore(sm + 2048, I64MIN())
            hx.lstore(sm + 2056, I64MIN())
        hx.lstore(sm + 4096 + t * 4, F32(0.0))
        gpu.barrier()
        rm = F32(0.0)
        br = I64MIN()
        bp = I64MIN()
        if prob:
            sres = seed_acc ^ I32(SEED_RES)
            for kk in range(PT):
                y = hx.lload(sm + ZOFF + (kk * 256 + t) * 4, F32)
                j = c0 + t + kk * 256
                gid = j + rank * V
                p = (y >= th).select(expf(y - M) / ZN, F32(0.0))
                wv = (j == xl).select(F32(0.0), p)
                if useq:
                    d = hx.lload(sm + ZOFF + PT * 1024 + (kk * 256 + t) * 4, F32)
                    qv = expf(d - MQ) / ZQ
                    wv = fx.max(Pk * p - qv, F32(0.0))
                valid = j < c0 + CH
                rm = rm + (valid & useq).select(wv, F32(0.0))
                g = gumbel(row, gid, sres)
                br = (valid & (wv > F32(0.0))).select(fx.max(br, pkey(ln(wv) + g, gid)), br)
                bp = (valid & (p > F32(0.0))).select(fx.max(bp, pkey(ln(p) + g, gid)), bp)
        hx.latomic_add_f32(sm + 4096 + (t % 32) * 4, rm)
        latomic_max64(sm + 2048, br)
        latomic_max64(sm + 2056, bp)
        gpu.barrier()
        r5 = rec1 + i64(row * CPR + c) * 32
        if act & (t == 0):
            rmc = F32(0.0)
            for q in range_constexpr(32):
                rmc = rmc + hx.lload(sm + 4096 + q * 4, F32)
            hx.gstore(r5, rmc)
            hx.gstore(r5 + i64(8), hx.lload(sm + 2048, fx.Int64))
            hx.gstore(r5 + i64(16), hx.lload(sm + 2056, fx.Int64))
        handoff(cnt + i64(20), t, NCT)
        rq = rec1 + i64(row * CPR + lq) * 32
        rmr = _bf_sum(hx.gload(rq, F32))
        brr = _bf_max64(hx.gload(rq + i64(8), fx.Int64))
        bpr = _bf_max64(hx.gload(rq + i64(16), fx.Int64))
        xchg6(recv, rank, 1, row, c, t, sm, [rmr.bitcast(I32), I32(brr & fx.Int64(0xFFFFFFFF)), I32(brr >> 32),
                                              I32(bpr & fx.Int64(0xFFFFFFFF)), I32(bpr >> 32), I32(0)], tag0 + 4)
        RM = F32(0.0)
        BR = I64MIN()
        BP = I64MIN()
        for o in range_constexpr(4):
            RM = RM + _lw(sm, o, 0)
            BR = fx.max(BR, (fx.Int64(_lw(sm, o, 2, I32)) << 32) | (fx.Int64(_lw(sm, o, 1, I32)) & fx.Int64(0xFFFFFFFF)))
            BP = fx.max(BP, (fx.Int64(_lw(sm, o, 4, I32)) << 32) | (fx.Int64(_lw(sm, o, 3, I32)) & fx.Int64(0xFFFFFFFF)))
        if act & (c == 0) & (t == 0):
            hx.gstore(prow + i64(row) * 32 + i64(8), RM)
            tok = (BR > I64MIN()).select(kid(BR), kid(BP))   # all-zero residual -> p's own draw
            hx.gstore(prow + i64(row) * 32 + i64(12), tok)
        handoff(cnt + i64(24), t, NCT)
    # D: per request on CTA 0 (every rank identically)
    if (b == 0) & (t < NREQ):
        rr = t
        r0 = rr * (ND + 1)
        sb = sp + i64(r0) * (SPW * 4)
        gq = hx.gload(sb + i64(12), I32) != I32(0)
        hq = hx.gload(sb + i64(20), I32) != I32(0)
        if fx.const_expr(qids is not None):
            qs_ = fx.max(hx.gload(sb + i64(24), I32), I32(0))
            for p_ in range_constexpr(ND):
                hq = hq & (hx.gload(dids + i64(rr * ND + p_) * 4, I32) == hx.gload(qids + i64(qs_ * ND + p_) * 4, I32))
        # a request whose p(x) / q(x) / rm is not finite (NaN draft or target logits) is verified as without q: the
        # block test's h = rm / (rm + 1 - P) took a NaN rm as 1 and accepted every draft (a NaN draft row's id-0 drafts were accepted)
        for p_ in range_constexpr(ND):
            for o_ in range_constexpr(3):
                fv = hx.gload(prow + i64(r0 + p_) * 32 + i64(o_ * 4), F32)
                hq = hq & (fv == fv) & (abs(fv) < F32(3e38))
        sacc = hx.gload(sb + i64(28), I32)
        tid = [hx.gload(ids + i64(r0 + p) * 4, I32) for p in range(ND + 1)]
        xs = [hx.gload(dids + i64(rr * ND + p) * 4, I32) for p in range(ND)]
        # greedy: first mismatch
        fg = I32(ND)
        for p in range_constexpr(ND - 1, -1, -1):
            fg = (xs[p] != tid[p]).select(I32(p), fg)
        # T > 0 with q: block test
        tau = I32(0)
        lpd = F32(0.0)
        for p in range_constexpr(ND):
            pxp = hx.gload(prow + i64(r0 + p) * 32, F32)
            qxp = hx.gload(prow + i64(r0 + p) * 32 + i64(4), F32)
            lrp = (pxp > F32(0.0)).select(ln(pxp) - ln(qxp), F32(NEG))
            lpd = fx.min(lpd + lrp, F32(0.0))
            Pn = expf(lpd)
            h = Pn
            if fx.const_expr(p < ND - 1):
                rmn = hx.gload(prow + i64(r0 + p + 1) * 32 + i64(8), F32)
                den = rmn + F32(1.0) - Pn
                h = (den > F32(0.0)).select(rmn / den, F32(1.0))
            uk = unif(r0 + p, I32(U_IDX), sacc)
            tau = (uk <= h).select(I32(p + 1), tau)
        # T > 0 without q: sequential one-hot rule
        fo = I32(ND)
        for p in range_constexpr(ND - 1, -1, -1):
            pxp = hx.gload(prow + i64(r0 + p) * 32, F32)
            uk = unif(r0 + p, I32(U_IDX), sacc)
            fo = (uk < pxp).select(fo, I32(p))
        # forced acceptance (ATOM's rejection_synthetic_sample_kernel): the first p with u_p >= c_p; the drafts before
        # it are emitted, its row's argmax corrects it (the rows k < 5 are argmax rows), no rejection -> the bonus sample
        sb5 = sb + i64(ND * SPW * 4)
        sim = hx.gload(sb5 + i64(20), I32) == I32(SIM_MAGIC)
        snf = hx.gload(sb5 + i64(24), I32)
        sfr = hx.gload(sb5 + i64(28), F32)
        fs = I32(ND)
        for p in range_constexpr(ND - 1, -1, -1):
            cp = (I32(p) < snf).select(F32(1.0), (I32(p) == snf).select(sfr, F32(0.0)))
            uk = unif(r0 + p, I32(U_IDX), sacc)
            fs = (uk < cp).select(fs, I32(p))
        f = sim.select(fs, gq.select(fg, hq.select(tau, fo)))
        gt = gq | sim
        # the token at f: greedy -> the target id; T > 0 -> the residual winner of row f, or the bonus sample at f = ND
        tf = tid[ND]
        for p in range_constexpr(ND):
            rt = hx.gload(prow + i64(r0 + p) * 32 + i64(12), I32)
            tf = (f == p).select(gt.select(tid[p], rt), tf)
        # a row whose logits are not finite has no p > 0 anywhere ->
        # no residual / p winner -> kid(INT64_MIN) = -1, which the draft entry then embeds (a page fault below emb):
        # fall back to the row's own sample, then to id 0, so a bad row costs a wrong token, not the GPU
        tff = tid[ND]
        for p in range_constexpr(ND):
            tff = (f == p).select(tid[p], tff)
        tf = (tf < I32(0)).select(fx.max(tff, I32(0)), tf)
        # outsl (when given): the host's copies -- outtok / verdict also into the step's slot of a 2-slot pair (slot
        # word: sparam row OUT_SLOT_ROW, word 7, host-written a step), read in place by the host's consumers (no clone)
        o_out, o_ver = [outtok], [verdict]
        if fx.const_expr(outsl is not None):
            sl_ = hx.gload(sp + i64(OUT_SLOT_ROW * SPW + 7) * 4, I32)
            o_out.append(outsl + i64(sl_) * OUT_SLOT_BYTES)
            o_ver.append(outsl + i64(2 * OUT_SLOT_BYTES) + i64(sl_) * VER_SLOT_BYTES)
        for o_ in range_constexpr(len(o_out)):
            for p in range_constexpr(ND):
                hx.gstore(o_out[o_] + i64(rr * (ND + 1) + p) * 4, (I32(p) < f).select(xs[p], (I32(p) == f).select(tf, I32(-1))))
            hx.gstore(o_out[o_] + i64(rr * (ND + 1) + ND) * 4, (f == ND).select(tf, I32(-1)))
            hx.gstore(o_ver[o_] + i64(rr) * 4, f)
            hx.gstore(o_ver[o_] + i64(NREQ + rr) * 4, I32(ND) - f)
            hx.gstore(o_ver[o_] + i64(2 * NREQ + rr) * 4, r0 + f)
        hx.gstore(anc + i64(rr) * 4, tf)
        if fx.const_expr(com is not None):
            commit(com, rr, f)


# tcom (i64 words, kernels/prep.py writes it each step): the Engram commit of the verify (ATOM's _commit_cursors) and
# each request's first position (the next step's device round advances from it)
COM_N, COM_CUR, COM_CAND, COM_CST, COM_CS0, COM_CS1, COM_W, COM_FIRSTP, COM_SLOT, COM_FIRST, COM_REQ = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24)
# the target launch's steady-counter stamps (100 MHz): prep's entry (COM_T0), each CTA's exit from the target launch
# (COM_EXIT + CTA): the host's poll-timeout check (``runner.launch``: no timing events on the step's stream)
COM_T0, COM_EXIT, COM_NEXIT = 25, 32, 256
COM_WMAX = 8                 # cursor row width bound (the history + 1: 4 for 4-grams)


@ASTRewriter.transform
def commit(com, rr, f):
    """Request rr (D's thread): its first position -> firstp; when the step staged cursor candidates (COM_N requests),
    its accepted prefix's candidate cursor (prefix f: the request's rows start at rr x 6) -> cursor[its slot]."""
    r64 = fx.Int64(rr)
    if r64 < hx.gload(com + i64(COM_REQ * 8), fx.Int64):
        fp = hx.gload(com + i64(COM_FIRSTP * 8), fx.Int64)
        hx.gstore(fp + r64 * 4, hx.gload(com + i64(COM_FIRST * 8) + r64 * 8, fx.Int64).to(I32))
    if r64 < hx.gload(com + i64(COM_N * 8), fx.Int64):
        cs0 = hx.gload(com + i64(COM_CS0 * 8), fx.Int64)
        cs1 = hx.gload(com + i64(COM_CS1 * 8), fx.Int64)
        src = hx.gload(com + i64(COM_CAND * 8), fx.Int64) + (r64 * cs0 + fx.Int64(f) * cs1) * 8
        slot = hx.gload(com + i64(COM_SLOT * 8) + r64 * 8, fx.Int64)
        dst = hx.gload(com + i64(COM_CUR * 8), fx.Int64) + slot * hx.gload(com + i64(COM_CST * 8), fx.Int64) * 8
        width = hx.gload(com + i64(COM_W * 8), fx.Int64)
        for w in range_constexpr(COM_WMAX):
            if fx.Int64(w) < width:
                hx.gstore(dst + i64(w * 8), hx.gload(src + i64(w * 8), fx.Int64))


def build(V, rank, rep=1):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def samp(lg: Int64, rec0: Int64, rec1: Int64, rec2: Int64, rec3: Int64, ids: Int64, cnt: Int64,
             recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64, sp: Int64, qlog: Int64, dids: Int64, prow: Int64,
             outtok: Int64, verdict: Int64, anc: Int64, tagbase: fx.Int32):
        t = fx.thread_idx.x
        b = fx.block_idx.x
        sm = hx.lds_base()
        recv = [recv0, recv1, recv2, recv3]
        for r in range(rep):
            sample_once_p(lg, rec0, rec1, rec2, rec3, ids, cnt + i64(r) * 64, t, b, sm, V, recv, rank, tagbase + r * 8,
                          sp, qlog, dids, prow, outtok, verdict, anc)

    @flyc.jit
    def launch(lg: Int64, rec0: Int64, rec1: Int64, rec2: Int64, rec3: Int64, ids: Int64, cnt: Int64,
               recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64, sp: Int64, qlog: Int64, dids: Int64, prow: Int64,
               outtok: Int64, verdict: Int64, anc: Int64, tagbase: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        samp(lg, rec0, rec1, rec2, rec3, ids, cnt, recv0, recv1, recv2, recv3, sp, qlog, dids, prow, outtok, verdict, anc,
             tagbase).launch(grid=(256,), block=(256,), smem=ZOFF + 2 * ((((V + CPR - 1) // CPR) + 255) // 256) * 1024, stream=stream)

    return launch
