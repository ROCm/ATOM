# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""Top-p sampling with the recipe's T > 0 verification (ATOM dspark_prob_sample: PROB_DRAFT + BLOCK_VERIFY) fused in.

Rows r = req * 6 + k (k = 0..4: the rows that verify draft x_k of the
request, k = 5: the bonus row), 5 CTAs a row, 4 ranks (vocab shards of VR columns), as sample_common.

sp [48][8] (16 + 16 B a row): invT f32, topp f32, seed i32, greedy i32 | invTd f32 (draft 1/T), hasq i32 (q of this request's
drafts kept by the kernel's prob drafter last step), qsrc i32 (its row in that step's qlog), seed_acc i32.
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
NEG = -3e38
ZOFF = 8192                  # LDS: this CTA's z = l / T chunk [PT][256] f32, then the q row's d * invTd chunk (read once in P0)


def ln(x):
    return F32(LN2) * log2f(x)


def unif(row, idx, seed):
    return (F32(lshr(hash32(row, idx, seed), 8)) + F32(0.5)) * F32(1.0 / 16777216.0)


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
    """6 i32 words of this rank's row record (2 tagged units of slot p) -> LDS HB + src * 24 for src 0..3."""
    if (c == 0) & (t == 0):
        v0 = fx.Vector.from_elements([ws[0], ws[1], ws[2], I32(tag)], I32)
        v1 = fx.Vector.from_elements([ws[3], ws[4], ws[5], I32(tag)], I32)
        for o in range_constexpr(4):
            if fx.const_expr(o != rank):
                hx.gstore(recv[o] + roff(p, rank, row, 0), v0, nt=True)
                hx.gstore(recv[o] + roff(p, rank, row, 1), v1, nt=True)
    if t == rank:
        for q in range_constexpr(6):
            hx.lstore(sm + HB + t * 24 + q * 4, ws[q])
    if fx.const_expr(mid is not None):   # other work (all threads) while the peers' words travel
        mid()
    if (t < 8) & ((t % 4) != rank):
        src = t % 4
        u = t // 4
        got = poll_unit(recv[rank] + roff(p, src, row, u), tag)
        for q in range_constexpr(3):
            hx.lstore(sm + HB + src * 24 + (u * 3 + q) * 4, got[q])
    gpu.barrier()


def _lw(sm, src, q, kind=F32):
    return hx.lload(sm + HB + src * 24 + q * 4, kind)


@ASTRewriter.transform
def sample_once_p(lg, rec0, rec1, rec2, rec3, ids, cnt, t, b, sm, V, recv, rank, tag0, sp, qlog, dids, prow, outtok,
                  verdict, anc, qids=None, ctxp=None):
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
    greedy = hx.gload(spb + i64(12), I32) != I32(0)
    invTd = hx.gload(spb + i64(16), F32)
    hasq = hx.gload(spb + i64(20), I32) != I32(0)
    qsrc = fx.max(hx.gload(spb + i64(24), I32), I32(0))
    seed_acc = hx.gload(spb + i64(28), I32)
    isd = k < ND                                   # a row that verifies draft x_k
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
    NCT = 256
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
    handoff(cnt, t, NCT)
    # P1: rank-row (m, s) and (mq, sq, dx) -> exchange -> global; histogram of z in [M - R, M)
    mr = F32(-1e30)
    for q in range_constexpr(CPR):
        mr = fx.max(mr, hx.gload(rec0 + i64(row * CPR + q) * 8, F32))
    sr = F32(0.0)
    for q in range_constexpr(CPR):
        sr = sr + hx.gload(rec0 + i64(row * CPR + q) * 8 + i64(4), F32) * expf(hx.gload(rec0 + i64(row * CPR + q) * 8, F32) - mr)
    qb = rec0 + i64(ROWS * CPR * 8) + i64(row * CPR) * 16
    mqr = F32(-1e30)
    dxr = F32(NEG)
    for q in range_constexpr(CPR):
        mqr = fx.max(mqr, hx.gload(qb + i64(q * 16), F32))
        dxr = fx.max(dxr, hx.gload(qb + i64(q * 16 + 8), F32))
    sqr = F32(0.0)
    for q in range_constexpr(CPR):
        sqr = sqr + hx.gload(qb + i64(q * 16 + 4), F32) * expf(hx.gload(qb + i64(q * 16), F32) - mqr)
    xchg6(recv, rank, 0, row, c, t, sm, [mr.bitcast(I32), sr.bitcast(I32), mqr.bitcast(I32), sqr.bitcast(I32),
                                          dxr.bitcast(I32), I32(0)], tag0, mid=_ctxmid(ctxp, 0))
    M = F32(-1e30)
    MQ = F32(-1e30)
    DX = F32(NEG)
    for o in range_constexpr(4):
        M = fx.max(M, _lw(sm, o, 0))
        MQ = fx.max(MQ, _lw(sm, o, 2))
        DX = fx.max(DX, _lw(sm, o, 4))
    Z = F32(0.0)
    ZQ = F32(0.0)
    for o in range_constexpr(4):
        Z = Z + _lw(sm, o, 1) * expf(_lw(sm, o, 0) - M)
        ZQ = ZQ + _lw(sm, o, 3) * expf(_lw(sm, o, 2) - MQ)
    qx = expf(DX - MQ) / ZQ
    gpu.barrier()
    hx.lstore(sm + t * 4, F32(0.0))
    gpu.barrier()
    lo1 = M - F32(R)
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
        u = (F32(lshr(hash32(row, j + rank * V, seed), 8)) + F32(0.5)) * F32(1.0 / 16777216.0)
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
    wb = I64MIN()
    znr = F32(0.0)
    zxr = F32(NEG)
    for q in range_constexpr(CPR):
        wb = fx.max(wb, hx.gload(rec3 + i64(row * CPR + q) * 8, fx.Int64))
        znr = znr + hx.gload(rec3 + i64(ROWS * CPR * 12) + i64(row * CPR + q) * 8, F32)
        zxr = fx.max(zxr, hx.gload(rec3 + i64(ROWS * CPR * 12) + i64(row * CPR + q) * 8 + i64(4), F32))
    xchg6(recv, rank, 3, row, c, t, sm, [I32(wb & fx.Int64(0xFFFFFFFF)), I32(wb >> 32), znr.bitcast(I32), zxr.bitcast(I32),
                                          I32(0), I32(0)], tag0 + 3, mid=_ctxmid(ctxp, 3))
    gb = I64MIN()
    ZN = F32(0.0)
    ZX = F32(NEG)
    for o in range_constexpr(4):
        gb = fx.max(gb, (fx.Int64(_lw(sm, o, 1, I32)) << 32) | (fx.Int64(_lw(sm, o, 0, I32)) & fx.Int64(0xFFFFFFFF)))
        ZN = ZN + _lw(sm, o, 2)
        ZX = fx.max(ZX, _lw(sm, o, 3))
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
        rmr = F32(0.0)
        brr = I64MIN()
        bpr = I64MIN()
        for q in range_constexpr(CPR):
            rq = rec1 + i64(row * CPR + q) * 32
            rmr = rmr + hx.gload(rq, F32)
            brr = fx.max(brr, hx.gload(rq + i64(8), fx.Int64))
            bpr = fx.max(bpr, hx.gload(rq + i64(16), fx.Int64))
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
        f = gq.select(fg, hq.select(tau, fo))
        # the token at f: greedy -> the target id; T > 0 -> the residual winner of row f, or the bonus sample at f = ND
        tf = tid[ND]
        for p in range_constexpr(ND):
            rt = hx.gload(prow + i64(r0 + p) * 32 + i64(12), I32)
            tf = (f == p).select(gq.select(tid[p], rt), tf)
        # a row whose logits are not finite has no p > 0 anywhere ->
        # no residual / p winner -> kid(INT64_MIN) = -1, which the draft entry then embeds (a page fault below emb):
        # fall back to the row's own sample, then to id 0, so a bad row costs a wrong token, not the GPU
        tff = tid[ND]
        for p in range_constexpr(ND):
            tff = (f == p).select(tid[p], tff)
        tf = (tf < I32(0)).select(fx.max(tff, I32(0)), tf)
        for p in range_constexpr(ND):
            hx.gstore(outtok + i64(rr * (ND + 1) + p) * 4, (I32(p) < f).select(xs[p], (I32(p) == f).select(tf, I32(-1))))
        hx.gstore(outtok + i64(rr * (ND + 1) + ND) * 4, (f == ND).select(tf, I32(-1)))
        hx.gstore(verdict + i64(rr) * 4, f)
        hx.gstore(verdict + i64(NREQ + rr) * 4, I32(ND) - f)
        hx.gstore(verdict + i64(2 * NREQ + rr) * 4, r0 + f)
        hx.gstore(anc + i64(rr) * 4, tf)


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
