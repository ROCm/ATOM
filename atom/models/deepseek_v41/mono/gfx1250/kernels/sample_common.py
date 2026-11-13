# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The 4-rank top-p sampling of the step tail (vocab-sharded logits, one GPU a rank), and the helpers sample / markov
share. The per-column passes are runtime loops re-reading the logits (L2), not unrolled register copies (the tail runs
once a step: cold instruction fetch of unrolled code is slow in the persistent kernel).
Every pass: per-CTA record -> in-rank device handoff -> rank-row record (5 CTAs, fixed order) -> tagged 16-byte pushes to the
3 peers (row scalar from CTA 0, histogram units split over the row's 5 CTAs) -> poll -> combine in rank order (identical
bits on every rank). Noise and ids use the global vocab index rank * V + j, so every rank draws the same winner.

Temperature + top-p sampling without the full logits:

logits[48][V] fp32. 5 CTAs a row (240 of 256 CTAs), each CTA a contiguous column range. Passes, device handoff between:
 P0  per CTA online (max, sum exp) of z = l / T                        -> rec0[row][cta]  (2 f32)
 P1  global (M, Z); 256-bin mass histogram of z in [M - R, M)         -> rec1[row][cta][256]
 P2  crossing bin b1 of the top-p mass p*Z (suffix scan); 256 sub-bins -> rec2[row][cta][256]
 P3  crossing sub-bin b2 -> threshold th; Gumbel-max of z + g over z >= th, g from a counter hash of (seed, row, id)
                                                                       -> rec3[row][cta] (u64 ordered key | ~id)
 P4  row winner -> ids[row]
Tokens below M - R (e^-R relative probability) are only kept when the whole range holds less than p (th = -inf).
The final sub-bin (width R / 65536 in z) is kept whole: a deviation of < R/65536 in the top-p cut, within tolerance."""
from __future__ import annotations


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64
from flydsl.compiler.ast_rewriter import ASTRewriter

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels.device import _v, gptr, lptr

ROWS, CPR, NB = D.T_MAX, 5, 256   # ROWS: the exchange layout's rows (the widest step); CPR: CTAs a row up to 48 rows
R = 16.0
F32, I32, I64 = fx.Float32, fx.Int32, fx.Int64
LN2 = 0.6931471805599453


def c32(v):
    return I32(v - (1 << 32) if v >= 1 << 31 else v)


def i64(x):
    return fx.Int64(x)


def log2f(x):
    return F32(hx.intrin(F32.ir_type, "llvm.log2.f32", [x]))


def exp2f(x):
    return F32(hx.intrin(F32.ir_type, "llvm.exp2.f32", [x]))


def expf(x):
    return exp2f(x * F32(1.4426950408889634))


def lshr(x, k):
    return (x >> k) & c32((1 << (32 - k)) - 1)


def I64MIN():
    return fx.Int64(-(1 << 63))


def hash32(row, idx, seed):
    h = idx * c32(0x9E3779B1) ^ (row * c32(0x85EBCA77)) ^ seed
    h = h ^ lshr(h, 16)
    h = h * c32(0x7FEB352D)
    h = h ^ lshr(h, 15)
    h = h * c32(0x846CA68B)
    h = h ^ lshr(h, 16)
    return h


def skey(f):
    """f32 -> i32 with the same (signed) order."""
    b = f.bitcast(I32)
    return (b < 0).select(b ^ I32(0x7FFFFFFF), b)


def latomic_max64(off, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.max, lptr(off), _v(I64(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup", alignment=8)


def handoff(cnt, t, nct, site=2):
    """A fenced device handoff of nct CTAs (device.handoff)."""
    hx.handoff(cnt, t, False, nct, site)


@ASTRewriter.transform
def suffix_scan(sm, t):
    """In-place inclusive suffix sum of the 256 f32 at LDS sm (thread t owns entry t)."""
    for k in range_constexpr(8):
        d = 1 << k
        v = hx.lload(sm + t * 4, F32)
        o = F32(0.0)
        if t + d < NB:
            o = hx.lload(sm + (t + d) * 4, F32)
        gpu.barrier()
        hx.lstore(sm + t * 4, v + o)
        gpu.barrier()


@ASTRewriter.transform
def crossing(sm, t, target):
    """sm holds a suffix-summed histogram; returns (b, rem): the largest bin b with suffix(b) >= target (-1 if none) and
    the mass still needed inside it, target - suffix(b + 1). Every thread gets the result (via LDS word sm + 1024)."""
    if t == 0:
        hx.lstore(sm + 1024, I32(-1))
    gpu.barrier()
    sv = hx.lload(sm + t * 4, F32)
    sn = F32(0.0)
    if t + 1 < NB:
        sn = hx.lload(sm + (t + 1) * 4, F32)
    if (sv >= target) & (sn < target):
        hx.lstore(sm + 1024, t)
        hx.lstore(sm + 1028, target - sn)
    gpu.barrier()
    bsel = hx.lload(sm + 1024, I32)
    rem = hx.lload(sm + 1028, F32)
    gpu.barrier()
    return bsel, rem


UNITS = 96          # 16-byte units a (pass, src, row) slot: 86 histogram units of 3 bins, or 1 scalar unit
HB = 4096           # LDS: per-source records [src][256] f32 (16 KB) / scalars [src] 8 B


def roff(p, src, row, u):
    return (i64(p * 4 + src) * ROWS + i64(row)) * (UNITS * 16) + i64(u) * 16


def zload(base, c0, CH, V, t, k, invT):
    j = c0 + t + k * 256
    jc = fx.min(j, I32(V - 1))
    z = hx.gload(base + i64(jc) * 4, F32) * invT
    return ((j < c0 + CH) & (j < V)).select(z, F32(-1e30))


@ASTRewriter.transform
def poll_unit(addr, tag, site=3):
    t0 = hx.clock()
    while (hx.gload(addr + i64(12), I32, nt=True) != tag) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
        hx.s_sleep(0)
    hx.note_timeout(t0, site)
    return hx.gload(addr, I32, 4, nt=True)


@ASTRewriter.transform
def xchg_scalar(recv, rank, p, row, c, t, sm, w0, w1, tag):
    """(w0, w1) i32 words of this rank's row record -> LDS HB + src * 8 for src 0..3 (own written directly)."""
    if (c == 0) & (t == 0):
        v = fx.Vector.from_elements([w0, w1, I32(0), I32(tag)], I32)
        for o in range_constexpr(4):
            if fx.const_expr(o != rank):
                hx.gstore(recv[o] + roff(p, rank, row, 0), v, nt=True)
    if t == rank:
        hx.lstore(sm + HB + t * 8, w0)
        hx.lstore(sm + HB + t * 8 + 4, w1)
    if (t < 4) & (t != rank):
        got = poll_unit(recv[rank] + roff(p, t, row, 0), tag)
        hx.lstore(sm + HB + t * 8, got[0])
        hx.lstore(sm + HB + t * 8 + 4, got[1])
    gpu.barrier()


@ASTRewriter.transform
def xchg_hist(recv, rank, p, row, c, t, sm, tag, mid=None, cpr=CPR):
    """This rank's row histogram at LDS sm[0:256] -> global sum over ranks (rank order) back into sm[0:256] (the 86
    pushed units split over the row's cpr CTAs)."""
    if (t < 86) & (t % cpr == c):
        vs = []
        for k in range_constexpr(3):
            bi = fx.min(t * 3 + k, I32(NB - 1))
            vv = hx.lload(sm + bi * 4, F32)
            vs.append((t * 3 + k < NB).select(vv, F32(0.0)).bitcast(I32))
        v = fx.Vector.from_elements([vs[0], vs[1], vs[2], I32(tag)], I32)
        for o in range_constexpr(4):
            if fx.const_expr(o != rank):
                hx.gstore(recv[o] + roff(p, rank, row, t), v, nt=True)
    hx.lstore(sm + HB + (rank * NB + t) * 4, hx.lload(sm + t * 4, F32))
    if fx.const_expr(mid is not None):   # other work (all threads) while the peers' histograms travel
        mid()
    for it in range_constexpr(2):
        idx = t + it * 256
        src = idx // 86
        u = idx % 86
        if (idx < 4 * 86) & (src != rank):
            got = poll_unit(recv[rank] + roff(p, src, row, u), tag)
            for k in range_constexpr(3):
                if u * 3 + k < NB:
                    hx.lstore(sm + HB + (src * NB + u * 3 + k) * 4, got[k])
    gpu.barrier()
    tot = hx.lload(sm + HB + t * 4, F32)
    for o in range_constexpr(1, 4):
        tot = tot + hx.lload(sm + HB + (o * NB + t) * 4, F32)
    gpu.barrier()
    hx.lstore(sm + t * 4, tot)
    gpu.barrier()


def _row_params(sp, row, invT, topp, seed):
    """Sampling semantics: sp = None -> the given constants (no greedy); else row's 16 B of sp: invT f32, topp f32,
    seed i32, greedy i32 (!= 0: no noise, no top-p cut -> argmax, lowest index on ties). A plain helper: the choice is
    made when the kernel is built."""
    if sp is None:
        return invT, topp, seed, None
    base = sp + i64(row) * 16
    return (hx.gload(base, F32), hx.gload(base + i64(4), F32), hx.gload(base + i64(8), I32),
            hx.gload(base + i64(12), I32) != I32(0))


def _greedy_noise(greedy, g):
    return g if greedy is None else greedy.select(F32(0.0), g)


def _greedy_th(greedy, th):
    return th if greedy is None else greedy.select(F32(-3e38), th)


@ASTRewriter.transform
def sample_once(lg, rec0, rec1, rec2, rec3, ids, cnt, t, b, sm, V, invT, topp, seed, recv, rank, tag0, sp=None):
    t = hx.opaque_v(t)
    b = hx.opaque_s(b)
    row = b // CPR
    c = b % CPR
    act = b < ROWS * CPR
    row = fx.min(row, I32(ROWS - 1))
    invT, topp, seed, greedy = _row_params(sp, row, invT, topp, seed)
    CH = (V + CPR - 1) // CPR
    PT = (CH + 255) // 256
    c0 = c * CH
    base = lg + i64(row) * (V * 4)
    NCT = 256
    # P0: (max, sum exp) of this CTA's columns
    m = F32(-1e30)
    s = F32(0.0)
    for k in range(PT):
        z = zload(base, c0, CH, V, t, k, invT)
        mn = fx.max(m, z)
        s = s * expf(m - mn) + expf(z - mn)
        m = mn
    hx.lstore(sm + t * 8, m)
    hx.lstore(sm + t * 8 + 4, s)
    gpu.barrier()
    for k in range_constexpr(8):
        d = 128 >> k
        if t < d:
            m1 = hx.lload(sm + t * 8, F32)
            s1 = hx.lload(sm + t * 8 + 4, F32)
            m2 = hx.lload(sm + (t + d) * 8, F32)
            s2 = hx.lload(sm + (t + d) * 8 + 4, F32)
            mn = fx.max(m1, m2)
            hx.lstore(sm + t * 8, mn)
            hx.lstore(sm + t * 8 + 4, s1 * expf(m1 - mn) + s2 * expf(m2 - mn))
        gpu.barrier()
    if act & (t < 2):
        hx.gstore(rec0 + i64(row * CPR + c) * 8 + i64(t) * 4, hx.lload(sm + t * 4, F32))
    handoff(cnt, t, NCT)
    # P1: rank-row (m, s) -> exchange -> global (M, Z) in rank order; histogram of z in [M - R, M)
    mr = F32(-1e30)
    for q in range_constexpr(CPR):
        mr = fx.max(mr, hx.gload(rec0 + i64(row * CPR + q) * 8, F32))
    sr = F32(0.0)
    for q in range_constexpr(CPR):
        sr = sr + hx.gload(rec0 + i64(row * CPR + q) * 8 + i64(4), F32) * expf(hx.gload(rec0 + i64(row * CPR + q) * 8, F32) - mr)
    xchg_scalar(recv, rank, 0, row, c, t, sm, mr.bitcast(I32), sr.bitcast(I32), tag0)
    M = F32(-1e30)
    for o in range_constexpr(4):
        M = fx.max(M, hx.lload(sm + HB + o * 8, F32))
    Z = F32(0.0)
    for o in range_constexpr(4):
        Z = Z + hx.lload(sm + HB + o * 8 + 4, F32) * expf(hx.lload(sm + HB + o * 8, F32) - M)
    gpu.barrier()
    hx.lstore(sm + t * 4, F32(0.0))
    gpu.barrier()
    lo1 = M - F32(R)
    sc1 = F32(NB / R)
    for k in range(PT):
        z = zload(base, c0, CH, V, t, k, invT)
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
    xchg_hist(recv, rank, 1, row, c, t, sm, tag0 + 1)
    suffix_scan(sm, t)
    b1, rem1 = crossing(sm, t, target)
    hx.lstore(sm + t * 4, F32(0.0))
    gpu.barrier()
    lo2 = lo1 + F32(b1) * F32(R / NB)
    sc2 = F32(NB * NB / R)
    for k in range(PT):
        z = zload(base, c0, CH, V, t, k, invT)
        if (b1 >= 0) & (z >= lo2):
            bi = fx.min(I32((z - lo2) * sc2), I32(NB - 1))   # z == M lands in the last sub-bin of the top bin
            if (z < lo2 + F32(R / NB)) | (b1 == NB - 1):
                hx.latomic_add_f32(sm + bi * 4, expf(z - M))
    gpu.barrier()
    if act:
        hx.gstore(rec2 + (i64(row * CPR + c) * NB + i64(t)) * 4, hx.lload(sm + t * 4, F32))
    handoff(cnt + i64(8), t, NCT)
    # P3: threshold, Gumbel-max over the kept set
    hs = F32(0.0)
    for q in range_constexpr(CPR):
        hs = hs + hx.gload(rec2 + (i64(row * CPR + q) * NB + i64(t)) * 4, F32)
    hx.lstore(sm + t * 4, hs)
    gpu.barrier()
    xchg_hist(recv, rank, 2, row, c, t, sm, tag0 + 2)
    suffix_scan(sm, t)
    b2, rem2 = crossing(sm, t, rem1)
    th = (b1 < 0).select(F32(-3e38), lo2 + F32(fx.max(b2, I32(0))) * F32(R / (NB * NB)))
    th = _greedy_th(greedy, th)
    if t == 0:
        hx.lstore(sm + 2048, I64MIN())
    gpu.barrier()
    best = I64MIN()
    for k in range(PT):
        z = zload(base, c0, CH, V, t, k, invT)
        j = c0 + t + k * 256
        u = (F32(lshr(hash32(row, j + rank * V, seed), 8)) + F32(0.5)) * F32(1.0 / 16777216.0)
        g = F32(-LN2) * log2f(F32(-LN2) * log2f(u))
        g = _greedy_noise(greedy, g)
        pk = fx.Int64(skey(z + g)) * fx.Int64(1 << 32) + (fx.Int64(c32(0xFFFFFFFF) - (j + rank * V)) & fx.Int64(0xFFFFFFFF))
        best = (z >= th).select(fx.max(best, pk), best)
    latomic_max64(sm + 2048, best)
    gpu.barrier()
    if act & (t == 0):
        hx.gstore(rec3 + i64(row * CPR + c) * 8, hx.lload(sm + 2048, fx.Int64))
        hx.gstore(rec3 + i64(ROWS * CPR * 8) + i64(row * CPR + c) * 4, th)
    handoff(cnt + i64(12), t, NCT)
    # P4: rank-row winner -> exchange -> global winner (every CTA 0 of a row, every rank)
    wb = I64MIN()
    for q in range_constexpr(CPR):
        wb = fx.max(wb, hx.gload(rec3 + i64(row * CPR + q) * 8, fx.Int64))
    xchg_scalar(recv, rank, 3, row, c, t, sm, I32(wb & fx.Int64(0xFFFFFFFF)), I32(wb >> 32), tag0 + 3)
    if act & (c == 0) & (t == 0):
        gb = I64MIN()
        for o in range_constexpr(4):
            gb = fx.max(gb, hx.lload(sm + HB + o * 8, fx.Int64))
        hx.gstore(ids + i64(row) * 4, c32(0xFFFFFFFF) - I32(gb & fx.Int64(0xFFFFFFFF)))


def build(V, rank, rep=1):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def samp(lg: Int64, rec0: Int64, rec1: Int64, rec2: Int64, rec3: Int64, ids: Int64, cnt: Int64,
             recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64,
             invT: fx.Float32, topp: fx.Float32, seed: fx.Int32, tagbase: fx.Int32):
        t = fx.thread_idx.x
        b = fx.block_idx.x
        sm = hx.lds_base()
        recv = [recv0, recv1, recv2, recv3]
        for r in range(rep):
            sample_once(lg, rec0, rec1, rec2, rec3, ids, cnt + i64(r) * 64, t, b, sm, V, invT, topp, seed, recv, rank,
                        tagbase + r * 4)

    @flyc.jit
    def launch(lg: Int64, rec0: Int64, rec1: Int64, rec2: Int64, rec3: Int64, ids: Int64, cnt: Int64,
               recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64,
               invT: fx.Float32, topp: fx.Float32, seed: fx.Int32, tagbase: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        samp(lg, rec0, rec1, rec2, rec3, ids, cnt, recv0, recv1, recv2, recv3, invT, topp, seed, tagbase).launch(
            grid=(256,), block=(256,), smem=HB + 4 * NB * 4, stream=stream)

    return launch
