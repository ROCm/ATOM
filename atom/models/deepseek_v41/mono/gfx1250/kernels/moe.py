# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The routed MoE as device functions in the attention style: two device-wide phases over all 256 CTAs (faster than one
16-CTA cluster an expert).

  union  every CTA builds the step's expert union and each expert's pick list (LDS atomics for the slots, then a
         16-wide sorting network per expert so every CTA gets the same slot order); experts with > 16 picks get overflow
         chunks; the shared expert joins as ceil(rows / 16) pseudo-experts (last in the union).
  UG     routed work unit = (expert, 32-col group g of 18) = 4 tiles (8 gate + 8 up rows each, one MXFP8 block of MID) over
         the full K = 5120 (40 steps of 128), one wave, FP4 x MXFP8 WMMA, register rings in fixed slots. The FP8 shared
         expert's units run on the last (dedicated) CTAs, K split over the 8 waves. The wave finishes SwiGLU (clamp 10),
         bf16, MXFP8 (code_ceil) of its block and writes fp8 + code to MID scratch.
  --     device-wide handoff (MID scratch)
  DN     work unit = (expert, 16-col output tile of 320): 5 K steps (K 576 padded to 640); the CTA stages the MID of its
         (<= 3) experts in LDS; route weight applied; per (row, pick) fp32 partial to global. The shared expert's down:
         on the dedicated CTAs (< 3 token chunks) or on all CTAs after the routed down (>= 3).

Routed expert weights: the production tensors as loaded (MXFP4, aiter shuffle_weight (16, 16): [E][N/16][K/32][2][16 rows]
[16 B]; w13 rows gate / up interleaved (GUGU), its N = 2 x 768 (576 padded); w2 K = 768 padded; E8M0 scales n32k4:
[E][N/32][K/128][32 rows][4 codes]). An A operand (16 rows x 128 K, unit (tile t, K step k)) is two 512-byte blocks and
32 x 4 codes (``lda``). The shared expert: FP8 E4M3 (32x32-block E8M0 scales) in its own plane c["shplane"] (UNIT8-byte
units: 18 UG work units x 160, then 320 DN tiles x 5).

`c` keys: t b w l sm (traced); pointers (Int64): w13 s13 w2 s2 shplane x8 xs ids wts out mid8 midc shcnt; python ints: NP (picks
used = rows x 6), nexp_max. Exchange stores are device-scope (hx.stx_dev, the fence-free handoff).
"""
from __future__ import annotations


import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels import moe_plan as MP
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import q8words, bf, e8m0_of, handoff, inv_pow2

H, I, TOPK, NE, CTAS = D.HID, D.MOE_I, D.TOPK, D.EXPERTS, D.CTAS
UGU, NUG, DNU, NDN = 160, 18, 5, 320
# the production routed-expert tensors (bytes): a 16-row tile, a 32-row scale super-row, an expert
IP = D.MOE_IPAD                             # I = 576 padded, as the production MoE stores its experts
W13_TS, W13_KC, W13_E, S13_E = 16 * H // 2, H // 128 * 128, 2 * IP * H // 2, 2 * IP * H // 32
W2_TS, W2_KC, W2_E, S2_E = 16 * IP // 2, IP // 128 * 128, H * IP // 2, H * IP // 32
MAXTOK, MROW, CROW, KMID = 16, 640, 32, 640
DUG = 16                                   # A units in flight (UG)
MSLOT = 3                                  # experts a CTA stages for the down
DN_PD = 2                                  # down tiles (x 5 A units) in flight ahead
F32, I32, I64 = fx.Float32, fx.Int32, fx.Int64
# LDS layout (bytes)
NOV = 16                                   # overflow chunks (virtual experts V_OV0..): <= 2 a routed expert (<= 48 picks)
V_OV0 = NE + 4
NEX = NE + 4 + NOV                         # + the shared expert as <= 3 pseudo-experts of <= 16 tokens (ids NE..), + overflow chunks
O_CNT = 0                                  # int[NEX]
O_PID = O_CNT + NEX * 4                    # int[288]
O_LST = O_PID + 288 * 4                    # int[NEX][16]
O_UNI = O_LST + NEX * MAXTOK * 4           # int[NEX]
O_NUNI = O_UNI + NEX * 4                   # int
O_MQ = O_NUNI + 16                         # fp8 [MSLOT][16][KMID]
O_MQS = O_MQ + MSLOT * MAXTOK * KMID       # int [MSLOT][16][5] (4 codes a K step)
O_WTS = O_MQS + MSLOT * MAXTOK * 5 * 4     # f32 [288 + 48] route weights (then 1.0 for the shared pseudo-picks) (LDS: a global load in the down loop would
                                           # wait, by the in-order loadcnt, for the prefetched weights)
O_VEXP = O_WTS + (288 + 48) * 4           # int[NOV]: an overflow chunk's routed expert
O_VBASE = O_VEXP + 16 * 4                  # int[NE]: the expert's first overflow chunk
O_CNT2 = O_VBASE + NE * 4                  # int[NE]: overflow pick counters
O_OVP = O_CNT2 + NE * 4                    # int[288]: the picks past 16
O_OVN = O_OVP + 288 * 4                    # int: their count
O_RED8 = O_OVN + 16                       # f32 [8 waves][4 tiles][8][32 lanes]: the FP8 shared UG unit's K-split partials
MOE_LDS = O_RED8 + 8 * 4 * 8 * 32 * 4

UNIT8 = 2176
SHARED_SLOT = 95
WIDE_NCH = MP.WIDE_NCH   # the wide schedule (moe_plan): the shared expert's UG early in the front's idle CTAs, its MID
                         # in the fixed last slots SHS0..
SHS0 = NE - WIDE_NCH                       # the shared MID slots (routed union entries < 288 + 16 overflow chunks < SHS0)


sh_early, sh8_upc, sh8_dnov = MP.sh_early, MP.sh8_upc, MP.sh8_dnov


def pslot(e, sm):
    """routed expert of (virtual) expert id e (the shared pseudo-experts -> SHARED_SLOT, an overflow chunk -> its expert)."""
    r = (e >= NE).select(I32(SHARED_SLOT), e)
    ov = hx.lload(sm + O_VEXP + fx.max(e - V_OV0, I32(0)) * 4, I32)
    return (e >= V_OV0).select(ov, r)


def prow(pk, NP, topk):
    """token row of a pick (routed: pk // 6; shared pseudo-pick: pk - NP)."""
    return (pk >= NP).select(pk - NP, pk // topk)


def _stx(addr, val):
    hx.stx_dev(addr, val)


def i64(x):
    return fx.Int64(x)


def ufl(x):
    return I32(hx.intrin(hx.i32(), "llvm.amdgcn.readfirstlane.i32", [I32(x)]))


def ballot(p):
    return I32(hx.intrin(hx.i32(), "llvm.amdgcn.ballot.i32", [p]))


def mbcnt(m):
    return I32(hx.intrin(hx.i32(), "llvm.amdgcn.mbcnt.lo", [m, I32(0)]))


def popc(m):
    return I32(hx.intrin(hx.i32(), "llvm.ctpop.i32", [m]))


def latomic_add(off, val):
    return I32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add, hx.lptr(off), hx._v(I32(val)), llvm.AtomicOrdering.monotonic,
                                syncscope="workgroup", alignment=4).result)


def sched_barrier():
    """__builtin_amdgcn_sched_barrier(0): the machine scheduler moves nothing across (keeps the register-ring loads
    issued where they are written; otherwise it sinks them next to their use and only ~1 unit stays in flight)."""
    hx.intrin(None, "llvm.amdgcn.sched.barrier", [I32(0)])


def wmma_fp4(a, sa, b, sb, c):
    """__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(4, a, 0, b, 0, c, 0, 0, sa, 0, 0, sb): A FP4 (8 dwords used of
    16), B FP8."""
    r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(hx.vty(8, F32.ir_type), hx._v(a), hx._v(b), hx._v(c), hx._v(I32(sa)),
                                              hx._v(I32(sb)), fmtA=4, fmtB=0)
    return fx.Vector(r, (8,), F32)


def wmma_fp8(a, sa, b, sb, c):
    """the scaled WMMA with A FP8 (all 16 dwords), B FP8."""
    r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(hx.vty(8, F32.ir_type), hx._v(a), hx._v(b), hx._v(c), hx._v(I32(sa)),
                                              hx._v(I32(sb)), fmtA=0, fmtB=0)
    return fx.Vector(r, (8,), F32)


def lda8(u, l):
    """FP8 A operand of the UNIT8 unit at byte address u: lane l's 64 B + its 4 scale codes."""
    p = u + i64(l * 64)
    return (hx.gload(p, I32, 4, nt=True), hx.gload(p + i64(16), I32, 4, nt=True), hx.gload(p + i64(32), I32, 4, nt=True),
            hx.gload(p + i64(48), I32, 4, nt=True), hx.gload(u + i64(2048) + i64(l * 4), I32, nt=True))


def avec8(o):
    return fx.Vector.from_elements([o[k][e] for k in range(4) for e in range(4)], I32)


def lane_offs(l, gugu):
    """Lane l's byte offsets in an A operand: (weights, scale codes). Lane l reads tile row r' (UG: lanes 0..7 the gate
    rows 0..7 = even rows, lanes 8..15 the up rows = odd rows; DN: row l % 16), K half l // 16."""
    r = l % 16
    rp = ((r < 8).select(2 * r, 2 * r - 15)) if gugu else r
    return i64(((l // 16) * 16 + rp) * 16), i64(rp * 4)


def lda(w, s, lo):
    """A operand: lane lo's 32 B of the unit's two 512-byte blocks at w, w + 512 (each a load of 4 cache lines) + its 4
    scale codes at s."""
    return (hx.gload(w + lo[0], I32, 4, nt=True), hx.gload(w + i64(512) + lo[0], I32, 4, nt=True),
            hx.gload(s + lo[1], I32, nt=True))


def ug_refill2(ra, wb, sb, lo, k1, kk, KB):
    """Ring slots kk - 1, kk refilled with K steps k1 - 1, k1, issued tile-major: a tile's two K steps are 2 KB
    contiguous (the 4 tiles of a K step are 40 KB apart in the production layout; one K step at a time was ~0.8 us a
    layer slower)."""
    na = {}
    for tile in range(4):
        for d in (1, 0):
            na[(kk - d) * 4 + tile] = lda(*ug_unit(wb, sb, k1 - d, tile), lo)
    return [na.get(i, ra[i]) for i in range(4 * KB)]


def ug_unit(wb, sb, ks, tile):
    """(weights, scales) of UG unit (K step ks, tile of 4) of the group whose first tile is at wb / sb."""
    return wb + i64(tile * W13_TS) + i64(ks) * 1024, sb + i64((tile // 2) * W13_KC + (tile % 2) * 64) + i64(ks) * 128


def avec(o):
    z = I32(0)
    return fx.Vector.from_elements([o[0][e] for e in range(4)] + [o[1][e] for e in range(4)] + [z] * 8, I32)


def ldb_g(row, srow, ks, l):
    """B operand (fp8 tokens, lane l -> token l%16, 16-byte chunks 2c + l/16 of the K step) from a global row."""
    p = row + i64(ks * 128) + i64((l // 16) * 16)
    parts = [hx.gload(p + i64(c * 32), I32, 4) for c in range(4)]
    return (fx.Vector.from_elements([parts[c][e] for c in range(4) for e in range(4)], I32),
            hx.gload(srow + i64(ks * 4), I32))


def ldb_l(row, ks, l):
    """The same from an LDS row (byte offset)."""
    p = row + ks * 128 + (l // 16) * 16
    parts = [hx.lload(p + c * 32, I32, 4, align=16) for c in range(4)]
    return fx.Vector.from_elements([parts[c][e] for c in range(4) for e in range(4)], I32)


def silu_clamp(g, u):
    gc = fx.min(g, F32(10.0))
    uc = fx.max(fx.min(u, F32(10.0)), F32(-10.0))
    e2 = F32(hx.intrin(F32.ir_type, "llvm.amdgcn.exp2.f32", [gc * F32(-1.4426950408889634)]))
    sig = F32(hx.intrin(F32.ir_type, "llvm.amdgcn.rcp.f32", [F32(1.0) + e2]))
    return bf(gc * sig * uc).to(F32)


def _sort_pairs(n):
    """Batcher odd-even merge sort network (compare-exchange index pairs) for n = 16."""
    pairs = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return pairs


SORT16 = _sort_pairs(16)
SORT8 = _sort_pairs(8)


@ASTRewriter.transform
def moe_union(c):
    """Expert union (S.uni, nuni) and each expert's pick list in pick order (S.lst), in this CTA's LDS."""
    t, w, l, sm, NP = c["t"], c["w"], c["l"], c["sm"], c["NP"]
    big = I32(0x7FFFFFFF)
    for k in range_constexpr(2):
        e = t + 256 * k
        if e < NEX:
            hx.lstore(sm + O_CNT + e * 4, I32(0))
            for q in range_constexpr(4):
                hx.lstore(sm + O_LST + (e * MAXTOK + 4 * q) * 4, fx.Vector.from_elements([big] * 4, I32), align=16)
        if e < NE:
            hx.lstore(sm + O_CNT2 + e * 4, I32(0))
    if t == 0:
        hx.lstore(sm + O_OVN, I32(0))
    for k in range_constexpr((NP + 255) // 256):
        p = t + 256 * k
        if p < NP:
            hx.lstore(sm + O_PID + p * 4, hx.gload(c["ids"] + i64(p) * 4, I32))
            hx.lstore(sm + O_WTS + p * 4, hx.gload(c["wts"] + i64(p) * 4, F32))
    gpu.barrier()
    for k in range_constexpr((NP + 255) // 256):
        p = t + 256 * k
        if p < NP:
            e = hx.lload(sm + O_PID + p * 4, I32)
            s = latomic_add(sm + O_CNT + e * 4, 1)
            if s < MAXTOK:
                hx.lstore(sm + O_LST + (e * MAXTOK + s) * 4, p)
            if s == MAXTOK:   # the 17th arrival: its expert overflows (listed once)
                hx.lstore(sm + O_OVP + latomic_add(sm + O_OVN, 1) * 4, e)
    gpu.barrier()
    # overflow chunks (hot experts: > 16 picks). The <= 16 overflowing experts (OVP[0 .. novf), arrival order) are put in
    # expert order by wave 0 (lane l holds one: its order = the listed experts with smaller ids; its chunks (1 or 2) and
    # chunk base = an exclusive prefix over them in that order). Then every overflowing expert's picks are re-placed by
    # their rank in pick order (the same in every CTA; the first 16 atomic arrivals differ by CTA): rank < 16 -> the
    # expert's list, else chunk rank // 16 - 1; the ranks from per-(round, wave, expert) ballot counts (O_OVP + 64).
    novf = hx.lload(sm + O_OVN, I32)
    if novf > 0:
        if w == 0:
            ok = l < novf
            e = ok.select(hx.lload(sm + O_OVP + fx.min(l, I32(15)) * 4, I32), I32(NE))
            cn = hx.lload(sm + O_CNT + fx.min(e, I32(NE - 1)) * 4, I32)
            nch = fx.min(fx.max(cn - 1, I32(0)) // MAXTOK, I32(2))
            r = I32(0)
            vb = I32(0)
            for q in range(novf):
                eq = hx.lload(sm + O_OVP + q * 4, I32)
                before = eq < e
                cq = hx.lload(sm + O_CNT + fx.min(eq, I32(NE - 1)) * 4, I32)
                r = r + before.select(I32(1), I32(0))
                vb = vb + before.select(fx.min(fx.max(cq - 1, I32(0)) // MAXTOK, I32(2)), I32(0))
            if ok:
                hx.lstore(sm + O_VBASE + e * 4, vb)
                hx.lstore(sm + O_CNT2 + e * 4, r)
                for j in range_constexpr(2):
                    if (j < nch) & (vb + j < NOV):
                        hx.lstore(sm + O_VEXP + (vb + j) * 4, e)
                        hx.lstore(sm + O_CNT + (V_OV0 + vb + j) * 4, fx.min(cn - MAXTOK * (j + 1), I32(MAXTOK)))
        gpu.barrier()
        NR = (NP + 255) // 256
        mine = []
        for k in range_constexpr(NR):
            p = t + 256 * k
            e = hx.lload(sm + O_PID + fx.min(p, I32(NP - 1)) * 4, I32)
            ov = (p < NP) & (hx.lload(sm + O_CNT + e * 4, I32) > MAXTOK)
            jj = ov.select(hx.lload(sm + O_CNT2 + e * 4, I32), I32(-1))
            rk = I32(0)
            for j in range(novf):
                m = ballot(jj == j)
                if l == 0:
                    hx.lstore(sm + O_OVP + 64 + ((k * 8 + w) * 16 + j) * 4, popc(m))
                rk = (jj == j).select(mbcnt(m), rk)
            mine.append((p, e, jj, rk))
        gpu.barrier()
        for k in range_constexpr(NR):
            p, e, jj, rk = mine[k]
            if jj >= 0:
                rank = rk
                for sl in range(k * 8 + w):   # the earlier (round, wave) slots
                    rank = rank + hx.lload(sm + O_OVP + 64 + (sl * 16 + jj) * 4, I32)
                inl = rank < MAXTOK
                vi = hx.lload(sm + O_VBASE + e * 4, I32) + rank // MAXTOK - 1
                dst = inl.select(e * MAXTOK + rank, (V_OV0 + vi) * MAXTOK + rank % MAXTOK)
                if inl | (vi < NOV):
                    hx.lstore(sm + O_LST + dst * 4, p)
        gpu.barrier()
    for k in range_constexpr(2):
        e = t + 256 * k
        # a wave's lists: no sort if every count <= 1, the 8-wide network if every count <= 8 (the slots past the
        # count hold the 'big' sentinel), else the 16-wide one
        ok = e < NEX
        ce = hx.lload(sm + O_CNT + fx.min(e, I32(NEX - 1)) * 4, I32)
        m2 = ballot(ok & (ce > 1))
        m9 = ballot(ok & (ce > 8))
        if ok & (m2 != 0):
            if m9 != 0:
                vs = [hx.lload(sm + O_LST + (e * MAXTOK + 4 * q) * 4, I32, 4, align=16) for q in range(4)]
                v = [vs[q][j] for q in range(4) for j in range(4)]
                for (i0, i1) in SORT16:
                    lo = fx.min(v[i0], v[i1])
                    hi = fx.max(v[i0], v[i1])
                    v[i0] = lo
                    v[i1] = hi
                for q in range_constexpr(4):
                    hx.lstore(sm + O_LST + (e * MAXTOK + 4 * q) * 4, fx.Vector.from_elements(v[4 * q:4 * q + 4], I32),
                              align=16)
            else:
                vs = [hx.lload(sm + O_LST + (e * MAXTOK + 4 * q) * 4, I32, 4, align=16) for q in range(2)]
                v = [vs[q][j] for q in range(2) for j in range(4)]
                for (i0, i1) in SORT8:
                    lo = fx.min(v[i0], v[i1])
                    hi = fx.max(v[i0], v[i1])
                    v[i0] = lo
                    v[i1] = hi
                for q in range_constexpr(2):
                    hx.lstore(sm + O_LST + (e * MAXTOK + 4 * q) * 4, fx.Vector.from_elements(v[4 * q:4 * q + 4], I32),
                              align=16)
    if w == 0:
        base = I32(0)
        cns = [hx.lload(sm + O_CNT + (cc * 32 + l) * 4, I32) for cc in range(NE // 32)]   # all issued first
        for cc in range_constexpr(NE // 32):
            e = cc * 32 + l
            has = cns[cc] > 0
            m = ballot(has)
            if has:
                hx.lstore(sm + O_UNI + (base + mbcnt(m)) * 4, e)
            base = base + popc(m)
        # overflow chunks (before the shared pseudo-experts: those stay last)
        ve = V_OV0 + (l % NOV)
        has = (l < NOV) & (hx.lload(sm + O_CNT + ve * 4, I32) > 0)
        m = ballot(has)
        if has:
            hx.lstore(sm + O_UNI + (base + mbcnt(m)) * 4, ve)
        base = base + popc(m)
        for k in range_constexpr((NP // c["topk"] + MAXTOK - 1) // MAXTOK):
            nk = min(MAXTOK, NP // c["topk"] - MAXTOK * k)
            if l == 0:
                hx.lstore(sm + O_UNI + (base + k) * 4, I32(NE + k))
                hx.lstore(sm + O_CNT + (NE + k) * 4, I32(nk))
            if l < nk:
                hx.lstore(sm + O_LST + ((NE + k) * MAXTOK + l) * 4, NP + MAXTOK * k + l)
                hx.lstore(sm + O_WTS + (NP + MAXTOK * k + l) * 4, F32(1.0))
        base = base + (NP // c["topk"] + MAXTOK - 1) // MAXTOK
        if l == 0:
            hx.lstore(sm + O_NUNI, base)
    gpu.barrier()


@ASTRewriter.transform
def moe_combine(c, TR, sm_out):
    """After the DN -> TP2 handoff: this CTA's TP2 slice (cols (b//4)*80 + (b%4)*20 .. +20, rows 0..TR-1) of the MoE output =
    sum over the row's 6 picks of out[pick] (already route-weighted) + the shared expert's row, as f32 [r][20]
    at sm_out (the layout tp_allreduce reads). One thread = 4 cols of a row (16-byte loads, all issued before the adds)."""
    t, b = c["t"], c["b"]
    NP = TR * c["topk"]
    if t < TR * 5:
        row = t // 5
        c4 = (t % 5) * 4
        col = (b // 4) * 80 + (b % 4) * 20 + c4
        vb = [hx.gload(c["out"] + (i64(row * c["topk"] + k) * H + i64(col)) * 2, fx.BFloat16, 4) for k in range(c["topk"])]
        vsh = hx.gload(c["out"] + (i64(NP + row) * H + i64(col)) * 4, F32, 4)   # the shared expert's row stays f32
        acc = [vb[0][j].to(F32) for j in range(4)]
        for k in range_constexpr(1, c["topk"]):
            acc = [acc[j] + vb[k][j].to(F32) for j in range(4)]
        acc = [acc[j] + vsh[j] for j in range(4)]
        hx.lstore(sm_out + (row * 20 + c4) * 4, fx.Vector.from_elements(acc, F32), align=16)
    gpu.barrier()


def ug_ring_init(wb, sb, lo, xrow, srow, l, KB):
    """The UG register rings, issued in the loop's order (per K step: 4 A units then the B) so the rolled loop's
    waits stay partial."""
    ra, rb = [], []
    ra = [None] * (4 * KB)
    for k0 in range(0, KB, 2):   # a tile's two K steps back to back (2 KB contiguous; see ug_refill2)
        for tile in range(4):
            for kk in (k0, k0 + 1):
                ra[kk * 4 + tile] = lda(*ug_unit(wb, sb, kk, tile), lo)
        rb = rb + [ldb_g(xrow, srow, k0, l), ldb_g(xrow, srow, k0 + 1, l)]
        sched_barrier()
    return ra, rb


@ASTRewriter.transform
def moe_ug(c):
    """UG: routed units (wave w: u0 + w, + 8, ...) on the first CR CTAs; the FP8 shared expert's NCH * 18 units on the
    last NDC (dedicated) CTAs, sh8_upc(NCH) a CTA; one epilogue copy."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    tl = l % 16
    nuni = ufl(hx.lload(sm + O_NUNI, I32))
    NCH = (c["NP"] // c["topk"] + MAXTOK - 1) // MAXTOK   # shared pseudo-experts (python int)
    SH8_UPC = sh8_upc(NCH)
    SH8_DNOV = sh8_dnov(NCH)
    NSHU = NCH * NUG   # FP8 shared units: the last NDC CTAs take SH8_UPC each (all 8 waves, K split)
    NDC = MP.dedicated_ctas(NCH, NUG)
    CR = CTAS - NDC
    U = (nuni - NCH) * NUG
    u0 = ufl(MP.block_lo(b, U, CR))
    u1 = ufl(MP.block_lo(b + 1, U, CR))
    u1 = (b < CR).select(u1, u0)
    wu = u0 + w
    ded = b >= CR
    if fx.const_expr(sh_early(NCH)):
        if b < NSHU:
            ug_shared_fin(c, b)
    STRIPE = MP.striped(NCH)
    nit = MP.unit_iters(STRIPE, U, CR, b, w, u1, wu)
    for it in range(ded.select(I32(SH8_UPC), fx.max(nit, I32(0)))):
        ui = I32(0)
        g = I32(0)
        n = I32(0)
        acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
        if ded:
            su0 = (b - CR) * SH8_UPC + it
            su = fx.min(su0, I32(NSHU - 1))
            ui = nuni - NCH + su // NUG
            g = su % NUG
            acc, n = ug_shared8(c, ui, su // NUG, g)
            n = (su0 < NSHU).select(n, I32(0))   # a dedicated CTA's slot past the shared units: no MID written
        else:
            wv = MP.unit_of(STRIPE, it, CR, b, w, wu)
            ui = wv // NUG
            g = wv % NUG
            e = hx.lload(sm + O_UNI + ui * 4, I32)
            n = fx.min(hx.lload(sm + O_CNT + e * 4, I32), I32(MAXTOK))
            pk = hx.lload(sm + O_LST + (e * MAXTOK + (tl < n).select(tl, I32(0))) * 4, I32)
            ex = i64(pslot(e, sm))
            acc = ug_body(c, pk, c["w13"] + ex * W13_E + i64(g) * (4 * W13_TS), c["s13"] + ex * S13_E + i64(g) * (2 * W13_KC))
        ug_epilogue(c, ui, g, n, acc)
    if fx.const_expr(SH8_DNOV):
        # the dedicated CTAs: once all NSHU shared units' MID is out (a counter among them), the shared down
        if ded:
            hx.wait_storecnt0()
            gpu.barrier()
            if t == 0:
                hx.atomic_add(c["shcnt"], 1)
                hx.wait_count(c["shcnt"], NDC)
            gpu.barrier()
            cd = {k: c[k] for k in c if k != "shcnt"}   # a plain dict (LazyC entries evaluated here)
            cd.update(dn8_bi=b - CR, dn8_nct=NDC)
            moe_dn_shared(cd)


@ASTRewriter.transform
def ug_epilogue(c, ui, g, n, acc):
    """UG unit epilogue: silu(gate) * up -> MXFP8 MID row (token l) of scratch slot ui, cols 32 g .. 32 g + 31."""
    l = c["l"]
    # epilogue: gate rows 0..7 in lanes 0..15 (token l), up rows in lanes 16..31 (token l - 16)
    v = []
    am = F32(0.0)
    for tile in range_constexpr(4):
        for i in range_constexpr(8):
            gv = acc[tile][i]
            uv = gv.shuffle_xor(16, 32)
            x = silu_clamp(gv, uv)
            v.append(x)
            am = fx.max(am, abs(x))
    if l < n:
        code = e8m0_of(am)
        inv = inv_pow2(code)
        words = q8words([v[j] for j in range(32)], inv)
        mrow = c["mid8"] + (i64(ui) * MAXTOK + i64(l)) * MROW
        _stx(mrow + i64(g * 32), fx.Vector.from_elements(words[:4], I32))
        _stx(mrow + i64(g * 32 + 16), fx.Vector.from_elements(words[4:], I32))
        crow = c["midc"] + (i64(ui) * MAXTOK + i64(l)) * CROW
        _stx(crow + i64(g), code.to(fx.Int8))
        if g == NUG - 1:   # the K pad of the down: cols 576..639 zero, codes 18, 19 = 1
            zv = fx.Vector.from_elements([I32(0)] * 4, I32)
            for k in range_constexpr(4):
                _stx(mrow + i64(I + 16 * k), zv)
            _stx(crow + i64(18), I32(0x0101).to(fx.Int16))


@ASTRewriter.transform
def ug_shared8(c, ui, k, g):
    """FP8 shared UG unit g of token chunk k on this (dedicated) CTA: wave w does K steps 5w .. 5w + 4 of the 4 tiles,
    the 8 partials are summed in wave order through LDS, wave 0 runs the epilogue."""
    w, l, sm = c["w"], c["l"], c["sm"]
    tl = l % 16
    # the shared pseudo-expert's picks are NP + 16 k + 0 .. n - 1 by construction (moe_union), so computed here, not
    # read from the union's LDS list (a served 4-request draft layer once read stale 0x7FFFFFFF slots there -> a
    # fault; the writer of that LDS state is not found)
    rows_ = c["NP"] // c["topk"]
    n = fx.min(I32(rows_) - MAXTOK * k, I32(MAXTOK))
    pk = c["NP"] + MAXTOK * k + (tl < n).select(tl, I32(0))
    row = prow(pk, c["NP"], c["topk"])
    xrow = c["x8"] + i64(row) * H
    srow = c["xs"] + i64(row) * (H // 32)
    ab = c["shplane"] + i64(g * UGU) * UNIT8
    acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
    ks0 = w * 5
    # a runtime loop over the wave's 5 K steps (layer-loop code size), the next step's A / B issued first
    ra = [lda8(ab + (i64(ks0) * 4 + tile) * UNIT8, l) for tile in range(4)]
    rb = ldb_g(xrow, srow, ks0, l)
    for kk in range(5):
        nk = ks0 + fx.min(kk + 1, I32(4))
        na = [lda8(ab + (i64(nk) * 4 + tile) * UNIT8, l) for tile in range(4)]
        nb = ldb_g(xrow, srow, nk, l)
        acc = [wmma_fp8(avec8(ra[tile]), ra[tile][4], rb[0], rb[1], acc[tile]) for tile in range(4)]
        ra = na
        rb = nb
    # partials [w][tile][lane][8] as 2 x 16 B; wave 0 sums them in wave order (a runtime loop over the waves)
    gpu.barrier()   # the previous unit's sum (wave 0) has read the partials
    for tile in range_constexpr(4):
        ow = sm + O_RED8 + ((w * 4 + tile) * 32 + l) * 32
        hx.lstore(ow, fx.Vector.from_elements([acc[tile][i] for i in range(4)], F32), align=16)
        hx.lstore(ow + 16, fx.Vector.from_elements([acc[tile][i] for i in range(4, 8)], F32), align=16)
    gpu.barrier()
    tot = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
    for ww in range(8):
        for tile in range_constexpr(4):
            orr = sm + O_RED8 + ((ww * 4 + tile) * 32 + l) * 32
            lo = hx.lload(orr, F32, 4, align=16)
            hi = hx.lload(orr + 16, F32, 4, align=16)
            tot[tile] = fx.Vector.from_elements([tot[tile][i] + lo[i] for i in range(4)] + [tot[tile][i + 4] + hi[i] for i in range(4)], F32)
    return tot, (w == 0).select(n, I32(0))


SHE_CTAS = 2   # CTAs a shared UG unit in the front's idle window (K split: one CTA's 348 KB unit is too slow)


@ASTRewriter.transform
def ug_shared_early(c, bi):
    """K-slice bi % SHE_CTAS of the FP8 shared expert's UG unit bi // SHE_CTAS on an idle CTA of the FFN front: wave w
    does the K steps of slot (bi % SHE_CTAS) * 8 + w of 40 (2..3), the 8 partials summed through LDS, the CTA's partial
    (4 tiles x 32 lanes x 8 f32) to c["out"] past the bf16 pick rows (the free upper half of the f32-sized pick area:
    over the pick rows themselves, a token's outputs read the partials whenever the down did not rewrite them) -- moe_ug sums the slices and runs the
    epilogue (ug_shared_fin). c: t w l sm NP x8 xs shplane out."""
    w, l, sm = c["w"], c["l"], c["sm"]
    tl = l % 16
    su = bi // SHE_CTAS
    k = su // NUG
    g = su % NUG
    rows_ = c["NP"] // c["topk"]
    n = fx.min(I32(rows_) - MAXTOK * k, I32(MAXTOK))
    row = prow(c["NP"] + MAXTOK * k + (tl < n).select(tl, I32(0)), c["NP"], c["topk"])
    xrow = c["x8"] + i64(row) * H
    srow = c["xs"] + i64(row) * (H // 32)
    ab = c["shplane"] + i64(g * UGU) * UNIT8
    gw = (bi % SHE_CTAS) * 8 + w
    NW = SHE_CTAS * 8
    k0 = (gw * 40) // NW
    k1 = ((gw + 1) * 40) // NW
    MAXS = (40 + NW - 1) // NW
    ra = [[lda8(ab + (i64(fx.min(k0 + j, I32(39))) * 4 + tile) * UNIT8, l) for tile in range(4)] for j in range(MAXS)]
    rb = [ldb_g(xrow, srow, fx.min(k0 + j, I32(39)), l) for j in range(MAXS)]
    acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
    for j in range_constexpr(MAXS):
        nacc = [wmma_fp8(avec8(ra[j][tile]), ra[j][tile][4], rb[j][0], rb[j][1], acc[tile]) for tile in range(4)]
        ok = k0 + j < k1
        acc = [fx.Vector.from_elements([ok.select(nacc[tile][i], acc[tile][i]) for i in range(8)], F32) for tile in range(4)]
    for tile in range_constexpr(4):
        ow = sm + O_RED8 + ((w * 4 + tile) * 32 + l) * 32
        hx.lstore(ow, fx.Vector.from_elements([acc[tile][i] for i in range(4)], F32), align=16)
        hx.lstore(ow + 16, fx.Vector.from_elements([acc[tile][i] for i in range(4, 8)], F32), align=16)
    gpu.barrier()
    if w == 0:
        for tile in range_constexpr(4):
            tot = fx.Vector.filled(8, 0.0, F32)
            for ww in range(8):
                orr = sm + O_RED8 + ((ww * 4 + tile) * 32 + l) * 32
                lo = hx.lload(orr, F32, 4, align=16)
                hi = hx.lload(orr + 16, F32, 4, align=16)
                tot = fx.Vector.from_elements([tot[i] + lo[i] for i in range(4)] + [tot[i + 4] + hi[i] for i in range(4)], F32)
            po = c["out"] + i64(c["NP"] * H * 2) + i64((((bi * 4) + tile) * 32 + l) * 32)
            _stx(po, fx.Vector.from_elements([tot[i] for i in range(4)], F32))
            _stx(po + i64(16), fx.Vector.from_elements([tot[i] for i in range(4, 8)], F32))


@ASTRewriter.transform
def ug_shared_fin(c, su):
    """The shared UG unit su's epilogue (wave 0): the SHE_CTAS K-slice partials (from ug_shared_early, in c["out"])
    summed, silu * up -> MXFP8 MID slot SHS0 + chunk."""
    w, l = c["w"], c["l"]
    k = su // NUG
    g = su % NUG
    n = fx.min(I32(c["NP"] // c["topk"]) - MAXTOK * k, I32(MAXTOK))
    tot = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
    for h in range_constexpr(SHE_CTAS):
        for tile in range_constexpr(4):
            po = c["out"] + i64(c["NP"] * H * 2) + i64(((((su * SHE_CTAS + h) * 4) + tile) * 32 + l) * 32)
            lo = hx.gload(po, F32, 4)
            hi = hx.gload(po + i64(16), F32, 4)
            tot[tile] = fx.Vector.from_elements([tot[tile][i] + lo[i] for i in range(4)] + [tot[tile][i + 4] + hi[i] for i in range(4)], F32)
    ug_epilogue(c, I32(SHS0) + k, g, (w == 0).select(n, I32(0)), tot)


@ASTRewriter.transform
def ug_body(c, pk, wb, sb):
    """One UG work unit (32 MID cols: 4 tiles x 40 K steps) of the expert, its first tile's weights / scales at wb / sb,
    for the tokens of pick list pk (lane tl's pick); returns the 4 tiles' accumulators."""
    l = c["l"]
    lo = lane_offs(l, True)
    row = prow(pk, c["NP"], c["topk"])
    xrow = c["x8"] + i64(row) * H
    srow = c["xs"] + i64(row) * (H // 32)
    KB = DUG // 4
    ra, rb = ug_ring_init(wb, sb, lo, xrow, srow, l, KB)
    # rolled over the 40 K steps in blocks of KB, fixed ring slots (slot kk*4+tile / kk refilled with K step
    # ks + KB right after K step ks used it): ~1/10 of the unrolled code, and no ring shifting through
    # loop-carried registers (that made the compiler wait for every load at the loop top)
    acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
    for j in range(40 // KB - 1):
        for kk in range_constexpr(KB):
            ks = j * KB + kk
            bq = rb[kk]
            acc = [wmma_fp4(avec(ra[kk * 4 + tile]), ra[kk * 4 + tile][2], bq[0], bq[1], acc[tile]) for tile in range(4)]
            if kk % 2 == 1:   # slots kk - 1, kk: K steps ks + KB - 1, ks + KB
                ra = ug_refill2(ra, wb, sb, lo, ks + KB, kk, KB)
                rb = rb[:kk - 1] + [ldb_g(xrow, srow, ks + KB - 1, l), ldb_g(xrow, srow, ks + KB, l)] + rb[kk + 1:]
            sched_barrier()
    for kk in range_constexpr(KB):
        bq = rb[kk]
        acc = [wmma_fp4(avec(ra[kk * 4 + tile]), ra[kk * 4 + tile][2], bq[0], bq[1], acc[tile]) for tile in range(4)]
    return acc


def dn_nuni(c, nuni):
    """Union entries in the routed down ranges (the shared pseudo-experts, last, go to moe_dn_shared)."""
    return nuni - (c["NP"] // c["topk"] + MAXTOK - 1) // MAXTOK


@ASTRewriter.transform
def moe_dn_stage(c):
    """Stage the MID (fp8 + codes) of this CTA's down experts (<= 3) in LDS: every load issued first (branch-free; the
    rows past an expert's pick count are zeroed by a mask)."""
    t, b, sm = c["t"], c["b"], c["sm"]
    nuni = ufl(hx.lload(sm + O_NUNI, I32))
    Ud = dn_nuni(c, nuni) * NDN
    d0 = ufl((b * Ud) // CTAS)
    x0 = d0 // NDN
    NCH = MAXTOK * KMID // 16   # 640 16-byte chunks a slot
    pend = []
    for sl in range_constexpr(MSLOT):
        ui = fx.min(x0 + sl, nuni - 1)
        e = hx.lload(sm + O_UNI + ui * 4, I32)
        n = fx.min(hx.lload(sm + O_CNT + e * 4, I32), I32(MAXTOK))
        for k in range_constexpr((NCH + 255) // 256):
            i = fx.min(t + 256 * k, I32(NCH - 1))
            tok = i // (KMID // 16)
            cc = i % (KMID // 16)
            v = hx.gload(c["mid8"] + (i64(ui) * MAXTOK + i64(tok)) * MROW + i64(cc * 16), I32, 4)
            m = (tok < n).select(I32(-1), I32(0))
            pend.append((sm + O_MQ + (sl * MAXTOK + tok) * KMID + cc * 16, v, m))
        tok = fx.min(t, I32(MAXTOK * 5 - 1)) // 5
        cc = fx.min(t, I32(MAXTOK * 5 - 1)) % 5
        cv = hx.gload(c["midc"] + (i64(ui) * MAXTOK + i64(tok)) * CROW + i64(cc * 4), I32)
        pend.append((sm + O_MQS + ((sl * MAXTOK + tok) * 5 + cc) * 4, cv, (tok < n)))
    for (dst, v, m) in pend:
        if fx.const_expr(isinstance(v, fx.Vector)):
            hx.lstore(dst, fx.Vector.from_elements([v[j] & m for j in range(4)], I32), align=16)
        else:
            hx.lstore(dst, m.select(v, I32(0x01010101)))
    gpu.barrier()


@ASTRewriter.transform
def moe_dn(c):
    """This wave's down tiles: a contiguous eighth [wt0, wt1) of the CTA's range (<= 2 experts), split at its (<= 1)
    expert boundary into segments. B (the MID of the segment's expert, 5 K steps) is held in registers: re-reading it
    from LDS for every tile made the down LDS-bandwidth bound; one B set a segment (85 VGPRs, not 2 + a
    16-dword select a K step), so more down tiles can be in flight (the down is in-flight bound). Per
    (row, pick) partials to out."""
    w, l, b, sm = c["w"], c["l"], c["b"], c["sm"]
    tl = l % 16
    lo = lane_offs(l, False)
    nuni = ufl(hx.lload(sm + O_NUNI, I32))
    Ud = dn_nuni(c, nuni) * NDN
    d0 = ufl((b * Ud) // CTAS)
    d1 = ufl(((b + 1) * Ud) // CTAS)
    x0 = d0 // NDN
    wt0 = d0 + (w * (d1 - d0)) // 8
    wt1 = d0 + ((w + 1) * (d1 - d0)) // 8
    last = fx.max(wt1 - 1, wt0)
    uiA = fx.min(wt0, d1 - 1) // NDN
    uiB = fx.min(last, d1 - 1) // NDN

    def bset(ui):
        sl = ui - x0
        mrow = sm + O_MQ + (sl * MAXTOK + tl) * KMID
        return [(ldb_l(mrow, ks, l), hx.lload(sm + O_MQS + ((sl * MAXTOK + tl) * 5 + ks) * 4, I32)) for ks in range(DNU)]

    def loads(inf):
        tile, ui, tt, e = inf
        ex = i64(pslot(e, sm))
        wb = c["w2"] + ex * W2_E + i64(tt) * W2_TS
        sb = c["s2"] + ex * S2_E + i64(tt // 2) * W2_KC + i64(tt % 2) * 64
        return [lda(wb + i64(ks * 1024), sb + i64(ks * 128), lo) for ks in range(DNU)]

    PD = DN_PD
    bnd = (uiA + 1) * NDN
    nseg = (uiB != uiA).select(I32(2), I32(1))
    for s in range(nseg):
        ts = (s == 0).select(wt0, fx.max(bnd, wt0))
        te = (s == 0).select(fx.min(wt1, bnd), wt1)
        B = bset((s == 0).select(uiA, uiB))

        def stinfo(j):
            tile = ts + j
            tc = fx.min(fx.min(tile, fx.max(te - 1, ts)), d1 - 1)
            ui = tc // NDN
            tt = tc % NDN
            e = hx.lload(sm + O_UNI + ui * 4, I32)
            return tile, ui, tt, e

        infos = []
        ops = []
        for k in range_constexpr(PD):
            infos.append(stinfo(I32(k)))
            ops.append(loads(infos[k]))
            sched_barrier()
        # PD fixed register slots (loop unrolled by PD): slot k computes tile j*PD + k, then is refilled with
        # tile (j+1)*PD + k -- no ring shift through loop-carried registers (that waited for the next tile too)
        for j in range((te - ts + PD - 1) // PD):
            for k in range_constexpr(PD):
                tile, ui, tt, e = infos[k]
                n = fx.min(hx.lload(sm + O_CNT + e * 4, I32), I32(MAXTOK))
                acc = fx.Vector.filled(8, 0.0, F32)
                for ks in range_constexpr(DNU):
                    bv, sb = B[ks]
                    aq = ops[k][ks]
                    acc = wmma_fp4(avec(aq), aq[2], bv, sb, acc)
                ni = stinfo(j * PD + k + PD)
                infos = infos[:k] + [ni] + infos[k + 1:]
                ops = ops[:k] + [loads(ni)] + ops[k + 1:]
                sched_barrier()
                if (tile < te) & (tl < n):
                    pk = hx.lload(sm + O_LST + (e * MAXTOK + tl) * 4, I32)
                    wt = hx.lload(sm + O_WTS + pk * 4, F32)
                    # routed partials as bf16 [NP][H] (half the stores: a hot expert's 16-pick tiles were store-bound)
                    op = c["out"] + (i64(pk) * H + i64(tt * 16 + (l // 16) * 8)) * 2
                    ov = [(acc[jj] * wt).to(fx.BFloat16) for jj in range(8)]
                    _stx(op, fx.Vector.from_elements(ov, fx.BFloat16))


@ASTRewriter.transform
def moe_dn_shared(c):
    """The FP8 shared expert's down: its NCH * 320 tiles split evenly over the CTAs (all of them, or the dedicated ones:
    c["dn8_bi"] / c["dn8_nct"]), one tile a wave (<= 8 a CTA); B (MID) straight from the global scratch (written before
    the UG -> DN handoff, or counted done among the dedicated CTAs)."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    tl = l % 16
    NP = c["NP"]
    NCH = (NP // c["topk"] + MAXTOK - 1) // MAXTOK
    Us = NCH * NDN
    nuni = ufl(hx.lload(sm + O_NUNI, I32))
    bi = c.get("dn8_bi", b)
    nct = c.get("dn8_nct", CTAS)
    d0 = (bi * Us) // nct
    d1 = ((bi + 1) * Us) // nct
    for it in range(fx.max((d1 - d0 - w + 7) // 8, I32(0))):
        tile = d0 + w + it * 8
        k = tile // NDN
        tt = tile % NDN
        ui = (I32(SHS0) + k) if sh_early(NCH) else nuni - NCH + k
        n = fx.min(I32(c["NP"] // c["topk"]) - MAXTOK * k, I32(MAXTOK))   # (as ug_shared8: not the LDS count)
        mrow = c["mid8"] + (i64(ui) * MAXTOK + i64(tl)) * MROW
        crow = c["midc"] + (i64(ui) * MAXTOK + i64(tl)) * CROW
        base = c["shplane"] + (i64(NUG * UGU) + i64(tt) * DNU) * UNIT8
        acc = fx.Vector.filled(8, 0.0, F32)
        ra = lda8(base, l)
        rb = ldb_g(mrow, crow, 0, l)
        for ks in range(DNU):   # a runtime loop (layer-loop code size), the next K step issued first
            nk = fx.min(ks + 1, I32(DNU - 1))
            na = lda8(base + i64(nk) * UNIT8, l)
            nb = ldb_g(mrow, crow, nk, l)
            acc = wmma_fp8(avec8(ra), ra[4], rb[0], rb[1], acc)
            ra = na
            rb = nb
        if tl < n:
            pk = NP + MAXTOK * k + tl
            op = c["out"] + (i64(pk) * H + i64(tt * 16 + (l // 16) * 8)) * 4
            _stx(op, fx.Vector.from_elements([acc[jj] for jj in range(4)], F32))
            _stx(op + i64(16), fx.Vector.from_elements([acc[jj] for jj in range(4, 8)], F32))


def moe(c, cnt_off):
    """The whole routed MoE (all 256 CTAs). cnt_off: byte address of this step's handoff counter (zeroed)."""
    moe_union(c)
    moe_ug(c)
    handoff(cnt_off, c["t"], True)
    moe_dn_stage(c)
    moe_dn(c)
    if fx.const_expr(not sh8_dnov((c["NP"] // c["topk"] + MAXTOK - 1) // MAXTOK)):
        moe_dn_shared(c)
