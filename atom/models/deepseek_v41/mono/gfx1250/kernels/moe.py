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
from atom.models.deepseek_v41.mono.gfx1250.kernels import stamps
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import q8words, bf, e8m0_of, handoff, inv_pow2, rsq

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
F32, I32, I64, BF16 = fx.Float32, fx.Int32, fx.Int64, fx.BFloat16
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


X8LO, X8LS = 220000, 250720   # (fx-v3) = mhc_v2.X8LO / X8LS: the CTA's own x8 [TR][5120] + codes [TR][160] in LDS


def ldb_x(c, row, ks, l):
    """(fx-v3) B operand from this CTA's LDS x8 (mhc_v2.x8_self) when c["x8l"], else from the global x8 row."""
    if fx.const_expr(c.get("x8l", False)):
        p = c["sm"] + X8LO + row * H + ks * 128 + (l // 16) * 16
        parts = [hx.lload(p + cc * 32, I32, 4, align=16) for cc in range(4)]
        return (fx.Vector.from_elements([parts[cc][e] for cc in range(4) for e in range(4)], I32),
                hx.lload(c["sm"] + X8LS + row * (H // 32) + ks * 4, I32))
    return ldb_g(c["x8"] + i64(row) * H, c["xs"] + i64(row) * (H // 32), ks, l)


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
    ids_g = "ids_lds" not in c or not c["ids_lds"]   # (v2: topk_v2 wrote ids / weights into this LDS)
    wts_g = "wts_lds" not in c or not c["wts_lds"]
    idp = c["ids"]
    wtp = c["wts"]
    for k in range_constexpr((NP + 255) // 256):
        p = t + 256 * k
        if p < NP:
            if fx.const_expr(ids_g):
                hx.lstore(sm + O_PID + p * 4, hx.gload(idp + i64(p) * 4, I32))
            if fx.const_expr(wts_g):
                hx.lstore(sm + O_WTS + p * 4, hx.gload(wtp + i64(p) * 4, F32))
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
def moe_combine(c, TR, sm_out, v3=False):
    """After the DN -> TP2 handoff: this CTA's TP2 slice (cols (b//4)*80 + (b%4)*20 .. +20, rows 0..TR-1) of the MoE output =
    sum over the row's 6 picks of out[pick] (already route-weighted) + the shared expert's row, as f32 [r][20]
    at sm_out (the layout tp_allreduce reads). One thread = 4 cols of a row (16-byte loads, all issued before the adds)."""
    t, b = c["t"], c["b"]
    NP = TR * c["topk"]
    OW = 32 if v3 else 20   # (fx-v3) owner b < 160: cols 32 b .. + 31, [r][32]; else 256 x 20
    NQ4 = OW // 4
    if (t < TR * NQ4) & ((b < 160) if v3 else (b >= 0)):
        row = t // NQ4
        c4 = (t % NQ4) * 4
        col = (b * 32 + c4) if v3 else ((b // 4) * 80 + (b % 4) * 20 + c4)
        vb = [hx.gload(c["out"] + (i64(row * c["topk"] + k) * H + i64(col)) * 2, fx.BFloat16, 4) for k in range(c["topk"])]
        vsh = hx.gload(c["out"] + (i64(NP + row) * H + i64(col)) * 4, F32, 4)   # the shared expert's row stays f32
        acc = [vb[0][j].to(F32) for j in range(4)]
        for k in range_constexpr(1, c["topk"]):
            acc = [acc[j] + vb[k][j].to(F32) for j in range(4)]
        acc = [acc[j] + vsh[j] for j in range(4)]
        hx.lstore(sm_out + (row * OW + c4) * 4, fx.Vector.from_elements(acc, F32), align=16)
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
    if fx.const_expr(sh_early(NCH) and DN_OVL and not SHE_IN_MOE):
        if b < NSHU:
            ug_shared_fin(c, b)
            if w == 0:
                hx.wait_storecnt0()
                if l == 0:
                    hx.put(c["ugflag"] + i64(SHFLAG0 + b) * 8, I32(0), c["ugtag"])
    elif fx.const_expr(sh_early(NCH)):
        if fx.const_expr(SHE_IN_MOE):   # (v2) the shared UG on the last NSHU CTAs (no routed groups at c1), one unit a CTA
            if b >= CTAS - NSHU * SHE_CTAS:   # K slices only: the sums + epilogue after the UG -> down handoff (moe)
                ug_shared_early(c, b - (CTAS - NSHU * SHE_CTAS))
        else:
            if b < NSHU:
                ug_shared_fin(c, b)
    STRIPE = MP.striped(NCH)
    nit = MP.unit_iters(STRIPE, U, CR, b, w, u1, wu)
    if fx.const_expr(not STRIPE):
        if fx.const_expr(TILE_UG):
            ug_routed_tile(c, U)
        else:
            if b < CR:
                ug_routed_ksplit(c, u0, u1)
    if fx.const_expr(not STRIPE):   # the routed units ran above (ug_routed_ksplit); the dedicated CTAs keep this loop
        nit = I32(0)
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
                hx.wait_count(c["shcnt"], NDC, 7)
            gpu.barrier()
            cd = {k: c[k] for k in c if k != "shcnt"}   # a plain dict (LazyC entries evaluated here)
            cd.update(dn8_bi=b - CR, dn8_nct=NDC)
            moe_dn_shared(cd)


IDLE_HOOK = hx.TraceSlot()   # (v2) trace-time hook run between the UG and the UG -> down handoff
TILE_UG = True   # (v2) the routed UG as TileRT's split: one 16-row tile a wave over the full K (a 40 KB contiguous stream)
UG1_KB = 8       # K steps in flight a wave (v2 tile UG)


@ASTRewriter.transform
def ug_body1(c, pk, wb, sb, tile):
    """One 16-row tile (8 gate + 8 up rows) of the group whose first tile is at wb / sb, all 40 K steps in this wave, a
    ring of UG1_KB K steps in flight (A: the tile's contiguous K stream; B: the picks' MXFP8 rows)."""
    l = c["l"]
    lo = lane_offs(l, True)
    row = prow(pk, c["NP"], c["topk"])
    xrow = c["x8"] + i64(row) * H
    srow = c["xs"] + i64(row) * (H // 32)
    KB = UG1_KB
    ra, rb = [], []
    for kk in range_constexpr(KB):
        ra.append(lda(*ug_unit(wb, sb, kk, tile), lo))
        rb.append(ldb_x(c, row, kk, l))
    sched_barrier()
    acc = fx.Vector.filled(8, 0.0, F32)
    nsteps = I32(40)
    for j in range(nsteps // KB - 1):
        for kk in range_constexpr(KB):
            ks = j * KB + kk
            acc = wmma_fp4(avec(ra[kk]), ra[kk][2], rb[kk][0], rb[kk][1], acc)
            ra = ra[:kk] + [lda(*ug_unit(wb, sb, ks + KB, tile), lo)] + ra[kk + 1:]
            rb = rb[:kk] + [ldb_x(c, row, ks + KB, l)] + rb[kk + 1:]
            sched_barrier()
    for kk in range_constexpr(KB):
        acc = wmma_fp4(avec(ra[kk]), ra[kk][2], rb[kk][0], rb[kk][1], acc)
    return acc


@ASTRewriter.transform
def ug_routed_tile(c, U):
    """(v2) Routed UG units = 32-col groups (expert, g) of 4 tiles; CTA b runs groups 2 b + w // 4 (+ 512 a round), wave
    w its tile w % 4 over the full K; the 4 tiles' accumulators meet in LDS and wave 0 / 4 of the group runs the
    epilogue (SwiGLU, the group's MXFP8 block, the MID row)."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tl = l % 16
    tile = w % 4
    for it in range((U + 2 * CTAS - 1) // (2 * CTAS)):
        G = it * (2 * CTAS) + b * 2 + w // 4
        live = G < U
        Gc = fx.min(G, U - 1)
        ui = Gc // NUG
        g = Gc % NUG
        e = hx.lload(sm + O_UNI + ui * 4, I32)
        n = fx.min(hx.lload(sm + O_CNT + e * 4, I32), I32(MAXTOK))
        pk = hx.lload(sm + O_LST + (e * MAXTOK + (tl < n).select(tl, I32(0))) * 4, I32)
        ex = i64(pslot(e, sm))
        acc1 = ug_body1(c, pk, c["w13"] + ex * W13_E + i64(g) * (4 * W13_TS), c["s13"] + ex * S13_E + i64(g) * (2 * W13_KC), tile)
        gpu.barrier()   # the previous round's epilogue has read the accumulators
        ow = sm + O_RED8 + (w * 32 + l) * 32
        hx.lstore(ow, fx.Vector.from_elements([acc1[i] for i in range(4)], F32), align=16)
        hx.lstore(ow + 16, fx.Vector.from_elements([acc1[i] for i in range(4, 8)], F32), align=16)
        gpu.barrier()
        if tile == 0:
            acc = []
            for q in range_constexpr(4):
                orr = sm + O_RED8 + ((w + q) * 32 + l) * 32
                lo_ = hx.lload(orr, F32, 4, align=16)
                hi_ = hx.lload(orr + 16, F32, 4, align=16)
                acc.append(fx.Vector.from_elements([lo_[i] for i in range(4)] + [hi_[i] for i in range(4)], F32))
            ug_epilogue(c, ui, g, live.select(n, I32(0)), acc)
            if fx.const_expr(DN_OVL):   # (v2) this group's MID cols are out: its LL flag (the down of expert ui waits for its NUG)
                hx.wait_storecnt0()
                if (l == 0) & live:
                    hx.put(c["ugflag"] + i64(ui * NUG + g) * 8, I32(0), c["ugtag"])


@ASTRewriter.transform
def ug_routed_ksplit(c, u0, u1):
    """A narrow routed CTA's units [u0, u1): KS = 4 / 2 / 1 waves a unit (units <= 2 / <= 4 / more), wave w
    does unit u0 + 8 / KS * round + w / KS over K steps (w % KS) * 40 / KS .. +40 / KS; the KS partials are summed in
    wave order through LDS (O_RED8) and the unit's first wave runs the epilogue."""
    w, l, sm = c["w"], c["l"], c["sm"]
    tl = l % 16
    units = fx.max(u1 - u0, I32(0))
    ks_ = (units <= 2).select(I32(4), (units <= 4).select(I32(2), I32(1)))
    per = I32(8) // ks_
    slot = w // ks_
    part = w % ks_
    nst = I32(40) // ks_
    # (K steps a part: multiples of the ring's KB = 4; 4 parts -> 12 / 12 / 8 / 8, 2 parts -> 20 / 20)
    k4 = ks_ == 4
    nstp = k4.select((part < 2).select(I32(12), I32(8)), nst)
    ks0p = k4.select((part < 2).select(part * 12, I32(24) + (part - 2) * 8), part * nst)
    for it in range((units + per - 1) // per):
        uu = u0 + it * per + slot
        live = uu < u1
        uc = fx.min(uu, u1 - 1)
        ui = uc // NUG
        g = uc % NUG
        e = hx.lload(sm + O_UNI + ui * 4, I32)
        n = fx.min(hx.lload(sm + O_CNT + e * 4, I32), I32(MAXTOK))
        pk = hx.lload(sm + O_LST + (e * MAXTOK + (tl < n).select(tl, I32(0))) * 4, I32)
        ex = i64(pslot(e, sm))
        acc = ug_body(c, pk, c["w13"] + ex * W13_E + i64(g) * (4 * W13_TS), c["s13"] + ex * S13_E + i64(g) * (2 * W13_KC),
                      ks0p, nstp)
        if ks_ > 1:
            gpu.barrier()   # the previous round's sums have read the partials
            for tile in range_constexpr(4):
                ow = sm + O_RED8 + ((w * 4 + tile) * 32 + l) * 32
                hx.lstore(ow, fx.Vector.from_elements([acc[tile][i] for i in range(4)], F32), align=16)
                hx.lstore(ow + 16, fx.Vector.from_elements([acc[tile][i] for i in range(4, 8)], F32), align=16)
            gpu.barrier()
            tot = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
            for q in range(ks_):
                for tile in range_constexpr(4):
                    orr = sm + O_RED8 + (((slot * ks_ + q) * 4 + tile) * 32 + l) * 32
                    lo_ = hx.lload(orr, F32, 4, align=16)
                    hi_ = hx.lload(orr + 16, F32, 4, align=16)
                    tot[tile] = fx.Vector.from_elements([tot[tile][i] + lo_[i] for i in range(4)] + [tot[tile][i + 4] + hi_[i] for i in range(4)], F32)
            acc = tot
        ug_epilogue(c, ui, g, (live & (part == 0)).select(n, I32(0)), acc)


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


DN_OVL = True   # (v2, TileRT) no device UG -> down handoff: LL flags a UG group, the down waits for its expert's
SHFLAG0 = 1024  # (flag slots: routed ui * NUG + g < SHFLAG0, then the shared units)
SHE_IN_MOE = False   # (v2) the shared expert's UG inside the MoE on idle CTAs (SHE_CTAS = 1), not early on CTAs 16..
SHE_SELFX8 = False   # (v2) the shared-UG CTAs quantize their own K slice of x8 (ug_shared_early selfx8)
SHE_CTAS = 2 if SHE_IN_MOE else 6   # CTAs a shared UG unit in the front's idle window (K split: one CTA's 348 KB unit is too slow)


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
    if fx.const_expr("selfx8" not in c):
        ra = [[lda8(ab + (i64(fx.min(k0 + j, I32(39))) * 4 + tile) * UNIT8, l) for tile in range(4)] for j in range(MAXS)]
    if fx.const_expr("pre_hook" in c):   # (v2) work that does not need x8 (the CTA's top-k), while the weights load
        c["pre_hook"]()
    if fx.const_expr("selfx8" in c):
        # (v2) this CTA's K slice of x8 made here from the owners' x1 + their 256 square-sum partials (bit-identical to
        # ffn_c2: same sum order, bf16 normed, e8m0 block scale, cvt): one hop (owners -> here), not two (-> c2 -> here)
        assert MAXS == 1
        ks = fx.min(k0, I32(39))
        cb = [ks * 128 + cc * 32 + (l // 16) * 16 for cc in range(4)]   # the lane's 16 cols of block cc
        if c["t"] == 0:   # wave 0 polls before it issues any load (loads return in order: a poll behind the weight
            hx.wait_count(c["postcnt"], 256, 6)   # loads would see the count only when they land); the other waves
        ra = [[lda8(ab + (i64(fx.min(k0 + j, I32(39))) * 4 + tile) * UNIT8, l) for tile in range(4)] for j in range(MAXS)]
        NV = [[hx.gload(c["nw"] + i64(cb[cc] + 8 * h) * 2, BF16, 8, align=16) for h in range(2)] for cc in range(4)]
        gpu.barrier()   # issued theirs before reaching this barrier
        if c["t"] == 0:
            hx.gstore(c["stp"] + i64(15 * 8), hx.clock())   # (study) owners' count seen by a shared-UG CTA
        XV = [[hx.gload(c["x1g"] + i64(row * H + cb[cc] + 8 * h) * 2, BF16, 8, align=16) for h in range(2)] for cc in range(4)]
        if w < rows_:
            pv = [hx.gload(c["part2"] + i64(((l + 32 * kq) * 8 + w) * 32 + 25) * 4, F32) for kq in range(256 // 32)]
            sq = F32(0.0)
            for kq in range_constexpr(256 // 32):
                sq = sq + pv[kq]
            for o in range_constexpr(5):
                sq = sq + sq.shuffle_xor(16 >> o, 32)
            if l == 0:
                hx.lstore(sm + O_WTS + w * 4, rsq(sq * F32(1.0 / H) + F32(c["norm_eps"])))
        gpu.barrier()
        rs = hx.lload(sm + O_WTS + row * 4, F32)
        words = []
        code4 = I32(0)
        for cc in range_constexpr(4):
            vals = [bf((XV[cc][h][e].to(F32) * rs) * NV[cc][h][e].to(F32)).to(F32) for h in range(2) for e in range(8)]
            am = F32(0.0)
            for e in range_constexpr(16):
                am = fx.max(am, abs(vals[e]))
            am = fx.max(am, am.shuffle_xor(16, 32))
            code = e8m0_of(am)
            words = words + q8words(vals, inv_pow2(code))
            code4 = code4 | (code << (8 * cc))
        rb = [(fx.Vector.from_elements(words, I32), code4)]
        gpu.barrier()   # (the rstd slots in O_WTS are the top-k's after this)
    else:
        if fx.const_expr("x8cnt" in c):   # (v2, TileRT) the weights issued before the wait for x8, not after it
            if fx.const_expr("xflag" in c):   # (v3 e10)
                hx.wait_flags(c["xflag"], c["x8n"], c["xtag"], w, l, 6)
            else:
                if c["t"] == 0:
                    hx.wait_count(c["x8cnt"], c["x8n"], 6)
            gpu.barrier()
            if c["t"] == 0:
                hx.gstore(c["stp"] + i64(15 * 8), hx.clock())   # (study) x8 seen by a shared-UG CTA
        rb = [ldb_x(c, row, fx.min(k0 + j, I32(39)), l) for j in range(MAXS)]
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
def ug_body(c, pk, wb, sb, ks0=None, nst=None):
    """One UG work unit (32 MID cols: 4 tiles x 40 K steps) of the expert, its first tile's weights / scales at wb / sb,
    for the tokens of pick list pk (lane tl's pick); returns the 4 tiles' accumulators."""
    l = c["l"]
    lo = lane_offs(l, True)
    row = prow(pk, c["NP"], c["topk"])
    xrow = c["x8"] + i64(row) * H
    srow = c["xs"] + i64(row) * (H // 32)
    KB = DUG // 4
    nsteps = I32(40)
    if fx.const_expr(ks0 is not None):   # K steps ks0 .. ks0 + nst - 1 only: every base shifted by ks0
        wb = wb + i64(ks0) * 1024
        sb = sb + i64(ks0) * 128
        xrow = xrow + i64(ks0 * 128)
        srow = srow + i64(ks0 * 4)
        nsteps = nst
    ra, rb = ug_ring_init(wb, sb, lo, xrow, srow, l, KB)
    # rolled over the 40 K steps in blocks of KB, fixed ring slots (slot kk*4+tile / kk refilled with K step
    # ks + KB right after K step ks used it): ~1/10 of the unrolled code, and no ring shifting through
    # loop-carried registers (that made the compiler wait for every load at the loop top)
    acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
    for j in range(nsteps // KB - 1):
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


DN_V2 = True   # (v2) the routed down as long contiguous weight streams: a wave an (expert, 128-col block)


@ASTRewriter.transform
def moe_dn_v2(c):
    """(v2) Routed down: unit (union expert ui, 128-col hidden block hb of 40) = 8 consecutive 16-row tiles (a ~48 KB
    contiguous weight span) in one wave; units spread one a wave over the CTAs (unit u -> CTA u % 256, wave u // 256,
    + 2048 a round); B (the expert's MID rows, 5 K steps) straight from global into registers once (no LDS staging);
    two tiles (10 A K steps) in flight. Per (row, pick) bf16 partials to out, the route weight applied (as moe_dn)."""
    w, l, b, sm = c["w"], c["l"], c["b"], c["sm"]
    tl = l % 16
    lo = lane_offs(l, False)
    nuni = ufl(hx.lload(sm + O_NUNI, I32))
    NB = NDN // 8
    Ud = dn_nuni(c, nuni) * NB
    for it in range(fx.max((Ud - (w * CTAS + b) + 8 * CTAS - 1) // (8 * CTAS), I32(0))):
        u = (it * 8 + w) * CTAS + b
        ui = u // NB
        hb = u % NB
        e = hx.lload(sm + O_UNI + ui * 4, I32)
        n = fx.min(hx.lload(sm + O_CNT + e * 4, I32), I32(MAXTOK))
        ex = i64(pslot(e, sm))
        mrow = c["mid8"] + (i64(ui) * MAXTOK + i64(tl)) * MROW
        crow = c["midc"] + (i64(ui) * MAXTOK + i64(tl)) * CROW
        pk = hx.lload(sm + O_LST + (e * MAXTOK + (tl < n).select(tl, I32(0))) * 4, I32)
        wt = hx.lload(sm + O_WTS + pk * 4, F32)

        def tile_loads(j):
            tt = hb * 8 + j
            wb = c["w2"] + ex * W2_E + i64(tt) * W2_TS
            sb = c["s2"] + ex * S2_E + i64(tt // 2) * W2_KC + i64(tt % 2) * 64
            return [lda(wb + i64(ks * 1024), sb + i64(ks * 128), lo) for ks in range(DNU)]

        ops = [tile_loads(0), tile_loads(1)]
        if fx.const_expr(DN_OVL):   # (v2) no UG -> down device handoff: wait for expert ui's NUG group flags (lane l < NUG)
            hx.get(c["ugflag"] + i64(ui * NUG + fx.min(l, I32(NUG - 1))) * 8, c["ugtag"], 7)
        B = [ldb_g(mrow, crow, ks, l) for ks in range(DNU)]
        sched_barrier()
        for j in range_constexpr(8):
            aq = ops[j % 2]
            acc = fx.Vector.filled(8, 0.0, F32)
            for ks in range_constexpr(DNU):
                acc = wmma_fp4(avec(aq[ks]), aq[ks][2], B[ks][0], B[ks][1], acc)
            if fx.const_expr(j + 2 < 8):
                ops = ops[:j % 2] + [tile_loads(j + 2)] + ops[j % 2 + 1:]
            sched_barrier()
            if tl < n:
                tt = hb * 8 + j
                op = c["out"] + (i64(pk) * H + i64(tt * 16 + (l // 16) * 8)) * 2
                ov = [(acc[jj] * wt).to(fx.BFloat16) for jj in range(8)]
                _stx(op, fx.Vector.from_elements(ov, fx.BFloat16))
        # this unit's 128 cols are out: count it in its block (the combine of those cols waits for its block only)
        hx.wait_storecnt0()
        if fx.const_expr("dnf" in c):   # (fx-v3 e13) the unit's LL flag: (block hb, union entry ui)
            if l == 0:
                hx.put(c["dnf"] + i64(hb * c["dnstr"] + ui) * 8, I32(0), c["dntag"])
        else:
            if l == 0:
                hx.atomic_add(c["dnblk"] + i64(hb) * 4, 1)


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
    if fx.const_expr(DN_OVL and not SHE_IN_MOE):   # the shared MID flags (ug_shared_fin on CTAs < NSHU)
        hx.get(c["ugflag"] + i64(SHFLAG0 + fx.min(l, I32(NCH * NUG - 1))) * 8, c["ugtag"], 7)
    if fx.const_expr(SHE_IN_MOE):   # the shared MID: written by the fin CTAs after the UG -> down handoff
        if t == 0:
            hx.wait_count(c["shfin"], NCH * NUG, 7)
        gpu.barrier()
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
        if fx.const_expr(DN_V2):   # this tile counted in its 128-col block (8 shared tiles a block, NCH chunks)
            hx.wait_storecnt0()
            if fx.const_expr("dnf" in c):   # (fx-v3 e13) the shared tile's LL flag: (block tt // 8, chunk k, tile tt % 8)
                if l == 0:
                    hx.put(c["dnf"] + i64((tt // 8) * c["dnstr"] + c["dnmaxu"] + k * 8 + tt % 8) * 8, I32(0), c["dntag"])
            else:
                if l == 0:
                    hx.atomic_add(c["dnblk"] + i64(tt // 8) * 4, 1)


@ASTRewriter.transform
def she_fin(c):
    NCHs = (c["NP"] // c["topk"] + MAXTOK - 1) // MAXTOK
    sb0 = CTAS - NCHs * NUG * SHE_CTAS
    b, t = c["b"], c["t"]
    if (b >= sb0) & (b < sb0 + NCHs * NUG):
        ug_shared_fin(c, b - sb0)
        hx.wait_storecnt0()
        gpu.barrier()
        if t == 0:
            hx.atomic_add(c["shfin"], 1)


def moe(c, cnt_off):
    """The whole routed MoE (all 256 CTAs). cnt_off: byte address of this step's handoff counter (zeroed). Phase stamps
    (stamps.point 10..14): union, UG, UG -> down handoff, down staging, down."""
    st = lambda k: stamps.point(10 + k)
    moe_union(c)
    st(0)
    moe_ug(c)
    st(1)
    if fx.const_expr(IDLE_HOOK[0] is not None):   # (v2) work for the CTAs without UG units (before the UG -> down handoff)
        IDLE_HOOK[0]()
    if fx.const_expr(not DN_OVL):
        handoff(cnt_off, c["t"], True)
    if fx.const_expr(SHE_IN_MOE):   # (v2) the shared UG units' K-slice sums + epilogue (partials of SHE_CTAS CTAs), counted
        she_fin(c)
    st(2)
    if fx.const_expr(DN_V2):
        st(4)
        moe_dn_v2(c)
    else:
        moe_dn_stage(c)
        st(4)
        moe_dn(c)
    if fx.const_expr(not sh8_dnov((c["NP"] // c["topk"] + MAXTOK - 1) // MAXTOK)):
        moe_dn_shared(c)
    st(3)


def _rv_wmax(v):   # (the router's wave max: front imports moe, so a call-time import)
    from atom.models.deepseek_v41.mono.gfx1250.kernels.front import _wmax
    return _wmax(v)


# ==============================================================================================================
# (merged from moe_v3.py)
# (fx-v3) The FFN / MoE block as TileRT's mega_afm organizes it (design 7: F0 / H / I / G / G' / J), TP4, c1.
# 
# Routed CTAs (240: b < 144 and b >= 160): i = b (< 144) or b - 16, row r = i // 40, slice s = i % 40 (rows >= TR idle).
#   F0   this row's x only: the 160 owners' flags, x1 row r + its 160 square-sum partials -> rstd -> MXFP8 in LDS (1 row)
#   top  wave 0: row r's 384 LL scores + bias -> its 6 picks (ids: the top-k or the fixed route table) + weights -> LDS
#   UG   row r's 108 groups (pick k, 32-col group g): CTA s groups [108 s / 40, 108 (s + 1) / 40), wave w the tile w % 4 of
#        group w // 4 (+ 2 a round), full K; the 4 tiles meet in LDS, SwiGLU + MXFP8 -> MID slot r * 6 + k, one LL flag
#   DN   hidden cols [128 s, 128 s + 128): wave k < 6 the row's pick k (8 tiles x 5 K steps, its first two tiles issued
#        before the wait for the pick's 18 group flags), wave 6 the shared expert (FP8) of row r (its 18 flags); x the
#        route weight, the 7 partials summed in fixed order in LDS -> 128 bf16
#   J    the 128-col partial pushed as tagged 16-byte words {w0, tag, w1, tag} to the 4 ranks' receive regions [r][s]
# Shared CTAs 144..159: x8 of all TR rows (x8_self), groups g = j, j + 16 (< 18) of the FP8 shared UG, K over the 8
#   waves, epilogue -> MID slot SHS0 (rows = tokens), one LL flag a group.
# Owners b < 160 (recv_owner): their 32 cols of every row: the 4 ranks' words summed in rank order 0..3 -> y2.

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off


NRC, NSC, SC0 = 40, 16, 144        # routed CTAs a row, shared CTAs, the first shared CTA
NG = TOPK * NUG                    # 108 UG groups a row
CTAS_R = 240                       # routed CTAs
DRED = 254000                      # LDS: the down's 7 x 128 f32 partials, then 64 packed words
SHDL = 110000                      # LDS: this CTA's shared-expert down slice (8 tiles x 5 K steps x UNIT8 = 87040 B)
NSHD = 8 * DNU * UNIT8 // 16       # its 16-byte chunks
DNPD = 4                           # routed down tiles in flight a wave
TP_TG = 40960
KB = 8                             # UG K steps in flight a wave


def rslot(b):
    """(row, slice) of routed CTA b (b in 144..159: the shared CTAs, row 99)."""
    i = (b < SC0).select(b, b - NSC)
    return (b >= SC0) & (b < SC0 + NSC), i // NRC, i % NRC


@ASTRewriter.transform
def f0_row(c, r):
    """This CTA's LDS x8 (row 0) = MXFP8 of bf(x1[r] rstd nw) (x8_self's arithmetic, one row)."""
    t, w, l, sm = c["t"], c["w"], c["l"], c["sm"]
    NV = [hx.gload(c["nw"] + i64(fx.min(t, I32(159)) * 32 + 8 * h) * 2, BF16, 8, align=16) for h in range(4)]
    hx.wait_flags(c["pflag"], 160, c["ptag"], w, l, 13)
    gpu.barrier()
    if w == 0:
        pv = [hx.gload(c["part2"] + i64(((l + 32 * k) * 8 + r) * 32 + 25) * 4, F32) for k in range(5)]
        s = F32(0.0)
        for k in range_constexpr(5):
            s = s + pv[k]
        for o in range_constexpr(5):
            s = s + s.shuffle_xor(16 >> o, 32)
        if l == 0:
            hx.lstore(sm + X8LS + 1024, rsq(s * F32(1.0 / H) + F32(c["norm_eps"])))
    gpu.barrier()
    if t < 160:
        rs = hx.lload(sm + X8LS + 1024, F32)
        xv = [hx.gload(c["x1g"] + i64(r * H + t * 32 + 8 * h) * 2, BF16, 8, align=16) for h in range(4)]
        vals = [bf((xv[h][e].to(F32) * rs) * NV[h][e].to(F32)).to(F32) for h in range(4) for e in range(8)]
        am = F32(0.0)
        for e in range_constexpr(32):
            am = fx.max(am, abs(vals[e]))
        code = e8m0_of(am)
        words = q8words(vals, inv_pow2(code))
        hx.lstore(sm + X8LO + t * 32, fx.Vector.from_elements(words[:4], I32), align=16)
        hx.lstore(sm + X8LO + t * 32 + 16, fx.Vector.from_elements(words[4:], I32), align=16)
        hx.lstore(sm + X8LS + t, code.to(fx.Int8))
    gpu.barrier()


def ldb_row(sm, ks, l):
    """B operand: LDS x8 row 0 for all 16 token lanes (K step ks)."""
    p = sm + X8LO + ks * 128 + (l // 16) * 16
    parts = [hx.lload(p + cc * 32, I32, 4, align=16) for cc in range(4)]
    return (fx.Vector.from_elements([parts[cc][e] for cc in range(4) for e in range(4)], I32),
            hx.lload(sm + X8LS + ks * 4, I32))


@ASTRewriter.transform
def shdn_prefetch(c, s):
    """(TileRT H: waves 1..6 during the top-k) the shared expert's down slice of cols [128 s, + 128) (one contiguous
    87 KB span of the plane) -> LDS SHDL; every load issued, then the stores."""
    t, sm = c["t"], c["sm"]
    src = c["shplane"] + (i64(NUG * UGU) + i64(s * 8 * DNU)) * UNIT8
    NIT = (NSHD + 255) // 256
    v = [hx.gload(src + i64(fx.min(t + 256 * k, I32(NSHD - 1))) * 16, I32, 4, nt=True) for k in range(NIT)]
    for k in range_constexpr(NIT):
        if t + 256 * k < NSHD:
            hx.lstore(sm + SHDL + (t + 256 * k) * 16, v[k], align=16)


def lda8_l(u, l):
    """lda8 from an LDS unit at byte offset u."""
    p = u + l * 64
    return (hx.lload(p, I32, 4, align=16), hx.lload(p + 16, I32, 4, align=16), hx.lload(p + 32, I32, 4, align=16),
            hx.lload(p + 48, I32, 4, align=16), hx.lload(u + 2048 + l * 4, I32))


@ASTRewriter.transform
def ug_tile_row(c, wb, sb, tile, ks0, NS, KBx):
    """K steps ks0 .. ks0 + NS - 1 of one 16-row tile of a group (first tile at wb / sb), B = the LDS row; KBx K steps
    in flight (NS, KBx: python ints, NS % KBx == 0)."""
    l, sm = c["l"], c["sm"]
    lo = lane_offs(l, True)
    ra, rb = [], []
    for kk in range_constexpr(KBx):
        ra.append(lda(*ug_unit(wb, sb, ks0 + kk, tile), lo))
        rb.append(ldb_row(sm, ks0 + kk, l))
    sched_barrier()
    acc = fx.Vector.filled(8, 0.0, F32)
    for j in range(I32(NS // KBx - 1)):
        for kk in range_constexpr(KBx):
            ks = ks0 + j * KBx + kk
            acc = wmma_fp4(avec(ra[kk]), ra[kk][2], rb[kk][0], rb[kk][1], acc)
            ra = ra[:kk] + [lda(*ug_unit(wb, sb, ks + KBx, tile), lo)] + ra[kk + 1:]
            rb = rb[:kk] + [ldb_row(sm, ks + KBx, l)] + rb[kk + 1:]
            sched_barrier()
    for kk in range_constexpr(KBx):
        acc = wmma_fp4(avec(ra[kk]), ra[kk][2], rb[kk][0], rb[kk][1], acc)
    return acc


def pick_id(c, k):
    """Row r's pick k: the top-k's (LDS slot k) or the fixed route table's."""
    if fx.const_expr(c["ids_lds"]):
        return hx.lload(c["sm"] + O_PID + (c["row"] * c.get("ktop", TOPK) + k) * 4, I32)
    return hx.gload(c["ids"] + i64(c["row"] * c.get("ktop", TOPK) + k) * 4, I32)


@ASTRewriter.transform
def ug_row(c, s):
    """Row r's UG groups [g0, g1) (2 or 3) on this CTA: round 0 groups g0, g0 + 1 (waves 0..3 / 4..7, tile w % 4, full
    K); round 1 (a third group) on all 8 waves, tile w // 2 over K half w % 2 (the halves summed in order): 60 K steps
    a wave at most, not 80."""
    w, l, sm, r = c["w"], c["l"], c["sm"], c["row"]
    g0 = (NG * s) // NRC
    g1 = (NG * (s + 1)) // NRC
    for rd in range_constexpr(2):
        if fx.const_expr(rd == 0):
            gi = g0 + w // 4
            tl4 = w % 4
        else:
            gi = g0 + 2 + w * 0
            tl4 = w // 2
        live = gi < g1
        gc = fx.min(gi, g1 - 1)
        k = gc // NUG
        g = gc % NUG
        ex = i64(pick_id(c, k))
        wb = c["w13"] + ex * W13_E + i64(g) * (4 * W13_TS)
        sb = c["s13"] + ex * S13_E + i64(g) * (2 * W13_KC)
        acc1 = fx.Vector.filled(8, 0.0, F32)
        if live:
            if fx.const_expr(rd == 0):
                acc1 = ug_tile_row(c, wb, sb, tl4, I32(0), 40, KB)
            else:
                acc1 = ug_tile_row(c, wb, sb, tl4, (w % 2) * 20, 20, 4)
        gpu.barrier()
        ow = sm + O_RED8 + (w * 32 + l) * 32
        hx.lstore(ow, fx.Vector.from_elements([acc1[i] for i in range(4)], F32), align=16)
        hx.lstore(ow + 16, fx.Vector.from_elements([acc1[i] for i in range(4, 8)], F32), align=16)
        gpu.barrier()
        if fx.const_expr(rd == 0):
            lead = (w % 4 == 0) & live
        else:
            lead = (w == 0) & live
        if lead:
            acc = []
            for q in range_constexpr(4):
                if fx.const_expr(rd == 0):
                    orr = sm + O_RED8 + ((w + q) * 32 + l) * 32
                    lo_ = hx.lload(orr, F32, 4, align=16)
                    hi_ = hx.lload(orr + 16, F32, 4, align=16)
                    acc.append(fx.Vector.from_elements([lo_[i] for i in range(4)] + [hi_[i] for i in range(4)], F32))
                else:
                    o0 = sm + O_RED8 + ((2 * q) * 32 + l) * 32
                    o1 = sm + O_RED8 + ((2 * q + 1) * 32 + l) * 32
                    a0 = hx.lload(o0, F32, 4, align=16)
                    a1 = hx.lload(o0 + 16, F32, 4, align=16)
                    b0 = hx.lload(o1, F32, 4, align=16)
                    b1 = hx.lload(o1 + 16, F32, 4, align=16)
                    acc.append(fx.Vector.from_elements([a0[i] + b0[i] for i in range(4)] + [a1[i] + b1[i] for i in range(4)], F32))
            ui = r * TOPK + k
            ug_epilogue(c, ui, g, I32(1), acc)
            hx.wait_storecnt0()
            if l == 0:
                hx.put(c["mflag"] + i64(ui * NUG + g) * 8, I32(0), c["mtag"])


@ASTRewriter.transform
def dn_row(c, s):
    """Row r's down of hidden cols [128 s, + 128): wave k < 6 pick k, wave 6 the shared expert; the 7 partials summed
    in order in LDS -> 64 packed bf16 words at DRED + 7 * 512."""
    t, w, l, sm, r = c["t"], c["w"], c["l"], c["sm"], c["row"]
    KT = c.get("ktop", TOPK)   # (draft: top-3; wave KT = the shared expert)
    tl = l % 16
    if w < KT:
        lo = lane_offs(l, False)
        p = r * KT + w
        e = pick_id(c, w)
        wt = hx.lload(sm + O_WTS + p * 4, F32)
        # (union UG) the pick's union entry ui (O_UNI[ui] == e) and its token slot pos (O_LST[e][pos] == p)
        nun = hx.lload(sm + O_NUNI, I32)
        ua = ((hx.lload(sm + O_UNI + l * 4, I32) == e) & (l < nun)).select(l, I32(-1))
        ub = ((hx.lload(sm + O_UNI + fx.min(l + 32, I32(NEX - 1)) * 4, I32) == e) & (l + 32 < nun)).select(l + 32, I32(-1))
        ui = ufl(_rv_wmax(fx.max(ua, ub)))
        ne_ = hx.lload(sm + O_CNT + e * 4, I32)
        pa = ((hx.lload(sm + O_LST + (e * MAXTOK + l % 16) * 4, I32) == p) & (l % 16 < ne_)).select(l % 16, I32(-1))
        pos = ufl(_rv_wmax(pa))

        def tile_loads(j):
            tt = s * 8 + j
            wb = c["w2"] + i64(e) * W2_E + i64(tt) * W2_TS
            sb = c["s2"] + i64(e) * S2_E + i64(tt // 2) * W2_KC + i64(tt % 2) * 64
            return [lda(wb + i64(ks * 1024), sb + i64(ks * 128), lo) for ks in range(DNU)]

        ops = [tile_loads(j) for j in range(DNPD)]
        hx.get(c["mflag"] + i64(ui * NUG + fx.min(l, I32(NUG - 1))) * 8, c["mtag"], 7)
        mrow = c["mid8"] + i64(ui * MAXTOK + pos) * MROW
        crow = c["midc"] + i64(ui * MAXTOK + pos) * CROW
        B = [ldb_g(mrow, crow, ks, l) for ks in range(DNU)]
        sched_barrier()
        for j in range_constexpr(8):
            aq = ops[j % DNPD]
            acc = fx.Vector.filled(8, 0.0, F32)
            for ks in range_constexpr(DNU):
                acc = wmma_fp4(avec(aq[ks]), aq[ks][2], B[ks][0], B[ks][1], acc)
            if fx.const_expr(j + DNPD < 8):
                ops = ops[:j % DNPD] + [tile_loads(j + DNPD)] + ops[j % DNPD + 1:]
            sched_barrier()
            if tl == 0:   # token 0: cols j * 16 + (l // 16) * 8 + jj, the pick's weighted bf16 partial
                hx.lstore(sm + DRED + (w * 128 + j * 16 + (l // 16) * 8) * 4,
                          fx.Vector.from_elements([(acc[jj] * wt).to(BF16).to(F32) for jj in range(8)], F32), align=16)
    if w == KT:   # the shared expert (FP8) of row r
        hx.get(c["mflag"] + i64(c["shflag0"] + fx.min(l, I32(NUG - 1))) * 8, c["mtag"], 7)
        mrow = c["mid8"] + i64(SHS0 * MAXTOK + r) * MROW
        crow = c["midc"] + i64(SHS0 * MAXTOK + r) * CROW
        B = [ldb_g(mrow, crow, ks, l) for ks in range(DNU)]
        for j in range(8):
            ra = [lda8_l(sm + SHDL + (j * DNU + ks) * UNIT8, l) for ks in range(DNU)]
            acc = fx.Vector.filled(8, 0.0, F32)
            for ks in range_constexpr(DNU):
                acc = wmma_fp8(avec8(ra[ks]), ra[ks][4], B[ks][0], B[ks][1], acc)
            if tl == 0:
                hx.lstore(sm + DRED + (KT * 128 + j * 16 + (l // 16) * 8) * 4,
                          fx.Vector.from_elements([acc[jj] for jj in range(8)], F32), align=16)
    gpu.barrier()
    if t < 64:   # 2 cols a thread: the 6 picks in order, then the shared row -> bf16 pair
        lo2 = [hx.lload(sm + DRED + (k * 128 + 2 * t) * 4, F32, 2) for k in range(KT + 1)]
        a0 = lo2[0][0]
        a1 = lo2[0][1]
        for k in range_constexpr(1, KT + 1):
            a0 = a0 + lo2[k][0]
            a1 = a1 + lo2[k][1]
        wd = a0.to(BF16).bitcast(fx.Int16).to(I32) & 0xFFFF | (a1.to(BF16).bitcast(fx.Int16).to(I32) << 16)
        hx.lstore(sm + DRED + 7 * 512 + t * 4, wd)
    gpu.barrier()


@ASTRewriter.transform
def push_row(c, s, region, tag):
    """The row's 128-col partial (64 words) as 32 tagged 16-byte words to the 4 ranks' region [r][s] (own rank too)."""
    t, sm, r, rank = c["t"], c["sm"], c["row"], c["rank"]
    if t < 32:
        w0 = hx.lload(sm + DRED + 7 * 512 + (2 * t) * 4, I32)
        w1 = hx.lload(sm + DRED + 7 * 512 + (2 * t + 1) * 4, I32)
        v = fx.Vector.from_elements([w0, I32(tag), w1, I32(tag)], I32)
        slot = (r * NRC + s) * 32 + t
        for o in range_constexpr(4):
            hx.gstore(c["recv"][o] + (i64((region * 4) * TP_TG) + i64(rank * TP_TG) + i64(slot)) * 16, v, nt=True)


@ASTRewriter.transform
def recv_owner(c, region, tag, TR):
    """Owner b < 160: its 32 cols of every row from the 4 ranks' pushes, summed in rank order -> y2 (bf16)."""
    t, b, rank = c["t"], c["b"], c["rank"]
    if t < TR * 8:
        r = t // 8
        q = t % 8
        slot = (r * NRC + b // 4) * 32 + 8 * (b % 4) + q
        gps = [c["recv"][rank] + (i64((region * 4) * TP_TG) + i64(o * TP_TG) + i64(slot)) * 16 for o in range(4)]
        t0 = hx.clock()
        got = [hx.gload(gp, I32, 4, nt=True) for gp in gps]
        bad = I32(0)
        for o in range_constexpr(4):
            bad = bad | ((got[o][1] != tag) | (got[o][3] != tag)).select(I32(1), I32(0))
        while (bad != 0) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
            hx.s_sleep(0)
            got = [hx.gload(gp, I32, 4, nt=True) for gp in gps]
            bad = I32(0)
            for o in range_constexpr(4):
                bad = bad | ((got[o][1] != tag) | (got[o][3] != tag)).select(I32(1), I32(0))
        hx.note_timeout(t0, 6)
        for h in range_constexpr(2):
            sl = F32(0.0)
            sh = F32(0.0)
            for o in range_constexpr(4):
                wv = got[o][2 * h]
                sl = sl + (wv << 16).bitcast(F32)
                sh = sh + (wv & I32(-65536)).bitcast(F32)
            wq = sl.to(BF16).bitcast(fx.Int16).to(I32) & 0xFFFF | (sh.to(BF16).bitcast(fx.Int16).to(I32) << 16)
            hx.stx_dev(c["ytgt"] + (i64(r) * H + i64(b * 32 + 4 * q + 2 * h)) * 2, wq)


@ASTRewriter.transform
def shared_ug(c, j, TR):
    """Shared CTA j: FP8 shared UG groups j, j + 16 (< 18) for the TR rows (tokens), K steps 5 w .. 5 w + 4 a wave, the
    8 partials summed in wave order (LDS), wave 0's epilogue -> MID slot SHS0 rows 0..TR-1, one flag a group."""
    w, l, sm = c["w"], c["l"], c["sm"]
    tl = l % 16
    rowx = fx.min(tl, I32(TR - 1))
    for gi in range_constexpr(2):
        g = j + 16 * gi
        live = g < NUG
        gcl = fx.min(g, I32(NUG - 1))
        ab = c["shplane"] + i64(gcl * UGU) * UNIT8
        acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
        if live:
            ks0 = w * 5
            for kk in range_constexpr(5):
                ra = [lda8(ab + (i64(ks0 + kk) * 4 + tile) * UNIT8, l) for tile in range(4)]
                p = sm + X8LO + rowx * H + (ks0 + kk) * 128 + (l // 16) * 16
                parts = [hx.lload(p + cc * 32, I32, 4, align=16) for cc in range(4)]
                bq = fx.Vector.from_elements([parts[cc][e] for cc in range(4) for e in range(4)], I32)
                bs = hx.lload(sm + X8LS + rowx * (H // 32) + (ks0 + kk) * 4, I32)
                acc = [wmma_fp8(avec8(ra[tile]), ra[tile][4], bq, bs, acc[tile]) for tile in range(4)]
        gpu.barrier()
        for tile in range_constexpr(4):
            ow = sm + O_RED8 + ((w * 4 + tile) * 32 + l) * 32
            hx.lstore(ow, fx.Vector.from_elements([acc[tile][i] for i in range(4)], F32), align=16)
            hx.lstore(ow + 16, fx.Vector.from_elements([acc[tile][i] for i in range(4, 8)], F32), align=16)
        gpu.barrier()
        if (w == 0) & live:
            tot = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
            for ww in range(8):
                for tile in range_constexpr(4):
                    orr = sm + O_RED8 + ((ww * 4 + tile) * 32 + l) * 32
                    lo = hx.lload(orr, F32, 4, align=16)
                    hi = hx.lload(orr + 16, F32, 4, align=16)
                    tot[tile] = fx.Vector.from_elements([tot[tile][i] + lo[i] for i in range(4)] + [tot[tile][i + 4] + hi[i] for i in range(4)], F32)
            ug_epilogue(c, I32(SHS0), gcl, I32(TR), tot)
            hx.wait_storecnt0()
            if l == 0:
                hx.put(c["mflag"] + i64(c["shflag0"] + gcl) * 8, I32(0), c["mtag"])


@ASTRewriter.transform
def x8_pub(c, r):
    """CTA (r, 0): its LDS x8 row -> the global x8 row r (fx8 / fxs) + the row's LL flag (the union UG's B rows)."""
    t = c["t"]
    if t < 160:
        v0 = hx.lload(c["sm"] + X8LO + t * 32, I32, 4, align=16)
        v1 = hx.lload(c["sm"] + X8LO + t * 32 + 16, I32, 4, align=16)
        hx.stx_dev(c["x8"] + i64(r * H + t * 32), v0)
        hx.stx_dev(c["x8"] + i64(r * H + t * 32 + 16), v1)
    if t < 40:   # the 160 codes as 40 words
        hx.stx_dev(c["xs"] + i64(r * (H // 32) + t * 4), hx.lload(c["sm"] + X8LS + t * 4, I32))
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.put(c["xflag"] + i64(r) * 8, I32(0), c["xtag"])


@ASTRewriter.transform
def ug_union(c, i, TR):
    """Routed CTA i (0..239): union groups [U i / 240, U (i + 1) / 240) (U = routed union entries x 18): 2 groups a
    round (waves 0..3 / 4..7, tile w % 4, full K), B = the expert's picks' x8 rows (global, after the TR row flags);
    the group epilogue -> MID slot ui (token slots = the expert's pick list), one LL flag (ui, g)."""
    w, l, sm = c["w"], c["l"], c["sm"]
    tl = l % 16
    hx.wait_flags(c["xflag"], TR, c["xtag"], w, l, 10)
    gpu.barrier()
    nuni = ufl(hx.lload(sm + O_NUNI, I32))
    U = dn_nuni(c, nuni) * NUG
    u0 = (U * i) // (CTAS_R)
    u1 = (U * (i + 1)) // (CTAS_R)
    for rd in range_constexpr(2):
        G = u0 + rd * 2 + w // 4
        live = G < u1
        Gc = fx.max(fx.min(G, u1 - 1), I32(0))
        ui = Gc // NUG
        g = Gc % NUG
        e = hx.lload(sm + O_UNI + ui * 4, I32)
        n = fx.min(hx.lload(sm + O_CNT + e * 4, I32), I32(MAXTOK))
        pk = hx.lload(sm + O_LST + (e * MAXTOK + (tl < n).select(tl, I32(0))) * 4, I32)
        ex = i64(pslot(e, sm))
        acc1 = fx.Vector.filled(8, 0.0, F32)
        if live:
            acc1 = ug_body1(c, pk, c["w13"] + ex * W13_E + i64(g) * (4 * W13_TS), c["s13"] + ex * S13_E + i64(g) * (2 * W13_KC), w % 4)
        gpu.barrier()
        ow = sm + O_RED8 + (w * 32 + l) * 32
        hx.lstore(ow, fx.Vector.from_elements([acc1[q] for q in range(4)], F32), align=16)
        hx.lstore(ow + 16, fx.Vector.from_elements([acc1[q] for q in range(4, 8)], F32), align=16)
        gpu.barrier()
        if (w % 4 == 0) & live:
            acc = []
            for q in range_constexpr(4):
                orr = sm + O_RED8 + ((w + q) * 32 + l) * 32
                lo_ = hx.lload(orr, F32, 4, align=16)
                hi_ = hx.lload(orr + 16, F32, 4, align=16)
                acc.append(fx.Vector.from_elements([lo_[j] for j in range(4)] + [hi_[j] for j in range(4)], F32))
            ug_epilogue(c, ui, g, n, acc)
            hx.wait_storecnt0()
            if l == 0:
                hx.put(c["mflag"] + i64(ui * NUG + g) * 8, I32(0), c["mtag"])
