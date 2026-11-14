# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The attention stages of the step kernel as device functions (each AST-rewritten so runtime if / while lower to scf
ops): A (wqkv_a GEMV over the K halves, the cluster sum, q / kv quantization), kv (kv norm, RoPE, the window ring row),
C (wq_b, RoPE), D (sparse attention over the selected and window keys), E (wo_a), F (wo_b in a 4-CTA cluster) and the TP
all-reduce of its output (tp_allreduce). Handoffs: device.handoff (counters: layout).

`c` is a plain dict of traced values / trace-time constants:
  t b w l sm (Int32)  TR MT RPC QV o SMEM (python)  scale (Float32)
  pointers (Int64): x8 xs wqkv wqkv_s qnw kvnw wqb wqb_s kv sel sink rcos rsin woa woa_s wob wob_s qkv_part qkv ss qr8 qrs
                    kvn q apart aml o8 os y8 ys ypart out opart (layer-specific ones already offset by the caller)
  stx(addr, val): the exchange store (plain, or device-scope for fence-free handoffs)
"""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D

T, HID, QL, KVD = D.T_MAX, D.HID, D.Q_RANK, D.KV_DIM
SS_LAYER_F32 = 2 * T   # per-layer ss region (f32 words): the q / kv sums of squares [2][T]
NQKV = QL + KVD
H, HD = D.HEADS, D.KV_DIM
NQ = H * HD
G, OR = 2, 1024
NWOA, KWOA = G * OR, 8 * HD
THREADS, CTAS = D.THREADS, D.CTAS
REQ, ROWS, KEYS, CL, POOL, KP = D.MAX_REQ, D.ROWS_PER_REQ, 640, D.CL, 2048, 64
KC = KEYS // CL
# the product's KV geometry (attention.py _keys / _key_row): a row's keys = its selected rows, then its window ring rows;
# per-token kv_len mask, absolute pool rows, the ring-row KV write
KG_TOPK = 512    # sel row stride (index_plan TOPK)
KG_RING = 133    # ring_slots (window 128 + 5): set by the build from the cache geometry (the runner checks it against the cache geometry)
KG_ROW = 512     # pool row stride in bf16 elements (geometry.row_bytes / 2)
KMETA = 4   # kmeta words a token: first, count, (table offset), base
WG = CL + 4   # LDS row of the merge weights (16 + 1/den, padded to 16 B)
HP = HD + 8
F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16


NORM_EPS = D.RMS_EPS   # q_norm / kv_norm eps (config.rms_norm_eps)


def rope_cs(c, r, fi):
    """cos / sin of rotation pair fi of row r: the product's bf16 cos_cache / sin_cache [pos][32] at the row's position
    rpos[r] (int64)."""
    p = hx.gload(c["rpos"] + i64(r) * 8, fx.Int64)
    return (hx.gload(c["rcos"] + (p * 32 + i64(fi)) * 2, BF16).to(F32),
            hx.gload(c["rsin"] + (p * 32 + i64(fi)) * 2, BF16).to(F32))


def rope_cs16(c, row, f0):
    """pairs f0..f0+15 of row: [4][4] f32 lists (cos, sin)."""
    p = hx.gload(c["rpos"] + i64(row) * 8, fx.Int64)
    cv = hx.gload(c["rcos"] + (p * 32 + i64(f0)) * 2, BF16, 16)
    sv = hx.gload(c["rsin"] + (p * 32 + i64(f0)) * 2, BF16, 16)
    return ([[cv[4 * k + e].to(F32) for e in range(4)] for k in range(4)],
            [[sv[4 * k + e].to(F32) for e in range(4)] for k in range(4)])


def i64(x):
    return fx.Int64(x)


V3_RPC1 = True


def smem_layout(TR):
    RPC = (1 if TR <= 8 else 2) if V3_RPC1 else (2 if TR <= 32 else 3)   # (v3) one row a cluster at c1: 6 x 16 attention CTAs
    QV = RPC * H
    o, off = {}, 0
    for name, nbytes in (("k", RPC * 48 * HP * 2), ("q", QV * HP * 2), ("s", QV * 48 * 4), ("p", QV * KP * 2),
                         ("m", QV * 4), ("l", QV * 4), ("red", QV * 5 * 4), ("red2", QV * 5 * 4),
                         ("wgt", QV * WG * 4), ("ids", RPC * KC * 4), ("kvl", 16)):
        o[name] = off
        off += nbytes
    return dict(TR=TR, MT=(TR + 15) // 16, RPC=RPC, QV=QV, o=o, SMEM=off, LDS=off + 512)


# ------------------------------------------------------------------ device math
def e8m0_of(amax):
    """code_ceil(max(amax, 1e-10) / 448) clamped to [1, 254] (product MXFP8 scale)."""
    v = fx.max(amax, F32(1e-10)) / F32(448.0)
    bits = v.bitcast(I32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(I32(1), I32(0))
    return fx.max(fx.min(e, I32(254)), I32(1))


def inv_pow2(code):
    return ((I32(254) - code) << 23).bitcast(F32)


def to_e4m3(v):
    v = fx.min(fx.max(v, F32(-448.0)), F32(448.0))
    return hx.cvt_pk_fp8(v) & 0xFF


def bf(x):
    return F32(x).to(BF16)


def rsq(x):
    return F32(hx.intrin(F32.ir_type, "llvm.amdgcn.rsq.f32", [x]))


def expf(x):
    """raw v_exp_f32 (no denormal-range fixup: differs only for results < 2^-126)."""
    return F32(hx.intrin(F32.ir_type, "llvm.amdgcn.exp2.f32", [x * F32(1.4426950408889634)]))


def ld_op(row, kb, l):
    p = row + i64(kb) + i64((l // 16) * 16)
    parts = [hx.gload(p + i64(c * 32), I32, 4) for c in range(4)]
    return fx.Vector.from_elements([parts[c][e] for c in range(4) for e in range(4)], I32)


def mx_loop(n, fa, fb, D=4):
    """n K128 MXFP8 WMMA steps, D operand pairs in flight. Sched barriers keep the D-deep load ring in flight (each refill
    issued right after the WMMA that freed its slot)."""
    A, SA, B, SB = [None] * D, [None] * D, [None] * D, [None] * D
    for i in range(min(D, n)):
        B[i], SB[i] = fb(i)
        A[i], SA[i] = fa(i)
    _sbar()
    c = fx.Vector.filled(8, 0.0, F32)
    for ks in range(0, n, D):
        for i in range(D):
            if ks + i < n:
                c = hx.wmma_mx(A[i], SA[i], B[i], SB[i], c)
                if ks + i + D < n:
                    A[i], SA[i] = fa(ks + i + D)
                    B[i], SB[i] = fb(ks + i + D)
                    _sbar()
    return c


def _sbar():
    hx.intrin(None, "llvm.amdgcn.sched.barrier", [I32(0)])


def q8words(vals, inv):
    """MXFP8 words of vals * inv: same clamp / cvt as to_e4m3 + pack4 (bit-identical), 2 cvt_pk_fp8 a word."""
    n = len(vals)
    cl = [fx.min(fx.max(vals[e] * inv, F32(-448.0)), F32(448.0)) for e in range(n)]
    out = []
    for k in range(n // 4):
        lo = hx.cvt_pk_fp8_w(0, cl[4 * k], cl[4 * k + 1], False)
        out.append(hx.cvt_pk_fp8_w(lo, cl[4 * k + 2], cl[4 * k + 3], True))
    return out


def pack4(b):
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


def cluster_sync():
    hx.fence_release("cluster")
    hx.cluster_barrier()
    hx.fence_acquire("cluster")


handoff = hx.handoff   # (device.handoff: every stage module imports it from here)


def selfx8_op(c, rowx, rsx, kb, l):
    """(v2) MXFP8 A operand of K step kb (lane l: token rowx, 16-col chunks cc * 32 + (l // 16) * 16) from the column
    owners' x1 (c["x1a"], bf16) and the row's rstd: front_C's arithmetic (bf(x1 rs nw), e8m0 of the 32-block amax across
    the lane pair l, l ^ 16). Returns (words, the 4 block codes as one word, the bf16 normed values a chunk)."""
    words = []
    code4 = I32(0)
    nbs = []
    for cc in range(4):
        col = kb + cc * 32 + (l // 16) * 16
        xv2 = [hx.gload(c["x1a"] + i64(rowx * HID + col + 8 * h) * 2, BF16, 8, align=16) for h in range(2)]
        nv2 = [hx.gload(c["anw"] + i64(col + 8 * h) * 2, BF16, 8, align=16) for h in range(2)]
        nb = [bf((xv2[h][e].to(F32) * rsx) * nv2[h][e].to(F32)) for h in range(2) for e in range(8)]
        vals = [nb[e].to(F32) for e in range(16)]
        am = F32(0.0)
        for e in range(16):
            am = fx.max(am, abs(vals[e]))
        am = fx.max(am, am.shuffle_xor(16, 32))
        code = e8m0_of(am)
        words = words + q8words(vals, inv_pow2(code))
        code4 = code4 | (code << (8 * cc))
        nbs.append(nb)
    return fx.Vector.from_elements(words, I32), code4, nbs


def pf_rows(base, stride, rows, nbytes, t):
    """global_prefetch of every 128-byte line of `rows` rows of `nbytes` (stride apart), strided over the CTA."""
    lines = nbytes // 128
    n = rows * lines
    for it in range_constexpr((n + THREADS - 1) // THREADS):
        i = t + it * THREADS
        ic = fx.min(i, I32(n - 1))
        hx.prefetch(base + i64(ic // lines) * stride + i64(ic % lines) * 128)


X8L, X8C = 16384, 16384 + 6 * 2560   # (v2) LDS: stage A's x8 K half [TR][2560] and its block codes [TR][80]
C_WIDE = True   # (v2) stage C's q_b GEMM on all 8 waves at one row tile
A_WIDE = True   # (v2) stage A's GEMV on 5 waves x 4 K steps at one row tile


# ------------------------------------------------------------------ A: wqkv_a + cluster sum / q quant / kv bf16
@ASTRewriter.transform
def stage_A(c):
    t, b, w, l, sm, MT, TR = c["t"], c["b"], c["w"], c["l"], c["sm"], c["MT"], c["TR"]
    if b < 224:
        nt = b // 2
        kh = b % 2
        n0 = nt * 16
        if fx.const_expr(MT == 1 and A_WIDE):
            # (v2) one 16-row tile: 5 waves x 4 K steps (all four operand pairs in flight), not 2 waves x 10 serial steps
            NEWA = "newA" in c
            # (contract 6.5) this wave's 4 wqkv K steps issued before any wait (waves 5..7 repeat wave 4's: L2 hits)
            kk0p = kh * 20 + fx.min(w, I32(4)) * 4
            browp = c["wqkv"] + i64(n0 + l % 16) * HID
            preB = [(ld_op(browp, (kk0p + i) * 128, l),
                     hx.gload(c["wqkv_s"] + i64(((n0 + l % 16) // 32) * (HID // 32) + (kk0p + i) * 4), I32)) for i in range(4)]
            PUB = "pubflag" in c   # (fx-v3) x8 made once by the 32 publisher CTAs (stage_Apub): wait for them, read it from global
            if fx.const_expr(NEWA and PUB):
                if c["newA"]:
                    hx.wait_flags(c["pubflag"], 32, c["pubtag"], w, l, 9)
                    gpu.barrier()
            if fx.const_expr(NEWA and not PUB):   # (v2) x1 from the TP2 column owners: wait for them, each row's rstd from 256 partials
                if c["newA"]:
                    # every load in one round: nw (weights) before the wait; after it the x1 blocks and the 256 partials
                    NB = (TR * 80 + 255) // 256
                    blks = [fx.min(t + it * 256, I32(TR * 80 - 1)) for it in range(NB)]
                    cols = [kh * 2560 + (blks[it] % 80) * 32 for it in range(NB)]
                    if fx.const_expr("apflag" in c):   # (v3 e1) the 256 TP2 owners' LL flags
                        hx.wait_flags(c["apflag"], 160, c["aptag"], w, l, 9)
                    else:
                        if t == 0:   # (the poll first: a load issued before it would hold the poll until it lands)
                            hx.wait_count(c["attpost"], 160, 9)
                    gpu.barrier()
                    nv2 = [[hx.gload(c["anw"] + i64(cols[it] + 8 * h) * 2, BF16, 8, align=16) for h in range(4)] for it in range(NB)]
                    xv2 = [[hx.gload(c["x1a"] + i64((blks[it] // 80) * HID + cols[it] + 8 * h) * 2, BF16, 8, align=16) for h in range(4)] for it in range(NB)]
                    wr = fx.min(w, I32(TR - 1))
                    pv = [hx.gload(c["part2"] + i64(((l + 32 * kq) * 8 + wr) * 32 + 25) * 4, F32) for kq in range(5)]   # (160 owners)
                    if w < TR:
                        sq = F32(0.0)
                        for kq in range_constexpr(5):
                            sq = sq + pv[kq]
                        for o in range_constexpr(5):
                            sq = sq + sq.shuffle_xor(16 >> o, 32)
                        if l == 0:
                            hx.lstore(sm + 8192 + w * 4, rsq(sq * F32(1.0 / HID) + F32(c["norm_eps"])))
                    gpu.barrier()
                    # this CTA's K half of x8 (TR rows x 80 blocks of 32) quantized once into LDS (front_C's arithmetic):
                    # few registers, then the GEMV reads its A operands from LDS
                    for it in range_constexpr(NB):
                        blk = t + it * 256
                        if blk < TR * 80:
                            rr = blk // 80
                            bc = blk % 80
                            col = kh * 2560 + bc * 32
                            rsr = hx.lload(sm + 8192 + rr * 4, F32)
                            nb = [bf((xv2[it][h][e].to(F32) * rsr) * nv2[it][h][e].to(F32)) for h in range(4) for e in range(8)]
                            vals = [nb[e].to(F32) for e in range(32)]
                            am = F32(0.0)
                            for e in range_constexpr(32):
                                am = fx.max(am, abs(vals[e]))
                            code = e8m0_of(am)
                            wds = q8words(vals, inv_pow2(code))
                            hx.lstore(sm + X8L + rr * 2560 + bc * 32, fx.Vector.from_elements(wds[:4], I32), align=16)
                            hx.lstore(sm + X8L + rr * 2560 + bc * 32 + 16, fx.Vector.from_elements(wds[4:], I32), align=16)
                            hx.lstore(sm + X8C + (rr * 80 + bc) * 4, code)
                            if fx.const_expr("axn" in c):   # the attention input normed (bf16) for the indexer / compressor
                                if b < 2:
                                    for h in range_constexpr(4):
                                        c["stx"](c["axn"] + i64(rr * HID + col + 8 * h) * 2,
                                                 fx.Vector.from_elements([nb[8 * h + e] for e in range(8)], BF16))
                    gpu.barrier()
            if w < 5:
                kk0 = kh * 20 + w * 4
                arow = c["x8"] + i64(fx.min(l % 16, I32(c["TR"] - 1))) * HID   # (rows past TR: the last row again, cached)
                brow = c["wqkv"] + i64(n0 + l % 16) * HID

                def fa(i):
                    kb = (kk0 + i) * 128
                    xv = hx.gload(c["xs"] + i64(fx.min(l % 16, I32(c["TR"] - 1)) * (HID // 32) + kb // 32) * 4, I32, 4)
                    return ld_op(arow, kb, l), (xv[0] & 255) | ((xv[1] & 255) << 8) | ((xv[2] & 255) << 16) | ((xv[3] & 255) << 24)

                def fb(i):
                    kb = (kk0 + i) * 128
                    return ld_op(brow, kb, l), hx.gload(c["wqkv_s"] + i64(((n0 + l % 16) // 32) * (HID // 32) + kb // 32), I32)

                if fx.const_expr(NEWA and not PUB):
                    acc = fx.Vector.filled(8, 0.0, F32)
                    if c["newA"]:
                        rowx = fx.min(l % 16, I32(TR - 1))

                        def fa2(i):   # the A operand from the LDS x8 (16-byte chunks c * 32 + (l // 16) * 16 of K step i)
                            kr = (w * 4 + i) * 128
                            pq = sm + X8L + rowx * 2560 + kr + (l // 16) * 16
                            parts = [hx.lload(pq + cc * 32, I32, 4, align=16) for cc in range(4)]
                            cq = [hx.lload(sm + X8C + (rowx * 80 + kr // 32 + cc) * 4, I32) for cc in range(4)]
                            return (fx.Vector.from_elements([parts[cc][e] for cc in range(4) for e in range(4)], I32),
                                    cq[0] | (cq[1] << 8) | (cq[2] << 16) | (cq[3] << 24))

                        acc = mx_loop(4, fa2, lambda i: preB[i])
                    else:
                        acc = mx_loop(4, fa, lambda i: preB[i])
                else:
                    acc = mx_loop(4, fa, lambda i: preB[i])
                hx.lstore(sm + ((w * 32 + l) * 8) * 4, acc)
            gpu.barrier()
            if w == 0:
                tot = hx.lload(sm + (l * 8) * 4, F32, 8)
                for ww in range_constexpr(1, 5):
                    rv = hx.lload(sm + ((ww * 32 + l) * 8) * 4, F32, 8)
                    tot = fx.Vector.from_elements([tot[i] + rv[i] for i in range(8)], F32)
                for i in range_constexpr(8):
                    hx.gstore(c["qkv_part"] + i64((kh * T + (l // 16) * 8 + i) * NQKV + n0 + l % 16) * 4, tot[i])
        else:
            if (w < 6) & (w % 3 < MT):
                mi = w % 3
                kk0 = kh * 20 + (w // 3) * 10
                arow = c["x8"] + i64(mi * 16 + l % 16) * HID
                brow = c["wqkv"] + i64(n0 + l % 16) * HID

                def fa(i):
                    kb = (kk0 + i) * 128
                    xv = hx.gload(c["xs"] + i64((mi * 16 + l % 16) * (HID // 32) + kb // 32) * 4, I32, 4)
                    return ld_op(arow, kb, l), (xv[0] & 255) | ((xv[1] & 255) << 8) | ((xv[2] & 255) << 16) | ((xv[3] & 255) << 24)

                def fb(i):
                    kb = (kk0 + i) * 128
                    return ld_op(brow, kb, l), hx.gload(c["wqkv_s"] + i64(((n0 + l % 16) // 32) * (HID // 32) + kb // 32), I32)

                acc = mx_loop(10, fa, fb)
                hx.lstore(sm + ((w * 32 + l) * 8) * 4, acc)
                gpu.barrier()
                if (w < 3) & (w < MT):
                    r0v = hx.lload(sm + ((w * 32 + l) * 8) * 4, F32, 8)
                    r1v = hx.lload(sm + (((w + 3) * 32 + l) * 8) * 4, F32, 8)
                    for i in range_constexpr(8):
                        v = r0v[i] + r1v[i]
                        hx.gstore(c["qkv_part"] + i64((kh * T + mi * 16 + (l // 16) * 8 + i) * NQKV + n0 + l % 16) * 4, v)
            else:
                gpu.barrier()
    cluster_sync()
    if (b < 224) & (b % 4 == 0):
        c0 = (b // 4) * 32
        rss = sm + c["SMEM"]
        gmx = rss + 48 * 4
        if t < 48:
            hx.lstore(rss + t * 4, F32(0.0))
            hx.lstore(gmx + t * 4, I32(0))
        gpu.barrier()
        r = fx.min(t // 4, I32(TR - 1))
        qq = t % 4
        cc = c0 + qq * 8
        act = t < TR * 4
        p0 = hx.gload(c["qkv_part"] + i64(r * NQKV + cc) * 4, F32, 8)
        p1 = hx.gload(c["qkv_part"] + i64((T + r) * NQKV + cc) * 4, F32, 8)
        hb = [bf(p0[e] + p1[e]) for e in range(8)]
        xsq = F32(0.0)
        am = F32(0.0)
        vv = []
        for e in range_constexpr(8):
            x = hb[e].to(F32)
            xsq = xsq + x * x
            v = (c0 < QL).select(x * hx.gload(c["qnw"] + i64(cc + e) * 4, F32), x)
            vv.append(v)
            am = fx.max(am, abs(v))
        if act:
            c["stx"](c["qkv"] + i64(r * NQKV + cc) * 2, fx.Vector.from_elements(hb, BF16))
            hx.latomic_add_f32(rss + r * 4, xsq)
            hx.latomic_max_u32(gmx + r * 4, am.bitcast(I32))
        gpu.barrier()
        if (c0 < QL) & act:
            code = e8m0_of(hx.lload(gmx + r * 4, I32).bitcast(F32))
            inv = inv_pow2(code)
            bts = [to_e4m3(vv[e] * inv) for e in range(8)]
            c["stx"](c["qr8"] + i64(r * QL + cc), fx.Vector.from_elements([pack4(bts[:4]), pack4(bts[4:])], I32))
            if qq == 0:
                c["stx"](c["qrs"] + i64(r * (QL // 32) + c0 // 32), code.to(fx.Int8))
        if t < TR:
            hx.atomic_fadd(c["ss"] + i64((c0 < QL).select(I32(0), I32(T)) + t) * 4, hx.lload(rss + t * 4, F32))
        if fx.const_expr("a2flag" in c):   # (v3 e3) this 32-col block's q / kv (+ its ss share) out: one LL flag
            hx.wait_storecnt0()
            gpu.barrier()
            if t == 0:
                hx.put(c["a2flag"] + i64(b // 4) * 8, I32(0), c["a2tag"])


@ASTRewriter.transform
def stage_Apub(c):
    """(fx-v3) CTAs 224..255 (no A GEMV): publisher p = b - 224 makes the x8 / xs of cols 160 p .. + 159 for every row
    (stage A's arithmetic: rstd from the 160 owners' partials, bf(x1 rstd nw), the 32-col MXFP8 block) -> global, the
    attention input normed (axn), one LL flag. c: t b w l sm TR x1a part2 anw apflag aptag x8 xs axn norm_eps stx pubflag pubtag."""
    t, b, w, l, sm, TR = c["t"], c["b"], c["w"], c["l"], c["sm"], c["TR"]
    if b >= 224:
        p = b - 224
        rr = fx.min(t // 5, I32(TR - 1))
        bc = p * 5 + t % 5
        col = bc * 32
        nv = [hx.gload(c["anw"] + i64(col + 8 * h) * 2, BF16, 8, align=16) for h in range(4)]
        hx.wait_flags(c["apflag"], 160, c["aptag"], w, l, 9)
        gpu.barrier()
        xv = [hx.gload(c["x1a"] + i64(rr * HID + col + 8 * h) * 2, BF16, 8, align=16) for h in range(4)]
        wr = fx.min(w, I32(TR - 1))
        pv = [hx.gload(c["part2"] + i64(((l + 32 * kq) * 8 + wr) * 32 + 25) * 4, F32) for kq in range(5)]
        if w < TR:
            sq = F32(0.0)
            for kq in range_constexpr(5):
                sq = sq + pv[kq]
            for o in range_constexpr(5):
                sq = sq + sq.shuffle_xor(16 >> o, 32)
            if l == 0:
                hx.lstore(sm + 8192 + w * 4, rsq(sq * F32(1.0 / HID) + F32(c["norm_eps"])))
        gpu.barrier()
        if t < TR * 5:
            rsr = hx.lload(sm + 8192 + rr * 4, F32)
            nb = [bf((xv[h][e].to(F32) * rsr) * nv[h][e].to(F32)) for h in range(4) for e in range(8)]
            vals = [nb[e].to(F32) for e in range(32)]
            am = F32(0.0)
            for e in range_constexpr(32):
                am = fx.max(am, abs(vals[e]))
            code = e8m0_of(am)
            wds = q8words(vals, inv_pow2(code))
            c["stx"](c["x8"] + i64(rr * HID + col), fx.Vector.from_elements(wds[:4], I32))
            c["stx"](c["x8"] + i64(rr * HID + col + 16), fx.Vector.from_elements(wds[4:], I32))
            c["stx"](c["xs"] + i64(rr * (HID // 32) + bc) * 4, code)   # (i32 codes: stage A reads the attention x8 scales as words)
            for h in range_constexpr(4):
                c["stx"](c["axn"] + i64(rr * HID + col + 8 * h) * 2, fx.Vector.from_elements([nb[8 * h + e] for e in range(8)], BF16))
        hx.wait_storecnt0()
        gpu.barrier()
        if t == 0:
            hx.put(c["pubflag"] + i64(p) * 8, I32(0), c["pubtag"])


# ------------------------------------------------------------------ kv RMSNorm + RoPE + window write
KV_CTA0 = 224   # the kv CTAs: the Apub publishers (no A GEMV), ceil(TR / 4) of them


@ASTRewriter.transform
def stage_kv(c):
    t, b, TR = c["t"], c["b"], c["TR"]
    kb = b - KV_CTA0   # (the kv CTAs are past the A GEMV's: on CTAs 0, 1 the kv rows put their C, then every D, ~2.5 us late)
    kvc = (kb >= 0) & (kb * 4 < TR)
    if fx.const_expr("a2flag" in c):   # (v3 e3) the kv blocks of A2 (and their ss share)
        if kvc & (t < 32):
            hx.get(c["a2flag"] + i64(QL // 32 + fx.min(t, I32(NQKV // 32 - QL // 32 - 1))) * 8, c["a2tag"], 8)
        gpu.barrier()
    if kvc:
        for it in range_constexpr((4 * (KVD // 8) + THREADS - 1) // THREADS):
            i = t + it * THREADS
            r = kb * 4 + i // (KVD // 8)
            c8 = (i % (KVD // 8)) * 8
            if (i < 4 * (KVD // 8)) & (r < TR):
                # (v2) every load issued first (one round trip, not a chain): ss, the row, kvnw x 8, rope cos / sin x 4, kvw
                ssv = hx.gload(c["ss"] + i64(T + r) * 4, F32)
                hv = hx.gload(c["qkv"] + i64(r * NQKV + QL + c8) * 2, BF16, 8)
                nwv = hx.gload(c["kvnw"] + i64(c8) * 4, F32, 8)
                ppos = hx.gload(c["rpos"] + i64(r) * 8, fx.Int64)
                kw = hx.gload(c["kvw"] + i64(r) * 4, I32)   # this token's ring row (-1: a pad row, not written)
                fi0 = fx.max((c8 - (HD - 64)) // 2, I32(0))
                cv4 = hx.gload(c["rcos"] + (ppos * 32 + i64(fi0)) * 2, BF16, 4)
                sv4 = hx.gload(c["rsin"] + (ppos * 32 + i64(fi0)) * 2, BF16, 4)
                rk = rsq(ssv / F32(KVD) + F32(NORM_EPS))
                o8v = [bf(hv[e].to(F32) * rk * nwv[e]) for e in range(8)]
                c["stx"](c["kvn"] + i64(r * KVD + c8) * 2, fx.Vector.from_elements(o8v, BF16))
                is_r = c8 >= HD - 64
                rot = []
                for e in range_constexpr(0, 8, 2):
                    cs = cv4[e // 2].to(F32)
                    sn = sv4[e // 2].to(F32)
                    x0 = o8v[e].to(F32)
                    x1 = o8v[e + 1].to(F32)
                    rot.append(is_r.select(x0 * cs - x1 * sn, x0).to(BF16))
                    rot.append(is_r.select(x0 * sn + x1 * cs, x1).to(BF16))
                # the product's window row: e4m3 QAT a 32-group (rope_window._quantized_row)
                xr = [rot[e].to(F32) for e in range(8)]
                am = abs(xr[0])
                for e in range_constexpr(1, 8):
                    am = fx.max(am, abs(xr[e]))
                am = fx.max(am, am.shuffle_xor(1, 32))
                am = fx.max(am, am.shuffle_xor(2, 32))
                code = e8m0_of(am)
                inv = inv_pow2(code)
                stp = (code << 23).bitcast(F32)
                cl = [fx.min(fx.max(xr[e] * inv, F32(-448.0)), F32(448.0)) for e in range(8)]
                rot = []
                for h in range_constexpr(2):
                    wd = hx.cvt_pk_fp8_w(hx.cvt_pk_fp8_w(0, cl[4 * h], cl[4 * h + 1], False), cl[4 * h + 2], cl[4 * h + 3], True)
                    a0, a1 = hx.fp8x2_to_f32(wd, False)
                    a2, a3 = hx.fp8x2_to_f32(wd, True)
                    rot += [(a0 * stp).to(BF16), (a1 * stp).to(BF16), (a2 * stp).to(BF16), (a3 * stp).to(BF16)]
                if kw >= 0:
                    c["stx"](c["kv"] + (i64(kw + c["kring_off"]) * KG_ROW + i64(c8)) * 2, fx.Vector.from_elements(rot, BF16))
    if fx.const_expr("a2flag" in c):   # (v3 e4) this CTA's kv ring rows out
        if kvc:
            hx.wait_storecnt0()
            gpu.barrier()
            if t == 0:
                hx.put(c["kvflag"] + i64(kb) * 8, I32(0), c["kvtag"])


# ------------------------------------------------------------------ C: wq_b (rstd after the GEMM, RoPE)
@ASTRewriter.transform
def stage_C(c):
    t, b, w, l, MT = c["t"], c["b"], c["w"], c["l"], c["MT"]
    if fx.const_expr(MT == 1 and C_WIDE):   # (contract 6.5) this wave's wqb K steps issued before the wait
        KQp = QL // 128 // 4
        n0p = (b + (w // 4) * CTAS) * 16
        brp = c["wqb"] + i64(n0p + l % 16) * QL
        preQ = [(ld_op(brp, ((w % 4) * KQp + i) * 128, l),
                 hx.gload(c["wqb_s"] + i64(((n0p + l % 16) // 32) * (QL // 32) + ((w % 4) * KQp + i) * 4), I32, align=4))
                for i in range(KQp)]
    if fx.const_expr("a2flag" in c):   # (v3 e3) the q blocks of A2 (wave 0: 2 flags a lane)
        hx.wait_flags(c["a2flag"], QL // 32, c["a2tag"], w, l, 8)
        gpu.barrier()
    if fx.const_expr(MT == 1 and C_WIDE):
        # (v2) the CTA's 2 tiles x 4 K quarters on the 8 waves (operands in flight), partials summed in LDS by waves 0 / 4
        KQ = QL // 128 // 4
        jq = w // 4
        kq = w % 4
        n0q = (b + jq * CTAS) * 16
        arq = c["qr8"] + i64(fx.min(l % 16, I32(c["TR"] - 1))) * QL   # (rows past TR: the last row again, cached)
        brq = c["wqb"] + i64(n0q + l % 16) * QL

        def faq(i):
            kb = (kq * KQ + i) * 128
            return ld_op(arq, kb, l), hx.gload(c["qrs"] + i64((l % 16) * (QL // 32) + kb // 32), I32, align=4)

        def fbq(i):
            kb = (kq * KQ + i) * 128
            return ld_op(brq, kb, l), hx.gload(c["wqb_s"] + i64(((n0q + l % 16) // 32) * (QL // 32) + kb // 32), I32, align=4)

        accq = mx_loop(KQ, faq, lambda i: preQ[i])
        hx.lstore(c["sm"] + ((w * 32 + l) * 8) * 4, accq)
        gpu.barrier()
    if (w < 6) & (w % 3 < MT):
        j = w // 3
        mi = w % 3
        n0 = (b + j * CTAS) * 16
        arow = c["qr8"] + i64(mi * 16 + l % 16) * QL
        brow = c["wqb"] + i64(n0 + l % 16) * QL

        def fa(i):
            kb = i * 128
            return ld_op(arow, kb, l), hx.gload(c["qrs"] + i64((mi * 16 + l % 16) * (QL // 32) + kb // 32), I32, align=4)

        def fb(i):
            kb = i * 128
            return ld_op(brow, kb, l), hx.gload(c["wqb_s"] + i64(((n0 + l % 16) // 32) * (QL // 32) + kb // 32), I32, align=4)

        if fx.const_expr(MT == 1 and C_WIDE):   # tile j's 4 quarter partials (waves 4 j .. 4 j + 3), in order
            acc = hx.lload(c["sm"] + (((4 * j) * 32 + l) * 8) * 4, F32, 8)
            for qq in range_constexpr(1, 4):
                rv = hx.lload(c["sm"] + (((4 * j + qq) * 32 + l) * 8) * 4, F32, 8)
                acc = fx.Vector.from_elements([acc[i] + rv[i] for i in range(8)], F32)
        else:
            acc = mx_loop(QL // 128, fa, fb)
        cv = [acc[i] * rsq(hx.gload(c["ss"] + i64(mi * 16 + (l // 16) * 8 + i) * 4, F32) / F32(QL) + F32(NORM_EPS)) for i in range(8)]
        is_rope = n0 % HD >= HD - 64
        fi = fx.max((n0 % HD + l % 16 - (HD - 64)) // 2, I32(0))
        odd = (l & 1) == 1
        res = []
        for i in range_constexpr(8):
            r = mi * 16 + (l // 16) * 8 + i
            x = cv[i].to(BF16).to(F32)
            y = cv[i].shuffle_xor(1, 32).to(BF16).to(F32)
            cs, sn = rope_cs(c, r, fi)
            res.append(is_rope.select(odd.select(y * sn + x * cs, x * cs - y * sn), cv[i]))
        for i in range_constexpr(8):
            c["stx"](c["q"] + i64((mi * 16 + (l // 16) * 8 + i) * NQ + n0 + l % 16) * 2, res[i].to(BF16))
    if fx.const_expr("a2flag" in c):   # (v3 e4) this CTA's two q tiles out
        hx.wait_storecnt0()
        gpu.barrier()
        if t == 0:
            hx.put(c["cflag"] + i64(b) * 8, I32(0), c["ctag"])


# ------------------------------------------------------------------ D: attention (gathered keys), o as MXFP8
@ASTRewriter.transform
def _kg_rows(c, t, r0, jj, sm, o, RPC, TR):
    """This CTA's KC keys of each of its RPC rows -> absolute pool rows in LDS ids (attention.py _key_row: the
    n_sel selected rows first, then the window rows oldest first base + ring_off + (first + key - n_sel) % ring_slots); keys
    past kv_len = n_sel + count clamp to the last valid key's row (masked in the softmax); kv_len -> LDS kvl."""
    if t < RPC * KC:
        rr = t // KC
        tok = r0 + rr
        if fx.const_expr(TR % RPC != 0):   # the last cluster's rows past TR: the last row's keys (their outputs are not stored)
            tok = fx.min(tok, I32(TR - 1))
        first = hx.gload(c["kmeta"] + i64(tok * KMETA) * 4, I32)
        count = hx.gload(c["kmeta"] + i64(tok * KMETA + 1) * 4, I32)
        base = hx.gload(c["kmeta"] + i64(tok * KMETA + 3) * 4, I32)
        nsel = fx.min(fx.max(hx.gload(c["knsel"] + i64(tok) * 4, I32), I32(0)), I32(KG_TOPK))   # index_plan.selected = winners(bound, TOPK)
        kvl = nsel + count
        key = fx.min(jj * KC + t % KC, kvl - 1)
        picked = key < nsel
        selr = hx.gload(c["ksel"] + i64(tok * KG_TOPK + fx.min(key, I32(KG_TOPK - 1))) * 4, I32)
        win = base + c["kring_off"] + (first + key - nsel) % KG_RING
        hx.lstore(sm + o["ids"] + t * 4, picked.select(selr, win))
        if t % KC == 0:
            hx.lstore(sm + o["kvl"] + rr * 4, kvl)


@ASTRewriter.transform
def stage_D(c):
    t, b, w, l, sm, o = c["t"], c["b"], c["w"], c["l"], c["sm"], c["o"]
    TR, RPC, QV = c["TR"], c["RPC"], c["QV"]
    cl = b // CL
    jj = b % CL
    r0 = cl * RPC
    # every row: ceil(TR / RPC) clusters (a draft width 5 B need not be a multiple of RPC)
    if fx.const_expr("cflag" in c):   # (v3 e4) the kv ring rows first: the key rows load while C still runs
        if cl < (TR + RPC - 1) // RPC:
            hx.wait_flags(c["kvflag"], (TR + 3) // 4, c["kvtag"], w, l, 8)
            gpu.barrier()
    if cl < (TR + RPC - 1) // RPC:
        _kg_rows(c, t, r0, jj, sm, o, RPC, TR)
        # the merge's RoPE cos / sin (last 64 dims; position -> table row: two dependent loads) fetched now, used at the end
        rrow = r0 + fx.min(t // 8, I32(QV - 1)) // H   # (the merge: 8 lanes a qv)
        if fx.const_expr(TR % RPC != 0):
            rrow = fx.min(rrow, I32(TR - 1))
        rf0 = fx.max(jj * 32 - (HD - 64), I32(0)) // 2
        rpp = hx.gload(c["rpos"] + i64(rrow) * 8, fx.Int64)
        rcv = hx.gload(c["rcos"] + (rpp * 32 + i64(rf0 + 2 * (t % 8))) * 2, BF16, 2)   # this lane's 2 pairs
        rsv = hx.gload(c["rsin"] + (rpp * 32 + i64(rf0 + 2 * (t % 8))) * 2, BF16, 2)   # this lane's 2 pairs
        gpu.barrier()
        for rr in range_constexpr(RPC):
            for it in range(KC * HD // 8 // THREADS):   # runtime loops in D: smaller code (I-cache, see icache-probe)
                i = t + it * THREADS
                key = i // (HD // 8)
                cc = i % (HD // 8)
                idv = hx.lload(sm + o["ids"] + (rr * KC + key) * 4, I32)   # absolute pool row
                hx.async_to_lds_b128(c["kv"] + (i64(idv) * KG_ROW + i64(cc * 8)) * 2,
                                     sm + o["k"] + ((rr * 48 + key) * HP + cc * 8) * 2, 0)
        NZ = RPC * (48 - KC) * HD // 8
        for it in range(hx.opaque_s(I32((NZ + THREADS - 1) // THREADS))):
            i = t + it * THREADS
            if i < NZ:
                bb = i // ((48 - KC) * HD // 8)
                rem = i % ((48 - KC) * HD // 8)
                hx.lstore(sm + o["k"] + ((bb * 48 + KC + rem // (HD // 8)) * HP + (rem % (HD // 8)) * 8) * 2, fx.Vector.filled(4, 0, I32))
        if fx.const_expr("cflag" in c):   # every C CTA's q tiles: only the q copies wait for them
            hx.wait_flags(c["cflag"], CTAS, c["ctag"], w, l, 8)
            gpu.barrier()
        for it in range(QV * HD // 8 // THREADS):
            i = t + it * THREADS
            qv = i // (HD // 8)
            cc = i % (HD // 8)
            qrow = r0 + qv // H
            if fx.const_expr(TR % RPC != 0):
                qrow = fx.min(qrow, I32(TR - 1))
            hx.async_to_lds_b128(c["q"] + i64((qrow * H + qv % H) * HD + cc * 8) * 2,
                                 sm + o["q"] + (qv * HP + cc * 8) * 2, 0)
        pp = c["apart"] + i64(b * QV * HD) * 2
        hx.wait_asynccnt0()
        gpu.barrier()
        for tt in range(hx.opaque_s(I32(2))):   # (opaque trip count: kept a loop)
            tile = w + tt * 8
            if tile < RPC * 3:
                mi = tile // 3
                ni = tile % 3
                acc = fx.Vector.filled(8, 0.0, F32)
                for ks in range(HD // 32):
                    d0 = ks * 32 + (l // 16) * 16
                    av = hx.lload(sm + o["q"] + ((mi * H + l % 16) * HP + d0) * 2, BF16, 16)
                    bv = hx.lload(sm + o["k"] + ((mi * 48 + ni * 16 + l % 16) * HP + d0) * 2, BF16, 16)
                    acc = hx.wmma_bf16(av, bv, acc)
                for i in range_constexpr(8):
                    hx.lstore(sm + o["s"] + ((mi * H + (l // 16) * 8 + i) * 48 + ni * 16 + l % 16) * 4, acc[i] * c["scale"])
        gpu.barrier()
        sg = t % 5
        a5 = t < QV * 5
        sq = fx.min(t // 5, I32(QV - 1))
        svv = hx.lload(sm + o["s"] + (sq * 48 + sg * 8) * 4, F32, 8)
        sv = [svv[e] for e in range(8)]
        kvl = hx.lload(sm + o["kvl"] + (sq // H) * 4, I32)   # keys >= this row's kv_len take no weight
        kval = [(jj * KC + sg * 8 + e) < kvl for e in range(8)]
        sv = [kval[e].select(sv[e], F32(-1e30)) for e in range(8)]
        mloc = sv[0]
        for e in range_constexpr(1, 8):
            mloc = fx.max(mloc, sv[e])
        if a5:
            hx.lstore(sm + o["red"] + (sq * 5 + sg) * 4, mloc)
        gpu.barrier()
        if a5:
            m = hx.lload(sm + o["red"] + sq * 5 * 4, F32)
            for g in range_constexpr(1, 5):
                m = fx.max(m, hx.lload(sm + o["red"] + (sq * 5 + g) * 4, F32))
            pb = [expf(sv[e] - m).to(BF16) for e in range(8)]
            pb = [kval[e].select(pb[e], BF16(0.0)) for e in range(8)]   # keys >= kv_len: no weight
            ssum = F32(0.0)
            for e in range_constexpr(8):
                ssum = ssum + pb[e].to(F32)
            hx.lstore(sm + o["p"] + (sq * KP + sg * 8) * 2, fx.Vector.from_elements(pb, BF16))
            if sg == 0:
                z = fx.Vector.filled(4, 0, I32)
                hx.lstore(sm + o["p"] + (sq * KP + 40) * 2, z)
                hx.lstore(sm + o["p"] + (sq * KP + 48) * 2, z)
                hx.lstore(sm + o["p"] + (sq * KP + 56) * 2, z)
                hx.lstore(sm + o["m"] + sq * 4, m)
            hx.lstore(sm + o["red2"] + (sq * 5 + sg) * 4, ssum)
        gpu.barrier()
        if t < QV:
            s5 = F32(0.0)
            for g in range_constexpr(5):
                s5 = s5 + hx.lload(sm + o["red2"] + (t * 5 + g) * 4, F32)
            hx.lstore(sm + o["l"] + t * 4, s5)
        hh = l // 16
        for rr in range(hx.opaque_s(I32(RPC))):   # runtime loops over the rows and the 4 N tiles (code size; opaque trip
            # counts: LLVM would unroll constant ones back)
            av = []
            for ks in range_constexpr(2):
                p0 = hx.lload(sm + o["p"] + ((rr * H + l % 16) * KP + ks * 32 + 8 * hh) * 2, BF16, 8)
                p1 = hx.lload(sm + o["p"] + ((rr * H + l % 16) * KP + ks * 32 + 16 + 8 * hh) * 2, BF16, 8)
                av.append(fx.Vector.from_elements([p0[e] for e in range(8)] + [p1[e] for e in range(8)], BF16))
            for n in range(hx.opaque_s(I32(4))):
                nd = w * 4 + n
                acc = fx.Vector.filled(8, 0.0, F32)
                for ks in range_constexpr(2):
                    parts = []
                    for blk in range_constexpr(2):
                        if fx.const_expr(ks * 32 + blk * 16 >= 48):
                            # keys 48..63 (p = 0) lie past this row's 48 key slots, in the next row's keys or the q
                            # region: zeros, not those values (0 x NaN there made a NaN row poison its neighbour --
                            # across requests in the draft's 3-row clusters)
                            parts += [BF16(0.0)] * 8
                        else:
                            v8 = hx.ds_load_tr16(sm + o["k"] + ((rr * 48 + ks * 32 + blk * 16 + 8 * hh + l % 8) * HP + nd * 16 + 8 * ((l // 8) % 2)) * 2)
                            parts += [v8[e] for e in range(8)]
                    acc = hx.wmma_bf16(av[ks], fx.Vector.from_elements(parts, BF16), acc)
                hx.gstore(pp + i64((nd * 16 + l % 16) * QV + rr * H + (l // 16) * 8) * 2,
                          fx.Vector.from_elements([acc[i].to(BF16) for i in range(8)], BF16))
        gpu.barrier()
        if t < QV:
            hx.gstore(c["aml"] + i64(b * QV * 2 + t * 2) * 4, hx.lload(sm + o["m"] + t * 4, F32))
            hx.gstore(c["aml"] + i64(b * QV * 2 + t * 2 + 1) * 4, hx.lload(sm + o["l"] + t * 4, F32))
        if fx.const_expr("a1flag" in c):   # (design 8: A1) the partials out: one LL flag; the merge runs on the A2 CTAs
            hx.wait_storecnt0()
            gpu.barrier()
            if t == 0:
                hx.put(c["a1flag"] + i64(cl * CL + jj) * 8, I32(0), c["a1tag"])
        else:
            cluster_sync()
            _d_merge(c, cl, jj, rcv, rsv)


@ASTRewriter.transform
def _d_merge(c, cl, jj, rcv, rsv):
    """D's merge of row cluster cl, dims jj * 32 .. + 31 of every head: the CL key segments' partials (apart / aml)
    log-sum-exp merged with the sink, RoPE of the last 64 dims, MXFP8 -> o8 / os, the e6 flag."""
    t, w, l, sm, o = c["t"], c["w"], c["l"], c["sm"], c["o"]
    TR, RPC, QV = c["TR"], c["RPC"], c["QV"]
    r0 = cl * RPC
    c0 = cl * CL
    # merge: thread g < QV*4 owns dim dd = g // (QV/8) of this CTA's 32 and 8 consecutive qv: one b128 load a partial
    QW = 4 if QV * 32 // 4 <= THREADS else 8   # qv a thread (all 256 threads busy at 2 rows a cluster)
    NG8 = QV * 32 // QW
    QG = QV // QW
    g8 = fx.min(t, I32(NG8 - 1))
    dd8 = g8 // QG
    q08 = (g8 % QG) * QW
    pv8 = [hx.gload(c["apart"] + i64(((c0 + p) * HD + jj * 32 + dd8) * QV + q08) * 2, BF16, QW) for p in range(CL)]
    if t < QV:
        snk = hx.gload(c["sink"] + i64(t % H) * 4, F32)
        mp = [hx.gload(c["aml"] + i64((c0 + p) * QV * 2 + t * 2) * 4, F32) for p in range(CL)]
        lp = [hx.gload(c["aml"] + i64((c0 + p) * QV * 2 + t * 2 + 1) * 4, F32) for p in range(CL)]
        M = snk
        for p in range_constexpr(CL):
            M = fx.max(M, mp[p])
        den = expf(snk - M)
        for p in range_constexpr(CL):
            f = expf(mp[p] - M)
            hx.lstore(sm + o["wgt"] + (t * WG + p) * 4, f)
            den = den + lp[p] * f
        hx.lstore(sm + o["wgt"] + (t * WG + CL) * 4, F32(1.0) / den)
    gpu.barrier()
    ob = sm + o["k"]
    if t < NG8:
        for e in range_constexpr(QW):
            qv = q08 + e
            wq = [hx.lload(sm + o["wgt"] + (qv * WG + 4 * k) * 4, F32, 4) for k in range(CL // 4)]
            num = F32(0.0)
            for p in range_constexpr(CL):
                num = num + pv8[p][e].to(F32) * wq[p // 4][p % 4]
            hx.lstore(ob + (qv * 32 + dd8) * 4, (num * hx.lload(sm + o["wgt"] + (qv * WG + CL) * 4, F32)).to(BF16).to(F32))
    gpu.barrier()
    # RoPE (the last 64 dims) and MXFP8 of each qv's 32 dims in registers: 8 lanes a qv (4 dims = 2 RoPE pairs each; the
    # block amax over the 8 lanes by xor shuffles -- the same values / code / words as one thread's 32: bit-identical)
    assert QV * 8 <= THREADS
    qm = fx.min(t // 8, I32(QV - 1))
    pq = t % 8
    live_q = t < QV * 8
    if fx.const_expr(TR % RPC != 0):   # the last cluster's rows past TR: not stored
        live_q = live_q & (r0 + qm // H < TR)
    v4 = hx.lload(ob + (qm * 32 + pq * 4) * 4, F32, 4)
    x = [v4[e] for e in range(4)]
    rot = []
    for k in range_constexpr(2):
        cs = rcv[k].to(F32)
        sn = rsv[k].to(F32)
        rot.append((x[2 * k] * cs + x[2 * k + 1] * sn).to(BF16).to(F32))
        rot.append((F32(0.0) - x[2 * k] * sn + x[2 * k + 1] * cs).to(BF16).to(F32))
    is_rope = jj * 32 >= HD - 64
    vals = [is_rope.select(rot[e], x[e]) for e in range(4)]
    amax = fx.max(fx.max(abs(vals[0]), abs(vals[1])), fx.max(abs(vals[2]), abs(vals[3])))
    for sh in range_constexpr(3):
        amax = fx.max(amax, amax.shuffle_xor(1 << sh, 32))
    code = e8m0_of(amax)
    word = q8words(vals, inv_pow2(code))[0]
    if live_q:
        row = r0 + qm // H
        if pq == 0:
            c["stx"](c["os"] + i64(row * (NQ // 32) + (qm % H) * (HD // 32) + jj), code.to(fx.Int8))
        c["stx"](c["o8"] + i64(row * NQ + (qm % H) * HD + jj * 32 + pq * 4), word)
    if fx.const_expr("dflag" in c):   # (v3 e6) this (row cluster, key segment)'s o8 / os out: one LL flag
        hx.wait_storecnt0()
        gpu.barrier()
        if t == 0:
            hx.put(c["dflag"] + i64(cl * CL + jj) * 8, I32(0), c["dtag"])


@ASTRewriter.transform
def stage_D2(c):
    """(design 8: A2) CTA b in DCT .. 2 DCT - 1: the merge of (row cluster, dims slice) (b - DCT) // CL, (b - DCT) % CL
    once that cluster's CL A1 flags are out; its RoPE rows fetched before the wait."""
    t, b, w, l, TR, RPC, QV = c["t"], c["b"], c["w"], c["l"], c["TR"], c["RPC"], c["QV"]
    DCT = ((TR + RPC - 1) // RPC) * CL
    bb = b - DCT
    if (bb >= 0) & (bb < DCT):
        cl = bb // CL
        jj = bb % CL
        r0 = cl * RPC
        rrow = r0 + fx.min(t // 8, I32(QV - 1)) // H   # (the merge: 8 lanes a qv)
        if fx.const_expr(TR % RPC != 0):
            rrow = fx.min(rrow, I32(TR - 1))
        rf0 = fx.max(jj * 32 - (HD - 64), I32(0)) // 2
        rpp = hx.gload(c["rpos"] + i64(rrow) * 8, fx.Int64)
        rcv = hx.gload(c["rcos"] + (rpp * 32 + i64(rf0 + 2 * (t % 8))) * 2, BF16, 2)   # this lane's 2 pairs
        rsv = hx.gload(c["rsin"] + (rpp * 32 + i64(rf0 + 2 * (t % 8))) * 2, BF16, 2)   # this lane's 2 pairs
        hx.wait_flags(c["a1flag"] + i64(cl * CL) * 8, CL, c["a1tag"], w, l, 8)
        gpu.barrier()
        _d_merge(c, cl, jj, rcv, rsv)


# ------------------------------------------------------------------ E: wo_a (K quarters, cluster sum, MXFP8 y)
@ASTRewriter.transform
def stage_E(c):
    t, b, w, l, sm, MT, TR = c["t"], c["b"], c["w"], c["l"], c["sm"], c["MT"], c["TR"]
    cgp = b // 4
    kq = b % 4
    g = cgp // 32
    nb = cgp * 32
    if fx.const_expr("dflag" in c):   # (v3 e6) every (row cluster, key segment) of D: wave 0, lanes poll 3 flags each
        NDF = ((TR + c["RPC"] - 1) // c["RPC"]) * CL
        hx.wait_flags(c["dflag"], NDF, c["dtag"], w, l, 8)
        gpu.barrier()
    if (w < 6) & (w % 3 < MT):
        jt = w // 3
        mi = w % 3
        n0 = nb + jt * 16
        arow = c["o8"] + i64((mi * 16 + l % 16) * NQ + g * KWOA + kq * 1024)
        brow = c["woa"] + i64((n0 + l % 16) * KWOA + kq * 1024)

        def fa(i):
            kb = i * 128
            return ld_op(arow, kb, l), hx.gload(c["os"] + i64((mi * 16 + l % 16) * (NQ // 32) + (g * KWOA + kq * 1024 + kb) // 32), I32, align=4)

        def fb(i):
            kb = i * 128
            return ld_op(brow, kb, l), hx.gload(c["woa_s"] + i64(((n0 + l % 16) // 32) * (KWOA // 32) + (kq * 1024 + kb) // 32), I32, align=4)

        acc = mx_loop(8, fa, fb)
        yp = c["ypart"] + i64(b * T * 32) * 4
        for i in range_constexpr(8):
            hx.gstore(yp + i64((mi * 16 + (l // 16) * 8 + i) * 32 + jt * 16 + l % 16) * 4, acc[i])
    cluster_sync()
    if (t < 12 * 16) & (kq * 12 + t // 16 < TR):
        r = kq * 12 + t // 16
        cp = (t % 16) * 2
        v0 = F32(0.0)
        v1 = F32(0.0)
        for qq in range_constexpr(4):
            ypq = c["ypart"] + i64(((cgp * 4 + qq) * T + r) * 32 + cp) * 4
            v0 = v0 + hx.gload(ypq, F32)
            v1 = v1 + hx.gload(ypq + i64(4), F32)
        hx.lstore(sm + ((t // 16) * 32 + cp) * 4, v0.to(BF16).to(F32))
        hx.lstore(sm + ((t // 16) * 32 + cp + 1) * 4, v1.to(BF16).to(F32))
    gpu.barrier()
    if (t < 12) & (kq * 12 + t < TR):
        r = kq * 12 + t
        vq = [hx.lload(sm + (t * 32 + 4 * k) * 4, F32, 4) for k in range(8)]
        vals = [vq[e // 4][e % 4] for e in range(32)]
        amax = F32(0.0)
        for e in range_constexpr(32):
            amax = fx.max(amax, abs(vals[e]))
        code = e8m0_of(amax)
        inv = inv_pow2(code)
        c["stx"](c["ys"] + i64(r * (NWOA // 32) + cgp), code.to(fx.Int8))
        words = q8words([vals[e] for e in range(32)], inv)
        yb = c["y8"] + i64(r * NWOA + nb)
        c["stx"](yb, fx.Vector.from_elements(words[:4], I32))
        c["stx"](yb + i64(16), fx.Vector.from_elements(words[4:], I32))
    if fx.const_expr("eflag" in c):   # (v3 e7) this (col group, K quarter)'s y8 rows out: one LL flag
        hx.wait_storecnt0()
        gpu.barrier()
        if t == 0:
            hx.put(c["eflag"] + i64(b) * 8, I32(0), c["etag"])


# ------------------------------------------------------------------ F (4-GPU): 64 col groups (80 cols) x 4 K quarters, cluster sum
@ASTRewriter.transform
def stage_F_cluster(c):
    """CTA kq of group cg ends with its 20 cols x TR rows (fp32) in LDS [r][20] and in `out` (the TP partial)."""
    t, b, w, l, sm, MT, TR = c["t"], c["b"], c["w"], c["l"], c["sm"], c["MT"], c["TR"]
    cg = b // 4
    kq = b % 4
    cb = cg * 80
    op = c["opart"] + i64(b * T * 80) * 4
    if fx.const_expr("eflag" in c):   # (v3 e7) y8 cols 512 kq .. + 511: wo_a CTAs (cgp, q) with cgp in 16 kq .. 16 kq + 15
        hx.wait_flags(c["eflag"] + i64(kq * 64) * 8, 64, c["etag"], w, l, 8)   # (cgp, q) = 4 cgp + q: 64 consecutive flags
        gpu.barrier()
    for pp_ in range_constexpr(2):
        pr = w + pp_ * 8
        nt = pr // 3
        mi = pr % 3
        if (pr < 15) & (mi < MT):
            n0 = cb + nt * 16
            arow = c["y8"] + i64((mi * 16 + l % 16) * NWOA + kq * 512)
            brow = c["wob"] + i64((n0 + l % 16) * NWOA + kq * 512)

            def fa(i):
                kb = i * 128
                return ld_op(arow, kb, l), hx.gload(c["ys"] + i64((mi * 16 + l % 16) * (NWOA // 32) + (kq * 512 + kb) // 32), I32, align=4)

            def fb(i):
                kb = i * 128
                return ld_op(brow, kb, l), hx.gload(c["wob_s"] + i64(((n0 + l % 16) // 32) * (NWOA // 32) + (kq * 512 + kb) // 32), I32, align=4)

            acc = mx_loop(4, fa, fb)
            for i in range_constexpr(8):
                hx.gstore(op + i64((mi * 16 + (l // 16) * 8 + i) * 80 + nt * 16 + l % 16) * 4, acc[i])
    cluster_sync()
    for it in range_constexpr((TR * 20 + THREADS - 1) // THREADS):
        i = t + it * THREADS
        if i < TR * 20:
            r = i // 20
            cc = kq * 20 + i % 20
            v = [hx.gload(c["opart"] + i64(((cg * 4 + q) * T + r) * 80 + cc) * 4, F32) for q in range(4)]
            s = (v[0] + v[1]) + (v[2] + v[3])
            hx.gstore(c["out"] + i64(r * HID + cb + cc) * 4, s)
            hx.lstore(sm + (r * 20 + (cc - kq * 20)) * 4, s)


# ------------------------------------------------------------------ TP push-then-reduce (tagged b128, all-reduce)
@ASTRewriter.transform
def tp_allreduce(c, region, tag):
    """This CTA's 20 cols x TR rows (fp32 in LDS [r][20]) -> bf16 pairs, 3 pairs + tag a 16-byte word, pushed
    nontemporally to every peer's receive region (region * 4 + src); the 3 peers' words polled together and summed ->
    tpout and (device scope) row-major bf16 [row][5120] at ytgt (the next front's input y)."""
    t, b, TR = c["t"], c["b"], c["TR"]
    TP_TG = 40960
    TPG = (TR * 10 + 2) // 3   # ceil: TR * 10 words need not be a multiple of 3 (draft rows 5 a request)
    NWD = TR * 10
    rank = c["rank"]
    if t < TPG:
        lo, hi, wv = [], [], []
        for k in range_constexpr(3):
            wi = t * 3 + k
            lo.append(hx.lload(c["sm"] + ((wi // 10) * 20 + 2 * (wi % 10)) * 4, F32))
            hi.append(hx.lload(c["sm"] + ((wi // 10) * 20 + 2 * (wi % 10) + 1) * 4, F32))
            wv.append(lo[k].to(BF16).bitcast(fx.Int16).to(I32) & 0xFFFF | (hi[k].to(BF16).bitcast(fx.Int16).to(I32) << 16))
        v = fx.Vector.from_elements([wv[0], wv[1], wv[2], I32(tag)], I32)
        for o in range_constexpr(4):
            if o != rank:
                hx.gstore(c["recv"][o] + (i64((region * 4) * TP_TG) + i64(rank * TP_TG) + i64(b * TPG + t)) * 16, v, nt=True)
        # the 3 peers' tags polled together (3 loads in flight an iteration), then the 3 words loaded together
        sl = [lo[k].to(BF16).to(F32) for k in range(3)]
        sh = [hi[k].to(BF16).to(F32) for k in range(3)]
        gps = [c["recv"][rank] + (i64((region * 4) * TP_TG) + i64(o * TP_TG) + i64(b * TPG + t)) * 16 for o in range(4) if o != rank]
        t0 = hx.clock()
        while ((hx.gload(gps[0] + i64(12), I32, nt=True) != tag) | (hx.gload(gps[1] + i64(12), I32, nt=True) != tag)
               | (hx.gload(gps[2] + i64(12), I32, nt=True) != tag)) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
            hx.s_sleep(0)
        gots = [hx.gload(gp, I32, 4, nt=True) for gp in gps]
        for got in gots:
            for q in range_constexpr(3):
                sl[q] = sl[q] + (got[q] << 16).bitcast(F32)
                sh[q] = sh[q] + (got[q] & I32(-65536)).bitcast(F32)
        for q in range_constexpr(3):
            wq = sl[q].to(BF16).bitcast(fx.Int16).to(I32) & 0xFFFF | (sh[q].to(BF16).bitcast(fx.Int16).to(I32) << 16)
            hx.gstore(c["tpout"] + i64(b * TPG * 3 + t * 3 + q) * 4, wq)
            wi = t * 3 + q
            col = (b // 4) * 80 + (b % 4) * 20 + 2 * (wi % 10)
            if fx.const_expr(NWD % 3 == 0 or q == 0):
                hx.stx_dev(c["ytgt"] + (i64(wi // 10) * HID + i64(col)) * 2, wq)
            else:
                if wi < NWD:   # the last thread's padding words: no row
                    hx.stx_dev(c["ytgt"] + (i64(wi // 10) * HID + i64(col)) * 2, wq)


# ==============================================================================================================
# (merged from v3tail.py)
# (fx-v3) The attention tail as TileRT's mega_afm partitions it (design 6.7), c1 (one 16-row tile), TP4:
# 
# E4   wo_a: CTAs 192..255 = (group g, 32-row out tile) over the full K = 8 heads x 512 (the head sum in the CTA);
#      its weights read into VGPRs before the wait for D's flags; bf16, MXFP8 of the 32-col block -> y8 / ys, one LL
#      flag a tile.
# F3   wo_b: CTA b < NOWN = 160 owns hidden cols 32 b .. 32 b + 31 over the full K = 2048: wave w = (N tile, K quarter),
#      weights in VGPRs before the wait for the 32 fold flags; the quarters summed ((q0 + q1) + (q2 + q3)) -> f32 [r][32]
#      at LDS 0 (the TP payload) and `out`.
# TP3  the TP all-reduce of that [r][32] slice (tagged 16-byte pushes, as attention.tp_allreduce, 16 words a row).

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off


NOWN, OWC = 160, 32          # the column owners (wo_b / TP / mHC post): 160 CTAs x 32 hidden cols
GROUP_ROWS = NWOA // 2       # wo_a out rows a group (1024)


@ASTRewriter.transform
def stage_F3(c):
    """wo_b of hidden cols 32 b .. + 31 (b < NOWN), full K: c: t b w l sm y8 ys wob wob_s eflag etag out TR.
    Ends with the f32 [r][32] slice at LDS sm + 0 (TP3's payload)."""
    t, b, w, l, sm, TR = c["t"], c["b"], c["w"], c["l"], c["sm"], c["TR"]
    NEF = c.get("NEF", NWOA // 64)
    if b < NOWN:
        nt = w // 4
        kq = w % 4
        n0 = b * OWC + nt * 16
        brow = c["wob"] + i64((n0 + l % 16) * NWOA + kq * 512)
        Bw = [ld_op(brow, s * 128, l) for s in range(4)]
        Bs = [hx.gload(c["wob_s"] + i64(((n0 + l % 16) // 32) * (NWOA // 32) + (kq * 512 + s * 128) // 32), I32, align=4)
              for s in range(4)]
        hx.wait_flags(c["eflag"], NEF, c["etag"], w, l, 8)
        gpu.barrier()
        lr = fx.min(l % 16, I32(TR - 1))   # (rows past TR: the last row again, cached)
        arow = c["y8"] + i64(lr * NWOA + kq * 512)
        acc = fx.Vector.filled(8, 0.0, F32)
        for s in range_constexpr(4):
            av = ld_op(arow, s * 128, l)
            sa = hx.gload(c["ys"] + i64(lr * (NWOA // 32) + (kq * 512 + s * 128) // 32), I32, align=4)
            acc = hx.wmma_mx(av, sa, Bw[s], Bs[s], acc)
        hx.lstore(sm + 8192 + ((w * 32 + l) * 8) * 4, acc)
        gpu.barrier()
        if kq == 0:
            q = [hx.lload(sm + 8192 + (((nt * 4 + qq) * 32 + l) * 8) * 4, F32, 8) for qq in range(4)]
            for i in range_constexpr(8):
                r = (l // 16) * 8 + i
                s_ = (q[0][i] + q[1][i]) + (q[2][i] + q[3][i])
                if r < TR:
                    hx.lstore(sm + (r * OWC + nt * 16 + l % 16) * 4, s_)
                    hx.gstore(c["out"] + i64(r * HID + n0 + l % 16) * 4, s_)
        gpu.barrier()


@ASTRewriter.transform
def tp3(c, region, tag):
    """attention.tp_allreduce for the [r][32] slice of owner b < NOWN (16 bf16-pair words a row, 3 a 16-byte push)."""
    t, b, TR = c["t"], c["b"], c["TR"]
    TP_TG = 40960
    WPR = OWC // 2
    NWD = TR * WPR
    TPG = (NWD + 2) // 3
    rank = c["rank"]
    if (b < NOWN) & (t < TPG):
        lo, hi, wv = [], [], []
        for k in range_constexpr(3):
            wi = fx.min(t * 3 + k, I32(NWD - 1))
            lo.append(hx.lload(c["sm"] + ((wi // WPR) * OWC + 2 * (wi % WPR)) * 4, F32))
            hi.append(hx.lload(c["sm"] + ((wi // WPR) * OWC + 2 * (wi % WPR) + 1) * 4, F32))
            wv.append(lo[k].to(BF16).bitcast(fx.Int16).to(I32) & 0xFFFF | (hi[k].to(BF16).bitcast(fx.Int16).to(I32) << 16))
        v = fx.Vector.from_elements([wv[0], wv[1], wv[2], I32(tag)], I32)
        for o in range_constexpr(4):
            if o != rank:
                hx.gstore(c["recv"][o] + (i64((region * 4) * TP_TG) + i64(rank * TP_TG) + i64(b * TPG + t)) * 16, v, nt=True)
        sl = [lo[k].to(BF16).to(F32) for k in range(3)]
        sh = [hi[k].to(BF16).to(F32) for k in range(3)]
        gps = [c["recv"][rank] + (i64((region * 4) * TP_TG) + i64(o * TP_TG) + i64(b * TPG + t)) * 16 for o in range(4) if o != rank]
        t0 = hx.clock()
        while ((hx.gload(gps[0] + i64(12), I32, nt=True) != tag) | (hx.gload(gps[1] + i64(12), I32, nt=True) != tag)
               | (hx.gload(gps[2] + i64(12), I32, nt=True) != tag)) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
            hx.s_sleep(0)
        gots = [hx.gload(gp, I32, 4, nt=True) for gp in gps]
        for got in gots:
            for q in range_constexpr(3):
                sl[q] = sl[q] + (got[q] << 16).bitcast(F32)
                sh[q] = sh[q] + (got[q] & I32(-65536)).bitcast(F32)
        for q in range_constexpr(3):
            wq = sl[q].to(BF16).bitcast(fx.Int16).to(I32) & 0xFFFF | (sh[q].to(BF16).bitcast(fx.Int16).to(I32) << 16)
            wi = t * 3 + q
            col = b * OWC + 2 * (wi % WPR)
            if wi < NWD:
                hx.gstore(c["tpout"] + i64(b * TPG * 3 + t * 3 + q) * 4, wq)
                hx.stx_dev(c["ytgt"] + (i64(wi // WPR) * HID + i64(col)) * 2, wq)


E4_CTA0 = 192   # wo_a CTAs 192..255 (D runs on 0..95)


@ASTRewriter.transform
def stage_E4(c):
    """wo_a with the head sum inside the CTA: CTA eb = b - 192 < 64 = (group g = eb // 32, 32-row
    out tile tt = eb % 32) over the group's full K = 8 heads x 512; wave w = (16-row half w // 4, K quarter w % 4 of 1024 =
    8 K steps), its weights (128 VGPRs) read before the wait for D's flags; the quarters summed in order in LDS, bf16,
    the tile's MXFP8 block -> y8 / ys, one LL flag (g, tt). c: t b w l sm o8 os woa woa_s dflag dtag NDF eflag etag stx TR."""
    t, b, w, l, sm, TR = c["t"], c["b"], c["w"], c["l"], c["sm"], c["TR"]
    eb = b - E4_CTA0   # (CTAs past D's: their weights land while D runs)
    if (eb >= 0) & (eb < 64):
        g = eb // 32
        tt = eb % 32
        nt = w // 4
        kq = w % 4
        n0 = g * GROUP_ROWS + tt * 32 + nt * 16
        brow = c["woa"] + i64((n0 + l % 16) * KWOA + kq * 1024)
        Bw = [ld_op(brow, s * 128, l) for s in range(8)]
        Bs = [hx.gload(c["woa_s"] + i64(((n0 + l % 16) // 32) * (KWOA // 32) + (kq * 1024 + s * 128) // 32), I32, align=4)
              for s in range(8)]
        hx.wait_flags(c["dflag"], c["NDF"], c["dtag"], w, l, 8)
        gpu.barrier()
        lr = fx.min(l % 16, I32(TR - 1))   # (rows past TR: the last row again, cached)
        arow = c["o8"] + i64(lr * NQ + g * KWOA + kq * 1024)
        acc = fx.Vector.filled(8, 0.0, F32)
        for s in range_constexpr(8):
            av = ld_op(arow, s * 128, l)
            sa = hx.gload(c["os"] + i64(lr * (NQ // 32) + (g * KWOA + kq * 1024 + s * 128) // 32), I32, align=4)
            acc = hx.wmma_mx(av, sa, Bw[s], Bs[s], acc)
        hx.lstore(sm + ((w * 32 + l) * 8) * 4, acc)
        gpu.barrier()
        if kq == 0:
            q = [hx.lload(sm + (((nt * 4 + qq) * 32 + l) * 8) * 4, F32, 8) for qq in range(4)]
            for i in range_constexpr(8):
                r = (l // 16) * 8 + i
                s_ = (q[0][i] + q[1][i]) + (q[2][i] + q[3][i])
                if r < TR:
                    hx.lstore(sm + 8192 + (r * 32 + nt * 16 + l % 16) * 4, s_.to(BF16).to(F32))
        gpu.barrier()
        if t < TR:
            vq = [hx.lload(sm + 8192 + (t * 32 + 4 * k) * 4, F32, 4) for k in range(8)]
            vals = [vq[e // 4][e % 4] for e in range(32)]
            amax = F32(0.0)
            for e in range_constexpr(32):
                amax = fx.max(amax, abs(vals[e]))
            code = e8m0_of(amax)
            words = q8words(vals, inv_pow2(code))
            blk = g * (GROUP_ROWS // 32) + tt
            c["stx"](c["ys"] + i64(t * (NWOA // 32) + blk), code.to(fx.Int8))
            yb = c["y8"] + i64(t * NWOA + blk * 32)
            c["stx"](yb, fx.Vector.from_elements(words[:4], I32))
            c["stx"](yb + i64(16), fx.Vector.from_elements(words[4:], I32))
        hx.wait_storecnt0()
        gpu.barrier()
        if t == 0:
            hx.put(c["eflag"] + i64(eb) * 8, I32(0), c["etag"])
