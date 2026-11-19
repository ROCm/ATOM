# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The DSpark draft's context KV (models/deepseek_v41/dspark.py write_context_kv / project_context /
DraftAttention.project_context, unfused path) as whole-step kernel stages, every row of the target step:

    stage_tap   (a lane a (row, 32-col group)): aux = bf16(mean over the hc streams of target layers 37 / 38 / 39's
                input residual) [T][3 H], MXFP8 (group 32, ceil E8M0: quantize_fp8) -> aux8 / auxs
    stage_proj  (a wave a (16-col tile, 16-row tile)): main_proj (FP8 [H][3 H], 32x32 E8M0) x aux8 -> hpre bf16 [T][H]
    stage_hnorm (a wave a row): main_norm (RMSNorm, eps) -> h, MXFP8 -> h8 / hs
    stage_wkv   (a wave a (draft layer, 16-col tile, 16-row tile)): wkv_d (FP8 [512][H], 32x32 E8M0) x h8 -> kvp bf16
    stage_kvrow (a wave a (draft layer, row)): kv_norm_d (RMSNorm 512, eps) -> bf16, RoPE of dims 448..511 (pairs
                (2i, 2i+1), the row's cos / sin), QAT e4m3 (group 32, ceil E8M0, dequantized: quantize_fp8(dequantize))
                -> bf16 -> the draft layer's ring row kvw[row] (+ the layer's ring offset; -1: a pad row, not written)

Each stage is a grid-wide phase (a handoff between them in the step kernel).
The GEMV operand layouts are stage_C's (plain row-major FP8, 32x32 block scales, the checkpoint's own)."""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import e8m0_of, inv_pow2, ld_op, q8words, rsq

F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16
H, NTAP, KVD, ROPE0, NDL = D.HID, 3, D.KV_DIM, 448, D.DRAFT_LAYERS
KA = NTAP * H                       # 15360
WAVES, CTAS = D.WAVES, D.CTAS
EPS = D.RMS_EPS   # config rms_norm_eps


def i64(x):
    return fx.Int64(x)


def bfr(x):
    return x.to(BF16).to(F32)


def e4m3_round(v):
    """v (|v| <= 448) rounded to the nearest e4m3 value (ties to even): the cvt's value, as f32."""
    bits = v.bitcast(I32)
    e = fx.max(((bits >> 23) & 0xFF) - 127, I32(-6))
    step = ((e - 3 + 127) << 23).bitcast(F32)
    q = F32(hx.intrin(F32.ir_type, "llvm.rint.f32", [v / step])) * step
    return q


def _row_rms(vals, n):
    """sum of squares over the wave (each lane holds its share), rsqrt(mean + eps)."""
    s = F32(0.0)
    for v in vals:
        s = s + v * v
    for o in range_constexpr(5):
        s = s + s.shuffle_xor(16 >> o, 32)
    return rsq(s / F32(n) + F32(EPS))


@ASTRewriter.transform
def stage_hnorm(c, T, tp=False):
    """a CTA a row (CTAs 0..T-1), thread g < 160 a 32-col group: its 32 values (hpref f32, bf16-rounded as the bf16 hpre
    was), the row's sum of squares (wave xor tree, then the 5 waves' partials in LDS), main_norm -> bf16 -> MXFP8 -> h8 /
    hs. The atomic-sum columns of hpref (tiles 256..319, cols 4096..) are zeroed after the read for the next step.
    tp (stage_proj_tp): the 32 values are the owner rank's bf16 hpre words in this rank's receive half (_hpre_units)."""
    b, t, w, l, sm = c["b"], c["t"], c["w"], c["l"], c["sm"]
    if b < T:
        r = b
        g = fx.min(t, I32(H // 32 - 1))
        col = g * 32
        live = t < H // 32
        wv = [hx.gload(c["mnw"] + i64(col + 8 * k) * 2, BF16, 8) for k in range(4)]
        if fx.const_expr(tp):
            xs = _hpre_units(c, T, r, col)
        else:
            hv = [hx.gload(c["hpref"] + (i64(r) * H + i64(col + 4 * k)) * 4, F32, 4) for k in range(8)]
            xs = [bfr(hv[k // 4][k % 4]) for k in range(32)]
            if live & (col >= CTAS * 16):
                for k in range_constexpr(8):
                    hx.gstore(c["hpref"] + (i64(r) * H + i64(col + 4 * k)) * 4, fx.Vector.filled(4, 0.0, F32))
        ss = F32(0.0)
        for v in xs:
            ss = ss + v * v
        ss = live.select(ss, F32(0.0))
        for o in range_constexpr(5):
            ss = ss + ss.shuffle_xor(16 >> o, 32)
        if l == 0:
            hx.lstore(sm + w * 4, ss)
        gpu.barrier()
        tot = F32(0.0)
        for ww in range_constexpr(5):   # (waves 0..4 hold the 160 groups)
            tot = tot + hx.lload(sm + ww * 4, F32)
        rs = rsq(tot / F32(H) + F32(EPS))
        vals = []
        am = F32(0.0)
        for k in range_constexpr(32):
            v = bfr(xs[k] * rs * wv[k // 8][k % 8].to(F32))
            vals.append(v)
            am = fx.max(am, abs(v))
        if live:
            code = e8m0_of(am)
            words = q8words(vals, inv_pow2(code))
            dst = c["h8"] + i64(r) * H + i64(col)
            hx.gstore(dst, fx.Vector.from_elements(words[:4], I32))
            hx.gstore(dst + i64(16), fx.Vector.from_elements(words[4:], I32))
            hx.gstore(c["hs"] + i64(r) * (H // 32) + i64(col // 32), code.to(fx.Int8))
        gpu.barrier()


def ctx_fp8_lane16(ro, inv):
    """(FP4 KV) a context row's 16 dims (BF16 values; lanes 2g, 2g + 1 a 32-group, inv = 2^(127 - its code)) -> 4 words
    of E4M3 bytes: clamp(x * inv, +-448), RNE (quantize_fp8's values, the QAT's e4m3_round)."""
    cl = [fx.min(fx.max(ro[i] * inv, F32(-448.0)), F32(448.0)) for i in range(16)]
    return [hx.cvt_pk_fp8_w(hx.cvt_pk_fp8_w(0, cl[4 * h], cl[4 * h + 1], False), cl[4 * h + 2], cl[4 * h + 3], True)
            for h in range(4)]


@ASTRewriter.transform
def stage_kvrow(c, T):
    """a wave a (draft layer, row): lane l holds cols 16 l .. 16 l + 15 (half a 32-group: lanes 2g, 2g+1 share one)."""
    b, w, l = c["b"], c["w"], c["l"]
    task = b * WAVES + w
    if task < NDL * T:
        d = task // T
        r = task % T
        c0 = l * 16
        hv = [hx.gload(c["kvp"] + ((i64(d) * T + i64(r)) * KVD + i64(c0 + 8 * k)) * 2, BF16, 8) for k in range(2)]
        xs = [hv[k][e].to(F32) for k in range(2) for e in range(8)]
        rs = _row_rms(xs, KVD)
        nw = (d == 0).select(c["knw0"], (d == 1).select(c["knw1"], c["knw2"]))
        wv = [hx.gload(nw + i64(c0 + 8 * k) * 2, BF16, 8) for k in range(2)]
        nv = [bfr(xs[i] * rs * wv[i // 8][i % 8].to(F32)) for i in range(16)]
        is_r = c0 >= ROPE0
        ro = []
        for i in range_constexpr(0, 16, 2):
            fi = fx.max((c0 + i - ROPE0) // 2, I32(0))
            cs = hx.gload(c["rcos"] + i64(r * 32 + fi) * 4, F32)
            sn = hx.gload(c["rsin"] + i64(r * 32 + fi) * 4, F32)
            ro.append(is_r.select(bfr(nv[i] * cs - nv[i + 1] * sn), nv[i]))
            ro.append(is_r.select(bfr(nv[i] * sn + nv[i + 1] * cs), nv[i + 1]))
        am = F32(0.0)
        for i in range_constexpr(16):
            am = fx.max(am, abs(ro[i]))
        am = fx.max(am, am.shuffle_xor(1, 32))
        sc = ((e8m0_of(am)) << 23).bitcast(F32)          # 2^(code - 127)
        inv = inv_pow2(e8m0_of(am))
        qv = [e4m3_round(fx.min(fx.max(ro[i] * inv, F32(-448.0)), F32(448.0))) * sc for i in range(16)]
        kw = hx.gload(c["kvw"] + i64(r) * 4, I32)
        roff = (d == 0).select(c["roff0"], (d == 1).select(c["roff1"], c["roff2"]))
        if fx.const_expr(c["kvfp4"]):
            # FP4 KV: the stage's window ring holds the MXFP8 row itself (quantize_fp8(dequantize=False): the same E4M3
            # values and code): 16 bytes at row kw of its data plane, the code (even lanes) in its scale plane
            wdp = (d == 0).select(c["wd0"], (d == 1).select(c["wd1"], c["wd2"]))
            wsp = (d == 0).select(c["ws0"], (d == 1).select(c["ws1"], c["ws2"]))
            w8 = ctx_fp8_lane16(ro, inv)
            kr8 = i64(kw) + i64(roff)
            if kw >= 0:
                hx.gstore(wdp + kr8 * 512 + i64(c0), fx.Vector.from_elements(w8, I32))
            if (kw >= 0) & (l % 2 == 0):
                hx.gstore(wsp + kr8 * 16 + i64(l // 2), e8m0_of(am).to(fx.Int8))
        else:
            if kw >= 0:
                dst = c["ring"] + ((i64(kw) + i64(roff)) * KVD + i64(c0)) * 2
                hx.gstore(dst, fx.Vector.from_elements([qv[i].to(BF16) for i in range(8)], BF16))
                hx.gstore(dst + i64(16), fx.Vector.from_elements([qv[i].to(BF16) for i in range(8, 16)], BF16))


@ASTRewriter.transform
def stage_block_state(c, B, W):
    """build_block_state (model_ops/deepseek_v41/dspark.py): draft row r = (request r // W, block position r % W) gets
    hc_mult (4) copies of the anchor token's embedding (position 0; the full bf16 table on every rank: the same bytes
    the TP embedding all-reduce yields) or the noise row (positions 1..W-1); pre_mix [r][4] = (1, 0, 0, 0). For the
    FlyDSL front's fold (no post owed at stage 0): y = 0, post = 0, comb = identity, so R_new = R exactly.
    A lane a 8 columns of a row; c: aid (int32 [B]), emb (bf16 [V][H]), noise (bf16 [H]), R (bf16 [B W][4][H]),
    y (bf16 [B W][H]), pre / post (f32 [B W][4]), comb (f32 [B W][4][4])."""
    t, b = c["t"], c["b"]
    gi = b * 256 + t
    NC = H // 8
    if gi < B * W * NC:
        r = gi // NC
        col = (gi % NC) * 8
        q = r // W
        tok = hx.gload(c["aid"] + i64(q) * 4, I32)
        src = ((r % W) == 0).select(c["emb"] + i64(tok) * (H * 2), c["noise"])
        v = hx.gload(src + i64(col) * 2, BF16, 8)
        for s in range_constexpr(4):
            hx.gstore(c["R"] + ((i64(r) * 4 + i64(s)) * H + i64(col)) * 2, v)
        hx.gstore(c["y"] + (i64(r) * H + i64(col)) * 2, fx.Vector.from_elements([F32(0.0).to(BF16)] * 8, BF16))
        if col == 0:
            for s in range_constexpr(4):
                hx.gstore(c["pre"] + (i64(r) * 4 + i64(s)) * 4, F32(1.0 if s == 0 else 0.0))
                hx.gstore(c["post"] + (i64(r) * 4 + i64(s)) * 4, F32(0.0))
                for j in range_constexpr(4):
                    hx.gstore(c["comb"] + ((i64(r) * 4 + i64(s)) * 4 + i64(j)) * 4, F32(1.0 if s == j else 0.0))


@ASTRewriter.transform
def draft_meta(c, B, W):
    """The draft block's step metadata from this step's verify (write_draft_step_meta, dspark_draft_runner.backbone):
    request q's anchor is its verify row q (W + 1) + f (f = accepted drafts, verdict[0][q]) at target position
    rpos[q (W + 1) + f]; its W block rows sit at anchor + 1 .. anchor + W, write their keys at slot * slot_rows +
    pos % ring_slots of each draft layer's ring, and see the window max(anchor - window + 1, 0) .. anchor + W. A request
    whose slot is outside 0 .. num_slots - 1 writes and reads nothing. c: verdict (i32 [3][8]), rpos (i64 [T]), dgeom
    (i32: window, slot_rows, ring_slots, num_slots, -, -, -, -, then the B requests' slots), kmeta_d (i32 [B W][4]),
    kvw_d (i32 [B W]), rpos_d (i64 [B W]). One thread a draft row (CTA 0)."""
    t = c["t"]
    if t < B * W:
        q = t // W
        k = t % W
        g = c["dgeom"]
        window = hx.gload(g, I32)
        slot_rows = hx.gload(g + i64(4), I32)
        ring = hx.gload(g + i64(8), I32)
        nslots = hx.gload(g + i64(12), I32)
        slot = hx.gload(g + i64(32) + i64(q) * 4, I32)
        f = hx.gload(c["verdict"] + i64(q) * 4, I32)
        anchor = hx.gload(c["rpos"] + i64(q * (W + 1) + f) * 8, fx.Int64)
        live = (slot >= 0) & (slot < nslots)
        slot = live.select(slot, I32(0))
        pos = anchor + fx.Int64(1) + fx.Int64(k)
        hx.gstore(c["rpos_d"] + i64(t) * 8, pos)
        rel = live.select(slot * slot_rows + I32(pos % fx.Int64(ring)), I32(-1))
        hx.gstore(c["kvw_d"] + i64(t) * 4, rel)
        first = fx.max(anchor - fx.Int64(window) + fx.Int64(1), fx.Int64(0))
        cnt = live.select(I32(anchor + fx.Int64(W) - first + fx.Int64(1)), I32(0))
        hx.gstore(c["kmeta_d"] + i64(t) * 16, I32(first))
        hx.gstore(c["kmeta_d"] + i64(t) * 16 + i64(4), cnt)
        hx.gstore(c["kmeta_d"] + i64(t) * 16 + i64(8), I32(0))
        hx.gstore(c["kmeta_d"] + i64(t) * 16 + i64(12), slot * slot_rows)


@ASTRewriter.transform
def stage_tap1(c, T, j, src):
    """stage_tap for one target layer's input residual (src: bf16 [T][4][H], the attention front's Rout): its 160 groups
    of aux row r (cols j H ..). Runs right after the attention front that produced it (inside the target loop)."""
    t, b = c["t"], c["b"]
    gi = b * 256 + t
    NGR = H // 32
    if gi < T * NGR:
        r = gi // NGR
        col = (gi % NGR) * 32
        vals = []
        for k in range_constexpr(4):
            hv = [hx.gload(src + (i64(r) * 4 * H + i64(hc) * H + i64(col + 8 * k)) * 2, BF16, 8) for hc in range(4)]
            for e in range_constexpr(8):
                s = ((hv[0][e].to(F32) + hv[1][e].to(F32)) + hv[2][e].to(F32)) + hv[3][e].to(F32)
                vals.append(bfr(s * F32(0.25)))
        am = F32(0.0)
        for e in range_constexpr(32):
            am = fx.max(am, abs(vals[e]))
        code = e8m0_of(am)
        words = q8words(vals, inv_pow2(code))
        g = j * NGR + col // 32
        dst = c["aux8"] + i64(r) * KA + i64(g) * 32
        hx.gstore(dst, fx.Vector.from_elements(words[:4], I32))
        hx.gstore(dst + i64(16), fx.Vector.from_elements(words[4:], I32))
        hx.gstore(c["auxs"] + i64(r) * (KA // 32) + i64(g), code.to(fx.Int8))


def _gemv_part_m(c, a8, as_, K, w8, ws, n0, T, k0, nk, MT, depth=4):
    """_gemv_part for all MT row tiles at once: each weight operand loaded once, MT WMMAs (acc list)."""
    l = c["l"]
    ras = [fx.min(mi * 16 + l % 16, I32(T - 1)) for mi in range(MT)]
    brow = w8 + i64(n0 + l % 16) * K
    D = min(depth, nk)   # K steps in flight (the WMMAs still accumulate in K order)
    accs = [fx.Vector.filled(8, 0.0, F32) for _ in range(MT)]

    def fa(i, mi):
        kb = (k0 + i) * 128
        return ld_op(a8 + i64(ras[mi]) * K, kb, l), hx.gload(as_ + i64(ras[mi] * (K // 32) + kb // 32), I32, align=4)

    def fb(i):
        kb = (k0 + i) * 128
        return ld_op(brow, kb, l), hx.gload(ws + i64(((n0 + l % 16) // 32) * (K // 32) + kb // 32), I32, align=4)

    A = [[None] * MT for _ in range(D)]
    B_ = [None] * D
    for i in range(min(D, nk)):
        B_[i] = fb(i)
        A[i] = [fa(i, mi) for mi in range(MT)]
    hx.intrin(None, "llvm.amdgcn.sched.barrier", [I32(0)])
    for ks in range(0, nk, D):
        for i in range(D):
            if ks + i < nk:
                for mi in range(MT):
                    accs[mi] = hx.wmma_mx(A[i][mi][0], A[i][mi][1], B_[i][0], B_[i][1], accs[mi])
                if ks + i + D < nk:
                    B_[i] = fb(ks + i + D)
                    A[i] = [fa(ks + i + D, mi) for mi in range(MT)]
                    hx.intrin(None, "llvm.amdgcn.sched.barrier", [I32(0)])
    return accs


@ASTRewriter.transform
def _ntile_splitk(c, a8, as_, K, w8, ws, n0, T, out, N, orow0):
    """_tile_splitk over all row tiles of a 16-col tile (the weight read once)."""
    w, l, sm = c["w"], c["l"], c["sm"]
    MT = (T + 15) // 16
    KS = K // 128 // WAVES
    accs = _gemv_part_m(c, a8, as_, K, w8, ws, n0, T, w * KS, KS, MT)
    for mi in range_constexpr(MT):
        o = sm + ((mi * WAVES + w) * 32 + l) * 32
        hx.lstore(o, fx.Vector.from_elements([accs[mi][i] for i in range(4)], F32), align=16)
        hx.lstore(o + 16, fx.Vector.from_elements([accs[mi][i] for i in range(4, 8)], F32), align=16)
    gpu.barrier()
    if w < MT:
        tot = [F32(0.0)] * 8
        for ww in range_constexpr(WAVES):
            lo = hx.lload(sm + ((w * WAVES + ww) * 32 + l) * 32, F32, 4, align=16)
            hi = hx.lload(sm + ((w * WAVES + ww) * 32 + l) * 32 + 16, F32, 4, align=16)
            tot = [tot[i] + lo[i] for i in range(4)] + [tot[i + 4] + hi[i] for i in range(4)]
        for i in range_constexpr(8):
            r = w * 16 + (l // 16) * 8 + i
            if r < T:
                hx.gstore(out + ((i64(orow0) + i64(r)) * N + i64(n0 + l % 16)) * 2, tot[i].to(BF16))
    gpu.barrier()


@ASTRewriter.transform
def stage_proj_skb(c, T):
    """main_proj -> hpref (f32 [T][H]): the 320 N tiles as 1.25 tiles a CTA. Tile b (< 256) whole (the 8 waves split K,
    LDS sum, plain f32 stores); tiles 256..319 in 4 K quarters, CTA b takes (256 + b // 4, quarter b % 4): waves 0..5
    five K128 steps each, LDS sum, f32 atomic adds (stage_hnorm zeroes those columns after reading them)."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    MT = (T + 15) // 16
    KS = KA // 128 // WAVES   # 15
    accs = _gemv_part_m(c, c["aux8"], c["auxs"], KA, c["mpw"], c["mps"], b * 16, T, w * KS, KS, MT)
    _lds_sum_out(c, accs, MT, WAVES, T, b * 16, False)
    n1 = (CTAS + b // 4) * 16
    kq = b % 4
    QS = KA // 128 // 4 // 6  # 5
    accq = _gemv_part_m(c, c["aux8"], c["auxs"], KA, c["mpw"], c["mps"], n1, T, kq * (KA // 128 // 4) + fx.min(w, I32(5)) * QS, QS, MT)
    if w >= 6:   # (waves 6, 7 computed a duplicate of wave 5's steps: zeroed)
        accq = [fx.Vector.filled(8, 0.0, F32) for _ in range(MT)]
    _lds_sum_out(c, accq, MT, WAVES, T, n1, True)


@ASTRewriter.transform
def _lds_sum_out(c, accs, MT, NW, T, n0, atomic):
    """the NW waves' partial accumulators of a 16-col tile summed (LDS) -> hpref (plain stores or f32 atomic adds)."""
    w, l, sm = c["w"], c["l"], c["sm"]
    for mi in range_constexpr(MT):
        o = sm + ((mi * WAVES + w) * 32 + l) * 32
        hx.lstore(o, fx.Vector.from_elements([accs[mi][i] for i in range(4)], F32), align=16)
        hx.lstore(o + 16, fx.Vector.from_elements([accs[mi][i] for i in range(4, 8)], F32), align=16)
    gpu.barrier()
    if w < MT:
        tot = [F32(0.0)] * 8
        for ww in range_constexpr(NW):
            lo = hx.lload(sm + ((w * WAVES + ww) * 32 + l) * 32, F32, 4, align=16)
            hi = hx.lload(sm + ((w * WAVES + ww) * 32 + l) * 32 + 16, F32, 4, align=16)
            tot = [tot[i] + lo[i] for i in range(4)] + [tot[i + 4] + hi[i] for i in range(4)]
        for i in range_constexpr(8):
            r = w * 16 + (l // 16) * 8 + i
            if r < T:
                dst = c["hpref"] + (i64(r) * H + i64(n0 + l % 16)) * 4
                if fx.const_expr(atomic):
                    hx.atomic_fadd(dst, tot[i])
                else:
                    hx.gstore(dst, tot[i])
    gpu.barrier()


@ASTRewriter.transform
def stage_wkv_skb(c, T):
    b = c["b"]
    NT = NDL * (KVD // 16)
    for it in range((NT - b + CTAS - 1) // CTAS):
        task = b + it * CTAS
        d = task // (KVD // 16)
        wk = (d == 0).select(c["wkv0"], (d == 1).select(c["wkv1"], c["wkv2"]))
        wks = (d == 0).select(c["wks0"], (d == 1).select(c["wks1"], c["wks2"]))
        _ntile_splitk(c, c["h8"], c["hs"], H, wk, wks, (task % (KVD // 16)) * 16, T, c["kvp"], KVD, d * T)


# ---------------------------------------------------------------- the step tail block (after the target layers)
CTPTRS = ['aux8', 'auxs', 'mpw', 'mps', 'mnw', 'hpre', 'h8', 'hs', 'wkv0', 'wkv1', 'wkv2', 'wks0', 'wks1', 'wks2',
          'knw0', 'knw1', 'knw2', 'kvp', 'rcos', 'rsin', 'kvw', 'ring', 'roff', 'hpref',
          'wd0', 'wd1', 'wd2', 'ws0', 'ws1', 'ws2']   # roff: int32 [3] ring row offsets; wd* / ws*: FP4 KV's ring planes (else 0)


def ctptr(tab, name):
    """pointer `name` of the context-KV table (wave-uniform)."""
    v = hx.gload(tab + i64(CTPTRS.index(name) * 8), I32, 2)
    rf = lambda x: I32(hx.intrin(hx.i32(), "llvm.amdgcn.readfirstlane.i32", [I32(x)]))
    return (fx.Int64(rf(v[1])) << 32) | (fx.Int64(rf(v[0])) & fx.Int64(0xFFFFFFFF))


# (ctx TP) main_proj split by output columns over the TP ranks: rank r computes the N tiles [r CX_TILES, (r + 1) CX_TILES)
# (each output's K sum exactly as before: a tile whole on a CTA, the 8 waves' K slices summed in LDS in wave order) and
# pushes them, bf16 (the value stage_hnorm rounds them to), to every rank (itself included) as tagged 16-byte units:
# unit ((tile * T + row) * 3 + j) of source slot r = columns 6 j .. 6 j + 5 of that tile row (3 words of bf16 pairs; j = 2:
# columns 12..15, the last word padding). The slots live in each rank's receive half at CX_OFF (zeroed with the half a
# step; tag: the step epoch's), between the sampling slots and the Markov words (tail.SOFF .. SOFF2).
TPR = 4
CX_TILES = H // 16 // TPR                     # 80 N tiles a rank
CX_TMAX = 8                                   # rows a step (c1: 6)
CX_SRC = CX_TILES * CX_TMAX * 3 * 16          # 30720 bytes a source slot
CX_OFF = 16 * 40960 * 16 + (512 << 10)        # bytes into a receive half (tail.SOFF + 512 KB)
CX_DEPTH = 6                                  # K steps in flight a wave (latency-bound: 4 took 5.1-5.6 us a tile)
CX_STAGE = 8192                               # LDS bytes: a tile's [T][16] bf16, past _lds_sum_out's partials


def _cx_unit(base, src, tile, T, r, j):
    return base + i64(CX_OFF) + i64(src * CX_SRC) + (i64(tile * T + r) * 3 + i64(j)) * 16


@ASTRewriter.transform
def poll_units(addrs, tag):
    """The 16-byte units at addrs (tag in word 3), polled together: (bounded: POLL_TICKS; a timeout marks STOP, as
    sample_common.poll_unit) -> their 3 data words each."""
    n = len(addrs)
    t0 = hx.clock()
    bad = I32(0)
    for i in range_constexpr(n):
        bad = bad + (hx.gload(addrs[i] + i64(12), I32, nt=True) != tag).select(I32(1), I32(0))
    st_ = I32(0)
    while (bad != 0) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)) & (st_ == I32(0)):
        hx.s_sleep(0)
        st_ = hx._stop_due(st_, t0)
        bad = I32(0)
        for i in range_constexpr(n):
            bad = bad + (hx.gload(addrs[i] + i64(12), I32, nt=True) != tag).select(I32(1), I32(0))
    hx._stop_mark((bad != 0) & (st_ == I32(0)))
    return [hx.gload(addrs[i], I32, 4, nt=True) for i in range(n)]


def _hpre_units(c, T, r, col):
    """Row r's bf16 hpre at columns col .. col + 31 (one owner rank: 1280 columns a rank, 32 | 1280) from the receive half
    -> 32 f32 (exactly bfr of the owner's f32 sums)."""
    own = col // (CX_TILES * 16)
    t0 = (col % (CX_TILES * 16)) // 16
    addrs = [_cx_unit(c["rmine"], own, t0 + q // 3, T, r, q % 3) for q in range(6)]
    us = poll_units(addrs, c["cxtag"])
    xs = []
    for k in range(32):
        cc = k % 16
        wd = us[(k // 16) * 3 + cc // 6][(cc % 6) // 2]
        xs.append(((wd << 16) if cc % 2 == 0 else (wd & I32(-65536))).bitcast(F32))
    return xs


@ASTRewriter.transform
def _lds_sum_push(c, accs, MT, T, tile, tl):
    """_lds_sum_out's wave sum of a 16-col tile (n0 = 16 tile), then its [T][16] bf16 pushed to the 4 ranks' source slot
    of this rank (local tile tl): thread j3 < 3 T a unit (row j3 // 3, columns 6 (j3 % 3) ..)."""
    w, l, sm, t = c["w"], c["l"], c["sm"], c["t"]
    for mi in range_constexpr(MT):
        o = sm + ((mi * WAVES + w) * 32 + l) * 32
        hx.lstore(o, fx.Vector.from_elements([accs[mi][i] for i in range(4)], F32), align=16)
        hx.lstore(o + 16, fx.Vector.from_elements([accs[mi][i] for i in range(4, 8)], F32), align=16)
    gpu.barrier()
    if w < MT:
        tot = [F32(0.0)] * 8
        for ww in range_constexpr(WAVES):
            lo = hx.lload(sm + ((w * WAVES + ww) * 32 + l) * 32, F32, 4, align=16)
            hi = hx.lload(sm + ((w * WAVES + ww) * 32 + l) * 32 + 16, F32, 4, align=16)
            tot = [tot[i] + lo[i] for i in range(4)] + [tot[i + 4] + hi[i] for i in range(4)]
        for i in range_constexpr(8):
            r = w * 16 + (l // 16) * 8 + i
            if r < T:
                hx.lstore(sm + CX_STAGE + (r * 16 + l % 16) * 2, tot[i].to(BF16))
    gpu.barrier()
    if t < 3 * T:
        rr = t // 3
        j = t % 3
        wd = hx.lload(sm + CX_STAGE + rr * 32 + j * 12, I32, 3, align=4)
        v = fx.Vector.from_elements([wd[0], wd[1], wd[2], c["cxtag"]], I32)
        for o in range_constexpr(TPR):
            hx.gstore(_cx_unit(c["recv"][o], c["rank"], tl, T, rr, j), v, nt=True)
    # (no barrier here: its store drain waited ~3.5 us for the pushes to land before the next tile. The consumer
    # validates each 16-byte unit by its own tag, the receive half is re-zeroed only after its owner polled every unit,
    # the partials were read before the barrier above and the staging is read and next written by wave 0 alone; the drain
    # moves to the next barrier -- the next tile's, or stage_hnorm's -- by then long landed)


@ASTRewriter.transform
def stage_proj_tp(c, T, nc):
    """main_proj's N tiles of this rank (c["rank"] * CX_TILES + tl, tl < CX_TILES) on nc CTAs, tl = vb + k nc (module
    comment above _cx_unit): each pushed to every rank (_lds_sum_push), read by stage_hnorm(tp=True) there."""
    b, w = c["b"], c["w"]
    MT = (T + 15) // 16
    KS = KA // 128 // WAVES   # 15
    for kk in range_constexpr((CX_TILES + nc - 1) // nc):
        tl = b + kk * nc
        if tl < CX_TILES:
            tile = c["rank"] * CX_TILES + tl
            accs = _gemv_part_m(c, c["aux8"], c["auxs"], KA, c["mpw"], c["mps"], tile * 16, T, w * KS, KS, MT, depth=CX_DEPTH)
            _lds_sum_push(c, accs, MT, T, tile, tl)


@ASTRewriter.transform
def stage_proj_nc(c, T, nc):
    """main_proj -> hpref on nc CTAs (virtual ids c["b"] 0..nc-1): the 320 N tiles whole, tile vb + k nc, the 8 waves
    splitting K (LDS sum, plain f32 stores)."""
    b, w = c["b"], c["w"]
    MT = (T + 15) // 16
    KS = KA // 128 // WAVES   # 15
    NTL = H // 16
    for kk in range_constexpr((NTL + nc - 1) // nc):
        tile = b + kk * nc
        if tile < NTL:
            accs = _gemv_part_m(c, c["aux8"], c["auxs"], KA, c["mpw"], c["mps"], tile * 16, T, w * KS, KS, MT, depth=CX_DEPTH)
            _lds_sum_out(c, accs, MT, WAVES, T, tile * 16, False)


@ASTRewriter.transform
def stage_wkv_nc(c, T, nc):
    """stage_wkv_skb's tasks strided by nc (virtual ids c["b"])."""
    b = c["b"]
    NT = NDL * (KVD // 16)
    for kk in range_constexpr((NT + nc - 1) // nc):
        task = b + kk * nc
        if task < NT:
            d = task // (KVD // 16)
            wk = (d == 0).select(c["wkv0"], (d == 1).select(c["wkv1"], c["wkv2"]))
            wks = (d == 0).select(c["wks0"], (d == 1).select(c["wks1"], c["wks2"]))
            _ntile_splitk(c, c["h8"], c["hs"], H, wk, wks, (task % (KVD // 16)) * 16, T, c["kvp"], KVD, d * T)


def ctx_split(tab, t, b, T, cnt, e0, nc, kv_fp4=False, recv=None, rank=None, tag=None):
    """The draft context KV on CTAs e0 .. e0 + nc - 1 alone (while the others sample: tail.target): main_proj -> main_norm
    -> 3 x wkv -> kv_norm / RoPE / QAT -> ring, the stages separated by nc-CTA handoffs on cnt, cnt + 4, cnt + 8 (zeroed
    a step). The CTAs' virtual ids b - e0 take ctx_tail's roles (hnorm: a row a CTA; kvrow: a wave a (layer, row)).
    recv (the 4 ranks' receive halves of this step), rank, tag: main_proj split over the ranks (stage_proj_tp: each rank's
    columns pushed to every rank, polled by stage_hnorm -- no handoff between the two)."""
    c = {k: ctptr(tab, k) for k in CTPTRS if k != 'roff'}
    ro = ctptr(tab, 'roff')
    c.update(t=t, b=b - e0, w=t // 32, l=t % 32, sm=hx.lds_base(), roff0=hx.gload(ro, I32), roff1=hx.gload(ro + i64(4), I32),
             roff2=hx.gload(ro + i64(8), I32), kvfp4=kv_fp4)
    if recv is not None:
        assert T <= CX_TMAX
        c.update(recv=recv, rank=rank, rmine=recv[rank], cxtag=tag)
        stage_proj_tp(c, T, nc)
        stage_hnorm(c, T, tp=True)
    else:
        stage_proj_nc(c, T, nc)
        hx.handoff(cnt, t, False, nc)
        stage_hnorm(c, T)
    hx.handoff(cnt + i64(4), t, False, nc)
    stage_wkv_nc(c, T, nc)
    hx.handoff(cnt + i64(8), t, False, nc)
    stage_kvrow(c, T)


def ctx_tail(tab, t, b, T, cnt, handoff, hdev=False, kv_fp4=False):
    """main_proj -> main_norm -> 3 x wkv -> kv_norm / RoPE / QAT -> ring (aux8 / auxs written by stage_tap1 in the target
    loop). cnt: 3 zeroed handoff counters (16 B apart)."""
    c = {k: ctptr(tab, k) for k in CTPTRS if k != 'roff'}
    ro = ctptr(tab, 'roff')
    c.update(t=t, b=b, w=t // 32, l=t % 32, sm=hx.lds_base(), roff0=hx.gload(ro, I32), roff1=hx.gload(ro + i64(4), I32),
             roff2=hx.gload(ro + i64(8), I32), kvfp4=kv_fp4)
    stage_proj_skb(c, T)   # split-K GEMV tiles, a weight operand shared by all row tiles
    handoff(cnt, t, hdev)
    stage_hnorm(c, T)
    handoff(cnt + i64(16), t, hdev)
    stage_wkv_skb(c, T)
    handoff(cnt + i64(32), t, hdev)
    stage_kvrow(c, T)
