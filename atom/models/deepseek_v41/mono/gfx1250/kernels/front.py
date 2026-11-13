# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The mHC front of the decode layer (FFN front and attention front, one copy of the code: front_any).

96 CTAs = 6 groups x 16 positions, one 16-CTA hw cluster a group; CTA (g, p) owns tokens 8g..8g+7 and hidden cols
[320p, 320p+320):
  A  R'_j = bf16(post_j*y + sum_h comb[h][j]*R_h), x1 = bf16(sum_j pre_j*R'_j), square sums      (mhc_fused_post_pre_delayed)
     (an attention front before an Engram layer: + gate . value, _engram_gate)
  B  mixes partial (bf16 WMMA, R' tokens as M x fn hi/lo outputs as N, K = 4 streams x 320)
  -- cluster barrier 1
  C  normed = bf16(x1 * rsqrt(mean x1^2 + eps) * w) -> MXFP8 + bf16 (block codes: int32 attention front, int8 FFN front)
  FFN front only:
  D  router partial (this CTA's gate columns, all loads issued first)
  -- cluster barrier 2
  E1 every CTA: experts 24p..24p+23 of its 8 tokens: logits sum (16 parts), sqrtsoftplus, local top-6 (64-bit keys)
  -- cluster barrier 3
  E2 CTA p=0 (FFN front): merge 16 x 6 candidates a token -> ids, weights (renorm x1.5); CTA p=1: Sinkhorn of the 8
     tokens (wave = token, lane = comb element)

`c` keys: t b w l sm (traced); pointers (Int64): R y pre post comb fnh fnl hsc hbs nw gate gbias Rout xn x8 xs part plog
cand cands npre npost ncomb ids wts; rms_eps hc_eps norm_eps (python floats); runtime flags isffn xs32 merge first last;
rk (the residual streams' LDS region, kept between the fronts); optional eng engkv enggw engpart (Engram).
"""
from __future__ import annotations


import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import q8words, bf, cluster_sync, e8m0_of, inv_pow2, pack4, rsq, to_e4m3
from atom.models.deepseek_v41.mono.gfx1250.kernels.moe import sched_barrier

T, H, NG, TG, CL, NE, TOPK = D.T_MAX, D.HID, 6, 8, D.CL, D.EXPERTS, D.TOPK


def router(c):
    """The router shape of front context c (c["ne"] experts, top c["topk"]; the target's 384 / 6 by default, the draft
    blocks' 128 / 3): (NE, TOPK, experts a CTA (E1), 32-lane rounds of the CL x TOPK candidates (E2))."""
    ne, topk = c.get("ne", NE), c.get("topk", TOPK)
    assert ne % CL == 0 and ne // CL <= 32 and CL * topk <= 96
    return ne, topk, ne // CL, (CL * topk + 31) // 32


COLS = H // CL          # 320
CP = COLS + 8           # padded LDS row (bf16)
FRONT_CTAS = NG * CL    # 96
F32, I32, I64, BF16 = fx.Float32, fx.Int32, fx.Int64, fx.BFloat16
# LDS layout (bytes)
O_RP = 0                                  # R' bf16 [16][4][CP] (rows 8..15 zero: the WMMA M padding)
O_XN = O_RP + 16 * 4 * CP * 2             # normed bf16 [16][CP] (rows 8..15 zero)
O_X1 = O_XN + 16 * CP * 2                 # x1 bf16 [8][CP]
O_RED = O_X1 + 8 * CP * 2                 # f32 [8 waves][2 tiles][32 lanes][8]
O_RSTD = O_RED + 8 * 2 * 32 * 8 * 4       # f32 [8]
FRONT_LDS = O_RSTD + 64
RK_BYTES = 8 * 4 * CP * 2                 # residual streams kept in LDS between fronts


def _stx(c, addr, val):
    """Exchange store of an output other CTAs read after a handoff: device scope (hx.stx_dev) in the fence-free step
    kernel (c["hdev"]), else hx.gstore. (A bool in `c`: a function there breaks the AST rewriter's scf.if states.)"""
    (hx.stx_dev if c.get("hdev", False) else hx.gstore)(addr, val)


def i64(x):
    return fx.Int64(x)


def frcp(x):
    return F32(hx.intrin(F32.ir_type, "llvm.amdgcn.rcp.f32", [x]))


def dv(a, b):
    """a / b as a * rcp(b) (the Sinkhorn divisions)."""
    return a * frcp(b)


def fexp2(x):
    """e^x with the raw v_exp_f32."""
    return F32(hx.intrin(F32.ir_type, "llvm.amdgcn.exp2.f32", [x * F32(1.4426950408889634)]))


def fexp(x):
    return F32(hx.intrin(F32.ir_type, "llvm.exp.f32", [x]))


def fsqrt(x):
    return F32(hx.intrin(F32.ir_type, "llvm.sqrt.f32", [x]))


def fma(a, b, c):
    return F32(hx.intrin(F32.ir_type, "llvm.fma.f32", [a, b, c]))


def shx(v, o):
    """xor-shuffle of an i32 across the wave (through f32 bits)."""
    return v.bitcast(F32).shuffle_xor(o, 32).bitcast(I32)


def key64(hi, lo):
    return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))





@ASTRewriter.transform
def _engram_gate(c, rq, tok, g, p, w, l, c0, cs, seg2):
    """The Engram gate of an attention front before an Engram layer (engram; gate.engram_post_wkv_reference):
    per branch j, f32 R'_j and key_j, rstd = rsqrt(mean R'^2 + 1e-20) rsqrt(mean key^2 + 1e-20), dot = sum(R' gw key)
    rstd / sqrt(H), gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))); returns R' <- bf16(R' + gate value). The three
    sums of a (token, branch) over the cluster's 16 x 320 columns: per-wave partials -> engpart -> cluster sync -> the 16
    records summed (same order on every CTA)."""
    ekv = c["engkv"] + i64(fx.min(tok, I32(c["TR"] - 1)) * (5 * H)) * 2
    kv = [[hx.gload(ekv + i64(h * H + c0 + cs[s_]) * 2, BF16, 8, align=16) for s_ in range(2)] for h in range(5)]
    gw = [[[hx.gload(c["enggw"] + i64(h * H + c0 + cs[s_] + 4 * u) * 4, F32, 4, align=16) for u in range(2)]
           for s_ in range(2)] for h in range(4)]
    pr = [F32(0.0)] * 12   # per branch: sum R'^2, sum key^2, sum R' gw key
    for s_ in range_constexpr(2):
        on = (s_ == 0) | seg2
        for e in range_constexpr(8):
            for j in range_constexpr(4):
                rf = rq[j][s_][e]
                kf = kv[j][s_][e].to(F32)
                gwe = gw[j][s_][e // 4][e % 4]
                pr[3 * j] = pr[3 * j] + on.select(rf * rf, F32(0.0))
                pr[3 * j + 1] = pr[3 * j + 1] + on.select(kf * kf, F32(0.0))
                pr[3 * j + 2] = pr[3 * j + 2] + on.select(rf * gwe * kf, F32(0.0))
    for o in range_constexpr(5):
        pr = [v + v.shuffle_xor(16 >> o, 32) for v in pr]
    rec = c["engpart"] + i64(((g * CL + p) * TG + w) * 16) * 4
    if l == 0:
        for k in range_constexpr(3):
            hx.gstore(rec + i64(16 * k), fx.Vector.from_elements(pr[4 * k:4 * k + 4], F32))
    cluster_sync()
    src = c["engpart"] + i64(((g * CL + fx.min(l, I32(CL - 1))) * TG + w) * 16) * 4
    got = [hx.gload(src + i64(16 * k), F32, 4) for k in range(3)]
    tot = [(l < CL).select(got[k // 4][k % 4], F32(0.0)) for k in range(12)]
    for o in range_constexpr(5):   # (lanes >= 16 hold 0: every lane gets the 16 records' sum)
        tot = [v + v.shuffle_xor(16 >> o, 32) for v in tot]
    live = tok < c["TR"]
    gate = []
    for j in range_constexpr(4):
        rstd = rsq(tot[3 * j] * F32(1.0 / H) + F32(D.RMS_EPS)) * rsq(tot[3 * j + 1] * F32(1.0 / H) + F32(D.RMS_EPS))
        dot = tot[3 * j + 2] * rstd * F32(1.0 / (H ** 0.5))
        mag = fsqrt(fx.max(abs(dot), F32(1e-6)))
        sg = (dot < F32(0.0)).select(-mag, mag)
        gate.append(live.select(frcp(F32(1.0) + fexp(-sg)), F32(0.0)))
    return [[[bf(rq[j][s_][e] + gate[j] * kv[4][s_][e].to(F32)).to(F32) for e in range(8)] for s_ in range(2)]
            for j in range(4)]


@ASTRewriter.transform
def front_A(c):
    """thread = token w, cols l*8..+8 and (lanes < 8) 256 + l*8..+8: 16-byte loads / stores. The residual streams stay in
    this CTA's LDS between the fronts (c["rk"]): from global at the first layer, to global (Rout) after the last."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    g = b // CL
    p = b % CL
    c0 = p * COLS
    tok = g * TG + w
    q = [hx.gload(c["pre"] + i64(tok * 4 + j) * 4, F32) for j in range(4)]
    pp = [hx.gload(c["post"] + i64(tok * 4 + j) * 4, F32) for j in range(4)]
    cm = [[hx.gload(c["comb"] + i64(tok * 16 + h * 4 + j) * 4, F32) for j in range(4)] for h in range(4)]
    seg2 = l < 8
    cs = [l * 8, 256 + fx.min(l, I32(7)) * 8]
    rv = [[hx.lload(c["rk"] + ((w * 4 + h) * CP + cs[s_]) * 2, BF16, 8, align=16) for s_ in range(2)] for h in range(4)]
    if c["first"]:   # layer 0: from global (the carried list is replaced)
        rv = [[hx.gload(c["R"] + i64((tok * 4 + h) * H + c0 + cs[s_]) * 2, BF16, 8, align=16) for s_ in range(2)] for h in range(4)]
    yv = [hx.gload(c["y"] + i64(tok * H + c0 + cs[s_]) * 2, BF16, 8, align=16) for s_ in range(2)]
    sched_barrier()   # every load issued before the math (in the whole-layer kernel the scheduler split them in two batches)
    # R'_j = bf16(post_j y + sum_h comb[h][j] R_h): rq[j][s_][e] (f32 of the bf16)
    rq = [[[None] * 8 for _ in range(2)] for _ in range(4)]
    for s_ in range_constexpr(2):
        for e in range_constexpr(8):
            x = yv[s_][e].to(F32)
            f = [rv[h][s_][e].to(F32) for h in range(4)]
            for j in range_constexpr(4):
                rq[j][s_][e] = bf(fma(cm[3][j], f[3], fma(cm[2][j], f[2], fma(cm[1][j], f[1], fma(cm[0][j], f[0], pp[j] * x))))).to(F32)
    if fx.const_expr("eng" in c):
        if c["eng"]:
            rq = _engram_gate(c, rq, tok, g, p, w, l, c0, cs, seg2)
    sr = F32(0.0)
    sx = F32(0.0)
    for s_ in range_constexpr(2):
        outs = [[None] * 8 for _ in range(4)]
        x1s = [None] * 8
        srs = F32(0.0)
        sxs = F32(0.0)
        for e in range_constexpr(8):
            gq = []
            for j in range_constexpr(4):
                rf = rq[j][s_][e]
                gq.append(rf)
                srs = fma(rf, rf, srs)
                outs[j][e] = rf.to(BF16)
            x1 = bf(fma(q[3], gq[3], fma(q[2], gq[2], fma(q[1], gq[1], q[0] * gq[0]))))
            x1f = x1.to(F32)
            sxs = fma(x1f, x1f, sxs)
            x1s[e] = x1
        if fx.const_expr(s_ == 0):
            sr = sr + srs
            sx = sx + sxs
        else:
            sr = sr + seg2.select(srs, F32(0.0))
            sx = sx + seg2.select(sxs, F32(0.0))
        if (s_ == 0) | seg2:
            for j in range_constexpr(4):
                pv = fx.Vector.from_elements(outs[j], BF16)
                hx.lstore(c["rk"] + ((w * 4 + j) * CP + cs[s_]) * 2, pv, align=16)
                if c["last"]:
                    _stx(c, c["Rout"] + i64((tok * 4 + j) * H + c0 + cs[s_]) * 2, pv)
                hx.lstore(sm + O_RP + ((w * 4 + j) * CP + cs[s_]) * 2, pv, align=16)
            hx.lstore(sm + O_X1 + (w * CP + cs[s_]) * 2, fx.Vector.from_elements(x1s, BF16), align=16)
    for o in range_constexpr(5):
        sr = sr + sr.shuffle_xor(16 >> o, 32)
        sx = sx + sx.shuffle_xor(16 >> o, 32)
    if l == 0:
        base = c["part"] + i64(((g * CL + p) * TG + w) * 32) * 4
        hx.gstore(base + i64(24 * 4), sr)
        hx.gstore(base + i64(25 * 4), sx)
    # zero the M-padding rows 8..15 of R' and normed (16-byte stores)
    zero = fx.Vector.filled(4, 0, I32)
    for it in range_constexpr((8 * 4 * CP * 2 // 16 + 255) // 256):
        i = t + it * 256
        if i < 8 * 4 * CP * 2 // 16:
            hx.lstore(sm + O_RP + 8 * 4 * CP * 2 + i * 16, zero)
    for it in range_constexpr((8 * CP * 2 // 16 + 255) // 256):
        i = t + it * 256
        if i < 8 * CP * 2 // 16:
            hx.lstore(sm + O_XN + 8 * CP * 2 + i * 16, zero)
    gpu.barrier()


@ASTRewriter.transform
def front_B(c):
    """mixes partial: wave w takes K steps 5w..5w+4 of 40 (stream kk/10, col block kk%10), N tiles 0/1, fn hi + lo."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    g = b // CL
    p = b % CL
    c0 = p * COLS
    acc = [fx.Vector.filled(8, 0.0, F32), fx.Vector.filled(8, 0.0, F32)]
    fw = []
    for u in range_constexpr(5):   # all 20 fn operands issued first
        kk = w * 5 + u
        j = kk // 10
        ks = kk % 10
        for n in range_constexpr(2):
            # only rows 0..23 of the 32 are used (part keeps o < 24): the N tile 1 lanes 8..15 re-read rows 16..23
            # (same cache lines as lanes 0..7) instead of fetching the unused rows 24..31
            fr = (n * 16 + l % 8) if n == 1 else (n * 16 + l % 16)
            off = i64(fr * 4 * H + j * H + c0 + ks * 32 + (l // 16) * 16) * 2
            fw.append((hx.gload(c["fnh"] + off, BF16, 16, align=16), hx.gload(c["fnl"] + off, BF16, 16, align=16)))
    sched_barrier()
    for u in range_constexpr(5):
        kk = w * 5 + u
        j = kk // 10
        ks = kk % 10
        av = hx.lload(sm + O_RP + (((l % 16) * 4 + j) * CP + ks * 32 + (l // 16) * 16) * 2, BF16, 16, align=16)
        for n in range_constexpr(2):
            fh, fl = fw[u * 2 + n]
            acc[n] = hx.wmma_bf16(av, fh, acc[n])
            acc[n] = hx.wmma_bf16(av, fl, acc[n])
    for n in range_constexpr(2):
        hx.lstore(sm + O_RED + (((w * 2 + n) * 32 + l) * 8) * 4, acc[n])
    gpu.barrier()
    if t < TG * 24:
        tok = t // 24
        o = t % 24
        n = o // 16
        lane = o % 16
        s = F32(0.0)
        for ww in range_constexpr(8):
            s = s + hx.lload(sm + O_RED + (((ww * 2 + n) * 32 + lane) * 8 + tok) * 4, F32)
        hx.gstore(c["part"] + i64(((g * CL + p) * TG + tok) * 32 + o) * 4, s)


@ASTRewriter.transform
def front_C(c):
    """norm + MXFP8 of this CTA's 320 cols (8 tokens)."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    g = b // CL
    p = b % CL
    c0 = p * COLS
    if t < TG:
        s = F32(0.0)
        for qq in range_constexpr(CL):
            s = s + hx.gload(c["part"] + i64(((g * CL + qq) * TG + t) * 32 + 25) * 4, F32)
        hx.lstore(sm + O_RSTD + t * 4, rsq(s * F32(1.0 / H) + F32(c["norm_eps"])))
    gpu.barrier()
    rs = hx.lload(sm + O_RSTD + w * 4, F32)
    cs = [l * 8, 256 + fx.min(l, I32(7)) * 8]
    for s_ in range_constexpr(2):
        if (s_ == 0) | (l < 8):
            xv = hx.lload(sm + O_X1 + (w * CP + cs[s_]) * 2, BF16, 8, align=16)
            nv = hx.gload(c["nw"] + i64(c0 + cs[s_]) * 2, BF16, 8, align=16)
            o8 = [bf((xv[e].to(F32) * rs) * nv[e].to(F32)) for e in range(8)]
            hx.lstore(sm + O_XN + (w * CP + cs[s_]) * 2, fx.Vector.from_elements(o8, BF16), align=16)
    gpu.barrier()
    if t < TG * 10:
        tok = t // 10
        gi = t % 10
        row = g * TG + tok
        vs = [hx.lload(sm + O_XN + (tok * CP + gi * 32 + 8 * k) * 2, BF16, 8, align=16) for k in range(4)]
        vals = [vs[k][e].to(F32) for k in range(4) for e in range(8)]
        am = F32(0.0)
        for e in range_constexpr(32):
            am = fx.max(am, abs(vals[e]))
        code = e8m0_of(am)
        inv = inv_pow2(code)
        words = q8words([vals[e] for e in range(32)], inv)
        _stx(c, c["x8"] + i64(row * H + c0 + gi * 32), fx.Vector.from_elements(words[:4], I32))
        _stx(c, c["x8"] + i64(row * H + c0 + gi * 32 + 16), fx.Vector.from_elements(words[4:], I32))
        if c["xs32"]:   # the attention front: one int32 a 32-col block (the wq_a / wkv input); the FFN front: int8
            _stx(c, c["xs"] + i64(row * (H // 32) + c0 // 32 + gi) * 4, code)
        else:
            _stx(c, c["xs"] + i64(row * (H // 32) + c0 // 32 + gi), code.to(fx.Int8))
        for k in range_constexpr(4):
            _stx(c, c["xn"] + i64(row * H + c0 + gi * 32 + 8 * k) * 2, vs[k])


def _grow(w, m, l, NE):
    """gate row of wave w's tile m, lane l (clamped to the NE rows: the draft router has 128)."""
    r = (w * 3 + m) * 16 + l % 16
    return fx.min(r, I32(NE - 1)) if NE < D.EXPERTS else r


@ASTRewriter.transform
def front_D(c):
    """router partial: experts as M (this CTA's 320 gate columns, all loads issued first), tokens as N."""
    NE, TOPK, EPC, NCI = router(c)
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    gw = [[hx.gload(c["gate"] + i64(_grow(w, m, l, NE) * H + (b % CL) * COLS + k * 32 + (l // 16) * 16) * 2,
                    BF16, 16, align=16) for k in range(10)] for m in range(3)]
    sched_barrier()
    g = b // CL
    p = b % CL
    for m in range_constexpr(3):
        acc = fx.Vector.filled(8, 0.0, F32)
        for k in range_constexpr(10):
            xb = hx.lload(sm + O_XN + ((l % 16) * CP + k * 32 + (l // 16) * 16) * 2, BF16, 16, align=16)
            acc = hx.wmma_bf16(gw[m][k], xb, acc)
        if fx.const_expr(NE < D.EXPERTS):   # draft router (NE experts): tiles past NE computed on clamped rows, not stored
            if (l % 16 < TG) & ((w * 3 + m) * 16 < NE):
                hx.gstore(c["plog"] + i64(((g * CL + p) * TG + l % 16) * NE + (w * 3 + m) * 16 + (l // 16) * 8) * 4, acc)
        else:
            if l % 16 < TG:
                hx.gstore(c["plog"] + i64(((g * CL + p) * TG + l % 16) * NE + (w * 3 + m) * 16 + (l // 16) * 8) * 4, acc)


@ASTRewriter.transform
def front_E1(c):
    """experts 24p..24p+23 of the 8 tokens (wave = token): logits sum, score, local top-6 -> candidates."""
    NE, TOPK, EPC, NCI = router(c)
    t, b, w, l = c["t"], c["b"], c["w"], c["l"]
    g = b // CL
    p = b % CL
    e = EPC * p + fx.min(l, I32(EPC - 1))
    s = F32(0.0)
    for qq in range_constexpr(CL):
        s = s + hx.gload(c["plog"] + i64(((g * CL + qq) * TG + w) * NE + e) * 4, F32)
    lg = bf(s).to(F32)
    # sqrt(softplus(lg)) with the raw v_exp / v_log / v_sqrt (no accurate-lowering fixups)
    sp = (lg > F32(20.0)).select(lg, F32(hx.intrin(F32.ir_type, "llvm.amdgcn.log.f32", [F32(1.0) + F32(hx.intrin(F32.ir_type, "llvm.amdgcn.exp2.f32", [lg * F32(1.4426950408889634)]))])) * F32(0.6931471805599453))
    sc = F32(hx.intrin(F32.ir_type, "llvm.amdgcn.sqrt.f32", [sp]))
    u = (sc + hx.gload(c["gbias"] + i64(e) * 4, F32)).bitcast(I32)
    hi = ((u >> 31) != 0).select(u ^ I32(0x7FFFFFFF), u)       # order-preserving signed key
    live = l < EPC
    mh = live.select(hi, I32(-2147483647 - 1))
    ml = live.select(I32(NE - 1) - e, I32(0))
    msc = sc
    for r in range(TOPK):   # runtime loop (layer-loop code size)
        bh = mh
        bl = ml
        bsc = msc
        for o in range_constexpr(5):
            oh = shx(bh, 16 >> o)
            ol = shx(bl, 16 >> o)
            os_ = bsc.shuffle_xor(16 >> o, 32)
            gt = (oh > bh) | ((oh == bh) & (ol > bl))
            bh = gt.select(oh, bh)
            bl = gt.select(ol, bl)
            bsc = gt.select(os_, bsc)
        if l == 0:
            ci = ((g * CL + p) * TG + w) * TOPK + r
            hx.gstore(c["cand"] + i64(ci) * 8, key64(bh, bl))
            hx.gstore(c["cands"] + i64(ci) * 4, bsc)
        mine = (mh == bh) & (ml == bl)
        mh = mine.select(I32(-2147483647 - 1), mh)
        ml = mine.select(I32(0), ml)


@ASTRewriter.transform
def front_E2(c):
    """CTA p=0 of a group merges the 16 x 6 candidates of each token (wave = token); CTA p=1 runs the Sinkhorn of the
    group's 8 tokens lane-parallel (wave = token). The merge runs for the FFN front only (c["merge"], runtime)."""
    NE, TOPK, EPC, NCI = router(c)
    t, b, w, l = c["t"], c["b"], c["w"], c["l"]
    g = b // CL
    p = b % CL
    if (p == 0) & c["merge"]:
        row = g * TG + w
        ch, cl_, cs = [], [], []
        for i in range_constexpr(NCI):
            ci = l + 32 * i
            if fx.const_expr(32 * (i + 1) > CL * TOPK):   # the last round's lanes past the CL x TOPK candidates: none
                ci = fx.min(ci, I32(CL * TOPK - 1))
            qq = ci // TOPK
            rr = ci % TOPK
            k = hx.gload(c["cand"] + i64(((g * CL + qq) * TG + w) * TOPK + rr) * 8, I64)
            kh = I32(k >> 32)
            kl = I32(k & fx.Int64(0xFFFFFFFF))
            if fx.const_expr(32 * (i + 1) > CL * TOPK):
                kh = (l + 32 * i < CL * TOPK).select(kh, I32(-2147483647 - 1))
                kl = (l + 32 * i < CL * TOPK).select(kl, I32(-1))
            ch.append(kh)
            cl_.append(kl)
            cs.append(hx.gload(c["cands"] + i64(((g * CL + qq) * TG + w) * TOPK + rr) * 4, F32))
        wsum = F32(0.0)
        sel = I32(0)
        swv = F32(0.0)
        for r in range(TOPK):   # runtime loop (layer-loop code size)
            bh = ch[0]
            bl = cl_[0]
            bsc = cs[0]
            for i in range_constexpr(1, NCI):
                gt = (ch[i] > bh) | ((ch[i] == bh) & (cl_[i] > bl))
                bh = gt.select(ch[i], bh)
                bl = gt.select(cl_[i], bl)
                bsc = gt.select(cs[i], bsc)
            for o in range_constexpr(5):
                oh = shx(bh, 16 >> o)
                ol = shx(bl, 16 >> o)
                os_ = bsc.shuffle_xor(16 >> o, 32)
                gt = (oh > bh) | ((oh == bh) & (ol > bl))
                bh = gt.select(oh, bh)
                bl = gt.select(ol, bl)
                bsc = gt.select(os_, bsc)
            for i in range_constexpr(NCI):
                hit = (ch[i] == bh) & (cl_[i] == bl)
                ch[i] = hit.select(I32(-2147483647 - 1), ch[i])
                cl_[i] = hit.select(I32(-1), cl_[i])
            mine = l == r
            sel = mine.select(I32(NE - 1) - bl, sel)
            swv = mine.select(bsc, swv)
            wsum = wsum + bsc
        if l < TOPK:
            _stx(c, c["ids"] + i64(row * TOPK + l) * 4, sel)
            # the sum floored as aiter's topk_gating (RENORM_SUM_FLOOR 1e-20): six selected scores of exactly 0 (logits
            # below ~-17: log(1 + exp(x)) rounds to 0) gave 0 x rcp(0) = NaN weights
            _stx(c, c["wts"] + i64(row * TOPK + l) * 4, swv * frcp(fx.max(wsum, F32(1e-20))) * F32(1.5))
    if p == 1:
        # Sinkhorn, lane-parallel: CTA p=1, wave w = token w, lane e%16 = comb element (r = e/4, cc = e%4)
        row = g * TG + w
        lm = l % 25
        sacc = F32(0.0)
        for qq in range_constexpr(CL):
            sacc = sacc + hx.gload(c["part"] + i64(((g * CL + qq) * TG + w) * 32 + lm) * 4, F32)
        sb = c["sm"] + O_RED + w * 128
        if l < 25:
            hx.lstore(sb + l * 4, sacc)
        gpu.barrier()
        rstd = rsq(hx.lload(sb + 24 * 4, F32) * F32(1.0 / (4 * H)) + F32(c["rms_eps"]))
        e = l % 16
        j4 = l % 4
        mj = hx.lload(sb + j4 * 4, F32)
        mq = hx.lload(sb + (4 + j4) * 4, F32)
        me = hx.lload(sb + (8 + e) * 4, F32)
        sc0 = hx.gload(c["hsc"], F32)
        sc1 = hx.gload(c["hsc"] + i64(4), F32)
        sc2 = hx.gload(c["hsc"] + i64(8), F32)
        bj = hx.gload(c["hbs"] + i64(j4) * 4, F32)
        bq = hx.gload(c["hbs"] + i64(4 + j4) * 4, F32)
        be = hx.gload(c["hbs"] + i64(8 + e) * 4, F32)
        if l < 4:
            _stx(c, c["npre"] + i64(row * 4 + j4) * 4, frcp(F32(1.0) + fexp2(-(mj * rstd * sc0 + bj))) + F32(c["hc_eps"]))
            _stx(c, c["npost"] + i64(row * 4 + j4) * 4, F32(2.0) * frcp(F32(1.0) + fexp2(-(mq * rstd * sc1 + bq))))
        eps = F32(c["hc_eps"])

        def rsum(v):   # sum over the 4 lanes of a row (cc)
            v = v + v.shuffle_xor(1, 32)
            return v + v.shuffle_xor(2, 32)

        def csum(v):   # sum over the 4 lanes of a column (r)
            v = v + v.shuffle_xor(4, 32)
            return v + v.shuffle_xor(8, 32)

        A = me * rstd * sc2 + be
        mx = fx.max(A, A.shuffle_xor(1, 32))
        mx = fx.max(mx, mx.shuffle_xor(2, 32))
        ex = fexp2(A - mx)
        P = dv(ex, rsum(ex)) + eps
        P = dv(P, eps + csum(P))
        for it in range(19):   # runtime loop: code size (the layer loop must stay in the I-cache)
            P = dv(P, eps + rsum(P))
            P = dv(P, eps + csum(P))
        if l < 16:
            _stx(c, c["ncomb"] + i64(row * 16 + e) * 4, P)


@ASTRewriter.transform
def front_any(c):
    """Both fronts with one copy of the code (c["isffn"] runtime: FFN front = + router D / E1, int8 block codes, top-6
    merge; attention front = int32 codes, Sinkhorn only). c["eng"] (optional, runtime):
    the attention front before an Engram layer (front_A's _engram_gate)."""
    front_A(c)
    front_B(c)
    cluster_sync()
    front_C(c)
    if fx.const_expr(not c.get("norouter", False)):   # (v2: the router runs on its own CTAs, router_v2)
        if c["isffn"]:
            front_D(c)
            cluster_sync()
            front_E1(c)
            cluster_sync()
    front_E2(c)
    gpu.barrier()


# ==============================================================================================================
# (merged from router_v2.py)
# (v2) The FFN router as TileRT structures it: dedicated router CTAs with the gate rows in registers before the input
# arrives, scores published as LL words, and every MoE CTA running its own top-k (no partial-logit fan-in, no cluster
# barriers, no merge CTA, no device handoff).
# 
# router_v2 (CTAs RCTA0 .. RCTA0 + NRC - 1): CTA r owns experts [EPR r, EPR r + EPR) of all TR rows over the full K:
#   gate rows -> VGPR (thread t: 16-byte chunks t, t + 256, t + 512 < 640 of each row: 18 loads, 72 VGPRs), issued first;
#   wait for the FFN front's normed x (X8_PUB = FRONT_CTAS); dot (fp32 FMA a thread, wave shuffle, LDS over the waves);
#   logit -> bf16 (as front_E1), sqrt(softplus) with the raw v_exp / v_log / v_sqrt (as front_E1) -> LL word of the score.
# topk_v2 (every MoE CTA, wave r < TR = row r): the row's NE scores (hx.get, lane l: experts l + 32 i), + bias, the
#   order-preserving key (as front_E1 / E2: ties to the smaller expert id), 6 rounds of lane-max + wave shuffle; ids and
#   weights (unbiased scores renormalized x 1.5, sum floored 1e-20, as front_E2) into the MoE LDS (O_PID / O_WTS).

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off


RCTA0, NRC = 160, 96   # the 96 CTAs past the owners: a router CTA that is also an owner started its MoE ~7 us late
TKPUB, TKCTA = False, 124
SHE_TOPK_FIRST = False   # (v2) the shared-UG CTAs run their top-k while their UG weights load, before waiting for x8   # (v2) one top-k CTA (idle: no shared UG, no router) publishes the picks as LL words
EPR = -(-NE // NRC)          # 4 experts a router CTA (the draft's 128: 2)
NCHUNK = H // 8          # 640 16-byte chunks a row
NI = NE // 32            # 12 experts a lane in the top-k


def _bf2(x):
    """an i32 as <2 x bfloat>."""
    return llvm.BitcastOp(hx.vty(2, BF16.ir_type), hx._v(x)).result


def dot2(a, b, acc):
    """acc + a.lo * b.lo + a.hi * b.hi (bf16 pairs in i32, fp32 accumulate): v_dot2_f32_bf16."""
    return F32(hx.intrin(F32.ir_type, "llvm.amdgcn.fdot2.f32.bf16", [_bf2(a), _bf2(b), acc, fx.Int32(0) != fx.Int32(0)]))


def xdpp(v, m):
    """i32 from lane ^ m (m < 16) through DPP row_xmask (gfx10+ 0x160 + m)."""
    return I32(llvm.call_intrinsic(I32.ir_type, "llvm.amdgcn.update.dpp.i32",
                                   [I32(v).ir_value(), I32(v).ir_value(), I32(0x160 + m).ir_value(), I32(0xF).ir_value(),
                                    I32(0xF).ir_value(), (fx.Int32(0) != fx.Int32(0)).ir_value()], [], []))


def ipf(kb):
    """instruction prefetch of the next kb KB of code (as step.iprefetch): issued before a wait, so the code after
    it runs warm (the layer loop is larger than the instruction cache; cold fetches under the MoE weight stream are slow)."""
    for off in range(0, kb * 1024, 4096):
        llvm.inline_asm(None, [], f"s_prefetch_inst_pc_rel {off}, null, 31", "", has_side_effects=True)


def _tree(v, f):
    """a pairwise (balanced) reduction of a Python list of values (outside the AST rewriter: plain Python loops)."""
    while len(v) > 1:
        v = [f(v[j], v[j + 1]) for j in range(0, len(v) - 1, 2)] + ([v[-1]] if len(v) % 2 else [])
    return v[0]


def _smax(a, b):
    return (b > a).select(b, a)


def _wmax(v):
    """signed i32 max over the wave32: xor 8 / 4 / 2 / 1 on DPP row_xmask, xor 16 on one shuffle."""
    for o in (8, 4, 2, 1):
        v = _smax(v, xdpp(v, o))
    return _smax(v, shx(v, 16))


def _score(lg):
    """sqrt(softplus(lg)) exactly as front_E1 (raw intrinsics)."""
    sp = (lg > F32(20.0)).select(lg, F32(hx.intrin(F32.ir_type, "llvm.amdgcn.log.f32", [F32(1.0) + F32(hx.intrin(F32.ir_type, "llvm.amdgcn.exp2.f32", [lg * F32(1.4426950408889634)]))])) * F32(0.6931471805599453))
    return F32(hx.intrin(F32.ir_type, "llvm.amdgcn.sqrt.f32", [sp]))


GL_OFF = 8192            # (past the owner post's 4096 + 2 KB sums: the gate rows land during the owner post) LDS: the CTA's EPR gate rows (bf16, row stride GL_STR), after a 4 KB reduction scratch
GL_STR = H * 2 + 16
XN_OFF = GL_OFF + EPR * GL_STR   # LDS: the rows' normed x (bf16 [TR][GL_STR / 2]) when the router normalizes x1 itself
RS_OFF = 2048                    # LDS: per (wave, row) square sums, then per row rstd
KW = H // 32 // 8        # 20 K steps of 32 a wave


@ASTRewriter.transform
def router_gate_pre(c):
    """(v2) The router CTA's gate rows -> LDS right after TP1, before the owner post (c: t b sm gate): the loads land
    while the owner post runs, not after it (router_v2 with gate_pre then skips them)."""
    t, b, sm = c["t"], c["b"], c["sm"]
    NE_ = c.get("ne", NE)   # (draft: 128 experts, 1 a router CTA)
    EPR_ = -(-NE_ // NRC)
    e0 = (b - RCTA0) * EPR_
    q = [t, t + 256, fx.min(t + 512, I32(NCHUNK - 1))]
    G = [[hx.gload(c["gate"] + i64(fx.min(e0 + e, I32(NE_ - 1)) * H + q[j] * 8) * 2, BF16, 8, align=16) for j in range(3)] for e in range(EPR_)]
    for e in range_constexpr(EPR_):
        for j in range_constexpr(3):
            hx.lstore(sm + GL_OFF + e * GL_STR + q[j] * 16, G[e][j], align=16)


@ASTRewriter.transform
def router_v2(c, TR):
    """c: t b w l sm (LDS: 4 KB scratch + EPR gate rows), gate (bf16 [NE][H]), xn (bf16 [T][H]), x8pub (counter),
    scores (LL [T][NE] x 8 B), tag (epoch32). The dot as bf16 WMMA 16x16x32 (experts as M, rows as N),
    the K split over the 8 waves (20 steps each), A from the LDS copy of the gate rows, B the rows' xn from global."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    NE_ = c.get("ne", NE)   # (draft: 128 experts, 1 a router CTA)
    EPR_ = -(-NE_ // NRC)
    e0 = (b - RCTA0) * EPR_
    q = [t, t + 256, fx.min(t + 512, I32(NCHUNK - 1))]
    if fx.const_expr("gate_pre" not in c):
        G = [[hx.gload(c["gate"] + i64(fx.min(e0 + e, I32(NE_ - 1)) * H + q[j] * 8) * 2, BF16, 8, align=16) for j in range(3)] for e in range(EPR_)]
    SELFN = "x1g" in c   # (v2) normalize x1 here (TileRT: the router reads x and computes the rms itself)
    if fx.const_expr(SELFN):   # the norm weight too, before the wait (not a second dependent trip after the x1 rows)
        NV = [hx.gload(c["nw"] + i64(q[j] * 8) * 2, BF16, 8, align=16) for j in range(3)]
    if fx.const_expr("gate_pre" not in c):
        for e in range_constexpr(EPR_):
            for j in range_constexpr(3):
                hx.lstore(sm + GL_OFF + e * GL_STR + q[j] * 16, G[e][j], align=16)
    ipf(16)
    if fx.const_expr(SELFN and "pflag" in c):   # (v3 e9) the 256 owners' LL flags
        hx.wait_flags(c["pflag"], 160, c["ptag"], w, l, 11)
    else:
        if t == 0:
            if fx.const_expr(SELFN):
                hx.wait_count(c["postcnt"], 160, 11)
            else:
                hx.wait_count(c["x8pub"], c.get("x8n", FRONT_CTAS), 11)
    gpu.barrier()
    if fx.const_expr(SELFN):
        # raw x1 rows -> LDS (low registers: nothing held across the barriers), the thread's square sums on the way
        v3 = t + 512 < NCHUNK
        XA = [[hx.gload(c["x1g"] + i64(r_ * H + q[j] * 8) * 2, BF16, 8, align=16) for j in range(3)] for r_ in range(TR)]
        for r_ in range_constexpr(TR):
            XR = XA[r_]
            s = F32(0.0)
            for j in range_constexpr(3):
                sj = F32(0.0)
                for e in range_constexpr(8):
                    xf = XR[j][e].to(F32)
                    sj = sj + xf * xf
                if fx.const_expr(j == 2):
                    sj = v3.select(sj, F32(0.0))
                s = s + sj
            for o in range_constexpr(5):
                s = s + s.shuffle_xor(16 >> o, 32)
            if l == 0:
                hx.lstore(sm + RS_OFF + (w * 8 + r_) * 4, s)
        gpu.barrier()
        if t < TR:
            s = F32(0.0)
            for ww in range_constexpr(8):
                s = s + hx.lload(sm + RS_OFF + (ww * 8 + t) * 4, F32)
            hx.lstore(sm + RS_OFF + 256 + t * 4, rsq(s * F32(1.0 / H) + F32(c["norm_eps"])))
        gpu.barrier()
        for r_ in range_constexpr(TR):
            rs = hx.lload(sm + RS_OFF + 256 + r_ * 4, F32)
            for j in range_constexpr(3):
                xr = XA[r_][j]   # (the raw rows kept in registers: no LDS round trip)
                o8 = [bf((xr[e].to(F32) * rs) * NV[j][e].to(F32)) for e in range(8)]
                hx.lstore(sm + XN_OFF + r_ * GL_STR + q[j] * 16, fx.Vector.from_elements(o8, BF16), align=16)
        gpu.barrier()
    am = fx.min(l % 16, I32(EPR_ - 1))        # A row (expert; rows past EPR_ repeat the last: their outputs unused)
    bn = fx.min(l % 16, I32(TR - 1))          # B row (token)
    kh = (l // 16) * 16
    acc = fx.Vector.filled(8, 0.0, F32)
    for ks in range_constexpr(KW):
        k0 = (w * KW + ks) * 32 + kh
        av = hx.lload(sm + GL_OFF + am * GL_STR + k0 * 2, BF16, 16, align=16)
        if fx.const_expr(SELFN):
            bv = hx.lload(sm + XN_OFF + bn * GL_STR + k0 * 2, BF16, 16, align=16)
        else:
            bv = hx.gload(c["xn"] + i64(bn * H + k0) * 2, BF16, 16, align=16)
        acc = hx.wmma_bf16(av, bv, acc)
    # lane l < 16: token l, experts 0..7 in acc[0..7]
    if (l < TR):
        for i in range_constexpr(EPR_):
            hx.lstore(sm + ((w * EPR_ + i) * 8 + l) * 4, acc[i])
    gpu.barrier()
    if t < TR * EPR_:
        r_ = t // EPR_
        e = t % EPR_
        s = F32(0.0)
        for ww in range_constexpr(8):
            s = s + hx.lload(sm + ((ww * EPR_ + e) * 8 + r_) * 4, F32)
        sc = _score(bf(s).to(F32))
        if e0 + e < NE_:   # (the draft's 128 experts: 2 a CTA, the CTAs past them none)
            hx.put(c["scores"] + i64(r_ * NE_ + e0 + e) * 8, sc.bitcast(I32), c["tag"])


@ASTRewriter.transform
def scores_batch(base, tag, NI=NI):
    """The NI LL words base + 256 i (lane's experts l + 32 i): all issued at once, re-polled together until every tag
    matches (bounded); returns the payloads."""
    t0 = hx.clock()
    pg = [hx.peek(base + i64(256 * i)) for i in range(NI)]
    bad = I32(0)
    for i in range_constexpr(NI):
        bad = bad | (pg[i][1] != tag).select(I32(1), I32(0))
    while (bad != 0) & (hx.clock() - t0 < I64(hx.POLL_TICKS)):
        hx.s_sleep(1)
        pg = [hx.peek(base + i64(256 * i)) for i in range(NI)]
        bad = I32(0)
        for i in range_constexpr(NI):
            bad = bad | (pg[i][1] != tag).select(I32(1), I32(0))
    return [pg[i][0] for i in range(NI)]


@ASTRewriter.transform
def topk_v2(c, TR, pid_off, wts_off, write_ids):
    """c: t w l sm (MoE LDS base), scores, gbias, tag. Wave r < TR: row r's top-TOPK -> LDS pid_off / wts_off [r * TOPK + k]
    (ids only if write_ids: a fixed-route fixture keeps its own ids)."""
    w, l, sm = c["w"], c["l"], c["sm"]
    NE_ = c.get("ne", NE)   # (draft: 128 experts, top-3)
    NI_ = NE_ // 32
    TOPK_ = c.get("ktop", TOPK)
    ONE = "row" in c   # (fx-v3 per-row MoE) wave 0 only, row c["row"], results at slots 0..5
    if fx.const_expr(ONE):
        act = w == 0
        row = c["row"]
        srow = 0
    else:
        act = w < TR
        row = w
        srow = w
    if act:
        mh, ml, ms = [], [], []
        bias = [hx.gload(c["gbias"] + i64(l + 32 * i) * 4, F32) for i in range(NI_)]   # before the scores arrive
        ipf(8)
        sw = scores_batch(c["scores"] + i64(row * NE_ + l) * 8, c["tag"], NI_)
        for i in range_constexpr(NI_):
            e = l + 32 * i
            sc = sw[i].bitcast(F32)
            u = (sc + bias[i]).bitcast(I32)
            mh.append(((u >> 31) != 0).select(u ^ I32(0x7FFFFFFF), u))
            ml.append(I32(NE_ - 1) - e)
            ms.append(sc)
        # as TileRT's expert CTA: the max key alone (v_max / v_max3 in the lane, max over DPP + one shuffle), then the
        # tie on the reversed id among the lanes at that key (max again): single-op steps, no 64-bit compare + selects
        wl = F32(0.0)
        sel = I32(0)
        for r in range_constexpr(TOPK_):
            bh = _wmax(_tree(list(mh), _smax))
            # (fast path) the lane's own tie-broken candidate; one lane holding the max -> read it there (scalar), not a
            # second wave max (ties across lanes: the wave max of the candidates, as before)
            bll = _tree([(mh[i] == bh).select(ml[i], I32(-1)) for i in range(NI_)], _smax)
            hm = I32(hx.intrin(hx.i32(), "llvm.amdgcn.ballot.i32", [bll >= I32(0)]))
            one = I32(hx.intrin(hx.i32(), "llvm.ctpop.i32", [hm])) == I32(1)
            bl = I32(0)
            if one:
                bl = I32(hx.intrin(hx.i32(), "llvm.amdgcn.readlane.i32",
                                   [bll, I32(hx.intrin(hx.i32(), "llvm.cttz.i32", [hm, fx.Int32(0) != fx.Int32(0)]))]))
            else:
                bl = _wmax(bll)
            hv = []
            for i in range_constexpr(NI_):
                hit = ml[i] == bl
                mh[i] = hit.select(I32(-2147483647 - 1), mh[i])
                ml[i] = hit.select(I32(-1), ml[i])
                hv.append(hit.select(ms[i], F32(0.0)))
            hsc = _tree(hv, lambda a, b: a + b)
            wl = wl + hsc
            if l == (I32(NE_ - 1) - bl) % 32:   # the lane holding the r-th pick: its raw score into slot r
                hx.lstore(sm + wts_off + (srow * TOPK_ + r) * 4, hsc)
            if fx.const_expr(write_ids):
                if l == r:
                    hx.lstore(sm + pid_off + (srow * TOPK_ + l) * 4, I32(NE_ - 1) - bl)
            sel = (l == r).select(I32(NE_ - 1) - bl, sel)
        for o in range_constexpr(5):
            if fx.const_expr(o < 4):
                wl = wl + xdpp(wl.bitcast(I32), 8 >> o).bitcast(F32)
            else:
                wl = wl + shx(wl.bitcast(I32), 16).bitcast(F32)
        if l < TOPK_:
            swv = hx.lload(sm + wts_off + (srow * TOPK_ + l) * 4, F32)
            wfin = swv * frcp(fx.max(wl, F32(1e-20))) * F32(1.5)
            hx.lstore(sm + wts_off + (srow * TOPK_ + l) * 4, wfin)
            if fx.const_expr("tkpub" in c):   # (v2) the top-k CTA publishes ids / weights as LL words for every other CTA
                hx.put(c["tkpub"] + i64((row * TOPK_ + l) * 2) * 8, sel, c["tktag"])
                hx.put(c["tkpub"] + i64((row * TOPK_ + l) * 2 + 1) * 8, wfin.bitcast(I32), c["tktag"])


@ASTRewriter.transform
def topk_get(c, TR, pid_off, wts_off, write_ids):
    """(v2) A CTA other than the top-k CTA: wave r < TR, lane l < TOPK takes row r's pick l (id, weight) from the LL
    words topk_v2 published (c: w l sm tkpub tktag) into the MoE LDS (O_PID / O_WTS) -- one hop instead of its own
    score reads and selection."""
    w, l, sm = c["w"], c["l"], c["sm"]
    if (w < TR) & (l < TOPK):
        a = c["tkpub"] + i64((w * TOPK + l) * 2) * 8
        pid = hx.get(a, c["tktag"], 12)
        pw = hx.get(a + i64(8), c["tktag"], 12)
        if fx.const_expr(write_ids):
            hx.lstore(sm + pid_off + (w * TOPK + l) * 4, pid)
        hx.lstore(sm + wts_off + (w * TOPK + l) * 4, pw.bitcast(F32))


# ------------------------------------------------------------------ (fx-v3) the router as TileRT's [E] (mega_afm 2.10)
RV3_CTA0 = 160           # router CTAs 160 .. 255 (not the wo_b / TP owners): 16 a row x 24 experts, rows < 6
RV3_PER_ROW = 16
RV3_EPC = NE // RV3_PER_ROW   # 24 experts a CTA


@ASTRewriter.transform
def router_v3(c, TR):
    """CTA b in RV3_CTA0 ..: row r = (b - RV3_CTA0) // 16, experts 24 s .. 24 s + 23 (s = (b - RV3_CTA0) % 16), the full K.
    The 24 gate rows go to VGPRs before the wait (thread t: 16-byte chunks t, t + 256, t + 512 < 640 of each row) with
    the norm weight; then the 160 owners' flags, the row's raw x1 (the same chunks), its rstd (CTA reduction), the
    normed bf16 x, 24 fp32 dot partials a thread, wave + LDS reductions in fixed order, sqrt(softplus(bf16 logit)) ->
    LL score words (the same words topk_v2 reads). c: t w l sm gate nw x1g pflag ptag scores tag norm_eps."""
    t, w, l, sm, b = c["t"], c["w"], c["l"], c["sm"], c["b"]
    r = (b - RV3_CTA0) // RV3_PER_ROW
    e0 = ((b - RV3_CTA0) % RV3_PER_ROW) * RV3_EPC
    q = [t, t + 256, fx.min(t + 512, I32(NCHUNK - 1))]
    v3 = t + 512 < NCHUNK
    G = [[hx.gload(c["gate"] + i64((e0 + e) * H + q[j] * 8) * 2, BF16, 8, align=16) for j in range(3)] for e in range(RV3_EPC)]
    NV = [hx.gload(c["nw"] + i64(q[j] * 8) * 2, BF16, 8, align=16) for j in range(3)]
    ipf(8)
    hx.wait_flags(c["pflag"], 160, c["ptag"], w, l, 11)
    gpu.barrier()
    rr = fx.min(r, I32(TR - 1))
    X = [hx.gload(c["x1g"] + i64(rr * H + q[j] * 8) * 2, BF16, 8, align=16) for j in range(3)]
    s = F32(0.0)
    for j in range_constexpr(3):
        sj = F32(0.0)
        for e in range_constexpr(8):
            xf = X[j][e].to(F32)
            sj = sj + xf * xf
        if fx.const_expr(j == 2):
            sj = v3.select(sj, F32(0.0))
        s = s + sj
    for o in range_constexpr(5):
        s = s + s.shuffle_xor(16 >> o, 32)
    if l == 0:
        hx.lstore(sm + RS_OFF + w * 4, s)
    gpu.barrier()
    tot = F32(0.0)
    for ww in range_constexpr(8):
        tot = tot + hx.lload(sm + RS_OFF + ww * 4, F32)
    rs = rsq(tot * F32(1.0 / H) + F32(c["norm_eps"]))
    xn = []
    for j in range_constexpr(3):
        for e in range_constexpr(8):
            v = bf((X[j][e].to(F32) * rs) * NV[j][e].to(F32)).to(F32)
            if fx.const_expr(j == 2):
                v = v3.select(v, F32(0.0))
            xn.append(v)
    part = []
    for e in range_constexpr(RV3_EPC):
        a = F32(0.0)
        for j in range_constexpr(3):
            for k in range_constexpr(8):
                a = a + xn[j * 8 + k] * G[e][j][k].to(F32)
        for o in range_constexpr(5):
            a = a + a.shuffle_xor(16 >> o, 32)
        part.append(a)
    if l == 0:
        for e in range_constexpr(RV3_EPC):
            hx.lstore(sm + RS_OFF + 64 + (w * RV3_EPC + e) * 4, part[e])
    gpu.barrier()
    if (t < RV3_EPC) & (r < TR):
        sacc = F32(0.0)
        for ww in range_constexpr(8):
            sacc = sacc + hx.lload(sm + RS_OFF + 64 + (ww * RV3_EPC + t) * 4, F32)
        sc = _score(bf(sacc).to(F32))
        hx.put(c["scores"] + i64(r * NE + e0 + t) * 8, sc.bitcast(I32), c["tag"])


# ==============================================================================================================
# (merged from mhc_v2.py)
# (v2) The FFN mHC as TileRT structures it: the post runs in the TP1 column owners right after their reduction (no
# 16-CTA front chain before the router).
# 
# ffn_post_owner (every CTA b, after its TP1 cols 80 (b // 4) + 20 (b % 4) .. + 19 are reduced into y1):
#   R'' = bf(post y + comb R'), x1 = bf(pre . R'') for its 20 cols x TR rows (the front_A arithmetic, fma order kept), R'' ->
#   Rout (the next attention front reads it from global), x1 -> x1g; per row partials: sum R''^2 (part2 [b][r][24]),
#   sum x1^2 ([25]) and the mix  sum_K fn[o] R'' (fnh + fnl, [o < 24]) over its 80 K entries (4 streams x 20 cols);
#   then counted (cnt).
# ffn_c2 (front CTAs g = 0, b < CL): waits for all CTAS owners, rstd of the full row from the 256 x1 partials, normed =
#   bf(x1 rs nw) of its 320 cols -> xn / x8 / xs (front_C's arithmetic), counted in x8pub.
# sinkhorn2 (CTA 1): the FFN mix of the 8 tokens from the 256 partials -> npre / npost / ncomb (front_E2's Sinkhorn).

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off


CTAS = 256
NMIX = 24
OC = 32                 # cols an owner (fx-v3: 160 owners x 32 cols, the wo_b / TP slices)
NOWN = 160              # the column owners
RQO = 140000            # LDS: the owner's R'' [TR][4][20] f32 (past the router's LDS: the router CTAs run their mix after it)


@ASTRewriter.transform
def ffn_post_owner(c, TR):
    """c: t b w l sm (LDS scratch >= 6 KB), y, R, Rout, x1g, pre, post, comb, fnh, fnl, part2, cnt."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    col0 = b * OC
    live = t < TR * OC
    r = fx.min(t // OC, I32(TR - 1))
    cc = col0 + t % OC
    x = hx.gload(c["y"] + i64(r * H + cc) * 2, BF16).to(F32)
    f = [hx.gload(c["R"] + i64((r * 4 + h) * H + cc) * 2, BF16).to(F32) for h in range(4)]
    q = [hx.gload(c["pre"] + i64(r * 4 + j) * 4, F32) for j in range(4)]
    pp = [hx.gload(c["post"] + i64(r * 4 + j) * 4, F32) for j in range(4)]
    cm = [[hx.gload(c["comb"] + i64(r * 16 + h * 4 + j) * 4, F32) for j in range(4)] for h in range(4)]
    rq = [bf(fma(cm[3][j], f[3], fma(cm[2][j], f[2], fma(cm[1][j], f[1], fma(cm[0][j], f[0], pp[j] * x))))).to(F32) for j in range(4)]
    if fx.const_expr("engkv" in c):
        # (fx-v3) the Engram gate of the layer after (front_A's _engram_gate in the column owners): per (row, branch j)
        # sum R'^2, sum key^2, sum R' gw key over this owner's 32 cols (wave w = row), LL-published; every owner sums the
        # 160 records, gate_j = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))), R'_j <- bf(R'_j + gate_j value)
        ekv = c["engkv"] + i64(r * (5 * H) + cc) * 2
        kvv = [hx.gload(ekv + i64(h * H) * 2, BF16).to(F32) for h in range(5)]
        if c["eng"]:
            gw = [hx.gload(c["enggw"] + i64(h * H + cc) * 4, F32) for h in range(4)]
            pr = []
            for j in range_constexpr(4):
                pr += [rq[j] * rq[j], kvv[j] * kvv[j], rq[j] * gw[j] * kvv[j]]
            for o in range_constexpr(5):
                pr = [v + v.shuffle_xor(16 >> o, 32) for v in pr]
            if (w < TR) & (l == 0):
                for k in range_constexpr(3):
                    hx.stx_dev(c["engp"] + i64((b * 8 + w) * 16 + 4 * k) * 4, fx.Vector.from_elements(pr[4 * k:4 * k + 4], F32))
            hx.wait_storecnt0()
            gpu.barrier()
            if t == 0:
                hx.put(c["engflag"] + i64(b) * 8, I32(0), c["engtag"])
            hx.wait_flags(c["engflag"], NOWN, c["engtag"], w, l, 13)
            gpu.barrier()
            if w < TR:
                tot = [F32(0.0)] * 12
                for kq in range_constexpr(NOWN // 32):
                    src = c["engp"] + i64(((l + 32 * kq) * 8 + w) * 16) * 4
                    got = [hx.gload(src + i64(16 * k), F32, 4) for k in range(3)]
                    tot = [tot[k] + got[k // 4][k % 4] for k in range(12)]
                for o in range_constexpr(5):
                    tot = [v + v.shuffle_xor(16 >> o, 32) for v in tot]
                if l < 4:
                    gsel = F32(0.0)
                    for j in range_constexpr(4):
                        rstd = rsq(tot[3 * j] * F32(1.0 / H) + F32(D.RMS_EPS)) * rsq(tot[3 * j + 1] * F32(1.0 / H) + F32(D.RMS_EPS))
                        dot = tot[3 * j + 2] * rstd * F32(1.0 / (H ** 0.5))
                        mag = fsqrt(fx.max(abs(dot), F32(1e-6)))
                        sg = (dot < F32(0.0)).select(F32(0.0) - mag, mag)
                        gsel = (l == j).select(frcp(F32(1.0) + fexp(F32(0.0) - sg)), gsel)
                    hx.lstore(sm + 6144 + (w * 4 + l) * 4, gsel)
            gpu.barrier()
        gts = [hx.lload(sm + 6144 + (r * 4 + j) * 4, F32) for j in range(4)]
        rq = [c["eng"].select(bf(rq[j] + gts[j] * kvv[4]).to(F32), rq[j]) for j in range(4)]
    srs = F32(0.0)
    for j in range_constexpr(4):
        srs = fma(rq[j], rq[j], srs)
    x1 = bf(fma(q[3], rq[3], fma(q[2], rq[2], fma(q[1], rq[1], q[0] * rq[0]))))
    x1f = x1.to(F32)
    if live:
        for j in range_constexpr(4):
            hx.stx_dev(c["Rout"] + i64((r * 4 + j) * H + cc) * 2, rq[j].to(BF16))
            hx.lstore(sm + RQO + ((r * 4 + j) * OC + t % OC) * 4, rq[j])    # R'' for the mix (LDS [r][4][20] f32)
        hx.stx_dev(c["x1g"] + i64(r * H + cc) * 2, x1)
        hx.lstore(sm + 4096 + t * 8, srs)
        hx.lstore(sm + 4096 + t * 8 + 4, x1f * x1f)
    gpu.barrier()
    # per-row sums of the two squares (rows r < TR: 20 entries each)
    if t < TR:
        a = F32(0.0)
        z = F32(0.0)
        for k in range_constexpr(OC):
            a = a + hx.lload(sm + 4096 + (t * OC + k) * 8, F32)
            z = z + hx.lload(sm + 4096 + (t * OC + k) * 8 + 4, F32)
        hx.stx_dev(c["part2"] + i64((b * 8 + t) * 32 + 24) * 4, a)
        hx.stx_dev(c["part2"] + i64((b * 8 + t) * 32 + 25) * 4, z)
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        if fx.const_expr("pflag" in c):   # (v3 e9 / e1) one LL flag an owner, not a 256-way counter
            hx.put(c["pflag"] + i64(b) * 8, I32(0), c["ptag"])
        else:
            hx.atomic_add(c["cnt"], 1)


@ASTRewriter.transform
def ffn_mix_owner(c, TR):
    """The owner's mix partials (after ffn_post_owner left R'' in its LDS; off the FFN critical path, the Sinkhorn of the
    next attention front): counted in cnt2. c: t b sm fnh fnl part2 cnt2."""
    t, b, sm = c["t"], c["b"], c["sm"]
    col0 = b * OC
    # thread t < 96 = fn row o * 4 + j: its 20 weights (hi + lo) x the rows' R''_j, summed over j by shuffles
    if t < NMIX * 4:
        o = t // 4
        j = t % 4
        wsum = [F32(0.0)] * 8
        # 8-byte loads (col0 * 2 is a multiple of 40 B): 5 + 5 a thread, not 40 two-byte loads
        fh = [hx.gload(c["fnh"] + i64((o * 4 + j) * H + col0 + 4 * k) * 2, BF16, 4, align=8) for k in range(OC // 4)]
        fl = [hx.gload(c["fnl"] + i64((o * 4 + j) * H + col0 + 4 * k) * 2, BF16, 4, align=8) for k in range(OC // 4)]
        fw = [fh[k // 4][k % 4].to(F32) + fl[k // 4][k % 4].to(F32) for k in range(OC)]
        for rr in range_constexpr(TR):
            s = F32(0.0)
            for k in range_constexpr(OC):
                s = fma(fw[k], hx.lload(sm + RQO + ((rr * 4 + j) * OC + k) * 4, F32), s)
            s = s + s.shuffle_xor(1, 32)
            s = s + s.shuffle_xor(2, 32)
            wsum[rr] = s
        if j == 0:
            for rr in range_constexpr(TR):
                hx.stx_dev(c["part2"] + i64((b * 8 + rr) * 32 + o) * 4, wsum[rr])
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(c["cnt2"], 1)


@ASTRewriter.transform
def ffn_c2(c, TR):
    """Front CTA p = b (< CL) of group 0: c: t b w l sm (the front LDS), x1g, part2, cnt, nw, xn, x8, xs, x8pub, norm_eps."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    p = b
    c0 = p * COLS
    if fx.const_expr("pflag" in c):
        hx.wait_flags(c["pflag"], NOWN, c["ptag"], w, l, 13)
    else:
        if t == 0:
            hx.wait_count(c["cnt"], NOWN, 13)
    gpu.barrier()
    # wave w = token: lane l sums owners l, l + 32, .. (8 loads in flight), then the wave sum
    pv = [hx.gload(c["part2"] + i64(((l + 32 * k) * 8 + w) * 32 + 25) * 4, F32) for k in range(NOWN // 32)]
    s = F32(0.0)
    for k in range_constexpr(NOWN // 32):
        s = s + pv[k]
    for o in range_constexpr(5):
        s = s + s.shuffle_xor(16 >> o, 32)
    if l == 0:
        hx.lstore(sm + O_RSTD + w * 4, rsq(s * F32(1.0 / H) + F32(c["norm_eps"])))
    # x1 of the 8 tokens x this CTA's 320 cols into LDS O_X1 (rows >= TR zero)
    cs = [l * 8, 256 + fx.min(l, I32(7)) * 8]
    tokc = fx.min(w, I32(TR - 1))
    for s_ in range_constexpr(2):
        if (s_ == 0) | (l < 8):
            v = hx.gload(c["x1g"] + i64(tokc * H + c0 + cs[s_]) * 2, BF16, 8, align=16)
            v = fx.Vector.from_elements([(w < TR).select(v[e], BF16(0.0)) for e in range(8)], BF16)
            hx.lstore(sm + O_X1 + (w * CP + cs[s_]) * 2, v, align=16)
    gpu.barrier()
    rs = hx.lload(sm + O_RSTD + w * 4, F32)
    for s_ in range_constexpr(2):
        if (s_ == 0) | (l < 8):
            xv = hx.lload(sm + O_X1 + (w * CP + cs[s_]) * 2, BF16, 8, align=16)
            nv = hx.gload(c["nw"] + i64(c0 + cs[s_]) * 2, BF16, 8, align=16)
            o8 = [bf((xv[e].to(F32) * rs) * nv[e].to(F32)) for e in range(8)]
            hx.lstore(sm + O_XN + (w * CP + cs[s_]) * 2, fx.Vector.from_elements(o8, BF16), align=16)
    gpu.barrier()
    if t < TG * 10:
        tok = t // 10
        gi = t % 10
        row = tok
        vs = [hx.lload(sm + O_XN + (tok * CP + gi * 32 + 8 * k) * 2, BF16, 8, align=16) for k in range(4)]
        vals = [vs[k][e].to(F32) for k in range(4) for e in range(8)]
        am = F32(0.0)
        for e in range_constexpr(32):
            am = fx.max(am, abs(vals[e]))
        code = e8m0_of(am)
        inv = inv_pow2(code)
        words = q8words([vals[e] for e in range(32)], inv)
        hx.stx_dev(c["x8"] + i64(row * H + c0 + gi * 32), fx.Vector.from_elements(words[:4], I32))
        hx.stx_dev(c["x8"] + i64(row * H + c0 + gi * 32 + 16), fx.Vector.from_elements(words[4:], I32))
        hx.stx_dev(c["xs"] + i64(row * (H // 32) + c0 // 32 + gi), code.to(fx.Int8))
        for k in range_constexpr(4):
            hx.stx_dev(c["xn"] + i64(row * H + c0 + gi * 32 + 8 * k) * 2, vs[k])
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        if fx.const_expr("xflag" in c):   # (v3 e10) one LL flag a c2 CTA
            hx.put(c["xflag"] + i64(b) * 8, I32(0), c["xtag"])
        else:
            hx.atomic_add(c["x8pub"], 1)


NL1 = 6          # level-1 summing CTAs of the mix partials (CTAs SK0 .. SK0 + 5: idle until the scores)
SK0 = 250   # (router CTAs past the UG units at c1: idle through the UG)
OPL = (NOWN + NL1 - 1) // NL1   # owners a level-1 CTA (43)


@ASTRewriter.transform
def mix_l1(c, j):
    """Level-1 CTA j (< NL1): thread t < TG * 25 = (token, value) sums owners [OPL j, OPL (j + 1)) of part2 -> part3 [j];
    counted in cnt3. c: t part2 part3 cnt2 cnt3."""
    t = c["t"]
    if t == 0:
        hx.wait_count(c["cnt2"], NOWN, 14)
    gpu.barrier()
    if t < TG * 25:
        tok = t // 25
        v = t % 25
        k0 = j * OPL
        pv = [hx.gload(c["part2"] + i64(((fx.min(k0 + k, I32(NOWN - 1))) * 8 + tok) * 32 + v) * 4, F32) for k in range(OPL)]
        s = F32(0.0)
        for k in range_constexpr(OPL):
            s = s + (k0 + k < NOWN).select(pv[k], F32(0.0))
        hx.stx_dev(c["part3"] + i64((j * 8 + tok) * 32 + v) * 4, s)
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(c["cnt3"], 1)


@ASTRewriter.transform
def sinkhorn2(c):
    """CTA 1: front_E2's Sinkhorn (wave = token, lane = comb element) from the 256 owners' partials.
    c: t w l sm, part2, cnt, hsc, hbs, npre, npost, ncomb, rms_eps, hc_eps."""
    t, w, l = c["t"], c["w"], c["l"]
    row = w
    if t == 0:
        hx.wait_count(c["cnt3"], NL1, 14)
    gpu.barrier()
    lm = l % 25
    # (counted done: ffn_c2 of this CTA waited before) lane l sums owners l, l + 32, .. of all 25 values, then wave sums
    sb = c["sm"] + O_RED + w * 128
    pv = [hx.gload(c["part3"] + i64((k * 8 + w) * 32 + lm) * 4, F32) for k in range(NL1)]
    sacc = F32(0.0)
    for k in range_constexpr(NL1):
        sacc = sacc + pv[k]
    if l < 25:
        hx.lstore(sb + l * 4, sacc)
    gpu.barrier()
    rstd = rsq(hx.lload(sb + 24 * 4, F32) * F32(1.0 / (4 * H)) + F32(c["rms_eps"]))
    e = l % 16
    j4 = l % 4
    mj = hx.lload(sb + j4 * 4, F32)
    mq = hx.lload(sb + (4 + j4) * 4, F32)
    me = hx.lload(sb + (8 + e) * 4, F32)
    sc0 = hx.gload(c["hsc"], F32)
    sc1 = hx.gload(c["hsc"] + i64(4), F32)
    sc2 = hx.gload(c["hsc"] + i64(8), F32)
    bj = hx.gload(c["hbs"] + i64(j4) * 4, F32)
    bq = hx.gload(c["hbs"] + i64(4 + j4) * 4, F32)
    be = hx.gload(c["hbs"] + i64(8 + e) * 4, F32)
    if l < 4:
        hx.stx_dev(c["npre"] + i64(row * 4 + j4) * 4, frcp(F32(1.0) + fexp2(-(mj * rstd * sc0 + bj))) + F32(c["hc_eps"]))
        hx.stx_dev(c["npost"] + i64(row * 4 + j4) * 4, F32(2.0) * frcp(F32(1.0) + fexp2(-(mq * rstd * sc1 + bq))))
    eps = F32(c["hc_eps"])

    def rsum(v):
        v = v + v.shuffle_xor(1, 32)
        return v + v.shuffle_xor(2, 32)

    def csum(v):
        v = v + v.shuffle_xor(4, 32)
        return v + v.shuffle_xor(8, 32)

    A = me * rstd * sc2 + be
    mx = fx.max(A, A.shuffle_xor(1, 32))
    mx = fx.max(mx, mx.shuffle_xor(2, 32))
    ex = fexp2(A - mx)
    P = dv(ex, rsum(ex)) + eps
    P = dv(P, eps + csum(P))
    for it in range(19):
        P = dv(P, eps + rsum(P))
        P = dv(P, eps + csum(P))
    if l < 16:
        hx.stx_dev(c["ncomb"] + i64(row * 16 + e) * 4, P)
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(c["done"], 1)


@ASTRewriter.transform
def mix_owner_g(c, TR):
    """(v2, attention side) ffn_mix_owner with R'' read back from the owner's own global writes (c["Rq"]: aRout, bf16,
    the same values ffn_post_owner kept in LDS): runs after stage A / C, when the LDS copy is gone. c: t b fnh fnl Rq
    part2 cnt2."""
    t, b = c["t"], c["b"]
    col0 = b * OC
    if t < NMIX * 4:
        o = t // 4
        j = t % 4
        wsum = [F32(0.0)] * 8
        fh = [hx.gload(c["fnh"] + i64((o * 4 + j) * H + col0 + 4 * k) * 2, BF16, 4, align=8) for k in range(OC // 4)]
        fl = [hx.gload(c["fnl"] + i64((o * 4 + j) * H + col0 + 4 * k) * 2, BF16, 4, align=8) for k in range(OC // 4)]
        rqv = [[hx.gload(c["Rq"] + i64((rr * 4 + j) * H + col0 + 4 * k) * 2, BF16, 4, align=8) for k in range(OC // 4)] for rr in range(TR)]
        fw = [fh[k // 4][k % 4].to(F32) + fl[k // 4][k % 4].to(F32) for k in range(OC)]
        for rr in range_constexpr(TR):
            s = F32(0.0)
            for k in range_constexpr(OC):
                s = fma(fw[k], rqv[rr][k // 4][k % 4].to(F32), s)
            s = s + s.shuffle_xor(1, 32)
            s = s + s.shuffle_xor(2, 32)
            wsum[rr] = s
        if j == 0:
            for rr in range_constexpr(TR):
                hx.stx_dev(c["part2"] + i64((b * 8 + rr) * 32 + o) * 4, wsum[rr])
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(c["cnt2"], 1)


X8LO, X8LS = 220000, 250720   # (fx-v3) LDS (past the MoE / router / Sinkhorn areas): this CTA's x8 [TR][5120] and codes [TR][160] (+1024: rstd)


@ASTRewriter.transform
def x8_self(c, TR):
    """(fx-v3, TileRT mega_afm F0) every CTA normalizes and quantizes the FFN input itself into its LDS (no c2 hop):
    waits for the 160 owners' flags, rstd of each row from their partials (ffn_c2's sum order), then bf(x1 rs nw) and
    the MXFP8 block of each 32 cols (front_C's arithmetic). c: t w l sm x1g part2 pflag ptag nw norm_eps."""
    t, w, l, sm = c["t"], c["w"], c["l"], c["sm"]
    hx.wait_flags(c["pflag"], NOWN, c["ptag"], w, l, 13)
    gpu.barrier()
    wr = fx.min(w, I32(TR - 1))
    pv = [hx.gload(c["part2"] + i64(((l + 32 * k) * 8 + wr) * 32 + 25) * 4, F32) for k in range(NOWN // 32)]
    if w < TR:
        s = F32(0.0)
        for k in range_constexpr(NOWN // 32):
            s = s + pv[k]
        for o in range_constexpr(5):
            s = s + s.shuffle_xor(16 >> o, 32)
        if l == 0:
            hx.lstore(sm + X8LS + 1024 + w * 4, rsq(s * F32(1.0 / H) + F32(c["norm_eps"])))
    gpu.barrier()
    NBK = TR * (H // 32)
    for it in range_constexpr((NBK + 255) // 256):
        blk = t + it * 256
        if blk < NBK:
            rr = blk // (H // 32)
            bc = blk % (H // 32)
            rs = hx.lload(sm + X8LS + 1024 + rr * 4, F32)
            xv = [hx.gload(c["x1g"] + i64(rr * H + bc * 32 + 8 * h) * 2, BF16, 8, align=16) for h in range(4)]
            nv = [hx.gload(c["nw"] + i64(bc * 32 + 8 * h) * 2, BF16, 8, align=16) for h in range(4)]
            vals = [bf((xv[h][e].to(F32) * rs) * nv[h][e].to(F32)).to(F32) for h in range(4) for e in range(8)]
            am = F32(0.0)
            for e in range_constexpr(32):
                am = fx.max(am, abs(vals[e]))
            code = e8m0_of(am)
            words = q8words(vals, inv_pow2(code))
            hx.lstore(sm + X8LO + rr * H + bc * 32, fx.Vector.from_elements(words[:4], I32), align=16)
            hx.lstore(sm + X8LO + rr * H + bc * 32 + 16, fx.Vector.from_elements(words[4:], I32), align=16)
            hx.lstore(sm + X8LS + rr * (H // 32) + bc, code.to(fx.Int8))
    gpu.barrier()


@ASTRewriter.transform
def xh_out(c, TR):
    """(fx-v3) the last layer's head input xh = bf(x1 rstd nw) (front_C's normed rows, as the attention front of the last
    layer wrote it) from the TP2 owners' x1 + square-sum partials: CTA b < 16 its 320 cols. c: t b w l sm x1g part2
    pflag ptag nw norm_eps xh."""
    t, b, w, l, sm = c["t"], c["b"], c["w"], c["l"], c["sm"]
    hx.wait_flags(c["pflag"], NOWN, c["ptag"], w, l, 13)
    gpu.barrier()
    wr = fx.min(w, I32(TR - 1))
    pv = [hx.gload(c["part2"] + i64(((l + 32 * k) * 8 + wr) * 32 + 25) * 4, F32) for k in range(NOWN // 32)]
    if w < TR:
        s = F32(0.0)
        for k in range_constexpr(NOWN // 32):
            s = s + pv[k]
        for o in range_constexpr(5):
            s = s + s.shuffle_xor(16 >> o, 32)
        if l == 0:
            hx.lstore(sm + 6144 + w * 4, rsq(s * F32(1.0 / H) + F32(c["norm_eps"])))
    gpu.barrier()
    for it in range_constexpr((TR * 40 + 255) // 256):
        i = t + it * 256
        if i < TR * 40:
            rr = i // 40
            cc = b * 320 + (i % 40) * 8
            rs = hx.lload(sm + 6144 + rr * 4, F32)
            xv = hx.gload(c["x1g"] + i64(rr * H + cc) * 2, BF16, 8, align=16)
            nv = hx.gload(c["nw"] + i64(cc) * 2, BF16, 8, align=16)
            o8 = [bf((xv[e].to(F32) * rs) * nv[e].to(F32)) for e in range(8)]
            hx.stx_dev(c["xh"] + i64(rr * H + cc) * 2, fx.Vector.from_elements(o8, BF16))
    hx.wait_storecnt0()
