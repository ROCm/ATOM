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
def front_any(c, stf):
    """Both fronts with one copy of the code (c["isffn"] runtime: FFN front = + router D / E1, int8 block codes, top-6
    merge; attention front = int32 codes, Sinkhorn only). stf(k): the FFN front's stamps. c["eng"] (optional, runtime):
    the attention front before an Engram layer (front_A's _engram_gate)."""
    front_A(c)
    stf(2)
    front_B(c)
    stf(3)
    cluster_sync()
    stf(4)
    front_C(c)
    stf(5)
    if c["isffn"]:
        front_D(c)
        cluster_sync()
        front_E1(c)
        stf(6)
        cluster_sync()
    front_E2(c)
    gpu.barrier()
    stf(7)
