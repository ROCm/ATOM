# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The indexer's query half for the whole-layer kernel (FULL / REINDEX layers): iw, iq, iquant.

The reduction orders are this kernel's own
(native gfx1250 is not the gfx950 wv_splitk order either); validation is the original tolerance.

    stage_iq    (CTAs 0..255, waves 0..2): iq rows b*16..+15 of the replicated indexer wq_b [4096][1280] (plain row-major
                FP8, 32x32 e8m0 block scales; the bind un-preshuffles the checkpoint's (16,16) layout) times the q latent
                qr8 / qrs (stage A's MXFP8, before q_norm's rsqrt, which the epilogue applies as stage C does), bf16, GPT-J
                RoPE of each 128-dim head's dims 64..127 (the layer's table, global for every indexer layer), bf16 -> IQ.
    stage_iw    (CTAs IW0..IW0+IW_SLICES-1, waves 0..5): a K slice of weights_proj [32][5120] bf16 times the normed rows
                (the attention front's axn), one 16x16 tile a wave over the slice's K steps -> f32 partials IWP[slice].
    stage_iquant (S x 32 (token, head) rows, a wave each): iw = bf16(sum of the slices' partials in slice order); the row's
                128 IQ values: amax (seed 1e-10), scale = amax * f32(1/448), inv = 1 / scale, e4m3(clamp(x inv)) -> iq_out
                [S][32][128]; iqs_out = scale; iw_out = (w scale) / 64.
"""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import CTAS, QL, HID, T, NORM_EPS, ld_op, mx_loop, rope_cs, rsq

F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16
HEADS, DIM = D.INDEX_HEADS, D.INDEX_DIM
IQ_ROWS = HEADS * DIM            # 4096
IW_SLICES = 32                   # K slices of weights_proj (5120 / 32 = 160 = 5 bf16 WMMA steps)
IW_K = HID // IW_SLICES
IW0 = 0                          # first CTA of stage_iw
AMAX_SEED = 1e-10
WSCALE = 1.0 / 64.0              # 128^-0.5 * 32^-0.5, exact


def i64(x):
    return fx.Int64(x)


@ASTRewriter.transform
def stage_iq(c):
    """iq rows n0 = b*16..+15 (one 16-row tile a CTA); waves 0..2 the three 16-token M tiles."""
    t, b, w, l, MT = c["t"], c["b"], c["w"], c["l"], c["MT"]
    if (w < 3) & (w < MT):
        mi = w
        n0 = b * 16
        arow = c["qr8"] + i64(mi * 16 + l % 16) * QL
        brow = c["iqw"] + i64(n0 + l % 16) * QL

        def fa(i):
            kb = i * 128
            return ld_op(arow, kb, l), hx.gload(c["qrs"] + i64((mi * 16 + l % 16) * (QL // 32) + kb // 32), I32, align=4)

        def fb(i):
            kb = i * 128
            return ld_op(brow, kb, l), hx.gload(c["iqws"] + i64(((n0 + l % 16) // 32) * (QL // 32) + kb // 32), I32, align=4)

        acc = mx_loop(QL // 128, fa, fb)
        cv = [acc[i] * rsq(hx.gload(c["ss"] + i64(mi * 16 + (l // 16) * 8 + i) * 4, F32) / F32(QL) + F32(NORM_EPS)) for i in range(8)]
        d = (n0 + l % 16) % DIM
        is_rope = d >= DIM - 64
        fi = fx.max((d - (DIM - 64)) // 2, I32(0))
        odd = (l & 1) == 1
        for i in range_constexpr(8):
            r = mi * 16 + (l // 16) * 8 + i
            x = cv[i].to(BF16).to(F32)
            y = cv[i].shuffle_xor(1, 32).to(BF16).to(F32)
            cs, sn = rope_cs(c, r, fi)
            v = is_rope.select(odd.select(y * sn + x * cs, x * cs - y * sn), x)
            c["stx"](c["iq"] + i64(r * IQ_ROWS + n0 + l % 16) * 2, v.to(BF16))


@ASTRewriter.transform
def stage_iw(c):
    """Slice s = b - IW0 of K: waves 0..5 = (M tile w % 3, N tile w // 3) of [tokens][32 heads]."""
    t, b, w, l, MT = c["t"], c["b"], c["w"], c["l"], c["MT"]
    s = b - IW0
    if (s >= 0) & (s < IW_SLICES) & (w < 6) & (w % 3 < MT):
        mi = w % 3
        ni = w // 3
        k0 = s * IW_K
        acc = fx.Vector.filled(8, 0.0, F32)
        av = [hx.gload(c["axn"] + i64((mi * 16 + l % 16) * HID + k0 + ks * 32 + (l // 16) * 16) * 2, BF16, 16)
              for ks in range(IW_K // 32)]
        bv = [hx.gload(c["iww"] + i64((ni * 16 + l % 16) * HID + k0 + ks * 32 + (l // 16) * 16) * 2, BF16, 16)
              for ks in range(IW_K // 32)]
        for ks in range_constexpr(IW_K // 32):
            acc = hx.wmma_bf16(av[ks], bv[ks], acc)
        for i in range_constexpr(8):
            tok = mi * 16 + (l // 16) * 8 + i
            c["stx"](c["iwp"] + i64((s * T + tok) * HEADS + ni * 16 + l % 16) * 4, acc[i])


@ASTRewriter.transform
def stage_iquant(c):
    """(token, head) row q = b * 8 + w (S x 32 rows over the grid's waves), 4 dims a lane."""
    t, b, w, l, TR = c["t"], c["b"], c["w"], c["l"], c["TR"]
    q = b * 8 + w
    if q < TR * HEADS:
        tok = q // HEADS
        h = q % HEADS
        ws = F32(0.0)
        for s in range_constexpr(IW_SLICES):
            ws = ws + hx.gload(c["iwp"] + i64((s * T + tok) * HEADS + h) * 4, F32)
        wv = ws.to(BF16).to(F32)
        xv = hx.gload(c["iq"] + i64(tok * IQ_ROWS + h * DIM + 4 * l) * 2, BF16, 4)
        x = [xv[e].to(F32) for e in range(4)]
        am = fx.max(fx.max(fx.max(abs(x[0]), abs(x[1])), fx.max(abs(x[2]), abs(x[3]))), F32(AMAX_SEED))
        for o in range_constexpr(5):
            am = fx.max(am, am.shuffle_xor(16 >> o, 32))
        scale = am * F32(1.0 / 448.0)
        inv = F32(1.0) / scale
        cl = [fx.min(fx.max(x[e] * inv, F32(-448.0)), F32(448.0)) for e in range(4)]
        lo = hx.cvt_pk_fp8_w(0, cl[0], cl[1], False)
        word = hx.cvt_pk_fp8_w(lo, cl[2], cl[3], True)
        c["stx"](c["iq_out"] + i64((tok * HEADS + h) * DIM + 4 * l), word)
        if l == 0:
            c["stx"](c["iqs_out"] + i64(tok * HEADS + h) * 4, scale)
            c["stx"](c["iw_out"] + i64(tok * HEADS + h) * 4, (wv * scale) * F32(WSCALE))
