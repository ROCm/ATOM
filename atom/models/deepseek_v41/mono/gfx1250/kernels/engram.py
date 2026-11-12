# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""Engram for the whole-layer kernel (layers 1 and 14). Semantics: atom/model_ops/engram/device/gate.py
(engram_post_wkv_reference).

Row k - 1 of the layer table carries lflags bit 7 ("the next layer has Engram") and that layer's Engram pointers: engw
[25600][6144] e4m3 (rows: 4 branch keys of 5120, then the shared value), engws [800][192] e8m0 (32 x 32 blocks), enggw
[4][5120] f32 (q_weight * k_weight), engemb [T][6144] bf16 (ATOM's staged rows, read in place).

    stage_engq   (layer k - 1, before A; every CTA): engemb -> MXFP8 eng8 [T][6144] / engs [T][192] (one 32-group a thread)
    stage_engkv  (layer k - 1, after A's handoff): kv = eng8 . engw^T, bf16 -> engkv [T][25600]; a wave task = (16-column
                 N tile, all MT M tiles)
The gate itself (R' += gate . value) is the attention front's (front._engram_gate)."""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from flydsl.expr import gpu
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import CTAS, T, e8m0_of, inv_pow2, ld_op, q8words

F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16
EK = 6144            # engram embedding width
EG = EK // 32        # its 32-groups
HC, HD5 = D.HC, D.HID    # branches, hidden
EN = (HC + 1) * HD5  # 25600 kv columns
NT = EN // 16        # N tiles


def i64(x):
    return fx.Int64(x)


@ASTRewriter.transform
def stage_engq(c):
    """Group gi = b * 256 + t of T x EG: 32 bf16 -> amax -> e8m0 code -> 8 MXFP8 words (the kernel's q8words / e8m0_of,
    as every other activation quantization here). Rows >= TR: zeros (code of 0), up to the build's row tiles (48 rows at
    least: the narrow builds' extent)."""
    t, b = c["t"], c["b"]
    gi = b * 256 + t
    if gi < max(D.NARROW_T, c["MT"] * 16) * EG:
        row = gi // EG
        g = gi % EG
        live = row < c["TR"]
        src = c["engemb"] + i64(fx.min(row, c["TR"] - 1) * EK + g * 32) * 2
        vs = [hx.gload(src + i64(16 * k), BF16, 8) for k in range(4)]
        vals = [live.select(vs[k][e].to(F32), F32(0.0)) for k in range(4) for e in range(8)]
        am = F32(0.0)
        for e in range_constexpr(32):
            am = fx.max(am, abs(vals[e]))
        code = e8m0_of(am)
        words = q8words(vals, inv_pow2(code))
        dst = c["eng8"] + i64(row * EK + g * 32)
        c["stx"](dst, fx.Vector.from_elements(words[:4], I32))
        c["stx"](dst + i64(16), fx.Vector.from_elements(words[4:], I32))
        c["stx"](c["engs"] + i64(row * EG + g), code.to(fx.Int8))


ED = 4                 # K128 steps of B operands in flight
KCH_N = 3072           # K bytes of A staged in LDS at a time up to 3 row tiles (2 chunks)
KCH_W = 1536           # past 3 row tiles (4 chunks: 96 rows of A in LDS)


@ASTRewriter.transform
def stage_engkv(c):
    """kv (see the module doc): a wave task = one 16-column N tile for all MT M tiles (1600 tasks = CTAs 0..199 x 8
    waves: every weight row read once). A (the MT*16 eng8 rows) is staged per CTA in LDS, KCH bytes at a time (async
    copies, issued after the chunk's first ED weight operands), so the A re-reads hit LDS, not L2. Runtime loops over
    the chunks and their K128 steps (code size: this sits in the layer loop). LDS [0, MT*16*RS + MT*16*KCH/32) is free
    here (between attention stages; the layer-resident residual streams sit at RKOFF)."""
    t, b, w, l, MT, sm = c["t"], c["b"], c["w"], c["l"], c["MT"], c["sm"]
    KCH = KCH_N if MT <= 3 else KCH_W
    SPC = KCH // 128       # K128 steps a chunk
    RS = KCH + 16          # padded LDS row of the A chunk (bank spread)
    RA = MT * 16
    oa = sm
    osc = sm + RA * RS
    if b * 8 < NT:
        n0 = (b * 8 + w) * 16
        brow = c["engw"] + i64(n0 + l % 16) * EK
        bsc = c["engws"] + i64(((n0 + l % 16) // 32) * EG)

        def fb(k):
            return ld_op(brow, k * 128, l), hx.gload(bsc + i64(k * 4), I32, align=4)

        acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(MT)]
        for ch in range(EK // KCH):
            k0 = ch * SPC
            ring = [fb(k0 + i) for i in range(ED)]
            for it in range((RA * KCH // 16 + 255) // 256):   # A chunk: RA rows x KCH bytes, 16 B a copy
                e = t + it * 256
                if e < RA * KCH // 16:
                    row = e // (KCH // 16)
                    col = e % (KCH // 16)
                    hx.async_to_lds_b128(c["eng8"] + i64(row * EK + ch * KCH + col * 16), oa + row * RS + col * 16)
            for it in range_constexpr((RA * KCH // 32 // 16 + 255) // 256):   # its scale codes: RA rows x KCH / 32 bytes
                e = t + it * 256
                if e < RA * KCH // 32 // 16:
                    row = e // (KCH // 32 // 16)
                    col = e % (KCH // 32 // 16)
                    hx.async_to_lds_b128(c["engs"] + i64(row * EG + ch * (KCH // 32) + col * 16), osc + row * (KCH // 32) + col * 16)
            hx.wait_asynccnt0()
            gpu.barrier()
            for ks in range(0, SPC, ED):
                for i in range_constexpr(ED):
                    bv, bs = ring[i]
                    kk = ks + i
                    for mi in range_constexpr(MT):
                        ar = oa + (mi * 16 + l % 16) * RS + kk * 128 + (l // 16) * 16
                        parts = [hx.lload(ar + cc * 32, I32, 4) for cc in range(4)]
                        av = fx.Vector.from_elements([parts[cc][e_] for cc in range(4) for e_ in range(4)], I32)
                        asc = hx.lload(osc + (mi * 16 + l % 16) * (KCH // 32) + kk * 4, I32)
                        acc[mi] = hx.wmma_mx(av, asc, bv, bs, acc[mi])
                    ring[i] = fb(k0 + fx.min(ks + i + ED, I32(SPC - 1)))   # (the chunk's last ED refills: unused)
            gpu.barrier()   # every wave done with this chunk before the next overwrites it
        for mi in range_constexpr(MT):
            for i in range_constexpr(8):
                r = mi * 16 + (l // 16) * 8 + i
                c["stx"](c["engkv"] + i64(r * EN + n0 + l % 16) * 2, acc[mi][i].to(BF16))
