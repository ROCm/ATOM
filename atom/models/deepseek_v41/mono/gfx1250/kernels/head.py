# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The vocab-sharded LM head of one TP rank.
logits[48][32320] fp32 = x[48][5120] bf16 . W[32320][5120]^T bf16 (this rank's vocab slice; no logits all-gather).
256 CTAs x 8 waves, CTA b owns vocab tiles 8b..8b+7 (2020 tiles of 16). The 8 waves form 8/TW groups of TW waves; a group
owns TW tiles and splits K in TW parts: wave kq of the group streams the TW tiles' weights over K part kq and reuses each
x fragment it reads from LDS for TW tiles (LDS operand traffic / TW: without reuse the LDS reads were a large share of the head).
Weights pre-packed host-side as [tile][k-step][lane][16 bf16] (one contiguous 1 KB a wave load). x staged through LDS in
NPH phases (double-buffered async copies; a phase holds every K part's next CK columns). Partial sums of the TW waves are
reduced through LDS at the end."""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.compiler.ast_rewriter import ASTRewriter

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
# the LM-head weights (331 MB a rank) are streamed non-temporal: default caching evicted the layer loop's hot scratch

T, HID, VR = 16, D.HID, D.VR   # (c1: 6 target / 5 draft rows) one 16-row WMMA group, not 48 rows
NN = T // 16                  # 16-row WMMA groups
NT = VR // 16                 # 2020 vocab tiles
F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16


def i64(x):
    return fx.Int64(x)


def geom(TW, NPH):
    CK = HID // (TW * NPH)          # columns of one K part staged per phase
    XS = CK * 2 + 16                # LDS row stride (bytes), padded
    SUB = T * XS                    # one K part's slab
    BUF = TW * SUB                  # one phase buffer
    return CK, XS, SUB, BUF


@ASTRewriter.transform
def head_once_r(x, wb, out, t, l, w, sm, b, D, TW, NPH):
    """head_once with the K phases as a runtime loop (1/NPH of the code: the step tail runs once a step, cold
    instruction fetch of the unrolled code is slow under the weight stream). Needs SPP % D == 0."""
    assert (HID // (TW * NPH) // 32) % D == 0
    CK, XS, SUB, BUF = geom(TW, NPH)
    NS = HID // TW // 32            # k-steps of a wave
    SPP = CK // 32                  # k-steps a phase
    t = hx.opaque_v(t)   # addresses recomputed each pass, not hoisted out of the repetition loop (VGPR blow-up)
    l = hx.opaque_v(l)
    w = hx.opaque_v(w)
    b = hx.opaque_s(b)
    kq = w % TW
    grp = w // TW
    tiles = [fx.min(b * 8 + grp * TW + i, I32(NT - 1)) for i in range(TW)]
    wt = [wb + i64(tiles[i]) * (16 * HID * 2) + i64(kq * NS) * 1024 + i64(l) * 32 for i in range(TW)]
    acc = [[fx.Vector.filled(8, 0.0, F32) for _ in range(NN)] for _ in range(TW)]

    def stage(ph):   # every K part's columns [part*HID/TW + ph*CK, +CK) -> buffer ph % 2 (async)
        for it in range_constexpr(T * TW * (CK // 8) // 256):
            i = t + it * 256
            part = i // (T * (CK // 8))
            r = i % (T * (CK // 8))
            row = r // (CK // 8)
            pc = r % (CK // 8)
            hx.async_to_lds_b128(x + (i64(row) * HID + i64(part * (HID // TW) + ph * CK + pc * 8)) * 2,
                                 sm + (ph % 2) * BUF + part * SUB + row * XS + pc * 16)

    gpu.barrier()
    stage(0)
    A = [[hx.gload(wt[i] + i64(s * 1024), BF16, 16, nt=True) for s in range(D)] for i in range(TW)]
    for ph in range(NPH):
        hx.wait_asynccnt0()
        gpu.barrier()
        if ph + 1 < NPH:
            stage(ph + 1)
        xb = sm + (ph % 2) * BUF + kq * SUB
        for kk in range_constexpr(SPP):
            s = ph * SPP + kk
            a = [A[i][kk % D] for i in range(TW)]
            sp = fx.min(s + D, I32(NS - 1))   # last phase re-loads the final step (unused) instead of a branch
            for i in range_constexpr(TW):
                A[i][kk % D] = hx.gload(wt[i] + i64(sp) * 1024, BF16, 16, nt=True)
            for n in range_constexpr(NN):
                bv = hx.lload(xb + (n * 16 + l % 16) * XS + (kk * 32 + (l // 16) * 16) * 2, BF16, 16)
                for i in range_constexpr(TW):
                    acc[i][n] = hx.wmma_bf16(a[i], bv, acc[i][n])
            hx.intrin(None, "llvm.amdgcn.sched.barrier", [fx.Int32(0)])
    if fx.const_expr(TW == 1):
        if b * 8 + w < NT:
            for n in range_constexpr(NN):
                for e in range_constexpr(8):
                    hx.gstore(out + (i64(n * 16 + l % 16) * VR + i64(tiles[0] * 16 + (l // 16) * 8 + e)) * 4, acc[0][n][e])
    else:
        # partials of the group through LDS: slot [tile i][kq][n][lane] vec8 f32; wave kq reduces tile kq
        gpu.barrier()
        gbase = sm + grp * (TW * TW * 3 * 32 * 32)
        for i in range_constexpr(TW):
            for n in range_constexpr(NN):
                hx.lstore(gbase + (((i * TW + kq) * 3 + n) * 32 + l) * 32, acc[i][n])
        gpu.barrier()
        tile = b * 8 + grp * TW + kq
        if tile < NT:
            for n in range_constexpr(NN):
                sacc = hx.lload(gbase + (((kq * TW + 0) * 3 + n) * 32 + l) * 32, F32, 8)
                for q in range_constexpr(1, TW):
                    sacc = sacc + hx.lload(gbase + (((kq * TW + q) * 3 + n) * 32 + l) * 32, F32, 8)
                for e in range_constexpr(8):
                    hx.gstore(out + (i64(n * 16 + l % 16) * VR + i64(tile * 16 + (l // 16) * 8 + e)) * 4, sacc[e])
