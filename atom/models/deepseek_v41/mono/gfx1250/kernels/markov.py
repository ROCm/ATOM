# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""Markov draft-block sampling with the recipe's probabilistic drafting (ATOM dspark_prob_sample.prob_markov_sample): a T > 0
request samples x_k ~ softmax(z / T_d) by Gumbel-max (noise hash32(req * 5 + k, global id, seed_acc ^ SEED_D)); every
request's pre-temperature z of this rank's shard -> qlog[req][k][VR] f32 for the next step's verification (sample).
sp: the step's sampling rows [48][8] (sample layout; request q reads row q * 6). Greedy requests: argmax as before.

DSpark draft-block sampling (Markov head) on vocab-sharded draft logits, 4 ranks.

ATOM dspark._head_and_sample_sharded + dspark_markov_argmax_tp: for k = 0..4 (block positions),
  logits_k[b] = base[b, k] + W1[x_{k-1}[b]] @ W2^T ,  x_k[b] = argmax (lowest index on ties),  x_{-1} = anchor;
W1, W2 [V, 256] bf16 replicated; each rank scores its 32320-row W2 shard against its base-logit shard and the ranks exchange
(max, id) per request. Here: per round, 8 W1 rows staged in LDS (B operand, rows 8..15 zero), every wave one 16-row W2 tile
(A operand, 8 bf16 WMMA 16x16x32), + base logits, packed (ordered key, ~id) max in LDS -> one global atomic max a CTA and
request -> device handoff -> CTA 0 pushes the 8 rank maxima to the peers (tagged 16 B) -> every CTA polls and takes the max
in rank order -> x_k stays in LDS for the next round; CTA 0 writes ids and the W1[x_k] rows (confidence-head input)."""
from __future__ import annotations


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64
from flydsl.compiler.ast_rewriter import ASTRewriter

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import dims as D
from atom.models.deepseek_v41.mono.gfx1250.kernels.device import _v, gptr
from atom.models.deepseek_v41.mono.gfx1250.kernels.sample_common import I64MIN, c32, handoff, latomic_max64, skey, poll_unit, hash32, lshr, LN2, log2f

SEED_D = 0x68E31DA4
SPW = 8

NREQ, TB, RK = D.MAX_REQ, D.DSPARK_W, 256          # requests, block positions, Markov rank
VR = D.VR                        # vocab rows a rank
NT = VR // 16
ES = RK * 2 + 16                  # LDS row stride of the staged W1 rows (bytes)
LX = 16 * ES                      # LDS: x_{k-1} ids [8] i32
LB = LX + 64                      # LDS: per-request CTA best [8] i64
LP = LB + 64                      # LDS: per-source maxima [4][8] i64
LDS = LP + 4 * NREQ * 8
F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16


def i64(x):
    return fx.Int64(x)


def gatomic_max64(addr, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.max, gptr(addr), _v(fx.Int64(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="agent", alignment=8)


def roff(k, src, req):
    return (i64(k * 4 + src) * NREQ + i64(req)) * 16


@ASTRewriter.transform
def markov_once(base, anchors, w1, w2s, ids_out, emb_out, best, cnt, recv, rank, tag0, t, b, sm, sp=None, qlog=None, qids=None):
    t = hx.opaque_v(t)
    b = hx.opaque_s(b)
    w = t // 32
    l = t % 32
    tile = fx.min(b * 8 + w, I32(NT - 1))
    act = b * 8 + w < NT
    wrow = w2s + (i64(tile * 16 + l % 16) * RK + i64((l // 16) * 16)) * 2
    # zero LDS rows 8..15 of the B operand; x_{-1} = anchors
    hx.lstore(sm + (8 + t // 32) * ES + (t % 32) * 16, fx.Vector.filled(4, 0, I32))
    if t < NREQ:
        hx.lstore(sm + LX + t * 4, hx.gload(anchors + i64(t) * 4, I32))
    anyprob = I32(1) > I32(0)
    if fx.const_expr(sp is not None):   # (uniform) any request of the step drafting with T > 0
        np_ = I32(0)
        for q in range_constexpr(NREQ):
            np_ = np_ + (hx.gload(sp + i64(q * (TB + 1) * SPW * 4 + 12), I32) == I32(0)).select(I32(1), I32(0))
        anyprob = np_ > I32(0)
    gpu.barrier()
    for k in range(TB):
        # stage W1[x_{k-1}] rows; prefetch this wave's W2 tile meanwhile
        a = [hx.gload(wrow + i64(ks * 64), BF16, 16) for ks in range(RK // 32)]
        req = t // 32
        x = hx.lload(sm + LX + req * 4, I32)
        hx.lstore(sm + req * ES + (t % 32) * 16, hx.gload(w1 + (i64(x) * RK + i64((t % 32) * 8)) * 2, BF16, 8))
        if t < NREQ:
            hx.lstore(sm + LB + t * 8, I64MIN())
        gpu.barrier()
        acc = fx.Vector.filled(8, 0.0, F32)
        for ks in range_constexpr(RK // 32):
            bv = hx.lload(sm + (l % 16) * ES + (ks * 32 + (l // 16) * 16) * 2, BF16, 16)
            acc = hx.wmma_bf16(a[ks], bv, acc)
        rq = fx.min(l % 16, I32(NREQ - 1))
        if fx.const_expr(sp is not None):   # this request's draft temperature / noise (row rq * 6 of the step's sp)
            spb = sp + i64(rq * (TB + 1)) * (SPW * 4)
            gq = hx.gload(spb + i64(12), I32) != I32(0)
            itd = hx.gload(spb + i64(16), F32)
            sd = hx.gload(spb + i64(28), I32) ^ I32(SEED_D)
        zs = []
        for i in range_constexpr(8):
            j = tile * 16 + (l // 16) * 8 + i
            zs.append(acc[i] + hx.gload(base + (i64(rq * TB + k) * VR + i64(j)) * 4, F32))
        bl = I64MIN()
        for i in range_constexpr(8):   # greedy: argmax of z
            gj = tile * 16 + (l // 16) * 8 + i + rank * VR
            bl = fx.max(bl, fx.Int64(skey(zs[i])) * fx.Int64(1 << 32) + (fx.Int64(c32(0xFFFFFFFF) - gj) & fx.Int64(0xFFFFFFFF)))
        if fx.const_expr(sp is not None):
            if anyprob:   # some request samples (T > 0): Gumbel-max of z / T_d (a batch of greedy requests skips this)
                bp = I64MIN()
                for i in range_constexpr(8):
                    gj = tile * 16 + (l // 16) * 8 + i + rank * VR
                    u = (F32(lshr(hash32(rq * TB + k, gj, sd), 8)) + F32(0.5)) * F32(1.0 / 16777216.0)
                    g = F32(-LN2) * log2f(F32(-LN2) * log2f(u))
                    bp = fx.max(bp, fx.Int64(skey(zs[i] * itd + g)) * fx.Int64(1 << 32) + (fx.Int64(c32(0xFFFFFFFF) - gj) & fx.Int64(0xFFFFFFFF)))
                bl = gq.select(bl, bp)
        if fx.const_expr(qlog is not None):   # q's pre-temperature logits, 8 consecutive columns a lane
            if act & (l % 16 < NREQ) & anyprob:
                qa = qlog + (i64(rq * TB + k) * VR + i64(tile * 16 + (l // 16) * 8)) * 4
                hx.gstore(qa, fx.Vector.from_elements(zs[:4], F32))
                hx.gstore(qa + i64(16), fx.Vector.from_elements(zs[4:], F32))
        if act & (l % 16 < NREQ):
            latomic_max64(sm + LB + rq * 8, bl)
        gpu.barrier()
        if t < NREQ:
            gatomic_max64(best + i64(k * NREQ + t) * 8, hx.lload(sm + LB + t * 8, fx.Int64))
        handoff(cnt + i64(k) * 16, t, 256, site=1)
        tag = tag0 + k
        if t < NREQ:
            mine = hx.gload(best + i64(k * NREQ + t) * 8, fx.Int64)
            hx.lstore(sm + LP + (rank * NREQ + t) * 8, mine)
            if b == 0:
                v = fx.Vector.from_elements([I32(mine & fx.Int64(0xFFFFFFFF)), I32(mine >> 32), I32(0), I32(tag)], I32)
                for o in range_constexpr(4):
                    if fx.const_expr(o != rank):
                        hx.gstore(recv[o] + roff(k, rank, t), v, nt=True)
        src = t // NREQ
        if (t < 4 * NREQ) & (src != rank):
            got = poll_unit(recv[rank] + roff(k, src, t % NREQ), tag, site=9)
            hx.lstore(sm + LP + t * 8, got[0])
            hx.lstore(sm + LP + t * 8 + 4, got[1])
        gpu.barrier()
        if t < NREQ:
            g = I64MIN()
            for o in range_constexpr(4):
                g = fx.max(g, hx.lload(sm + LP + (o * NREQ + t) * 8, fx.Int64))
            xi = c32(0xFFFFFFFF) - I32(g & fx.Int64(0xFFFFFFFF))
            hx.lstore(sm + LX + t * 4, xi)
            if b == 0:
                hx.gstore(ids_out + i64(t * TB + k) * 4, xi)
                if fx.const_expr(qids is not None):   # the drafts qlog was kept for (persistent; checked by the verify)
                    hx.gstore(qids + i64(t * TB + k) * 4, xi)
        gpu.barrier()
        if b == 0:   # W1[x_k] rows for the confidence head: [req][k][256] bf16
            x2 = hx.lload(sm + LX + req * 4, I32)
            hx.gstore(emb_out + (i64(req * TB + k) * RK + i64((t % 32) * 8)) * 2,
                      hx.gload(w1 + (i64(x2) * RK + i64((t % 32) * 8)) * 2, BF16, 8))


def build(rank, rep=1):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def mk(base: Int64, anchors: Int64, w1: Int64, w2s: Int64, ids_out: Int64, emb_out: Int64, best: Int64, cnt: Int64,
           recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64, sp: Int64, qlog: Int64, tagbase: fx.Int32):
        t = fx.thread_idx.x
        b = fx.block_idx.x
        sm = hx.lds_base()
        recv = [recv0, recv1, recv2, recv3]
        for r in range(rep):
            markov_once(base, anchors, w1, w2s, ids_out, emb_out, best + i64(r) * (TB * NREQ * 8), cnt + i64(r) * 128,
                        recv, rank, tagbase + r * 8, t, b, sm, sp, qlog)

    @flyc.jit
    def launch(base: Int64, anchors: Int64, w1: Int64, w2s: Int64, ids_out: Int64, emb_out: Int64, best: Int64, cnt: Int64,
               recv0: Int64, recv1: Int64, recv2: Int64, recv3: Int64, sp: Int64, qlog: Int64, tagbase: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        mk(base, anchors, w1, w2s, ids_out, emb_out, best, cnt, recv0, recv1, recv2, recv3, sp, qlog, tagbase).launch(
            grid=(256,), block=(256,), smem=LDS, stream=stream)

    return launch
