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
request -> count -> CTA 0 (alone waiting for the count) pushes the 8 rank maxima to the peers and to this rank's own slot
(tagged 16 B) -> every CTA polls and takes the max in rank order -> x_k stays in LDS for the next round; CTA 0 writes ids and
the W1[x_k] rows (confidence-head input)."""
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
MBZ = 16384                       # (M1) LDS: the rows' base logits of every position [TB][256][8] f32 (40 KB)
F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16


def i64(x):
    return fx.Int64(x)


def gatomic_max64(addr, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.max, gptr(addr), _v(fx.Int64(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="agent", alignment=8)


@ASTRewriter.transform
def head_local(t):
    """The draft head -> Markov hand-over (tail.draft) without an all-CTA handoff: Markov CTA b reads only the base logits
    head CTA b wrote (markov_once's tile b * 8 + w = the head's tiles 8b .. 8b + 7; its waves past the vocab read a clamped
    tile they never use) -> a CTA-local release (every wave's s_wait_storecnt 0, barrier) / acquire (t0's acquire_se)."""
    hx.wait_storecnt0()
    hx.bar()
    if t == 0:
        hx.acquire_se()
    hx.bar()


def roff(k, src, req):
    return (i64(k * 4 + src) * NREQ + i64(req)) * 16


@ASTRewriter.transform
def markov_once(base, anchors, w1, w2s, ids_out, emb_out, best, cnt, recv, rank, tag0, t, b, sm, sp=None, qlog=None, qids=None,
                ids64=None):
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
    hx.bar()
    # (M1) every load of the rounds that does not depend on x, once: this wave's W2 tile, the rows' base logits of every
    # position (TB x 8 a lane) and the rows' temperature / noise words (the same values the rounds read: bit-identical)
    a = [hx.gload(wrow + i64(ks * 64), BF16, 16) for ks in range(RK // 32)]
    rq = fx.min(l % 16, I32(NREQ - 1))
    for kk in range_constexpr(TB):   # -> LDS (past the sampler's own words): [TB][256 lanes][8] f32
        bzk = [hx.gload(base + (i64(rq * TB + kk) * VR + i64(tile * 16 + (l // 16) * 8 + i)) * 4, F32) for i in range(8)]
        hx.lstore(sm + MBZ + ((kk * 256 + t) * 8) * 4, fx.Vector.from_elements(bzk, F32))
    gq = I32(0) != I32(0)
    itd = F32(1.0)
    sd = I32(0)
    if fx.const_expr(sp is not None):
        spb = sp + i64(rq * (TB + 1)) * (SPW * 4)
        gq = hx.gload(spb + i64(12), I32) != I32(0)
        itd = hx.gload(spb + i64(16), F32)
        sd = hx.gload(spb + i64(28), I32) ^ I32(SEED_D)
    for k in range(TB):
        # stage W1[x_{k-1}] rows (the W2 tile: loaded before the rounds)
        req = t // 32
        x = hx.lload(sm + LX + req * 4, I32)
        hx.lstore(sm + req * ES + (t % 32) * 16, hx.gload(w1 + (i64(x) * RK + i64((t % 32) * 8)) * 2, BF16, 8))
        if t < NREQ:
            hx.lstore(sm + LB + t * 8, I64MIN())
        hx.bar()
        acc = fx.Vector.filled(8, 0.0, F32)
        for ks in range_constexpr(RK // 32):
            bv = hx.lload(sm + (l % 16) * ES + (ks * 32 + (l // 16) * 16) * 2, BF16, 16)
            acc = hx.wmma_bf16(a[ks], bv, acc)
        bzv = hx.lload(sm + MBZ + ((k * 256 + t) * 8) * 4, F32, 8)   # (M1: the base logits, landed before the rounds)
        zs = [acc[i] + bzv[i] for i in range(8)]
        bl = I64MIN()
        for i in range_constexpr(8):   # greedy: argmax of z
            gj = tile * 16 + (l // 16) * 8 + i + rank * VR
            bl = fx.max(bl, fx.Int64(skey(zs[i])) * fx.Int64(1 << 32) + (fx.Int64(c32(0xFFFFFFFF) - gj) & fx.Int64(0xFFFFFFFF)))
        if fx.const_expr(sp is not None):
            if anyprob:   # some request samples (T > 0): Gumbel-max of z / T_d (a batch of greedy requests skips this)
                bp = I64MIN()
                for i in range_constexpr(8):
                    gj = tile * 16 + (l // 16) * 8 + i + rank * VR
                    u = (F32(lshr(hash32(rq * TB + k, gj, sd), 9)) + F32(0.5)) * F32(1.0 / 8388608.0)   # (23 bits: u < 1 exactly; 24 bits + 0.5 rounded to 1.0 in fp32)
                    g = F32(-LN2) * log2f(F32(-LN2) * log2f(u))
                    bp = fx.max(bp, fx.Int64(skey(zs[i] * itd + g)) * fx.Int64(1 << 32) + (fx.Int64(c32(0xFFFFFFFF) - gj) & fx.Int64(0xFFFFFFFF)))
                bl = gq.select(bl, bp)
        if act & (l % 16 < NREQ):
            latomic_max64(sm + LB + rq * 8, bl)
        hx.bar()
        # the CTA's maxima -> the rank's (global atomics, wave 0) -> count; only CTA 0 waits for the count, reads the rank
        # maxima and pushes them to the peers and to this rank's own slot; every other CTA goes on to poll all 4 sources
        # (no 256-CTA wait on the counter, no 256-CTA re-read of the maxima on CTA 0's path)
        if t < NREQ:
            gatomic_max64(best + i64(k * NREQ + t) * 8, hx.lload(sm + LB + t * 8, fx.Int64))
        if t < 32:
            hx.wait_storecnt0()   # (the release of wave 0's atomics, as device.handoff's)
        if t == 0:
            hx.atomic_add(cnt + i64(k) * 16, 1)
        tag = tag0 + k
        if b == 0:
            if t == 0:
                hx.wait_count(cnt + i64(k) * 16, 256, 1)
            hx.bar()
            if t < NREQ:
                mine = hx.gload(best + i64(k * NREQ + t) * 8, fx.Int64)
                hx.lstore(sm + LP + (rank * NREQ + t) * 8, mine)
                v = fx.Vector.from_elements([I32(mine & fx.Int64(0xFFFFFFFF)), I32(mine >> 32), I32(0), I32(tag)], I32)
                for o in range_constexpr(4):
                    hx.gstore(recv[o] + roff(k, rank, t), v, nt=True)
        # q's pre-temperature logits, 8 consecutive columns a lane -- stored here, past the count and CTA 0's push, not before
        # the local maximum (~1 MB a round over the CTAs: there they held the round's computation 1.5 -> 1.0 us, draft-tail fixture)
        if fx.const_expr(qlog is not None):
            if act & (l % 16 < NREQ) & anyprob:
                qa = qlog + (i64(rq * TB + k) * VR + i64(tile * 16 + (l // 16) * 8)) * 4
                hx.gstore(qa, fx.Vector.from_elements(zs[:4], F32))
                hx.gstore(qa + i64(16), fx.Vector.from_elements(zs[4:], F32))
        src = t // NREQ
        if (t < 4 * NREQ) & ((src != rank) | (b != 0)):
            got = poll_unit(recv[rank] + roff(k, src, t % NREQ), tag, site=9)
            hx.lstore(sm + LP + t * 8, got[0])
            hx.lstore(sm + LP + t * 8 + 4, got[1])
        hx.bar()
        if t < NREQ:
            g = I64MIN()
            for o in range_constexpr(4):
                g = fx.max(g, hx.lload(sm + LP + (o * NREQ + t) * 8, fx.Int64))
            xi = c32(0xFFFFFFFF) - I32(g & fx.Int64(0xFFFFFFFF))
            hx.lstore(sm + LX + t * 4, xi)
            if b == 0:
                hx.gstore(ids_out + i64(t * TB + k) * 4, xi)
                if fx.const_expr(ids64 is not None):   # the host's draft ids (int64: ``runner.draft_ids``)
                    hx.gstore(ids64 + i64(t * TB + k) * 8, fx.Int64(xi))
                if fx.const_expr(qids is not None):   # the drafts qlog was kept for (persistent; checked by the verify)
                    hx.gstore(qids + i64(t * TB + k) * 4, xi)
        hx.bar()
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
            grid=(256,), block=(256,), smem=MBZ + TB * 256 * 8 * 4, stream=stream)

    return launch
