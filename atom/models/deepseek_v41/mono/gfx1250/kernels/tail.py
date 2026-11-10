# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The step tail as two callable blocks of the step kernel:

    for Ly in range(NL):            layer_body(Ly)        # the target loop stays tail-free (instruction cache)
    tail.target(tp, t, b, sm, st, cnt_tail, recv, rank)   # head -> 4-rank top-p sampling -> verify
    for Ly in range(NL, NL + 3):    layer_body(Ly)        # draft layers (second copy of the layer code)
    tail.draft(tp, t, b, sm, st, cnt_tail, recv, rank)    # draft head -> 4-rank Markov block sampling

Pointers come from a device table `tp` (TPTRS order, 8 B each), read wave-uniform like the layer table (kernel args
push SGPRs into spills). `recv` = the 4 ranks' fine-grained receive buffers; the tail uses offsets SOFF (sampling, 1.18 MB)
and SOFF2 (Markov, 2.5 KB) after the TP regions -> allocate >= 16*40960*16 + 2 MB. `cnt_tail` = 512 B of zeroed counters
(reset every step), stamps (if st != 0) at slots 19 head, 20 sampling, 21 verify, 22 draft head, 24 Markov.
Inputs/outputs (all per rank): xh [48][5120] bf16 final-normed target hidden; hw [32320 tiles packed][5120] bf16 head shard
(head v3 layout); logits [48][32320] f32; s0..s3 sampling records; sids [48] i32; dids [8][5] i32 previous drafts;
outtok [8][6], verdict [3][8], anc [8]; dh [48][5120] bf16 draft hidden (rows req*5+k); dlog [48][32320] f32;
mw1 [129280][256], mw2 shard [32320][256] bf16; mids [8][5], memb [8][5][256]; mbest [40] i64 (= INT64_MIN each step)."""
from __future__ import annotations


import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import layout as LY
from atom.models.deepseek_v41.mono.gfx1250.kernels import head as HF
from atom.models.deepseek_v41.mono.gfx1250.kernels import sample_common as SF
from atom.models.deepseek_v41.mono.gfx1250.kernels import sample as SPF
from atom.models.deepseek_v41.mono.gfx1250.kernels import markov as MPF
from atom.models.deepseek_v41.mono.gfx1250.kernels import draft_ctx as CX
from atom.models.deepseek_v41.mono.gfx1250.kernels.attention import handoff

TPTRS = ['xh', 'hw', 'logits', 's0', 's1', 's2', 's3', 'sids', 'dids', 'outtok', 'verdict', 'anc', 'dh', 'dlog', 'mw1',
         'mw2', 'mids', 'memb', 'mbest', 'sparam', 'qlog', 'prow', 'qids']
# sparam [48][8] (32 B a row, host-filled a step; sample.SPW): invT, topp, seed, greedy | invTd, hasq, qsrc, seed_acc.
# T > 0 rows use the recipe's prob drafting + block verification (markov / sample; ATOM dspark_prob_sample
# PROB_DRAFT + BLOCK_VERIFY), greedy rows the argmax match. qlog [8][5][32320] f32: this rank's draft logits of the step
# (read by the next step's verification), qids [8][5] i32 the drafts it was kept for (both persistent across steps);
# prow [48][8] f32 scratch.
SOFF = 16 * 40960 * 16
SOFF2 = SOFF + 4 * 4 * 48 * SF.UNITS * 16
HEAD_LDS = 2 * HF.geom(2, 4)[3]      # LDS the head needs (launch smem >= this)


def i64(x):
    return fx.Int64(x)


def _ufl(x):
    return fx.Int32(hx.intrin(hx.i32(), "llvm.amdgcn.readfirstlane.i32", [fx.Int32(x)]))


def tptr(tp, name):
    v = hx.gload(tp + i64(TPTRS.index(name) * 8), fx.Int32, 2)
    return (fx.Int64(_ufl(v[1])) << 32) | (fx.Int64(_ufl(v[0])) & fx.Int64(0xFFFFFFFF))


@ASTRewriter.transform
def _stamp(st, b, t, k, on):
    if fx.const_expr(on):
        if t == 0:
            hx.gstore(st + i64(b) * (32 * 8) + i64(k * 8), hx.clock())


def _ctxp(ctx, t, b, T):   # (plain python) the draft context KV's 4 stages, run inside the sampling's exchanges
    return None if ctx is None else CX.ctx_parts(ctx, t, b, T, 65536)


@ASTRewriter.transform
def target(tp, t, b, sm, st, tc, recv, rank, stamps=True, head_d=4, ctx=None, T=None):
    P = {k: tptr(tp, k) for k in ('xh', 'hw', 'logits', 's0', 's1', 's2', 's3', 'sids', 'dids', 'outtok', 'verdict', 'anc')}
    handoff(tc, t, False)   # LY.HEAD (offset 0)
    HF.head_once_r(P['xh'], P['hw'], P['logits'], t, t % 32, t // 32, sm, b, head_d, 2, 4)
    handoff(tc + i64(LY.HEAD_DONE), t, False)
    _stamp(st, b, t, 19, stamps)
    # sampling + verification in one (outtok / verdict / anc written by CTA 0 after its last handoff)
    SPF.sample_once_p(P['logits'], P['s0'], P['s1'], P['s2'], P['s3'], P['sids'], tc + i64(LY.SAMPLE), t, b, sm, HF.VR,
                      [r + i64(SOFF) for r in recv], rank, 1, tptr(tp, 'sparam'), tptr(tp, 'qlog'), P['dids'],
                      tptr(tp, 'prow'), P['outtok'], P['verdict'], P['anc'], tptr(tp, 'qids'),
                      ctxp=_ctxp(ctx, t, b, T))
    _stamp(st, b, t, 20, stamps)
    handoff(tc + i64(LY.VERIFY_DONE), t, False)
    _stamp(st, b, t, 21, stamps)


@ASTRewriter.transform
def draft(tp, t, b, sm, st, tc, recv, rank, stamps=True, head_d=4):
    P = {k: tptr(tp, k) for k in ('hw', 'anc', 'dh', 'dlog', 'mw1', 'mw2', 'mids', 'memb', 'mbest')}
    HF.head_once_r(P['dh'], P['hw'], P['dlog'], t, t % 32, t // 32, sm, b, head_d, 2, 4)
    handoff(tc + i64(LY.DRAFT_HEAD_DONE), t, False)
    _stamp(st, b, t, 22, stamps)
    MPF.markov_once(P['dlog'], P['anc'], P['mw1'], P['mw2'], P['mids'], P['memb'], P['mbest'], tc + i64(LY.MARKOV),
                    [r + i64(SOFF2) for r in recv], rank, 101, t, b, sm, tptr(tp, 'sparam'), tptr(tp, 'qlog'), tptr(tp, 'qids'))
    _stamp(st, b, t, 24, stamps)
