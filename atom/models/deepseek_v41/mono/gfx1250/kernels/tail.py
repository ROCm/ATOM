# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The step tail as two callable blocks of the step kernel:

    for Ly in range(NL):            layer_body(Ly)        # the target loop stays tail-free (instruction cache)
    tail.target(tp, t, b, sm, cnt_tail, recv, rank)   # head -> 4-rank top-p sampling -> verify
    for Ly in range(NL, NL + 3):    layer_body(Ly)        # draft layers (second copy of the layer code)
    tail.draft(tp, t, b, sm, cnt_tail, recv, rank)    # draft head -> 4-rank Markov block sampling

Pointers come from a device table `tp` (TPTRS order, 8 B each), read wave-uniform like the layer table (kernel args
push SGPRs into spills). `recv` = the 4 ranks' fine-grained receive buffers; the tail uses offsets SOFF (sampling, 1.18 MB)
and SOFF2 (Markov, 2.5 KB) after the TP regions -> allocate >= 16*40960*16 + 2 MB. `cnt_tail` = 512 B of zeroed counters
(reset every step).
Inputs/outputs (all per rank): xh [48][5120] bf16 final-normed target hidden; hw [32320 tiles packed][5120] bf16 head shard
(head v3 layout); logits [48][32320] f32; s0..s3 sampling records; sids [48] i32; dids [8][5] i32 previous drafts;
outtok [8][6], verdict [3][8], anc [8]; dh [48][5120] bf16 draft hidden (rows req*5+k); dlog [48][32320] f32;
mw1 [129280][256], mw2 shard [32320][256] bf16; mids [8][5] (i32; mids64 the same as i64), memb [8][5][256]; mbest [40] i64 (= INT64_MIN each step)."""
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
         'mw2', 'mids', 'memb', 'mbest', 'sparam', 'qlog', 'prow', 'qids', 'mids64', 'tcom', 'outsl', 'outh']
# sparam [48][8] (32 B a row, host-filled a step; sample.SPW): invT, topp, seed, greedy | invTd, hasq, qsrc, seed_acc.
# T > 0 rows use the recipe's prob drafting + block verification (markov / sample; ATOM dspark_prob_sample
# PROB_DRAFT + BLOCK_VERIFY), greedy rows the argmax match. qlog [8][5][32320] f32: this rank's draft logits of the step
# (read by the next step's verification), qids [8][5] i32 the drafts it was kept for (both persistent across steps);
# prow [48][8] f32 scratch.
SOFF = 16 * 40960 * 16
SOFF2 = SOFF + 4 * 4 * 48 * SF.UNITS * 16
# (ctx TP) draft_ctx's main_proj exchange slots: past the sampling slots actually used (4 passes x 4 sources x ROWS rows x
# UNITS units), before the Markov words
assert CX.CX_OFF >= SOFF + 4 * 4 * SF.ROWS * SF.UNITS * 16 and CX.CX_OFF + CX.TPR * CX.CX_SRC <= SOFF2
CX_TAG_LAYER = hx.LAYER_SLOTS - 1   # (ctx TP) the exchange's tag slot: past every layer's (target + draft + entries)
HEAD_LDS = 2 * HF.geom(2, 4)[3]      # LDS the head needs (launch smem >= this)


def i64(x):
    return fx.Int64(x)


def _ufl(x):
    return fx.Int32(hx.intrin(hx.i32(), "llvm.amdgcn.readfirstlane.i32", [fx.Int32(x)]))


def tptr(tp, name):
    v = hx.gload(tp + i64(TPTRS.index(name) * 8), fx.Int32, 2)
    return (fx.Int64(_ufl(v[1])) << 32) | (fx.Int64(_ufl(v[0])) & fx.Int64(0xFFFFFFFF))


# the target tail's code (head, sampling + verification, the draft context stages) -> L2 at its start (device.
# code_prefetch_near, bounded by the launch's recorded end; one CTA in 16: a cluster shares the L2): it runs once a step,
# after the layer loop's weight stream (ktrace gapS, 133K: the sampling passes' computation before each exchange 7-14 us)
TAIL_L2_KB = 96
TAIL_L2_ONE_IN = 16


@ASTRewriter.transform
def target(tp, t, b, sm, tc, recv, rank, head_d=4, ctx=None, T=None, kv_fp4=False, codeend=None, done=True, tep=None):
    P = {k: tptr(tp, k) for k in ('xh', 'hw', 'logits', 's0', 's1', 's2', 's3', 'sids', 'dids', 'outtok', 'verdict', 'anc')}
    if fx.const_expr(codeend is not None):   # the head's and the sampling's code -> L2 (see TAIL_L2_KB)
        pc0 = hx.getpc()
        if b % TAIL_L2_ONE_IN == 0:
            hx.code_prefetch_near(t, b, codeend, TAIL_L2_KB, pc=pc0)
    handoff(tc, t, False)   # LY.HEAD (offset 0)
    HF.head_once_r(P['xh'], P['hw'], P['logits'], t, t % 32, t // 32, sm, b, head_d, 2, 4)
    handoff(tc + i64(LY.HEAD_DONE), t, False)
    # sampling + verification (outtok / verdict / anc written by CTA 0 after its last handoff) on the ROWS x CPR sampling
    # CTAs; the draft context KV on the others meanwhile (it sat inside the sampling's exchanges, on its critical path:
    # ctx0 12.1 + wkv 4.6 us, ktrace gapS; on 64 CTAs it takes ~29-36 us, fixture ctx64.py)
    if b < SF.ROWS * SF.CPR:
        SPF.sample_once_p(P['logits'], P['s0'], P['s1'], P['s2'], P['s3'], P['sids'], tc + i64(LY.SAMPLE), t, b, sm, HF.VR,
                          [r + i64(SOFF) for r in recv], rank, 1, tptr(tp, 'sparam'), tptr(tp, 'qlog'), P['dids'],
                          tptr(tp, 'prow'), P['outtok'], P['verdict'], P['anc'], tptr(tp, 'qids'), com=tptr(tp, 'tcom'),
                          outsl=tptr(tp, 'outsl'), outh=tptr(tp, 'outh'))
    else:
        if fx.const_expr(ctx is not None):
            # (ctx TP, with the step epoch) main_proj split over the ranks, its columns exchanged in the receive halves
            CX.ctx_split(ctx, t, b, T, tc + i64(LY.CTX_SPLIT), SF.ROWS * SF.CPR, CX.CTAS - SF.ROWS * SF.CPR, kv_fp4,
                         **({} if tep is None else dict(recv=recv, rank=rank, tag=hx.epoch32(tep, fx.Int32(CX_TAG_LAYER), 0))))
    # done: the verify's outputs reach the rest of this launch (one launch: the draft entry reads them); a launch that
    # ends here leaves them to stream order (the draft launch, the host's copies after its end event)
    if fx.const_expr(done):
        handoff(tc + i64(LY.VERIFY_DONE), t, False)


@ASTRewriter.transform
def draft(tp, t, b, sm, tc, recv, rank, head_d=4):
    P = {k: tptr(tp, k) for k in ('hw', 'anc', 'dh', 'dlog', 'mw1', 'mw2', 'mids', 'memb', 'mbest')}
    HF.head_once_r(P['dh'], P['hw'], P['dlog'], t, t % 32, t // 32, sm, b, head_d, 2, 4)
    handoff(tc + i64(LY.DRAFT_HEAD_DONE), t, False)
    MPF.markov_once(P['dlog'], P['anc'], P['mw1'], P['mw2'], P['mids'], P['memb'], P['mbest'], tc + i64(LY.MARKOV),
                    [r + i64(SOFF2) for r in recv], rank, 101, t, b, sm, tptr(tp, 'sparam'), tptr(tp, 'qlog'), tptr(tp, 'qids'),
                    ids64=tptr(tp, 'mids64'))
