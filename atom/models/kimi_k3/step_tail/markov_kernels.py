# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The draft launch (TP8, one GPU a rank): the DSpark draft's LM head on this rank's vocab shard, then its greedy
Markov block sampling, production's ids (``DSparkMarkovHead.sample_block`` over the all-gathered logits).

  base[req T + k][VR] = the draft's hidden rows . W[VR]^T: ``kernels.head_pass`` (production's GEMM bits), rounded to
  bf16; a wave keeps its tile's 16 logits of every row in LDS -- the only logits it reads back are its own, so
  nothing is stored to memory and nothing is gathered.
  For k = 0 .. T - 1 (x_{-1} = the anchors):  z[req][v] = f32(base[req T + k][v]) + W1[x_{k-1}[req]] . W2[v],
  x_k[req] = argmax z, lowest id on ties, NaN never the maximum (an all-NaN row answers 0): the compiled Triton
  kernels' maxnum reduction. The dot product is production's: WMMA 16x16x32 steps over the rank in order from 0 (each
  two fused groups, K 0..15 with the incoming sum, then K 16..31), the base logit added last (probed:
  tools/steptail_markov_order.py). A wave's 16 W2 rows are staged in LDS once (TDM); a round stages W1[x_{k-1}] of every
  request (16 B loads), multiplies, and reduces lane -> CTA (LDS) -> ``rec`` (device scope), one device hand-off.
  CTA 0 then reduces every CTA's records to the rank's best a request, pushes one 16 B line a request to every peer
  (system scope, tagged with the launch's epoch), polls the peers' lines -- a lane a line, so a poll round is one
  load instruction a wave and one wait -- and takes the max over the ranks: x_k, identical on every rank. It hands
  x_k to the rank's other CTAs as the hand-offs do: device-scope stores, waited, then a token they wait on. (They do
  not read CTA 0's line back: on another XCD a polled system-scope line now and then stayed at its old contents for
  the whole 2 s bound after CTA 0 rewrote it -- tools/steptail_markov_check.py DIAG.) x_k stays in LDS for the next
  round.

A round's hand-off count is a running total compared with epoch * CTAS, one count a round ordinal: a block of fewer
than MAX_DRAFTS drafts adds CTAS to the counts of the rounds it does not run, so a rank may launch blocks of
different draft counts in any order (without it a 7-draft block after a 4-draft one waited out the whole bound in
rounds 4 to 6 of every launch: tools/steptail_runner_check.py).

Every bounded wait has ``kernels``' form (the read in the loop, waited for at once, the clock read once a chunk of
rounds).
Every poll is bounded (``poll_ticks``), and the other CTAs' wait for CTA 0's token by twice that (CTA 0 publishes
x_k after its own poll, which may have given up); an id read after a wait that gave up is clamped into the
vocabulary (it indexes W1).
"""
from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T, as_ir_value

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.step_tail import kernels as K
from atom.models.kimi_k3.step_tail import layout as L
from atom.models.kimi_k3.step_tail import markov_layout as M
from atom.mono.device.ops import CM_DEV, rsrc

F32, I32, I64, BF16 = fx.Float32, fx.Int32, fx.Int64, fx.BFloat16
i64, c32, i64min = K.i64, K.c32, K.i64min


def tdm_rows(address, stride, dest, live, width):
    """``kernels.tdm_rows`` for rows ``stride`` BF16 apart."""
    live = K._ufl(live)
    group0 = vector.from_elements(
        T.vec(4, T.i32),
        [as_ir_value(v) for v in ((live > 0).select(I32(1), I32(0)), I32(dest),
                                  I32(address & fx.Int64(0xFFFFFFFF)), I32(address >> fx.Int64(32)) | c32(2 << 30))])
    config = (1 << 16) | (1 << 20) | ((int(math.log2(width * 2 // 4)) - 1) << 22) | ((L.LDS_PAD * 2 // 4 - 1) << 25)
    group1 = vector.from_elements(
        T.vec(8, T.i32),
        [as_ir_value(v) for v in (c32(config), c32(width << 16), (live & 0xFFFF) << 16,
                                  (live >> 16) | c32(width << 16), live, c32(stride), c32(0), c32(0))])
    zero4 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(c32(0))] * 4)
    zero8 = vector.from_elements(T.vec(8, T.i32), [as_ir_value(c32(0))] * 8)
    rocdl_dialect.tensor_load_to_lds(group0, group1, zero4, zero4, zero8,
                                     cache_policy=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))


def operand(row, step, l):
    """A lane's 16 BF16 of K step ``step`` of the LDS row at byte offset ``row`` (the WMMA's K layout)."""
    at = row + (step * L.K_STEP + (l // 16) * 8) * 2
    lo = hx.lload(at, BF16, 8)
    hi = hx.lload(at + 32, BF16, 8)
    return fx.Vector.from_elements([lo[e] for e in range(8)] + [hi[e] for e in range(8)], BF16)


@ASTRewriter.transform
def emit_logits(sm, zl, t, n, l, values):
    """Row group n's 8 logits a lane, rounded to bf16 as production's LM head stores them, kept by the wave."""
    hx.lstore(sm + zl + ((t // 32) * L.MAX_ROWS + n * 16 + l % 16) * 32 + (l // 16) * 16, values.to(BF16))


def make_emit(sm, zl, t):   # (plain python: outside the AST rewriter)
    return lambda n, tile, l, values: emit_logits(sm, zl, t, n, l, values)


def diag_pointer(tab, diag):   # (plain python)
    return K.tptr(tab, "tap") if diag else None


@ASTRewriter.transform
def poll_line(mine, word, epoch, cnt, site, need, ticks, dg=None, slot=None, b=None):
    """``kernels.poll_line`` for a lane that waits only if ``need`` (a timing build waits for this rank's own line and
    takes the peers' as they are: one GPU has no peers)."""
    got = K.load_line(mine, word)
    K.settle()
    t0 = hx.clock()
    while need & (fx.Int32(got[3]) != epoch) & (hx.clock() - t0 < fx.Int64(ticks)):
        left = I32(K.POLL_CHUNK)
        while need & (fx.Int32(got[3]) != epoch) & (left > I32(0)):
            hx.s_sleep(0)
            got = K.load_line(mine, word)
            K.settle()
            left = left - 1
    if need & (fx.Int32(got[3]) != epoch):
        hx.atomic_add(cnt + i64(L.CNT_TIMEOUT * 4), 1)
        hx.gstore(cnt + i64(L.CNT_SITE * 4), I32(site))
        if fx.const_expr(dg is not None):
            # a diagnostic build: the waits that gave up a (round, thread) and a CTA, and the tag last seen
            hx.atomic_add(dg + i64(slot) * 4, 1)
            hx.atomic_add(dg + i64(L.MAX_DRAFTS * L.THREADS + b) * 4, 1)
            hx.gstore(dg + i64(L.MAX_DRAFTS * L.THREADS + L.CTAS + slot) * 4, fx.Int32(got[3]) - epoch)
    return got


@ASTRewriter.transform
def draft_markov(tab, x, anchors, rank, t, b, sm, requests, drafts, rk, splits, wait, ticks, diag=False):
    rows = requests * drafts
    lds = L.head_lds(rows)
    ml = M.markov_lds(rk)
    PITCH = M.pitch(rk)
    UNITS = requests * (rk // 8)          # 16 B units of a round's W1 rows
    t = hx.opaque_v(t)
    b = hx.opaque_s(b)
    wv = t // 32
    l = t % 32
    wave = fx.Int32(rocdl.wave_id())
    tile = b * L.TILE_WAVES + wv
    K.head_pass(x, K.tptr(tab, "w"), t, b, sm, rows, splits, make_emit(sm, lds.red, t))
    gpu.barrier()
    cnt = K.tptr(tab, "cnt")
    rec = K.tptr(tab, "rec")
    w1 = K.tptr(tab, "w1")
    w2 = K.tptr(tab, "w2")
    out = K.tptr(tab, "drafts")
    peers = K.tptr(tab, "peers")
    bases = [K.ptr64(peers, o) for o in range(L.TP)]
    mine_rs = rsrc(K.ptr64_at(peers, rank))
    recs = rsrc(rec)
    dg = diag_pointer(tab, diag)
    xkp = K.tptr(tab, "mbest")
    epoch = hx.gload(cnt + i64(M.CNT_MEPOCH * 4), I32, scope="agent") + 1
    bank = epoch & 1
    # this wave's 16 W2 rows (this rank's vocab shard), once
    tdm_rows(w2 + i64(rank * L.VR + (b * L.TILE_WAVES + wave) * L.TILE) * (rk * 2), rk,
             sm + ml.w2 + wave * (L.TILE * PITCH), I32(L.TILE), rk)
    if t < requests:
        hx.lstore(sm + ml.x + t * 4, fx.max(fx.min(hx.gload(anchors + i64(t) * 4, I32), I32(L.VOCAB - 1)), I32(0)))
    rocdl.s_wait_tensorcnt(0)
    gpu.barrier()
    for k in range(drafts):
        # W1[x_{k-1}] of every request -> LDS (the loads go out together)
        frags = []
        for it in range_constexpr(-(-UNITS // L.THREADS)):
            i = fx.min(t + it * L.THREADS, I32(UNITS - 1))
            frags.append(hx.gload(w1 + (i64(hx.lload(sm + ml.x + (i // (rk // 8)) * 4, I32)) * rk
                                        + i64((i % (rk // 8)) * 8)) * 2, BF16, 8))
        for it in range_constexpr(-(-UNITS // L.THREADS)):
            i = fx.min(t + it * L.THREADS, I32(UNITS - 1))
            hx.lstore(sm + ml.w1 + (i // (rk // 8)) * PITCH + (i % (rk // 8)) * 16, frags[it])
        gpu.barrier()
        acc = fx.Vector.filled(8, 0.0, F32)
        for s in range_constexpr(rk // L.K_STEP):
            a = operand(sm + ml.w2 + (wv * 16 + l % 16) * PITCH, s, l)
            bq = operand(sm + ml.w1 + (l % 16) * PITCH, s, l)
            hx.intrin(None, "llvm.amdgcn.sched.barrier", [fx.Int32(0)])
            acc = hx.wmma_bf16(a, bq, acc)
        # a lane: request l % 16, vocab ids tile 16 + (l // 16) 8 .. + 8; the base logit is added last
        zb = hx.lload(sm + lds.red + (wv * L.MAX_ROWS + (l % 16) * drafts + k) * 32 + (l // 16) * 16, BF16, 8)
        best = i64min()
        for e in range_constexpr(8):
            best = fx.max(best, K.pack(K.ord_key(zb[e].to(F32) + acc[e], True),
                                       tile * 16 + (l // 16) * 8 + e + rank * L.VR))
        hx.lstore(sm + ml.best + t * 8, best)
        gpu.barrier()
        if t < requests:
            held = [hx.lload(sm + ml.best + (q * 16 + t) * 8, I64) for q in range(2 * L.WAVES)]
            hx.intrin(None, "llvm.amdgcn.sched.barrier", [fx.Int32(0)])
            cta_best = held[0]
            for q in range_constexpr(1, 2 * L.WAVES):
                cta_best = fx.max(cta_best, held[q])
            hx.gstore(rec + (i64(b) * L.MAX_ROWS + i64(t)) * 8, cta_best, scope="agent", align=8)
        K.handoff(cnt, L.CNT_MARKOV + k, epoch * L.CTAS, t, K.SITE_MARKOV_HAND, ticks)
        token = epoch * 8 + k + 1
        if b == 0:
            # the rank's best of every request, from every CTA's record: lane -> wave -> thread, then a line a
            # request to every peer
            bests = []
            for it in range_constexpr(-(-L.CTAS // L.THREADS)):
                c = fx.min(t + it * L.THREADS, I32(L.CTAS - 1))
                for q in range_constexpr(requests):
                    bests.append(fx.Vector(bo.buffer_load(recs, (c * L.MAX_ROWS + q) * 2, vec_width=2, dtype=T.i32,
                                                          cache_modifier=CM_DEV)))
            for q in range_constexpr(requests):
                lane_best = i64min()
                for it in range_constexpr(-(-L.CTAS // L.THREADS)):
                    v = bests[it * requests + q]
                    lane_best = fx.max(lane_best, (fx.Int64(fx.Int32(v[1])) << 32)
                                       | (fx.Int64(fx.Int32(v[0])) & fx.Int64(0xFFFFFFFF)))
                hx.lstore(sm + ml.lane + ((wv * L.MAX_REQ + q) * 32 + l) * 8, lane_best)
            gpu.barrier()
            if l < requests:
                pairs = [hx.lload(sm + ml.lane + ((wv * L.MAX_REQ + l) * 32 + q * 2) * 8, I64, 2) for q in range(16)]
                hx.intrin(None, "llvm.amdgcn.sched.barrier", [fx.Int32(0)])
                wave_best = fx.max(pairs[0][0], pairs[0][1])
                for q in range_constexpr(1, 16):
                    wave_best = fx.max(wave_best, fx.max(pairs[q][0], pairs[q][1]))
                hx.lstore(sm + ml.wave + (wv * L.MAX_REQ + l) * 8, wave_best)
            gpu.barrier()
            if t < requests:
                mine = i64min()
                for q in range_constexpr(L.WAVES):
                    mine = fx.max(mine, hx.lload(sm + ml.wave + (q * L.MAX_REQ + t) * 8, I64))
                line = fx.Vector.from_elements([I32(mine & fx.Int64(0xFFFFFFFF)), I32(mine >> 32), I32(0), epoch], I32)
                at = i64(L.markov_line(bank, k, rank, t))
                hx.lstore(sm + ml.peer + (t * L.TP + rank) * 8, mine)   # the rank's own best stays in LDS
                for o in range_constexpr(L.TP):
                    if I32(o) != rank:
                        ar._store_line_sys(bases[o] + at, line)
            # the peers' lines of the round, a lane a line (one load instruction a wave a round)
            if t < requests * L.TP:
                src = t % L.TP
                if src != rank:
                    got = poll_line(mine_rs, L.markov_line(bank, k, src, t // L.TP) // 4, epoch, cnt, K.SITE_MARKOV,
                                    I32(1 if wait else 0) != I32(0), ticks, dg, k * L.THREADS + t, b)
                    hx.lstore(sm + ml.peer + t * 8, got[0])
                    hx.lstore(sm + ml.peer + t * 8 + 4, got[1])
            gpu.barrier()
            if t < requests:
                g = i64min()
                for o in range_constexpr(L.TP):
                    g = fx.max(g, hx.lload(sm + ml.peer + (t * L.TP + o) * 8, I64))
                xk = fx.max(fx.min(K.packed_id(g), I32(L.VOCAB - 1)), I32(0))
                hx.lstore(sm + ml.x + t * 4, xk)
                hx.gstore(out + (i64(t) * drafts + i64(k)) * 4, xk)
                hx.gstore(xkp + i64(t) * 4, xk, scope="agent", align=4)
            # x_k to the rank's other CTAs: device-scope stores, waited, then the round's token
            hx.wait_storecnt0()
            gpu.barrier()
            if t == 0:
                hx.gstore(cnt + i64(M.CNT_MFLAG * 4), token, scope="agent", align=4)
        else:
            if t == 0:
                # twice CTA 0's bound: its poll of a late peer may give up first, and it publishes x_k after that;
                # a CTA leaving before it would run the next round on another x_{k} than CTA 0
                t0 = hx.clock()
                flag = hx.gload(cnt + i64(M.CNT_MFLAG * 4), I32, scope="agent")
                K.settle()
                while ((flag - token) < I32(0)) & (hx.clock() - t0 < fx.Int64(2 * ticks)):
                    left = I32(K.POLL_CHUNK)
                    while ((flag - token) < I32(0)) & (left > I32(0)):
                        hx.s_sleep(1)
                        flag = hx.gload(cnt + i64(M.CNT_MFLAG * 4), I32, scope="agent")
                        K.settle()
                        left = left - 1
                K.note_timeout(cnt, t0, K.SITE_MARKOV, 2 * ticks)
            gpu.barrier()
            if t < requests:
                hx.lstore(sm + ml.x + t * 4,
                          fx.max(fx.min(hx.gload(xkp + i64(t) * 4, I32, scope="agent"), I32(L.VOCAB - 1)), I32(0)))
        gpu.barrier()
    if (b == 0) & (t == 0):
        # the rounds this block does not run: their hand-off counts keep step with the epoch, which a later, longer
        # block compares them with (a rank's blocks of 4 and of 7 drafts share the counters and the epoch)
        for k in range_constexpr(drafts, L.MAX_DRAFTS):
            hx.atomic_add(cnt + i64((L.CNT_MARKOV + k) * 4), L.CTAS)
        hx.gstore(cnt + i64(M.CNT_MEPOCH * 4), epoch, scope="agent", align=4)


def draft_markov_name(requests, drafts, rk, splits, wait=True, poll_ticks=K.POLL_TICKS, diag=False):
    return (f"k3_steptail_draft_markov_tp{L.TP}_b{requests}_t{drafts}_r{rk}_sk{len(splits) + 1}"
            + ("" if wait else "_nowait") + ("" if poll_ticks == K.POLL_TICKS else f"_t{poll_ticks}")
            + ("_diag" if diag else ""))


def build_draft_markov(requests: int, drafts: int, rk: int, splits: tuple, wait: bool = True, sources: str = "",
                       poll_ticks: int = K.POLL_TICKS, diag: bool = False):
    """The draft launch of ``requests`` blocks of ``drafts`` tokens at Markov rank ``rk``. ``splits``: production's
    LM head split of ``requests * drafts`` rows (``head_dispatch.head_splits``). Tool builds: ``wait=False`` takes the
    peers' lines as they are (timing on one GPU); ``poll_ticks`` shortens the bounded waits; ``diag`` records which
    polls gave up.
    Arguments: the pointer table (``layout.POINTERS``: w, w1, w2, rec, cnt, peers, drafts [requests][drafts] i32,
    mbest: x_k of the round [MAX_REQ] i32), the
    draft's final-normed rows [requests * drafts, HIDDEN] bf16 (row = request * drafts + position), the anchors
    [requests] i32, the TP rank."""
    why = M.draft_refusal(requests, drafts, rk)
    if why:
        raise ValueError(why)
    lds = L.head_lds(requests * drafts)
    keyed = (("requests", requests), ("drafts", drafts), ("rk", rk), ("splits", tuple(splits)), ("wait", wait),
             ("poll_ticks", poll_ticks), ("diag", diag), ("sources", sources))

    @flyc.kernel(name=draft_markov_name(requests, drafts, rk, splits, wait, poll_ticks, diag),
                 known_block_size=[L.THREADS, 1, 1])
    def kernel(tab: Int64, x: Int64, anchors: Int64, rank: fx.Int32):
        _ = keyed
        draft_markov(tab, x, anchors, rank, fx.thread_idx.x, fx.block_idx.x, hx.lds_base(), requests, drafts, rk,
                     tuple(splits), wait, poll_ticks, diag)

    @flyc.jit
    def launch(tab: Int64, x: Int64, anchors: Int64, rank: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        kernel(tab, x, anchors, rank).launch(grid=(L.CTAS,), block=(L.THREADS,), smem=lds.size, stream=stream)

    return launch
