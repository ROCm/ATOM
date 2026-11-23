# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""The step tail's gfx1250 kernels (TP8, one GPU a rank).

``build_head_verify``: one launch after the target's last layer.
  logits[rows][VR] = x[rows][HIDDEN] bf16 . W[VR][HIDDEN]^T bf16, this rank's vocab rows, in production's arithmetic
  (AITER's a16w16 GEMM at the width: fp32 WMMA 16x16x32 sums over K in 32-column steps from 0, a new partial sum at
  each of ``splits``, the partials added in order from +0, the result rounded to bf16). 256 CTAs x 5 waves; CTA b's
  wave w owns vocab tile 5 b + w (16 ids). W (in place, no repacked copy) and x are staged through LDS by TDM copies
  (``head_pass``). Nothing is stored: a lane reduces its 8 logits a row to (key, lowest id) -- AITER's
  ``topk_per_row_argmax`` order: the bf16 value widened, NaN highest, -0 = +0, ties to the lowest id -- the CTA to a
  best a row (LDS), stored to ``rec`` at device scope; one device hand-off.
  Then CTA q (q < requests) takes request q: its rows' rank bests from ``rec``, one 16 B line a row pushed to every
  peer (system scope, tagged with the launch's epoch), the peers' lines polled, the bests combined (max key, lowest
  id: identical on every rank), and the request verified as ``rejection_greedy_sample_kernel`` does: output ids
  [requests][qlen] and verdict [3][requests].

The poll: a lane holds one (row, source rank) line, so a poll round is ONE system-scope 16 B load instruction a wave
and one wait for it, whatever the rank count (not a load and a wait a rank); a lane leaves the loop when its line
carries the launch's epoch. The rank's own best stays in LDS: it is never read back through memory.
Every bounded wait (the polls, the hand-offs) has one form: the read is inside the loop and is waited for at once
(``settle``: ``s_wait_loadcnt 0``) before anything else is waited for, and the loop counts its rounds; the clock is
read once every ``POLL_CHUNK`` rounds, never with a read out. (A system-scope poll whose load was still out across
the clock message's wait, with ``s_sleep 0`` and three polling waves in every CTA, now and then read a line's old
contents for the whole bound in cached memory: tools/steptail_stale_matrix.py.)
Every poll is bounded (``poll_ticks``): a wait that gives up is counted in ``cnt`` and the launch goes on with what
it read (wrong ids), as DeepSeek-V4.1's step kernel does.
"""
from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, vector
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T, as_ir_value

from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.step_tail import layout as L
from atom.mono.device.ops import CM_DEV, rsrc

F32, I32, I64, BF16 = fx.Float32, fx.Int32, fx.Int64, fx.BFloat16
POLL_TICKS = 200000000  # 2 s of the 100 MHz clock
POLL_CHUNK = 256  # rounds of a bounded wait between two reads of the clock
SITE_HEAD, SITE_SAMPLE, SITE_MARKOV_HAND, SITE_MARKOV = 1, 2, 3, 4
# a timeline build's stamps of a verify CTA (100 MHz clock)
ST_ENTRY, ST_HEAD, ST_HANDOFF, ST_PUSHED, ST_POLLED, ST_END = 0, 1, 2, 3, 4, 5


def i64(x):
    return fx.Int64(x)


def c32(v):
    return I32(v - (1 << 32) if v >= 1 << 31 else v)


def i64min():
    return fx.Int64(-(1 << 63))


def _ufl(x):
    return fx.Int32(hx.intrin(hx.i32(), "llvm.amdgcn.readfirstlane.i32", [fx.Int32(x)]))


def ptr64(table, index):
    """Entry ``index`` of an i64 table, wave-uniform."""
    v = hx.gload(table + i64(index * 8), I32, 2)
    return (fx.Int64(_ufl(v[1])) << 32) | (fx.Int64(_ufl(v[0])) & fx.Int64(0xFFFFFFFF))


def tptr(tab, name):
    return ptr64(tab, L.POINTERS.index(name))


def latomic_max64(off, val):
    llvm.AtomicRMWOp(llvm.AtomicBinOp.max, hx.lptr(off), hx._v(I64(val)), llvm.AtomicOrdering.monotonic,
                     syncscope="workgroup", alignment=8)


def ord_key(v, nan_low=False):
    """f32 -> i32 of the same order, -0 = +0; NaN above +inf (AITER's ``ord_signed_f32``), or with ``nan_low``
    below -inf (a maxnum reduction: NaN never wins)."""
    bits = v.bitcast(I32)
    bits = (bits == c32(0x80000000)).select(I32(0), bits)
    ordered = bits ^ ((bits >> I32(31)) & I32(0x7FFFFFFF))
    is_nan = (bits & I32(0x7FFFFFFF)) > I32(0x7F800000)
    return is_nan.select(c32(0x80000000) if nan_low else I32(0x7FFFFFFF), ordered)


def pack(key, gid):
    """(key, lowest id on ties) as one ordered i64."""
    return fx.Int64(key) * fx.Int64(1 << 32) + (fx.Int64(c32(0xFFFFFFFF) - gid) & fx.Int64(0xFFFFFFFF))


def packed_id(g):
    return c32(0xFFFFFFFF) - I32(g & fx.Int64(0xFFFFFFFF))


def settle():
    """The loads issued so far are back (``s_wait_loadcnt 0``): right after a bounded wait's read, before the
    clock's message or any other wait."""
    llvm.inline_asm(None, [], "s_wait_loadcnt 0x0", "", has_side_effects=True)


@ASTRewriter.transform
def note_timeout(cnt, t0, site, ticks):
    if hx.clock() - t0 >= fx.Int64(ticks):
        hx.atomic_add(cnt + i64(L.CNT_TIMEOUT * 4), 1)
        hx.gstore(cnt + i64(L.CNT_SITE * 4), I32(site))


@ASTRewriter.transform
def handoff(cnt, word, target, t, site, ticks):
    """Every CTA's device-scope stores are out once the count at ``cnt[word]`` reaches ``target`` (a running total:
    CTAS a launch, compared modulo 2^32)."""
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        at = cnt + i64(word * 4)
        hx.atomic_add(at, 1)
        t0 = hx.clock()
        seen = hx.gload(at, I32, scope="agent")
        settle()
        while ((seen - target) < I32(0)) & (hx.clock() - t0 < fx.Int64(ticks)):
            left = I32(POLL_CHUNK)
            while ((seen - target) < I32(0)) & (left > I32(0)):
                hx.s_sleep(1)
                seen = hx.gload(at, I32, scope="agent")
                settle()
                left = left - 1
        note_timeout(cnt, t0, site, ticks)
    gpu.barrier()


def load_line(mine, word):
    return fx.Vector(bo.buffer_load(mine, word, vec_width=4, dtype=T.i32, cache_modifier=ar.SCOPE_SYS))


@ASTRewriter.transform
def poll_line(mine, word, epoch, cnt, site, wait, ticks):
    """The 16 B line at word ``word`` of this rank's peer region once it carries ``epoch`` (bounded). ``word`` is a
    lane's own line: the wave's lines of a round are one load instruction and one wait."""
    got = load_line(mine, word)
    settle()
    if fx.const_expr(wait):
        t0 = hx.clock()
        while (fx.Int32(got[3]) != epoch) & (hx.clock() - t0 < fx.Int64(ticks)):
            left = I32(POLL_CHUNK)
            while (fx.Int32(got[3]) != epoch) & (left > I32(0)):
                hx.s_sleep(0)
                got = load_line(mine, word)
                settle()
                left = left - 1
        if fx.Int32(got[3]) != epoch:   # this lane's line never came (the wave left the loop on the bound)
            hx.atomic_add(cnt + i64(L.CNT_TIMEOUT * 4), 1)
            hx.gstore(cnt + i64(L.CNT_SITE * 4), I32(site))
    return got


@ASTRewriter.transform
def stamp(stp, b, t, slot, requests):
    if fx.const_expr(stp is not None):
        if (b < requests) & (t == 0):
            hx.gstore(stp + i64(b * 8 + slot) * 8, hx.clock())


def tdm_rows(address, dest, live, width, cache_policy=0):
    """One TDM copy by this wave: ``live`` rows (wave-uniform; 0 copies nothing, and still counts as a copy) of
    ``width`` BF16 columns at ``address``, rows HIDDEN BF16 apart, to LDS byte offset ``dest``, rows
    ``width + LDS_PAD`` BF16 apart (the Kimi-K3 GEMV's descriptor: ``gemv_tdm_gfx1250.load_rows``)."""
    live = _ufl(live)
    group0 = vector.from_elements(
        T.vec(4, T.i32),
        [as_ir_value(v) for v in ((live > 0).select(I32(1), I32(0)), I32(dest),
                                  I32(address & fx.Int64(0xFFFFFFFF)), I32(address >> fx.Int64(32)) | c32(2 << 30))])
    config = (1 << 16) | (1 << 20) | ((int(math.log2(width * 2 // 4)) - 1) << 22) | ((L.LDS_PAD * 2 // 4 - 1) << 25)
    group1 = vector.from_elements(
        T.vec(8, T.i32),
        [as_ir_value(v) for v in (c32(config), c32(width << 16), (live & 0xFFFF) << 16,
                                  (live >> 16) | c32(width << 16), live, c32(L.HIDDEN), c32(0), c32(0))])
    zero4 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(c32(0))] * 4)
    zero8 = vector.from_elements(T.vec(8, T.i32), [as_ir_value(c32(0))] * 8)
    rocdl_dialect.tensor_load_to_lds(group0, group1, zero4, zero4, zero8,
                                     cache_policy=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_policy))


@ASTRewriter.transform
def head_pass(x, w, t, b, sm, rows, splits, emit):
    """This rank's logits of ``rows`` rows, production's fp32 bits before the bf16 rounding: ``emit(n, tile, lane,
    values)`` is called with row group n's 8 fp32 a lane -- row n 16 + lane % 16, vocab ids tile 16 + (lane // 16) 8 ..
    + 8 -- for the caller to round and reduce or store.

    Both operands go through LDS in PHASES phases of PHASE_STEPS K steps, NBUF buffers, DEPTH phases in flight: a
    wave's TDM copies of a phase are its tile's 16 weight rows (non-temporal) and its share of the x rows; every
    phase issues the copies of the one DEPTH ahead (empty past the last, so the wait is one count throughout).
    A WMMA 16x16x32 adds its 32 products in two fused groups, the incoming sum with K 0..15, then K 16..31: a lane
    (h = lane // 16) holds K h 8 .. + 8 of a step in the low half of its operand and K 16 + h 8 .. + 8 in the high
    half, two 16 B LDS reads."""
    NG = L.row_groups(rows)
    PITCH, WB, BUF = L.x_stride(), L.w_buffer(), L.phase_buffer(rows)
    SPP, NPH, DEPTH, NBUF = L.PHASE_STEPS, L.PHASES, L.DEPTH, L.NBUF
    CK = SPP * L.K_STEP
    XRPW = L.x_rows_per_wave(rows)
    t = hx.opaque_v(t)
    b = hx.opaque_s(b)
    wv = t // 32
    l = t % 32
    wave = fx.Int32(rocdl.wave_id())
    tile = b * L.TILE_WAVES + wv
    w0 = w + i64(b * L.TILE_WAVES + wave) * (L.TILE * L.HIDDEN * 2)
    x0 = x + i64(wave * XRPW) * (L.HIDDEN * 2)
    xlive = fx.max(fx.min(I32(rows) - wave * XRPW, I32(XRPW)), I32(0))
    zero = fx.Vector.filled(8, 0.0, F32)
    acc = [zero for _ in range(NG)]
    tot = [zero for _ in range(NG)]

    def issue(ph):   # this wave's two copies of phase ph into buffer ph % NBUF
        ph = I32(ph)
        on = ph < NPH
        col = i64(fx.min(ph, I32(NPH - 1)) * CK) * 2
        base = sm + (ph % NBUF) * BUF
        tdm_rows(w0 + col, base + wave * (L.TILE * PITCH), on.select(I32(L.TILE), I32(0)), CK, 1)
        tdm_rows(x0 + col, base + WB + wave * (XRPW * PITCH), on.select(xlive, I32(0)), CK, 0)

    def operand(row, kk):   # a lane's 16 BF16 of K step kk of the LDS row at byte offset `row`
        at = row + (kk * L.K_STEP + (l // 16) * 8) * 2
        lo = hx.lload(at, BF16, 8)
        hi = hx.lload(at + 32, BF16, 8)
        return fx.Vector.from_elements([lo[e] for e in range(8)] + [hi[e] for e in range(8)], BF16)

    def operands(base, kk):   # a step's weight fragment, then its x fragments, loaded back to back
        return [operand(base + (wv * 16 + l % 16) * PITCH, kk)] + [
            operand(base + WB + (n * 16 + l % 16) * PITCH, kk) for n in range(NG)]

    gpu.barrier()
    for p in range_constexpr(DEPTH):
        issue(p)
    for ph in range(NPH):
        rocdl.s_wait_tensorcnt(2 * DEPTH - 2)   # this wave's copies of phase ph are in
        gpu.barrier()                           # ... and every wave's: the buffer of phase ph - 1 is free
        issue(ph + DEPTH)
        base = sm + (ph % NBUF) * BUF
        for kk in range_constexpr(SPP):
            for bi in range_constexpr(len(splits)):
                if fx.const_expr(splits[bi] % SPP == kk):   # a new partial sum starts here in phase splits[bi] // SPP
                    at = ph == splits[bi] // SPP
                    for n in range_constexpr(NG):
                        s_ = tot[n] + acc[n]
                        tot[n] = fx.Vector.from_elements([at.select(s_[e], tot[n][e]) for e in range(8)], F32)
                        acc[n] = fx.Vector.from_elements([at.select(F32(0.0), acc[n][e]) for e in range(8)], F32)
            ops = operands(base, kk)
            hx.intrin(None, "llvm.amdgcn.sched.barrier", [fx.Int32(0)])   # one wait for the step's LDS reads
            for n in range_constexpr(NG):
                acc[n] = hx.wmma_bf16(ops[0], ops[1 + n], acc[n])
            hx.intrin(None, "llvm.amdgcn.sched.barrier", [fx.Int32(0)])
    rocdl.s_wait_tensorcnt(0)
    for n in range_constexpr(NG):
        if fx.const_expr(len(splits) > 0):
            emit(n, tile, l, tot[n] + acc[n])
        else:
            emit(n, tile, l, acc[n])


@ASTRewriter.transform
def emit_argmax(sm, red_off, rows, rank, tapp, t, n, tile, l, values):
    """Row group n's 8 logits a lane, rounded to bf16 as production stores them: the lane's best (key, lowest id) of
    its row to the reductions' scratch [n][thread]."""
    hv = values.to(BF16)
    row = n * 16 + l % 16
    col = tile * 16 + (l // 16) * 8
    best = i64min()
    for e in range_constexpr(8):
        best = fx.max(best, pack(ord_key(hv[e].to(F32)), col + e + rank * L.VR))
    hx.lstore(sm + red_off + (n * L.THREADS + t) * 8, best)
    if fx.const_expr(tapp is not None):
        if row < rows:
            hx.gstore(tapp + (i64(row) * L.VR + i64(col)) * 2, hv)


def make_emit(tab, tap, sm, red_off, rows, rank, t):   # (plain python: outside the AST rewriter)
    tapp = tptr(tab, "tap") if tap else None
    return lambda n, tile, l, values: emit_argmax(sm, red_off, rows, rank, tapp, t, n, tile, l, values)


def stamps_pointer(tab, stamps):   # (plain python)
    return tptr(tab, "stamps") if stamps else None


@ASTRewriter.transform
def head_verify(tab, x, ids, rank, t, b, sm, rows, qlen, splits, tap, wait, stamps, ticks):
    step = L.Step(rows, qlen)
    lds = L.head_lds(rows)
    NREQ, ND = step.requests, step.drafts
    t = hx.opaque_v(t)
    b = hx.opaque_s(b)
    stp = stamps_pointer(tab, stamps)
    stamp(stp, b, t, ST_ENTRY, NREQ)
    wv = t // 32
    l = t % 32
    emit = make_emit(tab, tap, sm, lds.red, rows, rank, t)
    head_pass(x, tptr(tab, "w"), t, b, sm, rows, splits, emit)
    gpu.barrier()
    stamp(stp, b, t, ST_HEAD, NREQ)
    # (nothing but the weights' pointer is read before the first copies go out; the launch's epoch is read here,
    # before this CTA counts into the hand-off)
    cnt = tptr(tab, "cnt")
    rec = tptr(tab, "rec")
    epoch = hx.gload(cnt + i64(L.CNT_EPOCH * 4), I32, scope="agent") + 1
    if t < rows:
        # thread t: row t's best of the CTA, from the ten lanes that hold the row (lane % 16 of every wave)
        held = [hx.lload(sm + lds.red + ((t // 16) * L.THREADS + q * 16 + t % 16) * 8, I64)
                for q in range(2 * L.WAVES)]
        hx.intrin(None, "llvm.amdgcn.sched.barrier", [fx.Int32(0)])
        cta_best = held[0]
        for q in range_constexpr(1, 2 * L.WAVES):
            cta_best = fx.max(cta_best, held[q])
        hx.gstore(rec + (i64(b) * L.MAX_ROWS + i64(t)) * 8, cta_best, scope="agent", align=8)
    # the verify CTAs' pointers, read while the hand-off waits for the slowest CTA (and before any remote store: a
    # load issued after one waits for its round trip)
    peers = tptr(tab, "peers")
    bases = [ptr64(peers, o) for o in range(L.TP)]
    mine_rs = rsrc(ptr64_at(peers, rank))
    recs = rsrc(rec)
    handoff(cnt, L.CNT_HEAD, epoch * L.CTAS, t, SITE_HEAD, ticks)
    stamp(stp, b, t, ST_HANDOFF, NREQ)
    if b < NREQ:
        # request b: its rows' rank bests, the exchange, the verdict
        bank = epoch & 1
        # every CTA's best of the request's rows: the loads go out together (device scope, one wait); a lane keeps
        # the max a row, a wave's lane p the wave's of row p, thread p the rank's (no LDS atomics: 160 lanes on one
        # slot took ~18 us)
        bests = []
        for it in range_constexpr(-(-L.CTAS // L.THREADS)):
            c = fx.min(t + it * L.THREADS, I32(L.CTAS - 1))   # (a CTA read twice changes no maximum)
            for p in range_constexpr(qlen):
                bests.append(fx.Vector(bo.buffer_load(recs, (c * L.MAX_ROWS + b * qlen + p) * 2, vec_width=2,
                                                      dtype=T.i32, cache_modifier=CM_DEV)))
        for p in range_constexpr(qlen):
            lane_best = i64min()
            for it in range_constexpr(-(-L.CTAS // L.THREADS)):
                v = bests[it * qlen + p]
                lane_best = fx.max(lane_best,
                                   (fx.Int64(fx.Int32(v[1])) << 32) | (fx.Int64(fx.Int32(v[0])) & fx.Int64(0xFFFFFFFF)))
            hx.lstore(sm + lds.red + ((wv * L.MAX_QLEN + p) * 32 + l) * 8, lane_best)
        gpu.barrier()
        if l < qlen:
            pairs = [hx.lload(sm + lds.red + ((wv * L.MAX_QLEN + l) * 32 + q * 2) * 8, I64, 2) for q in range(16)]
            hx.intrin(None, "llvm.amdgcn.sched.barrier", [fx.Int32(0)])
            wave_best = fx.max(pairs[0][0], pairs[0][1])
            for q in range_constexpr(1, 16):
                wave_best = fx.max(wave_best, fx.max(pairs[q][0], pairs[q][1]))
            hx.lstore(sm + lds.wbest + (wv * L.MAX_QLEN + l) * 8, wave_best)
        gpu.barrier()
        if t < qlen:
            mine = i64min()
            for q in range_constexpr(L.WAVES):
                mine = fx.max(mine, hx.lload(sm + lds.wbest + (q * L.MAX_QLEN + t) * 8, I64))
            hx.lstore(sm + lds.peer + (t * L.TP + rank) * 8, mine)
            line = fx.Vector.from_elements([I32(mine & fx.Int64(0xFFFFFFFF)), I32(mine >> 32), I32(0), epoch], I32)
            at = i64(L.sample_line(bank, rank, b * qlen + t))
            for o in range_constexpr(L.TP):
                if I32(o) != rank:
                    ar._store_line_sys(bases[o] + at, line)
        stamp(stp, b, t, ST_PUSHED, NREQ)
        if t < qlen * L.TP:
            src = t % L.TP
            if src != rank:
                got = poll_line(mine_rs, L.sample_line(bank, src, b * qlen + t // L.TP) // 4,
                                epoch, cnt, SITE_SAMPLE, wait, ticks)
                hx.lstore(sm + lds.peer + t * 8, got[0])
                hx.lstore(sm + lds.peer + t * 8 + 4, got[1])
        gpu.barrier()
        stamp(stp, b, t, ST_POLLED, NREQ)
        if t < qlen:
            g = i64min()
            for o in range_constexpr(L.TP):
                g = fx.max(g, hx.lload(sm + lds.peer + (t * L.TP + o) * 8, I64))
            hx.lstore(sm + lds.ids + t * 4, packed_id(g))
        gpu.barrier()
        if t == 0:
            outtok = tptr(tab, "outtok")
            verdict = tptr(tab, "verdict")
            tid = [hx.lload(sm + lds.ids + p * 4, I32) for p in range(qlen)]
            # draft p of the request: the input token after the row that verifies it
            xs = [hx.gload(ids + i64(b * qlen + p + 1) * 4, I32) for p in range(ND)]
            f = I32(ND)
            for p in range_constexpr(ND - 1, -1, -1):
                f = (xs[p] != tid[p]).select(I32(p), f)
            for p in range_constexpr(ND):
                hx.gstore(outtok + i64(b * qlen + p) * 4, (I32(p) <= f).select(tid[p], I32(-1)))
            hx.gstore(outtok + i64(b * qlen + ND) * 4, (f == ND).select(tid[ND], I32(-1)))
            hx.gstore(verdict + i64(b) * 4, f)
            hx.gstore(verdict + i64(NREQ + b) * 4, I32(ND) - f)
            hx.gstore(verdict + i64(2 * NREQ + b) * 4, b * qlen + f)
        stamp(stp, b, t, ST_END, NREQ)
    if (b == 0) & (t == 0):
        # the next launch's epoch: every CTA read this one's before the hand-off
        hx.gstore(cnt + i64(L.CNT_EPOCH * 4), epoch, scope="agent", align=4)


def ptr64_at(table, index):
    """Entry ``index`` (a wave-uniform i32) of an i64 table, wave-uniform."""
    v = hx.gload(table + i64(index) * 8, I32, 2)
    return (fx.Int64(_ufl(v[1])) << 32) | (fx.Int64(_ufl(v[0])) & fx.Int64(0xFFFFFFFF))


def head_verify_name(rows, qlen, splits, tap=False, wait=True, stamps=False, poll_ticks=POLL_TICKS):
    return (f"k3_steptail_head_verify_tp{L.TP}_s{rows}_q{qlen}_sk{len(splits) + 1}"
            + ("_tap" if tap else "") + ("" if wait else "_nowait") + ("_stamps" if stamps else "")
            + ("" if poll_ticks == POLL_TICKS else f"_t{poll_ticks}"))


def build_head_verify(rows: int, qlen: int, splits: tuple, tap: bool = False, wait: bool = True, sources: str = "",
                      stamps: bool = False, poll_ticks: int = POLL_TICKS):
    """The head + verify launch of a ``rows``-row step of ``qlen`` rows a request. ``splits``: the K steps where
    production's GEMM starts a new partial sum (``layout.split_steps``). Tool builds: ``tap`` also stores the bf16
    logits; ``wait=False`` takes the peers' lines as they are (timing on one GPU, which has no peers); ``stamps``
    stores the verify CTAs' clock at ST_*; ``poll_ticks`` shortens the bounded waits.
    Arguments: the pointer table (``layout.POINTERS``), x [rows, HIDDEN] bf16, the step's input ids [rows] i32, the TP
    rank."""
    why = L.step_refusal(rows, qlen)
    if why:
        raise ValueError(why)
    lds = L.head_lds(rows)
    keyed = (("rows", rows), ("qlen", qlen), ("splits", tuple(splits)), ("tap", tap), ("wait", wait),
             ("stamps", stamps), ("poll_ticks", poll_ticks), ("sources", sources))

    @flyc.kernel(name=head_verify_name(rows, qlen, splits, tap, wait, stamps, poll_ticks),
                 known_block_size=[L.THREADS, 1, 1])
    def kernel(tab: Int64, x: Int64, ids: Int64, rank: fx.Int32):
        _ = keyed
        head_verify(tab, x, ids, rank, fx.thread_idx.x, fx.block_idx.x, hx.lds_base(), rows, qlen, tuple(splits), tap,
                    wait, stamps, poll_ticks)

    @flyc.jit
    def launch(tab: Int64, x: Int64, ids: Int64, rank: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        kernel(tab, x, ids, rank).launch(grid=(L.CTAS,), block=(L.THREADS,), smem=lds.size, stream=stream)

    return launch
