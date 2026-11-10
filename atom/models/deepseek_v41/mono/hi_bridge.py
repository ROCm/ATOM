# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi hook's cache bridge, free of ATOM / aiter imports (the CPU
tests run the bridge kernels in the Triton interpreter:
``tests/models/deepseek_v41/test_mono_hi_bridge.py``); which layers it serves is
``hi_plan``'s.

**Cache bridge.** ATOM's pool stays the source of truth (prefill, the draft,
any step the hook does not serve, and ATOM's own layers read it). The owners'
compressed KV, FP4 index keys and compressor state the hi programs read and write
in place in ATOM's storage (``hi.device.gfx950.paged``: the main rows through the
block table, the FP4 plane's units, the compress ring ``[slot][p % 7][512]``),
every store masked by its row's liveness; the begin-step launch writes the
request words those programs take (``_req_words``: a request's batch id, slot and
liveness, a row's). Only the window rings are imported and exported:

| what | ATOM | hi | import (step start) | export (step end) |
|---|---|---|---|---|
| window ring, every hi layer | bf16 rows ``ring_start(l) + slot * slot_rows + p % ring_slots`` (ring_slots = 128 + drafts) | bf16 [1, 128, 512], slot ``p % 128`` | the 128 positions before the step | the step's rows |

So a rejected draft row never corrupts the next step: ATOM's ring is
position-addressed with draft slack, and the next import rebuilds the hi
runtime's 128-slot ring from it (its own 128-slot layout would lose the accepted
rows' keys under a rejection). Selections (``sel_out``) and layer 20's candidate bitmap are written and read
within the step by hi layers only.

The mHC hand-off (``_planar``): the hi runtime's layer reads
and writes (R [rows, 4, 5120] bf16, gate [rows, 24] = pre 4 | post 4 |
comb[i_in][o_out] 16, the pending MoE output [rows, 5120]); ATOM's
``SinglePassHCState`` holds the same with the gates planar.
"""

from __future__ import annotations

import triton
import triton.language as tl

HC = 4
MIX = HC * (HC + 2)
WINDOW = 128
HEAD_DIM = 512
FP4_PAGE = 8  # ATOM's FP4 index plane: keys a unit (``FP4_INDEX_PAGE_ROWS``)
IDX_DIM = 128


# ---------------------------------------------------------------- ring / step metadata
#
# A step serves ``NB`` requests of ``rq = rows // NB`` rows each (the hi runtime's ``Scenario(NB, rq)``: row r is request
# r // rq). Request b's rows start at row b * rq: its batch id, slot and first position are those of that row. The
# hi-side buffers hold a request per batch slot b: the rings [NMF, BM, WIN, DIM] (``BM`` the most requests a step,
# the buffers' stride), ``pos`` [BM], the request words ``req`` [3, BM] (``_req_words``), the bounds and ``row.live``
# per row. One request (``NB`` = 1, b = 0) is the single-request layout.


@triton.jit
def _request(b, rq, positions, batch_ids, slots, num_slots, max_pos):
    """Request b of the step (its rows b * rq ..): (its batch id, its slot, whether it is live, its first position,
    clamped to [0, max_pos])."""
    rb = tl.load(batch_ids + b * rq).to(tl.int64)
    slot = tl.load(slots + tl.maximum(rb, 0)).to(tl.int64)
    live = (rb >= 0) & (slot >= 0) & (slot < num_slots)
    ctx = tl.load(positions + b * rq).to(tl.int64)
    ctx = tl.minimum(tl.maximum(ctx, 0), max_pos)
    return rb, slot, live, ctx


# the request words (``req``, int32 ``REQ_WORDS``) of the programs that use ATOM's storage in place
# (``hi.device.gfx950.paged``, the same layout): the page geometry at word 0 (4 words, the hook's), then a request's
# batch id, slot and liveness (``MAX_REQ`` each), then a row's liveness (``MAX_ROWS``); the jit bodies spell the offsets
MAX_REQ, MAX_ROWS = 8, 64
REQ_BID, REQ_SLOT, REQ_LIVE, ROW_LIVE = 4, 12, 20, 28
REQ_WORDS = ROW_LIVE + MAX_ROWS


@triton.jit
def _req_words(
    b, rq, positions, batch_ids, slots, num_slots, max_pos, req, RQP: tl.constexpr
):
    """Request b's words (``REQ_*``): its batch id, its slot (0 when dead), whether it is live, and its rows' (a row of
    the request, live; ``RQP`` >= rq: a power of two)."""
    rb, slot, live, _ = _request(b, rq, positions, batch_ids, slots, num_slots, max_pos)
    tl.store(req + 4 + b, rb.to(tl.int32))
    tl.store(req + 12 + b, tl.where(live, slot, 0).to(tl.int32))
    tl.store(req + 20 + b, live.to(tl.int32))
    j = tl.arange(0, RQP)
    rj = tl.load(batch_ids + b * rq + j, mask=j < rq, other=-1).to(tl.int64)
    tl.store(req + 28 + b * rq + j, (live & (rj == rb)).to(tl.int32), mask=j < rq)


@triton.jit
def _dead_words(req, pos_out, rows, NB: tl.constexpr, BM: tl.constexpr):
    """The words of the batch slots past the step's ``NB`` requests up to ``BM`` (a band's dead ones: batch id -1, not
    live; first position 0, so their rows' reads of ATOM's RoPE caches stay in the table) and of the rows past its
    ``rows``: not live (none of their stores reach ATOM's storage)."""
    b = tl.arange(0, 8)
    dead = (b >= NB) & (b < BM)
    tl.store(pos_out + b, tl.zeros((8,), tl.int32), mask=dead)
    tl.store(req + 4 + b, tl.full((8,), -1, tl.int32), mask=dead)
    tl.store(req + 12 + b, tl.zeros((8,), tl.int32), mask=dead)
    tl.store(req + 20 + b, tl.zeros((8,), tl.int32), mask=dead)
    j = tl.arange(0, 64)
    tl.store(req + 28 + j, tl.zeros((64,), tl.int32), mask=j >= rows)


@triton.jit
def _ring_in(
    m, b, s, rq, pool, positions, batch_ids, slots, rings, layer_ids, pos_out, bounds_out,
    ring_start0, slot_rows, ring_slots, num_slots, max_pos,
    WIN: tl.constexpr, DIM: tl.constexpr, RB: tl.constexpr, BM: tl.constexpr,
):  # fmt: skip
    """The hi runtime layer m's ring slot s of request b <- ATOM's ring row of the
    position q in [ctx - WIN, ctx) with q % WIN == s (zeros below position 0 or
    on a pad request). (0, b, 0) also writes
    ``pos[b]`` (the request's first position) and its rows' bounds [3, RB]:
    (p + 1) // 2, (p + 1), ceil((p + 1) / 8)."""
    _, slot, live, ctx = _request(
        b, rq, positions, batch_ids, slots, num_slots, max_pos
    )
    _ring_row(m, b, s, ctx, slot, live, pool, rings, layer_ids, ring_start0, slot_rows, ring_slots, WIN, DIM,
              BM)  # fmt: skip
    if (m == 0) & (s == 0):
        _step_bounds(ctx, pos_out + b, bounds_out + b * rq, rq, RB)


@triton.jit
def _ring_row(
    m, b, s, ctx, slot, live, pool, rings, layer_ids, ring_start0, slot_rows, ring_slots,
    WIN: tl.constexpr, DIM: tl.constexpr, BM: tl.constexpr,
):  # fmt: skip
    """``_ring_in``'s copy: hi layer m's ring slot s of request b <- ATOM's ring row of position q in
    [ctx - WIN, ctx), q % WIN == s (zeros below 0 / on a pad request).
    """
    layer = tl.load(layer_ids + m).to(tl.int64)
    slot = tl.where(live, slot, 0)
    q = ctx - WIN + (s - ctx % WIN + WIN) % WIN
    ok = live & (q >= 0)
    row = (
        ring_start0
        + layer * ring_slots
        + slot * slot_rows
        + tl.maximum(q, 0) % ring_slots
    )
    d = tl.arange(0, DIM)
    v = tl.load(pool + row * DIM + d, mask=ok & (d < DIM), other=0.0)
    at = ((m * BM + b) * WIN + s).to(tl.int64)
    tl.store(rings + at * DIM + d, v)


@triton.jit
def _step_bounds(ctx, pos_out, bounds_out, rows, RB: tl.constexpr):
    """``pos`` (a request's first position) and its rows' bounds [3, RB]: (p + 1) // 2, (p + 1), ceil((p + 1) / 8)
    (``pos_out`` / ``bounds_out`` at the request's own word / first row)."""
    tl.store(pos_out, ctx.to(tl.int32))
    j = tl.arange(0, RB)
    p1 = ctx + j + 1
    jm = j < rows
    tl.store(bounds_out + j, (p1 // 2).to(tl.int32), mask=jm)
    tl.store(bounds_out + RB + j, p1.to(tl.int32), mask=jm)
    tl.store(bounds_out + 2 * RB + j, ((p1 + 7) // 8).to(tl.int32), mask=jm)


@triton.jit
def _ring_out(
    m, j, rq, pool, positions, batch_ids, slots, rings, layer_ids,
    ring_start0, slot_rows, ring_slots, num_slots, max_pos,
    WIN: tl.constexpr, DIM: tl.constexpr, BM: tl.constexpr,
):  # fmt: skip
    """The hi runtime layer m's key row of step row j (request b = j // rq, its ring
    slot (ctx_b + j - b rq) % WIN) -> ATOM's ring at that position (the
    request's own rows only)."""
    layer = tl.load(layer_ids + m).to(tl.int64)
    b = j // rq
    b0, slot, live, ctx = _request(
        b, rq, positions, batch_ids, slots, num_slots, max_pos
    )
    rj = tl.load(batch_ids + j).to(tl.int64)
    live = live & (rj == b0)
    slot = tl.where(live, slot, 0)
    p = ctx + j - b * rq
    d = tl.arange(0, DIM)
    v = tl.load(rings + ((m * BM + b) * WIN + p % WIN).to(tl.int64) * DIM + d)
    row = ring_start0 + layer * ring_slots + slot * slot_rows + p % ring_slots
    tl.store(pool + row * DIM + d, v, mask=live & (d < DIM))


# ---------------------------------------------------------------- Engram inputs


@triton.jit
def _eng_hist(cursor, cursor_stride, slots, positions, hist, NGRAM1: tl.constexpr):
    """The Engram table lookup's history of one request: the
    ``NGRAM1`` committed compressed ids before the step's first position (ATOM's
    cursor ``[slot, 1:]``, oldest first, -1 = none) into the hi runtime's ``hist`` [1, L]
    at positions ``p0 - NGRAM1 ..`` (the step's own rows: the launch writes them).
    ``slots`` / ``positions`` / ``hist``: the request's slot, its first row's position, its ``hist`` row.
    """
    slot = tl.load(slots).to(tl.int64)
    p0 = tl.load(positions)
    k = tl.arange(0, 4)
    q = p0 - NGRAM1 + k
    ok = (k < NGRAM1) & (q >= 0)
    z = tl.load(cursor + slot * cursor_stride + 1 + k, mask=ok, other=-1)
    tl.store(hist + q, z.to(tl.int32), mask=ok)


@triton.jit
def _eng_fold(b, rq, cursor, cursor_stride, slots, positions, ids_src, ids_dst, tmap, dead, cand, cand_stride,
              cand_req_stride, live, NGRAM1: tl.constexpr, RPOW: tl.constexpr):  # fmt: skip
    """The Engram fold, request b of the step (its ``rq`` rows from token b rq): its ids (ATOM's
    ``ids_src``) into the kv launch's ``ids_dst``, and the Engram cursor candidates ATOM's staging snapshot writes
    (``engram_snapshot``'s ``STAGE_CURSOR``, its ``engram_compress`` inlined): row l of the request, token
    b rq + l < ``live``, of ``cand`` (int [requests, rows, NGRAM1 + 1], strides ``cand_req_stride`` /
    ``cand_stride``) = the position after it, then the ``NGRAM1`` compressed ids ending at it, oldest first
    (``tmap[id]``, -1 for a dead row; before the request's first row ATOM's cursor ``[slot, 1:]``).
    """
    t0 = b * rq
    slot = tl.load(slots + b).to(tl.int64)
    p0 = tl.load(positions + t0)
    r = tl.arange(0, RPOW)
    tl.store(
        ids_dst + t0 + r,
        tl.load(ids_src + t0 + r, mask=r < rq, other=0).to(tl.int32),
        mask=r < rq,
    )
    lv = (r < rq) & (t0 + r < tl.load(live))
    out = cand + b * cand_req_stride + r * cand_stride
    tl.store(out, p0 + r + 1, mask=lv)
    for shift in tl.static_range(NGRAM1):
        back = r - shift
        own = lv & (back >= 0)
        tok = tl.load(ids_src + t0 + back, mask=own, other=0)
        z = tl.load(tmap + tok, mask=own, other=-1)
        z = tl.where(tl.load(dead + t0 + back, mask=own, other=0) != 0, -1, z)
        prev = tl.load(
            cursor + slot * cursor_stride + 1 + NGRAM1 + back,
            mask=lv & (back < 0),
            other=-1,
        )
        tl.store(out + NGRAM1 - shift, tl.where(back >= 0, z, prev), mask=lv)


@triton.jit
def _eng_row(j, src, src_stride, dst, KIN: tl.constexpr, BLOCK: tl.constexpr):
    """Row j of ATOM's staged Engram rows (bf16 [rows, KIN], row stride ``src_stride``) -> the kv launch's
    ``eng<L>.rows`` (contiguous)."""
    j64 = j.to(tl.int64)
    for o in range(0, KIN, BLOCK):
        d = o + tl.arange(0, BLOCK)
        tl.store(
            dst + j64 * KIN + d,
            tl.load(src + j64 * src_stride + d, mask=d < KIN),
            mask=d < KIN,
        )


# ---------------------------------------------------------------- the mHC hand-off


@triton.jit
def _planar(j, gate, pre, post, comb, NMIX: tl.constexpr, NHC: tl.constexpr):
    """Row j of the hi runtime's gate [rows, 24] -> ATOM's planar pre / post / comb."""
    g = tl.arange(0, 32)
    gv = tl.load(gate + j * NMIX + g, mask=g < NMIX, other=0.0)
    tl.store(pre + j * NHC + g, gv, mask=g < NHC)
    tl.store(post + j * NHC + (g - NHC), gv, mask=(g >= NHC) & (g < 2 * NHC))
    tl.store(comb + j * NHC * NHC + (g - 2 * NHC), gv, mask=(g >= 2 * NHC) & (g < NMIX))


@triton.jit
def _expand_row(
    j,
    hidden,
    hidden_stride,
    h_in,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    NHC: tl.constexpr,
):
    """Row j of the embedding expand into layer 0's ``h_in`` [rows, NHC, HIDDEN] (``from_embeddings``): ATOM's
    embedding row ``hidden[j]``."""
    src = hidden + j * hidden_stride
    for o in range(0, HIDDEN, BLOCK):
        d = o + tl.arange(0, BLOCK)
        v = tl.load(src + d)
        for c in range(NHC):
            tl.store(h_in + (j * NHC + c) * HIDDEN + d, v)


@triton.jit
def _collapse_chunk(j, c, gate, res, out, HIDDEN: tl.constexpr, BLOCK: tl.constexpr, NHC: tl.constexpr,
                    NMIX: tl.constexpr):  # fmt: skip
    """Row j, hidden block c of ATOM's ``collapse_streams`` (the end-step's ``COLLAPSE``): the bf16 streams
    ``res`` [rows, NHC, HIDDEN] weighted by the pre-mix (the gate's first NHC values, what ``_planar`` writes as
    ATOM's ``pre``), summed in stream order in fp32, stored bf16; a program a (row, block) as ATOM's kernel. Its
    launch keeps the multiply and the add apart (``enable_fp_fusion`` False), as ATOM's ``_collapse_kernel`` does,
    so the two agree to the bit."""
    d = c * BLOCK + tl.arange(0, BLOCK)
    keep = d < HIDDEN
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for m in tl.static_range(NHC):
        w = tl.load(gate + j * NMIX + m).to(tl.float32)
        x = tl.load(res + (j * NHC + m) * HIDDEN + d, mask=keep, other=0.0).to(
            tl.float32
        )
        acc += w * x
    tl.store(out + j * HIDDEN + d, acc, mask=keep)


@triton.jit
def _tap(j, r_a, aux, HIDDEN: tl.constexpr, BLOCK: tl.constexpr, NHC: tl.constexpr):
    """Row j: the DSpark tap of a layer's settled input, ATOM's K1 ``aux``: the
    mean over the bf16 streams (summed in stream order, then / 4)."""
    for o in range(0, HIDDEN, BLOCK):
        d = o + tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for h in range(NHC):
            acc += tl.load(r_a + (j * NHC + h) * HIDDEN + d).to(tl.float32)
        tl.store(
            aux + j * HIDDEN + d,
            (acc * (1.0 / NHC)).to(tl.bfloat16, fp_downcast_rounding="rtne"),
        )


@triton.jit
def _draft_begin_kernel(anchor, noise, h_in, anchor_positions, pos, step, hidden, max_pos, NHC: tl.constexpr,
                        BLOCK: tl.constexpr, BUMP: tl.constexpr = 1):  # fmt: skip
    """The hi block draft's inputs (one request): row r of the block's streams =
    the anchor's embedding (r = 0: ``anchor``, ATOM's embedding row) or the noise token's, in each of the ``NHC``
    streams (``build_block_state``'s); program (0, 0) also writes the block's first position (anchor + 1, int32),
    clamped to [0, max_pos] as the target's first position is (``_request``: a capture's dummy or a stale anchor reads
    in-range RoPE rows; every real position is below it, so the same bits), and (``BUMP``) bumps the draft's
    launch-tag step (a draft of B requests in one launch set: once, with its first request).
    """
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < hidden
    a = tl.load(anchor + offs, mask=m, other=0.0)
    v = tl.where(r == 0, a, tl.load(noise + offs, mask=m, other=0.0))
    for j in tl.static_range(NHC):
        tl.store(h_in + (r * NHC + j) * hidden + offs, v, mask=m)
    if (r == 0) & (tl.program_id(1) == 0):
        p = tl.load(anchor_positions).to(tl.int64) + 1
        tl.store(pos, tl.minimum(tl.maximum(p, 0), max_pos).to(tl.int32))
        if BUMP:
            tl.store(step, tl.load(step) + 1)


# ---------------------------------------------------------------- the bridge in two launches a step
#
# ``_begin_step_kernel`` / ``_end_step_kernel``: one launch each for what would be 6 / 8 separate launches (every small
# launch costs ~4.5-5 us in a replayed graph, launch-bound). A 1-D grid split into segments, each running one of the
# bodies above (the CPU tests run each body alone as a reference). The segments touch disjoint data (the ring import and
# request words / the embedding expand; the ring export / taps / gates), so their order inside the launch is free.
# Up to 3 taps (t0 / t1 / t2; layers 37-39).

MAX_TAPS = 3

# ---------------------------------------------------------------- the incremental import
#
# The full import rebuilds every hi layer's 128 ring slots each step. Between two served steps of one request,
# though, the hi runtime's rings differ from ATOM's only where the earlier step's rows wrote: its ring slots
# (ctx0 + j) % 128, j < rows (rejected rows overwrote slots the window still needs). ATOM's ring keeps the window's
# other rows (its draft slack). So a step continuing the last served one (``imp`` = [slot, ctx0, rows]: the same slot,
# ctx0 <= ctx <= ctx0 + rows) imports only those ring slots; any other step (a new request, a slot change, a pad step,
# the first after a forward the hook did not serve: the hook clears ``imp`` then) imports all of them, as the full
# import. A fixed grid (graph replay) of RP programs a layer loops over the work.

IMP_WORDS = 4  # ``imp``: int32 [slot (-1: none), ctx0, rows, 0] a request (batch slot b at words 4 b ..)


@triton.jit
def _invalidate_kernel(imp, slots, n, BLOCK: tl.constexpr):
    """Program b: request slot b's record ``imp[b]`` cleared (slot -1: its next served step imports in full) when its
    ATOM slot is one of the n slots an unserved forward ran (``slots``: that forward's ``BatchStep.slots``); every
    other record kept."""
    b = tl.program_id(0)
    rec = tl.load(imp + b * 4)
    j = tl.arange(0, BLOCK)
    s = tl.load(slots + j, mask=j < n, other=-2)
    hit = tl.sum(((s == rec) & (rec >= 0)).to(tl.int32), axis=0) > 0
    if hit:
        tl.store(imp + b * 4, -1)


@triton.jit
def _import_state(b, rq, positions, batch_ids, slots, num_slots, max_pos, imp):
    """(rb, slot, live, ctx, incr, ctx0, rows0) of request b: its batch id / slot / liveness / first position, and
    whether it continues the last served step of its batch slot (``imp[b]``, module comment above).
    """
    rb, slot, live, ctx = _request(
        b, rq, positions, batch_ids, slots, num_slots, max_pos
    )
    rec = imp + b * 4
    pslot = tl.load(rec).to(tl.int64)
    pctx = tl.load(rec + 1).to(tl.int64)
    prow = tl.load(rec + 2).to(tl.int64)
    incr = live & (pslot >= 0) & (pslot == slot) & (ctx >= pctx) & (ctx - pctx <= prow)
    return rb, slot, live, ctx, incr, pctx, prow


@triton.jit
def _ring_in_incr(
    m, b, r, rq, positions, pool, batch_ids, slots, rings, layer_ids, pos_out, bounds_out,
    ring_start0, slot_rows, ring_slots, num_slots, max_pos, imp,
    WIN: tl.constexpr, DIM: tl.constexpr, RB: tl.constexpr, RP: tl.constexpr, BM: tl.constexpr,
):  # fmt: skip
    """Program (m, b, r) of the incremental ring import: layer m's slots r, r + RP, .. of request b that the last
    served step's rows wrote (all of them unless ``incr``), each ``_ring_row``; (0, b, 0) also pos / bounds.
    """
    _, slot, live, ctx, incr, pctx, prow = _import_state(
        b, rq, positions, batch_ids, slots, num_slots, max_pos, imp
    )
    for s in range(r, WIN, RP):
        if (~incr) | (((s - pctx % WIN + WIN) % WIN) < prow):
            _ring_row(m, b, s, ctx, slot, live, pool, rings, layer_ids, ring_start0, slot_rows, ring_slots, WIN, DIM,
                      BM)  # fmt: skip
    if (m == 0) & (r == 0):
        _step_bounds(ctx, pos_out + b, bounds_out + b * rq, rq, RB)


@triton.jit
def _begin_step_kernel(
    pool, positions, batch_ids, slots, rings, layer_ids, pos_out, bounds_out,
    ring_start0, slot_rows, ring_slots, num_slots, max_pos, rows,
    hidden, hidden_stride, h_in, step, req,
    e0_src, e0_src_stride, e0_dst, e1_src, e1_src_stride, e1_dst, cursor, cursor_stride, hist, imp,
    NMF: tl.constexpr, WIN: tl.constexpr, DIM: tl.constexpr, RB: tl.constexpr,
    HIDDEN: tl.constexpr, BLOCK: tl.constexpr, NHC: tl.constexpr,
    NE0: tl.constexpr = 0, NE1: tl.constexpr = 0, KIN: tl.constexpr = 6144, EHIST: tl.constexpr = 0,
    INCR: tl.constexpr = 0, RP: tl.constexpr = 128,
    NB: tl.constexpr = 1, BM: tl.constexpr = 1, RQP: tl.constexpr = 1,
    fold_ids=None, fold_dst=None, fold_tmap=None, fold_dead=None, fold_cand=None, fold_cand_stride=0,
    fold_live=None, hist_stride=0, fold_cand_req_stride=0, EFOLD: tl.constexpr = 0, RPOW: tl.constexpr = 16,
):  # fmt: skip
    """Grid NMF * NB * RP + rows + NE0 + NE1 + EHIST:
    the ring import of each of the step's ``NB`` requests (and its pos /
    bounds; program (0, b, 0) also request b's words, ``_req_words``), a row
    each of the embedding expand
    into layer 0's ``h_in`` (``SinglePassHCState.from_embeddings``; ``hidden``:
    ATOM's embedding rows), then the
    Engram inputs: a row each of ATOM's staged
    rows of the two Engram layers (``_eng_row``; NE* = rows, or 0), and
    (EHIST = the step's requests, the table lookup) each request's history from ATOM's cursor
    (``_eng_hist``, ``hist`` rows ``hist_stride`` apart; with ``EFOLD`` the same program also copies the request's ids
    from ATOM's ``fold_ids`` into the kv launch's ``ids`` (``fold_dst``) and writes its ATOM Engram cursor
    candidates, ``_eng_fold``); program 0 also bumps the launch-tag ``step`` and clears the words of the batch slots
    and rows past the step's (``_dead_words``). Request b = rows b * rows / NB .. (``_request``); its
    hi-side buffers are batch slot b of buffers holding ``BM``.

    The full import (``INCR`` = 0): a program a ring slot (RP = WIN). The incremental import (``INCR`` = 1, the hook's
    served path): RP programs a layer loop over only what changed since ``imp[b]`` (the request slot's last served
    step, ``_end_step_kernel``'s ``TRACK``), or over everything when the request does not continue it
    (``_import_state``)."""
    pid = tl.program_id(0)
    rq = rows // NB
    n0 = NMF * NB * RP
    n5 = n0 + rows
    if pid < n0:
        m = pid // (NB * RP)
        b = (pid % (NB * RP)) // RP
        r = pid % RP
        if INCR:
            _ring_in_incr(m, b, r, rq, positions, pool, batch_ids, slots, rings, layer_ids, pos_out, bounds_out,
                          ring_start0, slot_rows, ring_slots, num_slots, max_pos, imp, WIN, DIM, RB, RP,
                          BM)  # fmt: skip
        else:
            _ring_in(m, b, r, rq, pool, positions, batch_ids, slots, rings, layer_ids, pos_out, bounds_out,
                     ring_start0, slot_rows, ring_slots, num_slots, max_pos, WIN, DIM, RB, BM)  # fmt: skip
        if (m == 0) & (r == 0):
            _req_words(b, rq, positions, batch_ids, slots, num_slots, max_pos, req, RQP)
        if pid == 0:
            tl.store(step, tl.load(step) + 1)
            _dead_words(req, pos_out, rows, NB, BM)
    elif pid < n5:
        _expand_row(
            (pid - n0).to(tl.int64), hidden, hidden_stride, h_in, HIDDEN, BLOCK, NHC
        )
    elif pid < n5 + NE0:
        _eng_row(pid - n5, e0_src, e0_src_stride, e0_dst, KIN, BLOCK)
    elif pid < n5 + NE0 + NE1:
        _eng_row(pid - n5 - NE0, e1_src, e1_src_stride, e1_dst, KIN, BLOCK)
    elif pid < n5 + NE0 + NE1 + EHIST:
        b = pid - n5 - NE0 - NE1  # request b's history (EHIST = the step's requests)
        _eng_hist(
            cursor,
            cursor_stride,
            slots + b,
            positions + b * rq,
            hist + b * hist_stride,
            3,
        )
        if EFOLD:
            _eng_fold(b, rq, cursor, cursor_stride, slots, positions, fold_ids, fold_dst, fold_tmap, fold_dead,
                      fold_cand, fold_cand_stride, fold_cand_req_stride, fold_live, 3, RPOW)  # fmt: skip


@triton.jit
def _end_step_kernel(
    pool, positions, batch_ids, slots, rings, layer_ids,
    ring_start0, slot_rows, ring_slots, num_slots, max_pos, rows,
    t0_r_a, t0_aux, t1_r_a, t1_aux, t2_r_a, t2_aux, gate, pre, post, comb, imp,
    NMF: tl.constexpr, WIN: tl.constexpr, DIM: tl.constexpr,
    HIDDEN: tl.constexpr, BLOCK: tl.constexpr, NHC: tl.constexpr, NMIX: tl.constexpr,
    NTAPS: tl.constexpr, PLANAR: tl.constexpr, TRACK: tl.constexpr = 0,
    NB: tl.constexpr = 1, BM: tl.constexpr = 1,
    col_res=None, col_out=None, COLLAPSE: tl.constexpr = 0,
):  # fmt: skip
    """Grid (NMF + NTAPS + PLANAR (+ HIDDEN / BLOCK with COLLAPSE)) * rows, ``rows`` programs a
    segment: the ring export, the DSpark taps, and (PLANAR) the last
    hi layer's gates planar for ATOM's collapse (``COLLAPSE``: then the
    collapse itself into ``col_out``, a segment a BLOCK of the hidden,
    ``_collapse_chunk``). Row j belongs to request
    j // (rows / NB) (``_ring_out``). ``TRACK`` (the
    incremental import): program b < NB records request b's step in ``imp[b]``
    (int32 [slot, ctx, rows a request]; slot -1 on a pad request) for the next
    step's import (``_begin_step_kernel``'s ``INCR``)."""
    pid = tl.program_id(0)
    seg = pid // rows
    j = pid % rows
    rq = rows // NB
    if TRACK:  # noqa: SIM102 - Triton: a constexpr and a runtime if
        if pid < NB:
            _, s0, live0, c0 = _request(
                pid, rq, positions, batch_ids, slots, num_slots, max_pos
            )
            tl.store(imp + pid * 4, tl.where(live0, s0, -1).to(tl.int32))
            tl.store(imp + pid * 4 + 1, c0.to(tl.int32))
            tl.store(imp + pid * 4 + 2, rq)
    if seg < NMF:
        _ring_out(seg, j, rq, pool, positions, batch_ids, slots, rings, layer_ids, ring_start0, slot_rows, ring_slots,
                  num_slots, max_pos, WIN, DIM, BM)  # fmt: skip
    else:
        t = seg - NMF
        if t < NTAPS:
            if t == 0:
                _tap(j, t0_r_a, t0_aux, HIDDEN, BLOCK, NHC)
            elif t == 1:
                _tap(j, t1_r_a, t1_aux, HIDDEN, BLOCK, NHC)
            else:
                _tap(j, t2_r_a, t2_aux, HIDDEN, BLOCK, NHC)
        elif t < NTAPS + PLANAR:
            _planar(j, gate, pre, post, comb, NMIX, NHC)
        elif COLLAPSE:  # a segment a BLOCK of the hidden: row j's chunk
            _collapse_chunk(
                j, t - NTAPS - PLANAR, gate, col_res, col_out, HIDDEN, BLOCK, NHC, NMIX
            )
