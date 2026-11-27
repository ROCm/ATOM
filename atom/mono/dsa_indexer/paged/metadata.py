# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The per-token integers of a decode step the mono kernels read, each in one
launch: each token's window ring row (K1 writes its KV row there) and its window key
metadata (K2a's attention, ``attention._keys``): a target verify step's
(``write_step_meta``; as a torch chain these were ~19 elementwise kernels a
step), with the FP4 score stage's FULL-layer task plan a ratio, and a DSpark
draft block's (``write_draft_step_meta``, with its rows' positions)."""

import triton
import triton.language as tl

from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.selection_plan import SHORT_VERIFY_CTX, SPLIT_MIN
from atom.mono.plan.execution import BLOCKS

# a token's window key metadata words: first window position, window count,
# block table offset, slot row base
KMETA = 4


@triton.jit
def _request_groups(batch_ids, lo, hi, BLOCK: tl.constexpr):
    t = lo + tl.arange(0, BLOCK)
    live = t < hi
    ids = tl.load(batch_ids + t, mask=live, other=-1)
    prev = tl.load(batch_ids + t - 1, mask=live & (t > lo), other=-1)
    boundary = live & ((t == lo) | (ids != prev))
    start = boundary & (ids >= 0)
    first = tl.associative_scan(tl.where(start, t, lo), 0, _running_max)
    ends = tl.associative_scan(tl.where(boundary, t, hi), 0, _running_min, reverse=True)
    next_row = tl.gather(ends, tl.minimum(tl.arange(0, BLOCK) + 1, BLOCK - 1), 0)
    next_row = tl.where(t + 1 < hi, next_row, hi)
    end = tl.gather(next_row, first - lo, 0)
    ordinal = tl.cumsum(start.to(tl.int32), 0) - 1
    return (
        t,
        live & (ids >= 0),
        start,
        first,
        end,
        ordinal,
        tl.sum(start.to(tl.int32), 0),
    )


@triton.jit
def _rank_chunks(
    batch_ids,
    positions,
    lo,
    hi,
    top,
    lead,
    TOPK: tl.constexpr,
    RATIO: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    SHORT_CTX: tl.constexpr,
    SPLIT_CTX: tl.constexpr,
):
    _t, live, start, first, end, request, requests = _request_groups(
        batch_ids, lo, hi, BLOCK
    )
    last_pos = tl.load(positions + end - 1, mask=live, other=-1)
    request_bound = ((last_pos + 1) // RATIO).to(tl.int32)
    first_pos = tl.load(positions + first, mask=start, other=-1)
    first_bound = ((first_pos + 1) // RATIO).to(tl.int32)
    local = tl.sum((start & (first_bound > SHORT_CTX)).to(tl.int32), 0) == 0
    total = tl.sum(tl.where(start, request_bound, 0), 0)
    split = ((requests < WORLD) & (request_bound > SHORT_CTX)) | (
        (request_bound > SPLIT_CTX) & (2 * WORLD * request_bound > total)
    )
    span = ((request_bound + CHUNK - 1) // CHUNK + WORLD - 1) // WORLD
    offset = tl.where(~local & split, RANK * span, 0)
    need = tl.maximum(tl.minimum((top + CHUNK - 1) // CHUNK - offset, span), 0)
    owns = request % WORLD == RANK
    need = tl.where(
        local,
        (top + CHUNK - 1) // CHUNK,
        tl.where(split, need, tl.where(owns, (top + CHUNK - 1) // CHUNK, 0)),
    )
    need = tl.where(lead & live & (top > TOPK), need, 0)
    return offset, need


@triton.jit
def _running_max(a, b):
    return tl.maximum(a, b)


@triton.jit
def _running_min(a, b):
    return tl.minimum(a, b)


@triton.jit
def _walk_tasks(
    batch_ids,
    positions,
    lo,
    hi,
    RATIO: tl.constexpr,
    CTAS: tl.constexpr,
    WALK_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    SHORT_CTX: tl.constexpr,
    SPLIT_CTX: tl.constexpr,
):
    t = lo + tl.arange(0, BLOCK)
    m = t < hi
    owner = tl.load(batch_ids + t, mask=m, other=0)
    prev = tl.load(batch_ids + t - 1, mask=m & (t > lo), other=0)
    start = m & ((t == lo) | (owner != prev))
    first_row = tl.associative_scan(tl.where(start, t, 0), 0, _running_max)
    lead = m & (start | ((t - first_row) % WALK_ROWS == 0))
    next_lead = tl.associative_scan(
        tl.where(lead, t, hi), 0, _running_min, reverse=True
    )
    ii = tl.arange(0, BLOCK)
    end = tl.gather(next_lead, tl.minimum(ii + 1, BLOCK - 1), 0)
    end = tl.where(t + 1 < hi, end, hi)
    top = tl.full((BLOCK,), 0, tl.int32)
    for r in tl.static_range(WALK_ROWS):
        p = tl.load(positions + t + r, mask=lead & (t + r < end), other=-1)
        top = tl.maximum(top, ((p + 1) // RATIO).to(tl.int32))
    need = tl.where(lead & (top > TOPK), (top + CHUNK - 1) // CHUNK, 0)
    offset = tl.full((BLOCK,), 0, tl.int32)
    if WORLD > 1:
        first_pos = tl.load(positions + t, mask=start & (owner >= 0), other=-1)
        if tl.max((first_pos + 1) // RATIO, 0) > SHORT_CTX:
            offset, need = _rank_chunks(
                batch_ids,
                positions,
                lo,
                hi,
                top,
                lead,
                TOPK,
                RATIO,
                CHUNK,
                BLOCK,
                WORLD,
                RANK,
                SHORT_CTX,
                SPLIT_CTX,
            )
    total = tl.sum(need, 0)
    width = tl.maximum((total + CTAS - 1) // CTAS, 1)
    while tl.sum((need + width - 1) // width, 0) > CTAS:
        width += 1
    counts = (need + width - 1) // width
    first = tl.cumsum(counts, 0) - counts
    return t, end, need, offset, counts, first


@triton.jit
def _walk_plan(
    plan,
    batch_ids,
    positions,
    lo,
    hi,
    RATIO: tl.constexpr,
    CTAS: tl.constexpr,
    WALK_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    WORLD: tl.constexpr = 1,
    RANK: tl.constexpr = 0,
    SHORT_CTX: tl.constexpr = SHORT_VERIFY_CTX,
    SPLIT_CTX: tl.constexpr = SPLIT_MIN,
):
    t, end, need, offset, counts, first = _walk_tasks(
        batch_ids,
        positions,
        lo,
        hi,
        RATIO,
        CTAS,
        WALK_ROWS,
        TOPK,
        CHUNK,
        BLOCK,
        WORLD,
        RANK,
        SHORT_CTX,
        SPLIT_CTX,
    )
    s = tl.arange(0, CTAS)[:, None]
    hit = (s >= first[None, :]) & (s < (first + counts)[None, :])
    task_lead = tl.sum(tl.where(hit, t[None, :], 0), 1)
    task_span = tl.sum(tl.where(hit, (end - t)[None, :], 0), 1)
    task_need = tl.sum(tl.where(hit, need[None, :], 0), 1)
    task_k = tl.maximum(tl.sum(tl.where(hit, counts[None, :], 0), 1), 1)
    i = tl.arange(0, CTAS) - tl.sum(tl.where(hit, first[None, :], 0), 1)
    live = tl.sum(hit.to(tl.int32), 1) > 0
    row = plan + tl.arange(0, CTAS) * 4
    tl.store(row, tl.where(live, task_lead, 0))
    tl.store(row + 1, tl.where(live, task_span, 0))
    task_offset = tl.sum(tl.where(hit, offset[None, :], 0), 1)
    tl.store(row + 2, tl.where(live, task_offset + i * task_need // task_k, 0))
    tl.store(row + 3, tl.where(live, task_offset + (i + 1) * task_need // task_k, 0))


@triton.jit
def _walk_plan_one(
    plan,
    batch_ids,
    positions,
    lo,
    hi,
    task,
    RATIO: tl.constexpr,
    CTAS: tl.constexpr,
    WALK_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    WORLD: tl.constexpr = 1,
    RANK: tl.constexpr = 0,
    SHORT_CTX: tl.constexpr = SHORT_VERIFY_CTX,
    SPLIT_CTX: tl.constexpr = SPLIT_MIN,
):
    t, end, need, offset, counts, first = _walk_tasks(
        batch_ids,
        positions,
        lo,
        hi,
        RATIO,
        CTAS,
        WALK_ROWS,
        TOPK,
        CHUNK,
        BLOCK,
        WORLD,
        RANK,
        SHORT_CTX,
        SPLIT_CTX,
    )
    hit = (task >= first) & (task < first + counts)
    tok = tl.sum(tl.where(hit, t, 0), 0)
    span = tl.sum(tl.where(hit, end - t, 0), 0)
    chunks = tl.sum(tl.where(hit, need, 0), 0)
    num = tl.maximum(tl.sum(tl.where(hit, counts, 0), 0), 1)
    i = task - tl.sum(tl.where(hit, first, 0), 0)
    live = tl.sum(hit.to(tl.int32), 0) > 0
    row = plan + task * 4
    tl.store(row, tl.where(live, tok, 0))
    tl.store(row + 1, tl.where(live, span, 0))
    chunk_offset = tl.sum(tl.where(hit, offset, 0), 0)
    tl.store(row + 2, tl.where(live, chunk_offset + i * chunks // num, 0))
    tl.store(row + 3, tl.where(live, chunk_offset + (i + 1) * chunks // num, 0))


@triton.jit
def _step_meta_kernel(
    batch_ids,
    slots,
    positions,
    ring_rel,
    key_meta,
    walk_plans,
    walk_lo,
    walk_hi,
    table_stride,
    slot_rows,
    ring_slots,
    window_size,
    n,
    reset_buffers,
    ix_meta,
    peer_reset,
    reset_sizes: tl.constexpr,
    reset_split_only: tl.constexpr,
    PEER_BYTES: tl.constexpr,
    PEER_NONSPLIT_BYTES: tl.constexpr,
    SPLIT_RESETS: tl.constexpr,
    WORDS: tl.constexpr,
    RATIOS: tl.constexpr,
    CTAS: tl.constexpr,
    WALK_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    RESET_CTAS: tl.constexpr,
    PARALLEL_PLANS: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    SHORT_CTX: tl.constexpr,
    SPLIT_CTX: tl.constexpr,
):
    meta_ctas = 1 + len(RATIOS) * (CTAS if PARALLEL_PLANS else 1)
    if tl.program_id(0) >= meta_ctas:
        u = (tl.program_id(0) - meta_ctas) * 1024 + tl.arange(0, 1024)
        long_step = tl.full((), False, tl.int1)
        split_step = tl.full((), False, tl.int1)
        if PEER_BYTES or SPLIT_RESETS:
            t = tl.arange(0, BLOCK)
            live = t < n
            ids = tl.load(batch_ids + t, mask=live, other=-1)
            prev = tl.load(batch_ids + t - 1, mask=live & (t > 0), other=-1)
            start = live & (ids >= 0) & ((t == 0) | (ids != prev))
            p = tl.load(positions + t, mask=live & (ids >= 0), other=-1)
            long_step = tl.max(tl.where(start, p + 1, 0), 0) > SHORT_CTX
            # Conservative over all compression ratios: only low-request
            # steps or a possible dominant >256K request can split keys.
            # Owners never read radix histograms, including after a split.
            split_step = long_step & (
                (tl.sum(start.to(tl.int32), 0) < WORLD) | (tl.max(p + 1, 0) > SPLIT_CTX)
            )
        for r in tl.static_range(len(reset_sizes)):
            if not reset_split_only[r] or split_step:
                for base in range(0, reset_sizes[r], RESET_CTAS * 1024):
                    tl.store(
                        reset_buffers[r] + base + u, 0, mask=base + u < reset_sizes[r]
                    )
        if PEER_BYTES:  # noqa: SIM102
            if long_step:
                peer_words = tl.where(split_step, PEER_BYTES, PEER_NONSPLIT_BYTES) // 4
                for base in range(0, peer_words, RESET_CTAS * 1024):
                    tl.store(peer_reset + base + u, 0, mask=base + u < peer_words)
        return
    # program 0 the per-token integers, program 1 + r ratio r's walk plan
    for r in tl.static_range(len(RATIOS)):
        if PARALLEL_PLANS:
            if (tl.program_id(0) >= 1 + r * CTAS) & (
                tl.program_id(0) < 1 + (r + 1) * CTAS
            ):
                _walk_plan_one(
                    walk_plans[r],
                    batch_ids,
                    positions,
                    walk_lo,
                    walk_hi,
                    tl.program_id(0) - 1 - r * CTAS,
                    RATIOS[r],
                    CTAS,
                    WALK_ROWS,
                    TOPK,
                    CHUNK,
                    BLOCK,
                    WORLD,
                    RANK,
                    SHORT_CTX,
                    SPLIT_CTX,
                )
        elif tl.program_id(0) == r + 1:
            _walk_plan(
                walk_plans[r],
                batch_ids,
                positions,
                walk_lo,
                walk_hi,
                RATIOS[r],
                CTAS,
                WALK_ROWS,
                TOPK,
                CHUNK,
                BLOCK,
                WORLD,
                RANK,
                SHORT_CTX,
                SPLIT_CTX,
            )
    if tl.program_id(0) != 0:
        return
    if WORLD > 1:
        t, valid, start, first, end, request, requests = _request_groups(
            batch_ids, 0, n, BLOCK
        )
        tl.store(ix_meta + t * 4, tl.where(valid, request, -1), mask=t < n)
        tl.store(ix_meta + t * 4 + 1, first, mask=t < n)
        tl.store(ix_meta + t * 4 + 2, end - 1, mask=t < n)
        first_pos = tl.load(positions + t, mask=start, other=-1)
        short = tl.sum((start & (first_pos + 1 > SHORT_CTX)).to(tl.int32), 0) == 0
        tl.store(ix_meta + t * 4 + 3, requests | tl.where(short, 65536, 0), mask=t < n)
    i = tl.arange(0, BLOCK)
    m = i < n
    batch = tl.load(batch_ids + i, mask=m, other=-1).to(tl.int64)
    live = batch >= 0
    row = tl.maximum(batch, 0)
    slot = tl.load(slots + row, mask=m, other=0).to(tl.int64)
    pos = tl.load(positions + i, mask=m, other=0).to(tl.int64)
    # the bf16 pool's WindowParams: slot * slot_rows + pos % ring_slots; -1 a pad
    rel = slot * slot_rows + pos % ring_slots
    tl.store(ring_rel + i, tl.where(live, rel, -1).to(tl.int32), mask=m)
    # the original index build's (indices._indices) integers for a decode
    first = tl.maximum(pos - window_size + 1, 0)
    count = tl.where(live, pos + 1 - first, 0)
    tl.store(key_meta + i * WORDS, first.to(tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 1, count.to(tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 2, (row * table_stride).to(tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 3, (slot * slot_rows).to(tl.int32), mask=m)


def write_step_meta(
    step,
    geometry,
    window,
    ring_rel,
    key_meta,
    walk_plans,
    walk_rows,
    reset_buffers=(),
    parallel_plans=False,
    ix_meta=None,
    tp_world=1,
    tp_rank=0,
    peer_reset=None,
    split_reset_buffers=(),
    peer_nonsplit_bytes=None,
) -> None:
    """``ring_rel`` [S] int32: each token's window row before the layer's ring
    start, -1 for a pad row. ``key_meta`` [S, KMETA] int32: its first window
    position, window count (0 for a pad), block table offset and slot row base.
    ``walk_plans``: ratio -> [BLOCKS, 4] int32, the FP4 score stage's FULL-layer
    tasks over this step's rows (``index_plan.fill_walk_plan``); none on the
    FP8 plane. ``walk_rows``: (first row, rows) the plans cover."""
    n = ring_rel.shape[0]
    walk_lo, walk_n = walk_rows
    assert key_meta.shape == (n, KMETA) and key_meta.is_contiguous()
    if peer_nonsplit_bytes is not None:
        assert peer_reset is not None
        assert 0 <= peer_nonsplit_bytes <= peer_reset.nbytes
        assert peer_nonsplit_bytes % 4 == 0
    # a piece a row at most, and each needs a task of its own: else the width
    # search never ends
    assert n < BLOCKS or not walk_plans, n
    ratios = tuple(walk_plans)
    split_only = (False,) * len(reset_buffers) + (True,) * len(split_reset_buffers)
    resets = tuple(reset_buffers) + tuple(split_reset_buffers)
    reset_ctas = 32 if resets or peer_reset is not None else 0
    _step_meta_kernel[
        (1 + len(ratios) * (BLOCKS if parallel_plans else 1) + reset_ctas,)
    ](
        step.batch_ids,
        step.slots,
        step.positions,
        ring_rel,
        key_meta,
        tuple(walk_plans.values()),
        walk_lo,
        walk_lo + walk_n,
        step.block_tables.stride(0),
        window.slot_rows,
        window.ring_slots,
        geometry.window_size,
        n,
        resets,
        ix_meta if ix_meta is not None else ring_rel,
        peer_reset if peer_reset is not None else ring_rel,
        tuple(t.numel() for t in resets),
        split_only,
        PEER_BYTES=peer_reset.nbytes if peer_reset is not None else 0,
        PEER_NONSPLIT_BYTES=(
            (peer_reset.nbytes if peer_nonsplit_bytes is None else peer_nonsplit_bytes)
            if peer_reset is not None
            else 0
        ),
        SPLIT_RESETS=bool(split_reset_buffers),
        WORDS=KMETA,
        RATIOS=ratios,
        CTAS=BLOCKS,
        WALK_ROWS=ip.WALK_ROWS,
        TOPK=ip.TOPK,
        CHUNK=ip.CHUNK,
        BLOCK=triton.next_power_of_2(n),
        RESET_CTAS=reset_ctas,
        PARALLEL_PLANS=parallel_plans,
        WORLD=tp_world,
        RANK=tp_rank,
        SHORT_CTX=SHORT_VERIFY_CTX,
        SPLIT_CTX=SPLIT_MIN,
    )


@triton.jit
def _draft_meta_kernel(
    anchors,
    slots,
    positions,
    ring_rel,
    key_meta,
    slot_rows,
    ring_slots,
    window_size,
    width,
    num_slots,
    n,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    m = i < n
    # row i: request i // width's block row i % width
    request = i // width
    anchor = tl.load(anchors + request, mask=m, other=0).to(tl.int64)
    slot = tl.load(slots + request, mask=m, other=0).to(tl.int64)
    # a slot outside the pool (a warmup batch's rows past the ones its metadata
    # wrote) writes no key and reads none
    live = (slot >= 0) & (slot < num_slots)
    slot = tl.where(live, slot, 0)
    pos = anchor + 1 + i % width
    tl.store(positions + i, pos, mask=m)
    rel = tl.where(live, slot * slot_rows + pos % ring_slots, -1)
    tl.store(ring_rel + i, rel.to(tl.int32), mask=m)
    # every row: its request's window up to the anchor and whole block, both ways
    first = tl.maximum(anchor - window_size + 1, 0)
    count = tl.where(live, anchor + width - first + 1, 0)
    tl.store(key_meta + i * WORDS, first.to(tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 1, count.to(tl.int32), mask=m)
    # no block table offset: the draft has no indexer, so nothing selects through it
    tl.store(key_meta + i * WORDS + 2, tl.zeros([BLOCK], dtype=tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 3, (slot * slot_rows).to(tl.int32), mask=m)


def write_draft_step_meta(
    anchors, slots, num_slots, geometry, window, positions, ring_rel, key_meta
) -> None:
    """The DSpark draft blocks of ``anchors``' requests, request after request,
    ``width`` = rows / requests rows each: a block's rows sit at anchor + 1 ..
    anchor + width (``positions`` int64), their keys go to its slot's ring at
    those positions (``ring_rel``: the ring has window + width slots, so they
    never overwrite the window), and every row sees its request's window up to
    the anchor and all of its block (``key_meta``). A request whose slot is not
    one of the pool's ``num_slots`` writes and reads no key (``ring_rel`` -1,
    count 0); a padded request's slot, a real one, would be another's."""
    n = ring_rel.shape[0]
    requests = anchors.numel()
    width = n // requests
    assert width * requests == n and slots.numel() >= requests
    assert key_meta.shape == (n, KMETA) and key_meta.is_contiguous()
    assert window.ring_slots >= geometry.window_size + width
    _draft_meta_kernel[(1,)](
        anchors,
        slots,
        positions,
        ring_rel,
        key_meta,
        window.slot_rows,
        window.ring_slots,
        geometry.window_size,
        width,
        num_slots,
        n,
        WORDS=KMETA,
        BLOCK=triton.next_power_of_2(n),
    )
