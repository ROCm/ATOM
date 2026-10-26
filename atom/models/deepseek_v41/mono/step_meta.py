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

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.mono.plan.execution import BLOCKS

# a token's window key metadata words: first window position, window count,
# block table offset, slot row base
KMETA = 4


@triton.jit
def _running_max(a, b):
    return tl.maximum(a, b)


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
):
    """``index_plan.fill_walk_plan`` over ``index_plan.walk_pieces`` of rows lo
    .. hi, each row's bound its visible columns at ``RATIO``: ``plan`` [CTAS, 4]
    gets a (lead, span, ch0, ch1) a task, piece after piece, and zeros past the
    last."""
    t = lo + tl.arange(0, BLOCK)
    m = t < hi
    owner = tl.load(batch_ids + t, mask=m, other=0)
    prev = tl.load(batch_ids + t - 1, mask=m & (t > lo), other=0)
    run_start = m & ((t == lo) | (owner != prev))
    run_first = tl.associative_scan(tl.where(run_start, t, 0), 0, _running_max)
    lead = m & (run_start | ((t - run_first) % WALK_ROWS == 0))
    # a lead's piece runs to the next lead
    after = t[None, :] > t[:, None]
    end = tl.min(tl.where(lead[None, :] & after, t[None, :], hi), 1)
    pos = tl.load(positions + t, mask=m, other=0)
    bound = ((pos + 1) // RATIO).to(tl.int32)
    inside = (t[None, :] >= t[:, None]) & (t[None, :] < end[:, None])
    top = tl.max(tl.where(inside, bound[None, :], 0), 1)
    need = tl.where(lead & (top > TOPK), (top + CHUNK - 1) // CHUNK, 0)
    # the narrowest width a chunk run that leaves every piece a task fits in CTAS
    total = tl.sum(need, 0)
    width = tl.maximum((total + CTAS - 1) // CTAS, 1)
    while tl.sum((need + width - 1) // width, 0) > CTAS:
        width += 1
    k = (need + width - 1) // width
    first = tl.cumsum(k, 0) - k
    s = tl.arange(0, CTAS)[:, None]
    hit = (s >= first[None, :]) & (s < (first + k)[None, :])
    task_lead = tl.sum(tl.where(hit, t[None, :], 0), 1)
    task_span = tl.sum(tl.where(hit, (end - t)[None, :], 0), 1)
    task_need = tl.sum(tl.where(hit, need[None, :], 0), 1)
    task_k = tl.maximum(tl.sum(tl.where(hit, k[None, :], 0), 1), 1)
    i = tl.arange(0, CTAS) - tl.sum(tl.where(hit, first[None, :], 0), 1)
    live = tl.sum(hit.to(tl.int32), 1) > 0
    row = plan + tl.arange(0, CTAS) * 4
    tl.store(row, tl.where(live, task_lead, 0))
    tl.store(row + 1, tl.where(live, task_span, 0))
    tl.store(row + 2, tl.where(live, i * task_need // task_k, 0))
    tl.store(row + 3, tl.where(live, (i + 1) * task_need // task_k, 0))


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
    WORDS: tl.constexpr,
    RATIOS: tl.constexpr,
    CTAS: tl.constexpr,
    WALK_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # program 0 the per-token integers, program 1 + r ratio r's walk plan
    for r in tl.static_range(len(RATIOS)):
        if tl.program_id(0) == r + 1:
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
            )
    if tl.program_id(0) != 0:
        return
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
    step, geometry, window, ring_rel, key_meta, walk_plans, walk_rows
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
    # a piece a row at most, and each needs a task of its own: else the width
    # search never ends
    assert n < BLOCKS or not walk_plans, n
    ratios = tuple(walk_plans)
    _step_meta_kernel[(1 + len(ratios),)](
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
        WORDS=KMETA,
        RATIOS=ratios,
        CTAS=BLOCKS,
        WALK_ROWS=ip.WALK_ROWS,
        TOPK=ip.TOPK,
        CHUNK=ip.CHUNK,
        BLOCK=triton.next_power_of_2(n),
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
