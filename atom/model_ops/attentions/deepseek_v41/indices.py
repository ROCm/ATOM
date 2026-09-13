# SPDX-License-Identifier: MIT
"""Ragged PAGE/STATE addressing for the unchanged V4 BF16 CSR kernels."""

import torch
import triton
import triton.language as tl

from atom.model_ops.deepseek_v41.paged_indices import _indptr
from atom.model_ops.v4_kernels.pool_index import window_constexprs, window_row


@triton.jit
def _indices(
    selected,
    pptr,
    prefix,
    eptr,
    extend,
    positions,
    batches,
    cu,
    slots,
    tables,
    table_stride,
    global_offset,
    ring_start,
    DECODE: tl.constexpr,
    ROWS_PER_PAGE: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
    PACKED: tl.constexpr,
    MAIN_ROW_BYTES: tl.constexpr,
):
    t = tl.program_id(0)
    batch = tl.load(batches + t)
    start = tl.load(cu + batch)
    pos = tl.load(positions + t)
    first = tl.maximum(0, pos - RING_SLOTS + 1)
    history_end = pos + 1 if DECODE else tl.load(positions + start)
    window_count = tl.maximum(0, history_end - first)
    pbegin, pend = tl.load(pptr + t), tl.load(pptr + t + 1)
    i = tl.arange(0, BLOCK)
    if TOPK:
        ids = tl.load(selected + t * TOPK + i, i < TOPK, other=-1)
        valid = (i < TOPK) & (ids >= 0)
        pages = tl.load(
            tables + batch * table_stride + ids // ROWS_PER_PAGE, valid, other=0
        )
        rank = tl.cumsum(valid.to(tl.int32)) - 1
        if PACKED:
            address = (
                pages.to(tl.int64) * PAGE_ROWS
                + global_offset
                + (ids % ROWS_PER_PAGE) * MAIN_ROW_BYTES
            )
            row = address << 1
        else:
            row = pages * PAGE_ROWS + global_offset + ids % ROWS_PER_PAGE
        tl.store(prefix + pbegin + rank, row, valid)
    slot = tl.load(slots + batch)
    row = window_row(
        slot.to(tl.int64) if PACKED else slot,
        first + i,
        ring_start,
        RING_SLOTS,
        SLOT_ROWS,
        RING_STRIDE,
        RUN_ROWS,
    )
    if PACKED:
        row = (row << 1) | 1
    tl.store(prefix + pend - window_count + i, row, i < window_count)
    if not DECODE:
        count = tl.minimum(t - start + 1, RING_SLOTS)
        begin = tl.load(eptr + t)
        tl.store(extend + begin + i, t - count + 1 + i, i < count)


def build_indices(selected, step, geometry, window, owner, ratio):
    positions = step.positions
    first = (positions - window.ring_slots + 1).clamp_min(0)
    starts = step.cu_seqlens_q[:-1][step.batch_ids.long()]
    history_end = positions + 1 if step.decode else positions[starts.long()]
    counts = (history_end - first).clamp_min(0)
    topk = 0 if selected is None else selected.shape[-1]
    if topk:
        counts = counts + (selected >= 0).sum(-1, dtype=torch.int32).flatten()
    pptr = _indptr(counts)
    prefix = torch.empty(
        step.length * (topk + window.ring_slots),
        dtype=torch.int64 if geometry.packed else torch.int32,
        device=positions.device,
    )
    extend_counts = (
        torch.arange(step.length, device=positions.device) - starts + 1
    ).clamp_max(window.ring_slots)
    eptr = pptr if step.decode else _indptr(extend_counts)
    extend = torch.empty(
        0 if step.decode else step.length * min(step.max_length, window.ring_slots),
        dtype=torch.int32,
        device=positions.device,
    )
    if step.length:
        _indices[(step.length,)](
            selected if topk else prefix,
            pptr,
            prefix,
            eptr,
            extend,
            positions,
            step.batch_ids,
            step.cu_seqlens_q,
            step.slots,
            step.block_tables,
            step.block_tables.stride(0),
            geometry.main_offset(owner) if ratio else 0,
            window.ring_start,
            DECODE=step.decode,
            ROWS_PER_PAGE=geometry.block_size // (ratio or 1),
            PAGE_ROWS=geometry.page_bytes
            // (1 if geometry.packed else geometry.row_bytes),
            PACKED=geometry.packed,
            MAIN_ROW_BYTES=geometry.main_row_bytes,
            TOPK=topk,
            BLOCK=triton.next_power_of_2(max(topk, window.ring_slots)),
            **window_constexprs(window),
        )
    return prefix, pptr, extend, eptr
