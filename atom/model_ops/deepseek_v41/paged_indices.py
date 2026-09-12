# SPDX-License-Identifier: MIT
"""Map the eager CSA2 cache to the existing V4 sparse attention CSR inputs."""

import torch
import triton
import triton.language as tl

from atom.model_ops.v4_kernels.pool_index import window_constexprs, window_row


@triton.jit
def _write_indices(
    selected_ptr,
    prefix_indptr,
    prefix_indices,
    extend_indptr,
    extend_indices,
    position,
    global_start,
    ring_start,
    LENGTH: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
):
    t = tl.program_id(0)
    batch, offset = t // LENGTH, t % LENGTH
    pos = position + offset
    first = tl.maximum(0, pos - RING_SLOTS + 1)
    history_end = pos + 1 if LENGTH == 1 else position
    window_count = tl.maximum(0, history_end - first)
    prefix_begin = tl.load(prefix_indptr + t)
    prefix_end = tl.load(prefix_indptr + t + 1)
    i = tl.arange(0, BLOCK)
    if TOPK:
        selected = tl.load(selected_ptr + t * TOPK + i, i < TOPK, other=-1)
        valid = (i < TOPK) & (selected >= 0)
        rank = tl.cumsum(valid.to(tl.int32)) - 1
        tl.store(
            prefix_indices + prefix_begin + rank,
            batch * SLOT_ROWS + global_start + selected,
            valid,
        )
    tl.store(
        prefix_indices + prefix_end - window_count + i,
        window_row(
            batch, first + i, ring_start, RING_SLOTS, SLOT_ROWS, RING_STRIDE, RUN_ROWS
        ),
        i < window_count,
    )
    if LENGTH > 1:
        count = tl.minimum(offset + 1, RING_SLOTS)
        begin = tl.load(extend_indptr + t)
        tl.store(extend_indices + begin + i, t - count + 1 + i, i < count)


def _indptr(counts):
    result = torch.empty(counts.numel() + 1, dtype=torch.int32, device=counts.device)
    result[0] = 0
    torch.cumsum(counts.flatten(), 0, dtype=torch.int32, out=result[1:])
    return result


def build_sparse_indices(
    selected, *, position, length, batch_size, window, global_start, device
):
    """Produce valid CSR rows without copying KV or synchronizing device counts.

    Decode reads both SWA and global rows from the pool. Prefill reads global
    and prior SWA rows from the pool, and in-chunk SWA rows from the extend KV.
    Capacity is a shape-only upper bound; indptr controls the consumed region.
    The indexer may supply trailing padding or an empty global selection.
    """
    offset = torch.arange(length, dtype=torch.int32, device=device)
    first = (position + offset - window.ring_slots + 1).clamp_min(0)
    window_counts = ((position + 1 if length == 1 else position) - first).clamp_min(0)
    counts = window_counts.expand(batch_size, -1)
    topk = 0 if selected is None else selected.shape[-1]
    if topk:
        counts = counts + (selected >= 0).sum(-1, dtype=torch.int32)
    prefix_indptr = _indptr(counts)
    tokens = batch_size * length
    prefix = torch.empty(
        tokens * (topk + window.ring_slots), dtype=torch.int32, device=device
    )
    extend_counts = (offset + 1).clamp_max(window.ring_slots).expand(batch_size, -1)
    extend_indptr = _indptr(extend_counts) if length > 1 else prefix_indptr
    extend = torch.empty(
        tokens * min(length, window.ring_slots) if length > 1 else 0,
        dtype=torch.int32,
        device=device,
    )
    _write_indices[(tokens,)](
        selected if topk else prefix,
        prefix_indptr,
        prefix,
        extend_indptr,
        extend,
        position,
        global_start,
        window.ring_start,
        LENGTH=length,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(max(topk, window.ring_slots)),
        **window_constexprs(window),
    )
    return prefix, prefix_indptr, extend, extend_indptr
