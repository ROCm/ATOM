# SPDX-License-Identifier: MIT
"""Ragged PAGE/STATE addressing for the unchanged V4 BF16 CSR kernels."""

import torch
import triton
import triton.language as tl

from atom.model_ops.v4_kernels.pool_index import window_constexprs, window_row


@triton.jit
def _indptr_scan(
    batches,
    positions,
    cu,
    pptr,
    eptr,
    tokens,
    DECODE: tl.constexpr,
    WINDOW: tl.constexpr,
    RATIO: tl.constexpr,
    TOPK: tl.constexpr,
    EXTEND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program: a running offset over the forward's whole token axis.

    V4's `_v4_decode_indptr_kernel`, for CSA2's classes: a prefix sum has to
    see every earlier token, so this is one serial scan rather than a grid, and
    `tokens` is a runtime argument so a fresh token count does not make a fresh
    JIT variant.

    A padding token contributes 0, which leaves the tail of each indptr flat --
    the zero-length slice every reader downstream bails on, and the reason the
    writer below can skip those rows without leaving a hole.

    A row's selection count is closed-form rather than counted: the scorers
    emit `min(visible, columns)` ids and pad the rest with -1, which
    `test_selection_count_matches_its_closed_form` pins. It holds only while
    the candidates can hold a whole top-k, which `build_attention_topology`
    refuses a config without.
    """
    tl.store(pptr, 0)
    if EXTEND:
        tl.store(eptr, 0)
    prefix_total = 0
    extend_total = 0
    for base in tl.range(0, tokens, BLOCK):
        idx = base + tl.arange(0, BLOCK)
        in_range = idx < tokens
        bid = tl.load(batches + idx, mask=in_range, other=-1)
        live = in_range & (bid >= 0)
        pos = tl.load(positions + idx, mask=live, other=0).to(tl.int32)
        start = tl.load(cu + bid, mask=live, other=0)
        first = tl.maximum(pos - WINDOW + 1, 0)
        # Decode sees its own token; a prefill chunk sees only what its first
        # token already had, the rest arriving as the extend segment.
        history_end = (
            pos + 1
            if DECODE
            else tl.load(positions + start, mask=live, other=0).to(tl.int32)
        )
        count = tl.maximum(history_end - first, 0)
        if TOPK:
            count += tl.minimum((pos + 1) // RATIO, TOPK)
        count = tl.where(live, count, 0)
        tl.store(pptr + idx + 1, prefix_total + tl.cumsum(count, axis=0), mask=in_range)
        prefix_total += tl.sum(count)
        if EXTEND:
            reach = tl.where(live, tl.minimum(idx - start + 1, WINDOW), 0)
            tl.store(
                eptr + idx + 1, extend_total + tl.cumsum(reach, axis=0), mask=in_range
            )
            extend_total += tl.sum(reach)


@triton.jit
def _indptr_scan_all(
    batches,
    positions,
    cu,
    prefixes,
    extends,
    tokens,
    DECODE: tl.constexpr,
    WINDOW: tl.constexpr,
    RATIOS: tl.constexpr,
    TOPKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Ratios have independent scans, but share one launch. Each program owns
    # one output pair; no cross-program prefix or synchronization is needed.
    for i in tl.static_range(len(RATIOS)):
        if tl.program_id(0) == i:
            _indptr_scan(
                batches,
                positions,
                cu,
                prefixes[i],
                extends[i],
                tokens,
                DECODE,
                WINDOW,
                RATIOS[i] or 1,
                TOPKS[i],
                not DECODE,
                BLOCK,
            )


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
    layer_stride,
    plane,
    DECODE: tl.constexpr,
    ROWS_PER_PAGE: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    WINDOW: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
    PACKED: tl.constexpr,
    MAIN_ROW_BYTES: tl.constexpr,
):
    t = tl.program_id(0)
    # One program row per layer of the group. Everything a layer's indices
    # depend on is shared across the run except its ring, which sits exactly
    # `layer_stride` further along -- so the axis is an offset, not a lookup.
    layer = tl.program_id(1)
    ring_start += layer * layer_stride
    prefix += layer * plane
    batch = tl.load(batches + t)
    # Defined only where the batch id is, as V4's decode writer is: a padding
    # row owns no slot to address and was given no room to write into.
    if batch < 0:
        return
    start = tl.load(cu + batch)
    pos = tl.load(positions + t)
    first = tl.maximum(0, pos - WINDOW + 1)
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
        count = tl.minimum(t - start + 1, WINDOW)
        begin = tl.load(eptr + t)
        tl.store(extend + begin + i, t - count + 1 + i, i < count)


def fill_step_indptrs(step, geometry, buffers):
    """`{ratio: (prefix indptr, extend indptr, reserved top-k)}` for this step.

    Into the caller's fixed buffers, before any layer runs. Both halves are
    load-bearing: a table a layer fills on a miss is one a capture's recorded
    pass skips, and a table at a fresh address each forward is one its replay
    reads at the capture's.
    """
    ratios = geometry.layer_ratios
    built = {}
    for ratio in ratios:
        prefix, extend = buffers[ratio]
        pptr = prefix[: step.width + 1]
        eptr = pptr if step.decode else extend[: step.width + 1]
        built[ratio] = (pptr, eptr, geometry.batch_topk(ratio))
    if ratios:
        _indptr_scan_all[(len(ratios),)](
            step.batch_ids,
            step.positions,
            step.cu_seqlens_q,
            tuple(built[ratio][0] for ratio in ratios),
            tuple(built[ratio][1] for ratio in ratios),
            step.width,
            DECODE=step.decode,
            WINDOW=geometry.window_size,
            RATIOS=ratios,
            TOPKS=tuple(built[ratio][2] for ratio in ratios),
            BLOCK=min(1024, triton.next_power_of_2(max(step.width, 1))),
        )
    return built


def build_indices(selected, step, geometry, window, owner, ratio, layers=1, stride=0):
    """`layers` consecutive layers' indices, starting at `window`'s.

    Only a decode groups: its `extend` is empty, so the run needs no second
    plane, and it is the pass whose cost is the launch rather than the work.
    """
    if layers > 1 and not step.decode:
        raise ValueError("Only a decode batches its index build across layers")
    topk = 0 if selected is None else selected.shape[-1]
    pptr, eptr, reserved = step.indptrs[ratio]
    if topk != reserved:
        # Each row's reserve is closed-form in the width, and was taken before
        # any scorer ran. Another width leaves every row a hole its reader
        # dereferences.
        raise ValueError(f"Scorer width {topk} is not the {reserved} reserved")
    plane = step.width * (topk + geometry.window_size)
    prefix = torch.empty(
        layers * plane,
        dtype=torch.int64 if geometry.packed else torch.int32,
        device=step.positions.device,
    )
    extend = torch.empty(
        0 if step.decode else step.width * min(step.max_q_len, geometry.window_size),
        dtype=torch.int32,
        device=step.positions.device,
    )
    if step.width:
        _indices[(step.width, layers)](
            selected if topk else prefix,
            pptr,
            prefix,
            eptr,
            extend,
            step.positions,
            step.batch_ids,
            step.cu_seqlens_q,
            step.slots,
            step.block_tables,
            step.block_tables.stride(0),
            geometry.main_offset(owner) if ratio else 0,
            window.ring_start,
            stride,
            plane,
            DECODE=step.decode,
            ROWS_PER_PAGE=geometry.block_size // (ratio or 1),
            PAGE_ROWS=geometry.page_bytes
            // (1 if geometry.packed else geometry.row_bytes),
            PACKED=geometry.packed,
            MAIN_ROW_BYTES=geometry.main_row_bytes,
            TOPK=topk,
            BLOCK=triton.next_power_of_2(max(topk, geometry.window_size)),
            WINDOW=geometry.window_size,
            **window_constexprs(window),
        )
    return prefix.view(layers, plane), pptr, extend, eptr
