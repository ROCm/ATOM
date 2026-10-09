# SPDX-License-Identifier: MIT
"""Score inside one layer's candidate blocks instead of masking a full width.

A request's blocks are its PAGE table, each entry `units_per_page` blocks; this
narrows them to the blocks an earlier layer kept, so a consumer scores
`topk_blocks` blocks and not a whole context.

A candidate id IS a logical block id, so the translation is block `c % units`
of the PAGE its request's column `c // units` names -- true only while a
candidate block and an index block are the same length, which is why
`V41PoolGeometry.index_block_rows` is declared rather than derived.

Candidates are valid logical blocks, ascending with a `-1` tail. Only the
newest visible block can be partly filled, and it is last if selected, so a
plain length describes the compacted visibility exactly. Selection normally
pins that block; the length remains valid if it is absent and all kept blocks
are full.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_table_kernel(
    candidates,
    block_tables,
    batch_ids,
    visible,
    table,
    context,
    candidate_stride,
    page_stride,
    batch_stride,
    topk_blocks,
    UNITS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token: translate its kept blocks, then bound them.

    A slot past the row's kept count, and every slot of a padding token (batch
    id -1, its PAGE table never read), is written as block 0 rather than left
    at `-1`: `context` stops short of it so the scorer never reads it, and a
    negative id would be a real out-of-bounds read if anything ever did.
    """
    token = tl.program_id(0)
    batch = tl.load(batch_ids + token * batch_stride)
    slots = tl.arange(0, BLOCK)
    mask = slots < topk_blocks
    cand = tl.load(candidates + token * candidate_stride + slots, mask=mask, other=-1)
    live = cand >= 0

    page = tl.load(
        block_tables + batch * page_stride + cand // UNITS,
        mask=live & (batch >= 0),
        other=0,
    )
    unit = tl.where(live & (batch >= 0), page * UNITS + cand % UNITS, 0)
    tl.store(table + token * candidate_stride + slots, unit, mask=mask)

    seen = tl.load(visible + token).to(tl.int32)
    kept = tl.sum(live.to(tl.int32))
    last_selected = tl.max(tl.where(live, cand, -1))
    # A kept block before the request's newest block is full, not the entire
    # gap to `seen`. Bound its contribution by one block even if the pin was
    # lost upstream; empty rows still have no visible columns.
    last_span = tl.minimum(tl.maximum(seen - last_selected * ROWS, 0), ROWS)
    bound = (kept - 1) * ROWS + last_span
    tl.store(context + token, tl.where(kept > 0, bound, 0))


@triton.jit
def _lift_kernel(
    selected,
    candidates,
    selected_stride,
    candidate_stride,
    topk,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compacted columns back to compressed-row ids, in place, `-1` preserved.

    In place because the map takes a slot to itself -- column `s` of a row
    reads and writes column `s` -- so a second buffer would be a copy of the
    answer and nothing else.

    Order is kept without re-sorting: `candidates` is ascending, so a higher
    compacted column is always a higher real row, which is the order the
    attention kernel sums its prefix in.
    """
    token = tl.program_id(0)
    slots = tl.arange(0, BLOCK)
    mask = slots < topk
    column = tl.load(selected + token * selected_stride + slots, mask=mask, other=-1)
    live = column >= 0
    block = tl.load(
        candidates + token * candidate_stride + column // ROWS, mask=live, other=0
    )
    tl.store(
        selected + token * selected_stride + slots,
        tl.where(live, block * ROWS + column % ROWS, -1),
        mask=mask,
    )


def candidate_block_table(
    candidates, block_tables, batch_ids, units_per_page, visible, *, rows_per_block
):
    """`(table, context)` for scoring only the blocks `candidates` kept.

    `candidates` are each token's logical blocks, `block_tables` [B, columns]
    its request's PAGE table (`batch_ids` the token's request, -1 a padding
    token), a PAGE `units_per_page` blocks. `table` is `[tokens, topk_blocks]`
    physical block ids and `context` how far into `topk_blocks *
    rows_per_block` columns each row may read. Both are fresh allocations, so
    a caller inside a graph capture gets the address the replay will read.
    """
    tokens, topk_blocks = candidates.shape
    table = torch.empty_like(candidates)
    context = torch.empty(tokens, dtype=torch.int32, device=candidates.device)
    if tokens:
        _candidate_table_kernel[(tokens,)](
            candidates,
            block_tables,
            batch_ids,
            visible,
            table,
            context,
            candidates.stride(0),
            block_tables.stride(0),
            batch_ids.stride(0),
            topk_blocks,
            UNITS=units_per_page,
            ROWS=rows_per_block,
            BLOCK=triton.next_power_of_2(topk_blocks),
        )
    return table, context


class CandidateBlocks(NamedTuple):
    """An earlier layer's kept blocks, bound to one plane's tiles.

    `ids` are the logical blocks a selection lifts back through, `table` and
    `context` the scorer's block table and bounds over them, and
    `rows_per_block` the length all three were built at: a plane paged at any
    other cannot take them as a block table.
    """

    ids: torch.Tensor
    table: torch.Tensor
    context: torch.Tensor
    rows_per_block: int


def bind_candidates(
    ids, block_tables, batch_ids, units_per_page, visible, *, rows_per_block
):
    """`CandidateBlocks` for `ids` over their requests' PAGE tables
    (`candidate_block_table`), for every layer they bound."""
    table, context = candidate_block_table(
        ids, block_tables, batch_ids, units_per_page, visible,
        rows_per_block=rows_per_block,
    )  # fmt: skip
    return CandidateBlocks(ids, table, context, rows_per_block)


def lift_candidate_selection(selected, candidates, *, rows_per_block):
    """Rewrite compacted columns as the compressed-row ids they stand for."""
    tokens, topk = selected.shape
    if not tokens:
        return
    _lift_kernel[(tokens,)](
        selected,
        candidates,
        selected.stride(0),
        candidates.stride(0),
        topk,
        ROWS=rows_per_block,
        BLOCK=triton.next_power_of_2(topk),
    )


def candidate_block_table_reference(
    candidates, block_tables, batch_ids, units_per_page, visible, *, rows_per_block
):
    """Pure-torch twin, a loop over tokens.

    The property under test is which physical block each kept candidate names
    and how far into them a row may read; a loop states both directly, and a
    vectorised rework would be a second chance to make the same addressing
    mistake in the same shape.
    """
    table = torch.zeros_like(candidates)
    context = torch.zeros(
        candidates.shape[0], dtype=torch.int32, device=candidates.device
    )
    for token in range(candidates.shape[0]):
        kept = [c for c in candidates[token].tolist() if c >= 0]
        batch = int(batch_ids[token])
        for slot, cand in enumerate(kept):
            if batch >= 0:
                page = int(block_tables[batch, cand // units_per_page])
                table[token, slot] = page * units_per_page + cand % units_per_page
        if kept:
            seen = int(visible[token])
            context[token] = sum(
                min(rows_per_block, max(seen - cand * rows_per_block, 0))
                for cand in kept
            )
    return table, context


def lift_candidate_selection_reference(selected, candidates, *, rows_per_block):
    """Pure-torch twin of the lift, in place like the kernel."""
    for token in range(selected.shape[0]):
        for slot, column in enumerate(selected[token].tolist()):
            if column < 0:
                continue
            block = int(candidates[token, column // rows_per_block])
            selected[token, slot] = block * rows_per_block + column % rows_per_block
