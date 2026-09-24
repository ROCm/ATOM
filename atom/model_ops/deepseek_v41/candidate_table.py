# SPDX-License-Identifier: MIT
"""Score inside one layer's candidate blocks instead of masking a full width.

`unit_table` gives a token every block its request owns; this narrows it to the
blocks an earlier layer kept, so a consumer scores `topk_blocks` blocks and not
a whole context.

A candidate id IS a logical block id, so the translation is a gather out of
`tiles` and needs no PAGE table of its own -- true only while a candidate block
and an index block are the same length, which is why
`V41PoolGeometry.index_block_rows` is declared rather than derived.

The visibility a row carries into the compacted space is the property nothing
else states: candidates come back ascending and `pick_candidate_blocks` pins
the newest block, so the partly-filled block is the LAST kept slot and never an
interior one. That is what makes a plain length exact where an interior partial
block would need a per-column mask.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_table_kernel(
    candidates,
    tiles,
    visible,
    table,
    context,
    candidate_stride,
    tiles_stride,
    topk_blocks,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token: gather its kept blocks, then bound them.

    A slot past the row's kept count is written as block 0 rather than left at
    `-1`: `context` stops short of it so the scorer never reads it, and a
    negative id would be a real out-of-bounds read if anything ever did.
    """
    token = tl.program_id(0)
    slots = tl.arange(0, BLOCK)
    mask = slots < topk_blocks
    cand = tl.load(candidates + token * candidate_stride + slots, mask=mask, other=-1)
    live = cand >= 0

    unit = tl.load(tiles + token * tiles_stride + cand, mask=live, other=0)
    tl.store(table + token * candidate_stride + slots, unit, mask=mask)

    seen = tl.load(visible + token).to(tl.int32)
    kept = tl.sum(live.to(tl.int32))
    newest = tl.max(tl.where(live, cand, -1))
    bound = (kept - 1) * ROWS + (seen - newest * ROWS)
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


def candidate_block_table(candidates, tiles, visible, *, rows_per_block):
    """`(table, context)` for scoring only the blocks `candidates` kept.

    `table` is `[tokens, topk_blocks]` physical block ids and `context` how far
    into `topk_blocks * rows_per_block` columns each row may read. Both are
    fresh allocations, so a caller inside a graph capture gets the address the
    replay will read -- the same reason `unit_table` is built per forward.
    """
    tokens, topk_blocks = candidates.shape
    table = torch.empty_like(candidates)
    context = torch.empty(tokens, dtype=torch.int32, device=candidates.device)
    if tokens:
        _candidate_table_kernel[(tokens,)](
            candidates,
            tiles,
            visible,
            table,
            context,
            candidates.stride(0),
            tiles.stride(0),
            topk_blocks,
            ROWS=rows_per_block,
            BLOCK=triton.next_power_of_2(topk_blocks),
        )
    return table, context


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


def candidate_block_table_reference(candidates, tiles, visible, *, rows_per_block):
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
        for slot, cand in enumerate(kept):
            table[token, slot] = tiles[token, cand]
        if kept:
            seen = int(visible[token])
            context[token] = (len(kept) - 1) * rows_per_block + (
                seen - kept[-1] * rows_per_block
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
