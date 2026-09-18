# SPDX-License-Identifier: MIT
"""CSA2 candidate block selection: level one of the two-level index top-k.

A source layer picks the blocks, and the layers after it score their own rows
only inside them (`restrict_to_candidates`). Both are read out of the FP8 plane
by `paged_scoring`, which is the only index scorer there is.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.topk import top_k_per_row_decode


def _blocks(width, block_size):
    """Candidate blocks a `width`-column row spans, checking the block is sane."""
    if block_size & (block_size - 1):
        raise ValueError(f"A {block_size}-row candidate block is not a power of two")
    return triton.cdiv(width, block_size)


@triton.jit
def _restrict_kernel(
    logits,
    candidates,
    width,
    topk_blocks,
    logits_stride,
    candidate_stride,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    STEPS: tl.constexpr,
):
    """-inf the columns of the blocks this row did not choose.

    A row's candidates are ascending with a `-1` tail, so reading that tail as a
    value past every block id makes the row monotone and one binary search
    answers membership.
    """
    row = tl.program_id(0)
    ids = tl.program_id(1) * TILE + tl.arange(0, TILE)
    lo = tl.zeros((TILE,), tl.int32)
    hi = tl.full((TILE,), topk_blocks, tl.int32)
    for _ in tl.static_range(STEPS):
        mid = (lo + hi) // 2
        got = tl.load(
            candidates + row * candidate_stride + mid, mid < topk_blocks, other=-1
        )
        below = tl.where(got < 0, width, got) < ids
        lo = tl.where(below, mid + 1, lo)
        hi = tl.where(below, hi, mid)
    got = tl.load(candidates + row * candidate_stride + lo, lo < topk_blocks, other=-1)
    columns = ids[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    tl.store(
        logits + row * logits_stride + columns,
        float("-inf"),
        (columns < width) & (got != ids)[:, None],
    )


def restrict_to_candidates(logits, candidates, block_size, tile=64):
    """-inf every column outside the candidate blocks, in place.

    The blocks are an earlier layer's, picked from that layer's own scores, so
    this is not redundant with the top-k below it: a row this layer ranks
    highly can sit in a block that layer did not keep, and dropping it is the
    mechanism. Masking a fully scored width selects the same rows as scoring
    only the kept ones, and the blocks stay a list of ids because the column
    mask they stand for is the largest allocation on a long-context step.
    """
    rows, width = logits.shape
    topk_blocks = candidates.shape[-1]
    _restrict_kernel[(rows, triton.cdiv(_blocks(width, block_size), tile))](
        logits,
        candidates,
        width,
        topk_blocks,
        logits.stride(0),
        candidates.stride(0),
        BLOCK=block_size,
        TILE=tile,
        STEPS=max(topk_blocks - 1, 1).bit_length() + 1,
    )


@triton.jit
def _block_maxima_kernel(
    logits,
    visible,
    maxima,
    ends,
    blocks,
    logits_stride,
    maxima_stride,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    """Each block's best visible score, `+inf` on the row's newest block.

    The newest block holds the most recent tokens and is only partly filled, so
    it is pinned rather than left to be outscored. `ends` is the top-k's row
    bound in blocks, written here because this pass already knows it.
    """
    row = tl.program_id(0)
    seen = tl.load(visible + row).to(tl.int32)
    ids = tl.program_id(1) * TILE + tl.arange(0, TILE)
    columns = ids[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    scores = tl.load(
        logits + row * logits_stride + columns,
        columns < seen,
        other=float("-inf"),
    )
    best = tl.max(scores, axis=1)
    newest = (seen - 1) // BLOCK
    best = tl.where((seen > 0) & (ids == newest), float("inf"), best)
    tl.store(maxima + row * maxima_stride + ids, best, ids < blocks)
    if tl.program_id(1) == 0:
        tl.store(ends + row, (seen + BLOCK - 1) // BLOCK)


def pick_candidate_blocks(logits, visible, block_size, topk_blocks, tile=64):
    """The `topk_blocks` best blocks per row, ascending, `-1` padded.

    A block's score is its best visible row's. The bound and the ordering both
    belong to kernels that already take them, so neither is applied here.
    """
    rows, width = logits.shape
    blocks = _blocks(width, block_size)
    maxima = torch.empty(rows, blocks, dtype=torch.float32, device=logits.device)
    ends = torch.empty(rows, dtype=torch.int32, device=logits.device)
    _block_maxima_kernel[(rows, triton.cdiv(blocks, tile))](
        logits,
        visible,
        maxima,
        ends,
        blocks,
        logits.stride(0),
        maxima.stride(0),
        BLOCK=block_size,
        TILE=tile,
    )
    chosen = torch.empty(rows, topk_blocks, dtype=torch.int32, device=logits.device)
    # The decode entry point: its prefill sibling would want a row-start array
    # as well, and every row here starts at zero.
    top_k_per_row_decode(
        maxima,
        1,
        ends,
        chosen,
        rows,
        maxima.stride(0),
        maxima.stride(1),
        k=topk_blocks,
        stable=True,
    )
    return chosen
