# SPDX-License-Identifier: MIT
"""CSA2 candidate block selection: level one of the two-level index top-k.

A source layer picks the blocks, and the layers after it score their own rows
only inside them -- by paging over the kept list (`candidate_table`), which is
what makes the mask below a twin rather than a step. The picking is read out
of the FP8 plane by `paged_scoring`, which is the only index scorer there is --
and which bands the plane so that this kernel's `row * stride` stays inside
int32.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.topk import top_k_per_row_decode
from triton.language.extra import libdevice


def _blocks(width, block_size):
    """Candidate blocks a `width`-column row spans, checking the block is sane."""
    if block_size & (block_size - 1):
        raise ValueError(f"A {block_size}-row candidate block is not a power of two")
    return triton.cdiv(width, block_size)


@triton.jit
def _restrict_kernel(
    logits,
    candidates,
    visible,
    width,
    topk_blocks,
    logits_stride,
    candidate_stride,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    STEPS: tl.constexpr,
    LANES: tl.constexpr,
):
    """-inf the columns of the blocks this row did not choose.

    A row's candidates are ascending with a `-1` tail, so reading that tail as a
    value past every block id makes the row monotone and one binary search
    answers membership.

    Only out to the row's own visibility: the top-k below reads no further, so
    a -inf past it is a store nobody loads. A row's tiles are walked by `LANES`
    programs striding over them rather than one program each, so the grid stays
    a function of the batch alone -- what a captured replay reruns -- while the
    work follows the context a row actually has.
    """
    row = tl.program_id(0)
    seen = tl.load(visible + row).to(tl.int32)
    for tile in range(tl.program_id(1), tl.cdiv(seen, TILE * BLOCK), LANES):
        ids = tile * TILE + tl.arange(0, TILE)
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
        got = tl.load(
            candidates + row * candidate_stride + lo, lo < topk_blocks, other=-1
        )
        columns = ids[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
        tl.store(
            logits + row * logits_stride + columns,
            float("-inf"),
            (columns < seen) & (got != ids)[:, None],
        )


def _restrict_lanes(rows):
    """Programs per row, so the grid fills the machine and stops there.

    A tall batch wants few -- one program per row keeps that row's candidate
    list hot across its tiles -- and a short one wants many, having too few
    rows to occupy anything. `rows` is the only quantity known on the host and
    fixed for a captured shape. A fixed count is off by 12x at one end of
    `/app/logs_claude/restrict_lanes_sweep.md`, this by at most a third.
    """
    return min(256, max(1, 16384 // max(rows, 1)))


def restrict_to_candidates_reference(logits, candidates, visible, block_size, tile=64):
    """-inf every column outside the candidate blocks, in place.

    Nothing calls this: `candidate_table` reaches the same selection by paging
    over the kept list instead. Kept as the judge of that -- restating it in the
    test would be the same addressing written twice.

    The blocks are an earlier layer's, picked from that layer's own scores, so
    this is not redundant with the top-k after it: a row this layer ranks
    highly can sit in a block that layer did not keep, and dropping it is the
    mechanism.
    """
    rows, width = logits.shape
    topk_blocks = candidates.shape[-1]
    lanes = min(_restrict_lanes(rows), triton.cdiv(_blocks(width, block_size), tile))
    _restrict_kernel[(rows, lanes)](
        logits,
        candidates,
        visible,
        width,
        topk_blocks,
        logits.stride(0),
        candidates.stride(0),
        BLOCK=block_size,
        TILE=tile,
        STEPS=max(topk_blocks - 1, 1).bit_length() + 1,
        LANES=lanes,
    )


def _maxima_lanes(rows):
    """Programs per row for `_block_maxima_kernel`, about 2048 in the grid.

    A tile is one load and one max, so extra lanes have little latency to hide
    and an idle one is pure launch cost: the grid wants to fill the machine
    once and no more. Within about 10% of the best lane count from 6 to 8191
    rows and 2k to 256k context.
    """
    return min(256, max(1, 2048 // max(rows, 1)))


@triton.jit(do_not_specialize=["lanes"])
def _block_maxima_kernel(
    logits,
    visible,
    maxima,
    ends,
    logits_stride,
    maxima_stride,
    lanes,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    """Each block's best visible score, exclusive `+inf` on the newest block.

    The newest block holds the most recent tokens and is only partly filled, so
    it is pinned rather than left to be outscored. `ends` is the top-k's row
    bound in blocks, written here because this pass already knows it.

    Reserve `+inf` for that pin: another infinite score can tie it out of
    top-k, and NaNs can defeat the ordering entirely. Either would violate
    `candidate_table`'s newest-block invariant and let its computed context
    exceed the compacted allocation. Ignore NaNs and cap other blocks at the
    largest finite score before inserting the pin.

    Nothing past a row's `ends` is read or written: the top-k stops there, and
    the width is the block table's -- `max_model_len` -- not the row's context.
    `lanes` programs stride over a row's tiles, so the grid is fixed for a
    captured shape while the work follows the context. It is a runtime value
    and never specialized: it follows the row count, which on an eager prefill
    is any token count, and one compile per value would land in serving.
    """
    row = tl.program_id(0)
    seen = tl.load(visible + row).to(tl.int32)
    last = tl.cdiv(seen, BLOCK)
    for tile in range(tl.program_id(1), tl.cdiv(last, TILE), lanes):
        ids = tile * TILE + tl.arange(0, TILE)
        columns = ids[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
        scores = tl.load(
            logits + row * logits_stride + columns,
            columns < seen,
            other=float("-inf"),
        )
        best = tl.max(tl.where(libdevice.isnan(scores), float("-inf"), scores), axis=1)
        best = tl.minimum(best, 3.4028234663852886e38)
        best = tl.where(ids == last - 1, float("inf"), best)
        tl.store(maxima + row * maxima_stride + ids, best, ids < last)
    if tl.program_id(1) == 0:
        tl.store(ends + row, last)


def pick_candidate_blocks(logits, visible, block_size, out, tile=64):
    """Write the best blocks per row into `out`, ascending, `-1` padded.

    `out` is as wide as the count to keep. A block's score is its best visible
    row's; the bound and the ordering both belong to kernels that already take
    them, so neither is applied here.
    """
    rows, width = logits.shape
    topk_blocks = out.shape[-1]
    blocks = _blocks(width, block_size)
    maxima = torch.empty(rows, blocks, dtype=torch.float32, device=logits.device)
    ends = torch.empty(rows, dtype=torch.int32, device=logits.device)
    lanes = min(_maxima_lanes(rows), triton.cdiv(blocks, tile))
    _block_maxima_kernel[(rows, lanes)](
        logits,
        visible,
        maxima,
        ends,
        logits.stride(0),
        maxima.stride(0),
        lanes,
        BLOCK=block_size,
        TILE=tile,
    )
    # The decode entry point: its prefill sibling would want a row-start array
    # as well, and every row here starts at zero.
    top_k_per_row_decode(
        maxima,
        1,
        ends,
        out,
        rows,
        maxima.stride(0),
        maxima.stride(1),
        k=topk_blocks,
        stable=True,
    )
