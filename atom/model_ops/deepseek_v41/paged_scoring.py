# SPDX-License-Identifier: MIT
"""Decode-time CSA2 index scoring, straight out of the FP8 paged plane.

The tiled scorer in `indexer.py` reads keys as BF16 and holds a tile of them;
this one hands the plane to `deepgemm_fp8_paged_mqa_logits` and never
materializes a key. What it costs is expressiveness: visibility has to be a
per-row prefix and ties go to the smaller position, which is why the other
formats and the large-position tie-break stay on the tiled path.

Each query row is its own batch item (`next_n=1`), which is how a speculative
block gets an exact bound per drafted token rather than one shared by the
request -- the same shape DeepSeek-V4's `_score_topk_decode` passes.
"""

import torch
from aiter.ops.quant import dynamic_per_token_scaled_quant
from aiter.ops.topk import top_k_per_row_decode
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

from atom.model_ops.v4_kernels import scale_indexer_weights

from .indexer import ascending_with_padding


def quantize_query_rows(query):
    """E4M3 with one FP32 scale per `(token, head)` row.

    Not ATOM's `quantize_fp8`, whose grid is group-32 E8M0: the scorer
    dequantizes Q by folding one scale into that head's weight, so the row is
    the quantization block and the scale is a plain float.
    """
    rows = query.reshape(-1, query.shape[-1])
    stored = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows.shape[0], 1), dtype=torch.float32, device=rows.device)
    dynamic_per_token_scaled_quant(stored, rows, scale)
    return stored.view_as(query), scale


def round_query_rows(query):
    """The same rounding, kept in BF16 for the scorer that has no FP8 path.

    So that a tiled run and a paged run differ in how they accumulate and not
    in what they accumulate: the query a BF16 einsum sees is the one the FP8
    kernel would have dequantized.
    """
    stored, scale = quantize_query_rows(query)
    return (stored.float() * scale.view(*query.shape[:-1], 1)).to(query.dtype)


def unit_table(block_tables, batch_ids, units_per_page):
    """One row per query token, naming the tiles of that request's PAGEs.

    A PAGE holds `units_per_page` consecutive tiles, so this is the request's
    PAGE table with each entry expanded in place -- the translation the plane's
    region-major layout buys and the reason a block id is not a PAGE id.

    Defined only where the batch id is: a padding row's tiles are whatever
    row -1 gathers, and the scorer bails on its zero visible count before it
    loads one. Naming a real request there would state a guarantee this does
    not give.
    """
    pages = block_tables[batch_ids.long()].long()
    tiles = torch.arange(units_per_page, device=pages.device)
    return (pages[..., None] * units_per_page + tiles).flatten(-2).int()


def _as_blocks(logits, block_size):
    """`[rows, blocks, block_size]`, a view -- so a write through it lands.

    Always a view, never a padded copy: the scored width is a whole number of
    tiles and a tile is a whole number of candidate blocks, so a width that
    does not divide means the two were configured apart and the mask below
    would be written into a temporary and lost.
    """
    rows, width = logits.shape
    if width % block_size:
        raise ValueError(
            f"{width} scored columns is not a whole number of {block_size}-row "
            "candidate blocks"
        )
    return logits.view(rows, -1, block_size)


def restrict_to_candidates(logits, candidates, block_size):
    """-inf every row outside the candidate blocks, in place.

    The consumer layers attend inside the blocks their candidate source chose,
    so scoring the whole width and removing the rest is the same selection --
    a kept row's score is not a function of what else was scored. Padding
    lands in a column past the real ones rather than on block 0, which a
    `clamp` would silently keep.

    Broadcast over the block axis rather than expanding the mask to one bool
    per column: at a million-token context that expansion is the largest
    allocation on the step.
    """
    blocks = -(-logits.shape[1] // block_size)
    keep = torch.zeros(
        logits.shape[0], blocks + 1, dtype=torch.bool, device=logits.device
    )
    ids = torch.where(candidates >= 0, candidates, blocks).long()
    keep.scatter_(1, ids, torch.ones_like(ids, dtype=torch.bool))
    _as_blocks(logits, block_size).masked_fill_(~keep[:, :blocks, None], -torch.inf)


def pick_candidate_blocks(logits, visible, block_size, count):
    """The `count` best blocks per row, ascending, with the newest pinned.

    The block's score is its best row's, and the block holding the most recent
    visible row is kept whatever it scored -- both are the tiled path's rules,
    since this only changes where the scores come from.
    """
    maxima = _as_blocks(logits, block_size).amax(-1)
    ids = torch.arange(maxima.shape[-1], device=logits.device).expand_as(maxima)
    seen = visible.long()[:, None]
    maxima = maxima.masked_fill(ids >= -(-seen // block_size), -torch.inf)
    maxima = maxima.masked_fill(
        (seen > 0) & (ids == (seen - 1) // block_size), torch.inf
    )
    # Stable, descending: ties go to the smaller block, as the tiled merge does.
    order = maxima.sort(stable=True, dim=-1, descending=True).indices[:, :count]
    chosen = ids.gather(-1, order)
    return ascending_with_padding(
        torch.where(maxima.gather(-1, order) > -torch.inf, chosen, -1)
    )


def score_topk_decode(
    query,
    weights,
    plane,
    tiles,
    visible,
    *,
    topk,
    weights_scale,
    candidates=None,
    block_size=8,
    candidate_count=0,
):
    """`(selected, candidate blocks)` for a whole decode batch.

    `selected` is `[rows, topk]` ascending compressed-row ids, -1 padded, the
    layout `build_indices` reads. Rows past a row's own visibility are never
    picked: both kernels take `visible` as the bound, so the columns the
    scorer left unwritten are outside it.

    The scored width and the block the kernel pages by both come off `plane`,
    which is the only place either is a fact rather than a restatement.
    """
    rows, heads = weights.shape
    tile = plane.shape[1]
    width = tiles.shape[1] * tile
    q_fp8, q_scale = quantize_query_rows(query)
    logits = torch.empty(rows, width, dtype=torch.float32, device=query.device)
    deepgemm_fp8_paged_mqa_logits(
        q_fp8.view(rows, 1, heads, q_fp8.shape[-1]),
        plane.unsqueeze(-2),
        # Q's scale is dequantized by folding it into its head's weight, which
        # is the only place the kernel has for it.
        scale_indexer_weights(
            weights.contiguous(), q_scale.view(rows, heads, 1), weights_scale
        ),
        logits,
        visible,
        tiles,
        width,
        KVBlockSize=tile,
        Preshuffle=True,
    )
    chosen = None
    if candidate_count:
        chosen = pick_candidate_blocks(logits, visible, block_size, candidate_count)
    if candidates is not None:
        restrict_to_candidates(logits, candidates, block_size)
    selected = torch.empty(rows, topk, dtype=torch.int32, device=query.device)
    top_k_per_row_decode(
        logits,
        1,
        visible,
        selected,
        rows,
        logits.stride(0),
        logits.stride(1),
        k=topk,
        stable=True,
    )
    # Ascending, because the attention kernel sums its prefix in the order it
    # is given and the tiled path emits ascending: a different order is a
    # different rounding, not a different selection.
    return ascending_with_padding(selected), chosen
