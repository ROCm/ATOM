# SPDX-License-Identifier: MIT
"""CSA2 index scoring, straight out of the paged index plane, through the
shared scorer (`atom.model_ops.indexer.score_topk`).

What is CSA2's here is how a layer's rows read the plane, said as the shared
`IndexRows`: FP8 rows each off their own tile list (`unit_tiles`); FP4 rows a
request's together off its PAGE table (`Fp4MqaRaggedMetadata`), their logits
packed by visibility (`PackedRows`) where the step laid them out so. A layer
bounded by an earlier layer's candidates scores only those blocks: the kept
list is handed over as the block table (`candidate_table`), so the width is
`candidate_topk_blocks` blocks and not a whole context, and its columns are
lifted back to compressed-row ids after. A layer producing candidates picks
them off each band's logits before its top-k.

Each query row is its own batch item (`next_n=1`) with its own bound, so
nothing here is a function of the batch's composition: a prefill token, a
decode token and a drafted token are the same row to it, and a ragged batch
needs no uniform query length.

Scoring a prefill batch here rather than by concatenating the batch's keys was
measured on the production geometry: `/app/logs_claude/v41_index_scorer_record.md`.
"""

import torch

from atom.model_ops.indexer import (
    IndexCache,
    fold_query_scale,
    fp4_rows,
    fp8_rows,
    quantize_query,
    quantize_query_rows,
    score_topk,
)

from .candidate_table import lift_candidate_selection
from .index_plane import FP4_FLYDSL
from .indexer import pick_candidate_blocks, pick_candidate_blocks_packed

__all__ = [
    "quantize_query_fp4",
    "quantize_query_rows",
    "score_topk_paged",
    "score_topk_quantized",
]


def quantize_query_fp4(query, rope, positions, *, layout=FP4_FLYDSL):
    """`(values, scales)` of the index query before its RoPE, `query` [rows,
    heads, dim], in one launch, the rotated values rounded to `query`'s dtype
    first as V4.1's RoPE writes them, then the official FP4 arithmetic (E2M1,
    one ceil-E8M0 a 32-dim group). Each row its own sequence of one."""
    return quantize_query(
        query,
        rope.cos_cache,
        rope.sin_cache,
        positions,
        rope.rope_dim,
        fp4=True,
        layout=layout,
        round_rope=True,
    )


def score_topk_paged(
    query,
    weights,
    units,
    tiles,
    visible,
    *,
    topk,
    weights_scale,
    candidates=None,
    block_size=8,
    candidate_count=0,
    workspace=None,
):
    """`(selected, candidate blocks)` for a whole forward's FP8 query rows:
    `query` rotated, quantized here, its scale folded into the head weights.

    `selected` is `[rows, topk]` ascending compressed-row ids, -1 padded, the
    layout `build_indices` reads. `candidates` (`CandidateBlocks`) limits this
    layer to an earlier layer's blocks and `candidate_count` makes this layer
    that earlier one; never both. The plane is `workspace`'s when given.
    """
    rows, heads = weights.shape
    q_fp8, q_scale = quantize_query_rows(query)
    scaled = fold_query_scale(weights, q_scale.view(rows, heads, 1), weights_scale)
    return score_topk_quantized(
        q_fp8,
        scaled,
        units,
        tiles,
        visible,
        topk=topk,
        candidates=candidates,
        block_size=block_size,
        candidate_count=candidate_count,
        workspace=workspace,
    )


def score_topk_quantized(
    query,
    scaled,
    units,
    tiles,
    visible,
    *,
    topk,
    candidates=None,
    block_size=8,
    candidate_count=0,
    workspace=None,
    weight_scale=1.0,
    ragged=None,
    packed=None,
):
    """`score_topk_paged` past its query quantization, in `units`' format.
    FP8: `query` [rows, heads, dim] e4m3 and `scaled` [rows, heads] fp32, the
    head weights with the query scale and `weights_scale` folded in. FP4:
    `query` is `quantize_query_fp4`'s pair and `scaled` the head weights, the
    kernel applying `weight_scale` to them. The mono decode computes both
    itself.

    `ragged` (`Fp4MqaRaggedMetadata`, FP4) scores each of its sequences' rows
    off the sequence's block table, and `tiles` is then unused. Without it
    each row is its own sequence on its own `tiles` (or candidate) row.
    `packed` (`PackedRows`, with `ragged`) scores into `workspace`'s packed
    logits rather than a plane, the step's rows laid out band by band.
    """
    # Producing candidates reads scores at their real columns, so a layer that
    # also scored inside someone else's would have two meanings for a column.
    assert not (candidate_count and candidates is not None)
    tile = units.values.shape[1]
    cache = IndexCache(
        units.values, units.scales if units.fp4 else None, tile, units.layout
    )
    if candidates is not None:
        # One number reached by two paths: the geometry pages the plane at
        # `candidate_block_size`, and this is where they have to agree.
        assert candidates.rows_per_block == tile, (
            f"a candidate block is {candidates.rows_per_block} rows but the "
            f"index plane is paged at {tile}; the candidate list cannot be a "
            "block table"
        )
        assert ragged is None, "candidate rows keep their own tables"
        table, bound = candidates.table, candidates.context
        width = candidates.ids.shape[1] * tile
    elif ragged is not None:
        # Every block of each request, PAGE by PAGE, and the row's own
        # visibility into it.
        table, bound = ragged.block_tables, visible
        width = ragged.block_tables.shape[1] * ragged.pages_per_block * tile
    else:
        # Every block the request owns, and the row's own visibility into it.
        table, bound, width = tiles, visible, tiles.shape[1] * tile
    if packed is not None:
        assert ragged is not None and workspace is not None and workspace.packed
        bound = packed.visible
    rows = _index_rows(cache, scaled, table, bound, width, ragged, packed, workspace)
    count = scaled.shape[0]
    chosen = (
        torch.empty(count, candidate_count, dtype=torch.int32, device=scaled.device)
        if candidate_count
        else None
    )
    selected = score_topk(
        query,
        scaled,
        cache,
        rows,
        topk=topk,
        weight_scale=weight_scale,
        stable=True,
        on_band=(
            None
            if not candidate_count
            else _picker(bound, block_size, chosen, packed, workspace, width)
        ),
    )
    if candidates is not None:
        # Columns of the compacted width, which only this function knows how to
        # read; every caller past it wants compressed-row ids.
        lift_candidate_selection(selected, candidates.ids, rows_per_block=tile)
    # Ascending already: `stable=True` is aiter's deterministic ascending,
    # smallest-index-first emit with `-1` for a short row, which is the order
    # the attention kernel sums its prefix in.
    return selected, chosen


def _index_rows(cache, scaled, table, bound, width, ragged, packed, workspace):
    """This layer's rows as the shared scorer reads them."""
    if not cache.fp4:
        return fp8_rows(bound, width, workspace, table)
    count, heads = scaled.shape
    starts = (
        None
        if ragged is not None
        else (
            workspace.row_starts(count)
            if workspace is not None
            else torch.arange(count + 1, dtype=torch.int32, device=scaled.device)
        )
    )
    return fp4_rows(
        bound,
        width,
        workspace,
        ragged,
        heads=heads,
        page_size=cache.rows_per_block,
        layout=cache.layout,
        row_tables=None if ragged is not None else table,
        row_starts=starts,
        bands=None if packed is None else packed.bands,
        packed_offsets=None if packed is None else packed.offsets,
    )


def _picker(bound, block_size, chosen, packed, workspace, width):
    """Each band's candidate blocks, off its logits before their top-k."""
    if packed is None:

        def pick(scores, span):
            pick_candidate_blocks(scores, bound[span], block_size, chosen[span])

        return pick

    def pick_packed(scores, span):
        pick_candidate_blocks_packed(
            scores,
            packed.offsets[span],
            workspace.packed_maxima(),
            packed.block_offsets[span],
            packed.visible[span],
            block_size,
            chosen[span],
            width,
        )

    return pick_packed
