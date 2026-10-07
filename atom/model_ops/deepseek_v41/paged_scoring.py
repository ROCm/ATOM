# SPDX-License-Identifier: MIT
"""CSA2 index scoring, straight out of the paged index plane.

The only index scorer there is: it hands the plane to
`deepgemm_fp8_paged_mqa_logits` (FP8) or `flydsl_pa_mqa_logits_fp4` (FP4,
`index_plane`) and never materializes a key, and what it asks in return is
that visibility be a per-row prefix.

A layer bounded by an earlier layer's candidates scores only those blocks: the
kept list is handed over as the block table (`candidate_table`), so the width
is `candidate_topk_blocks` blocks and not a whole context.

Each query row is its own batch item (`next_n=1`) with its own bound and its
own tile list, so nothing here is a function of the batch's composition: a
prefill token, a decode token and a drafted token are the same row to it, and
a ragged batch needs no uniform query length. DeepSeek-V4's own paged scorer
reshapes `[bs, next_n]` and does demand one, which is why V4 keeps a separate
concatenated-key scorer for prefill and this file does not. FP4 is the
exception where it can be: a layer scoring requests' whole contexts hands the
kernel a request's rows as one sequence (`Fp4MqaRaggedMetadata`), sharing
each key load, each row still with its own bound.

Scoring a prefill batch here rather than by concatenating the batch's keys was
measured on the production geometry: `/app/logs_claude/v41_index_scorer_record.md`.
The two arrangements pick identical rows and run within 15% of each other, and
this one's logits are `1/batch` of the other's because its columns are one
request's rows rather than every request's.
"""

import torch
from aiter import dtypes
from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4
from aiter.ops.quant import dynamic_per_token_scaled_quant, rope_rotate_activation
from aiter.ops.topk import top_k_per_row_decode
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata
from atom.model_ops.v4_kernels import scale_indexer_weights

from .candidate_table import lift_candidate_selection
from .indexer import pick_candidate_blocks
from .score_workspace import plane_rows


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


def quantize_query_fp4(query, rope, positions):
    """`(values, scales)` of the index query before its RoPE, `query` [rows,
    heads, dim], in one launch: V4's fused RoPE + FP4 quant (aiter
    `rope_rotate_activation`), the rotated values rounded to `query`'s dtype
    first (`round_rope`) as V4.1's RoPE writes them, then the official FP4
    arithmetic (E2M1, one ceil-E8M0 a 32-dim group). Each row its own sequence
    of one: values [rows, 1, heads, dim / 2], scales in the row-group scorer's
    packed order."""
    rows, heads, dim = query.shape
    values = torch.empty(
        rows, 1, heads, dim // 2, dtype=torch.uint8, device=query.device
    )
    scales = torch.empty(
        rows, 1, dim // 128, 4, 16, -(-(heads // 16) // 4) * 4,
        dtype=torch.uint8, device=query.device,
    )  # fmt: skip
    rope_rotate_activation(
        values.view(dtypes.fp4x2),
        query,
        rope.cos_cache,
        rope.sin_cache,
        positions,
        rope.rope_dim,
        out_scale=scales,
        group_size=32,
        shuffle_scale=True,
        do_rotate_act=False,
        round_rope=True,
    )
    return values, scales


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
    """`(selected, candidate blocks)` for a whole forward's query rows.

    `selected` is `[rows, topk]` ascending compressed-row ids, -1 padded, the
    layout `build_indices` reads. Rows past a row's own visibility are never
    picked: both kernels take `visible` as the bound, so the columns the
    scorer left unwritten are outside it.

    `units` is the owner's FP8 `IndexUnits` (an FP4 query is quantized with
    its RoPE, `quantize_query_fp4`, and scored by `score_topk_quantized`). The
    block the kernel pages by comes off it, which is the only place it is a
    fact rather than a restatement. The width follows it: a row's whole
    context, or the kept blocks alone. Which one does not reach the caller --
    `selected` is compressed-row ids either way.

    `candidates` (`CandidateBlocks`, bound once a forward and shared by every
    layer they bound) limits this layer to an earlier layer's blocks and
    `candidate_count` makes this layer that earlier one; never both.

    Rows run in bands of `plane_rows(width)`, a bound rather than a knob: a
    width that fits the whole batch in one band gets one. The band is taken
    from `workspace` (`ScoreWorkspace`) when given, else allocated.
    """
    rows, heads = weights.shape
    q_fp8, q_scale = quantize_query_rows(query)
    # Q's scale is dequantized by folding it into its head's weight, which is
    # the only place the kernel has for it. Elementwise, so the whole batch's
    # goes in one launch and a band takes its slice.
    scaled = scale_indexer_weights(
        weights.contiguous(), q_scale.view(rows, heads, 1), weights_scale
    )
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
):
    """`score_topk_paged` past its query quantization, in `units`' format.
    FP8: `query` [rows, heads, dim] e4m3 and `scaled` [rows, heads] fp32, the
    head weights with the query scale and `weights_scale` folded in. FP4:
    `query` is `quantize_query_fp4`'s pair and `scaled` the head weights, the
    kernel applying `weight_scale` to them. The mono decode computes both
    itself. The rest, `workspace` included, as there.

    `ragged` (`Fp4MqaRaggedMetadata`, FP4) scores each of its sequences' rows
    off the sequence's block table, sharing each key load, and `tiles` is then
    unused. Without it each row is its own sequence on its own `tiles` (or
    candidate) row; a candidate-bounded layer's rows always are.
    """
    rows = scaled.shape[0]
    tile = units.values.shape[1]
    # Producing candidates reads scores at their real columns, so a layer that
    # also scored inside someone else's would have two meanings for a column.
    assert not (candidate_count and candidates is not None)
    compacted = candidates is not None
    if compacted:
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
        width = table.shape[1] * ragged.pages_per_block * tile
    else:
        # Every block the request owns, and the row's own visibility into it.
        table, bound, width = tiles, visible, tiles.shape[1] * tile
    device = scaled.device
    selected = torch.empty(rows, topk, dtype=torch.int32, device=device)
    chosen = (
        torch.empty(rows, candidate_count, dtype=torch.int32, device=device)
        if candidate_count
        else None
    )
    band = plane_rows(width)
    # One band's plane, reused; a short last band is a prefix of it.
    logits = (
        torch.empty(min(rows, band), width, dtype=torch.float32, device=device)
        if workspace is None
        else workspace.logits(min(rows, band), width)
    )
    for start in range(0, rows, band):
        count = min(band, rows - start)
        span = slice(start, start + count)
        seen, scores = bound[span], logits[:count]
        by_sequence = ragged is not None
        if by_sequence:
            band_ragged = ragged.band(start, count, rows)
        elif units.fp4:
            # every row its own sequence on its own table row
            starts = (
                torch.arange(count + 1, dtype=torch.int32, device=device)
                if workspace is None
                else workspace.row_starts(count)
            )
            band_ragged = Fp4MqaRaggedMetadata(starts, 1, table[span])
        else:
            band_ragged = None
        _band_logits(
            query,
            scaled,
            units,
            span,
            scores,
            seen,
            None if by_sequence else table[span],
            width,
            weight_scale,
            band_ragged,
        )
        if candidate_count:
            pick_candidate_blocks(scores, seen, block_size, chosen[span])
        top_k_per_row_decode(
            scores,
            1,
            seen,
            selected[span],
            count,
            scores.stride(0),
            scores.stride(1),
            k=topk,
            stable=True,
        )
    if compacted:
        # Columns of the compacted width, which only this function knows how to
        # read; every caller past it wants compressed-row ids.
        lift_candidate_selection(selected, candidates.ids, rows_per_block=tile)
    # Ascending already: `stable=True` is aiter's deterministic ascending,
    # smallest-index-first emit with `-1` for a short row, which is the order
    # the attention kernel sums its prefix in. Re-sorting it here was a kernel
    # that changed nothing, ties included.
    return selected, chosen


def _band_logits(
    query, scaled, units, span, scores, seen, table, width, weight_scale, ragged
):
    """Rows `span`'s logits into `scores`, each row seeing `seen` columns: FP8
    of the pages its `table` row lists, FP4 by `ragged`'s sequences, a
    sequence's rows sharing each key load."""
    page_rows = units.values.shape[1]
    if units.fp4:
        values, scales = query
        flydsl_pa_mqa_logits_fp4(
            values[span],
            scales[span],
            units.values,
            units.scales,
            weights=scaled[span],
            max_seq_len=width,
            weight_scale=weight_scale,
            kv_block_size=page_rows,
            out=scores,
            **ragged.kernel_args(
                seen, heads=scaled.shape[1], page_size=page_rows, max_seq_len=width
            ),
        )
        return
    count, heads = scores.shape[0], scaled.shape[1]
    deepgemm_fp8_paged_mqa_logits(
        query[span].view(count, 1, heads, query.shape[-1]),
        units.values.unsqueeze(-2),
        scaled[span],
        scores,
        seen,
        table,
        width,
        KVBlockSize=page_rows,
        Preshuffle=True,
    )
