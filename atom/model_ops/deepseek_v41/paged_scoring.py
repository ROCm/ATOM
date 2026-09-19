# SPDX-License-Identifier: MIT
"""CSA2 index scoring, straight out of the FP8 paged plane.

The only index scorer there is: it hands the plane to
`deepgemm_fp8_paged_mqa_logits` and never materializes a key, and what it asks
in return is that visibility be a per-row prefix.

Each query row is its own batch item (`next_n=1`) with its own bound and its
own tile list, so nothing here is a function of the batch's composition: a
prefill token, a decode token and a drafted token are the same row to it, and
a ragged batch needs no uniform query length. DeepSeek-V4's own paged scorer
reshapes `[bs, next_n]` and does demand one, which is why V4 keeps a separate
concatenated-key scorer for prefill and this file does not.

Scoring a prefill batch here rather than by concatenating the batch's keys was
measured on the production geometry: `/app/logs_claude/v41_index_scorer_record.md`.
The two arrangements pick identical rows and run within 15% of each other, and
this one's logits are `1/batch` of the other's because its columns are one
request's rows rather than every request's.
"""

import torch
from aiter.ops.quant import dynamic_per_token_scaled_quant
from aiter.ops.topk import top_k_per_row_decode
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

from atom.model_ops.v4_kernels import scale_indexer_weights

from .indexer import (
    pick_candidate_blocks,
    restrict_to_candidates,
)


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


def plane_rows(width):
    """Query rows a `width`-column logits plane may hold at once.

    Its readers reach a row as `row * stride` in int32, so a plane past 2**31
    elements wraps to a negative address. `width` follows the model-length cap
    and not the live context, so only a long-context configuration can get
    there; wherever a batch already fits, this leaves it in one piece.
    """
    return max(1, (2**31 - 1) // width)


def score_topk_paged(
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
    """`(selected, candidate blocks)` for a whole forward's query rows.

    `selected` is `[rows, topk]` ascending compressed-row ids, -1 padded, the
    layout `build_indices` reads. Rows past a row's own visibility are never
    picked: both kernels take `visible` as the bound, so the columns the
    scorer left unwritten are outside it.

    The scored width and the block the kernel pages by both come off `plane`,
    which is the only place either is a fact rather than a restatement.

    `candidates` bounds this layer to an earlier layer's blocks and
    `candidate_count` makes this layer that earlier one; never both.

    Rows run in bands of `plane_rows(width)`, a bound rather than a knob: a
    width that fits the whole batch in one band gets one.
    """
    rows, heads = weights.shape
    tile = plane.shape[1]
    width = tiles.shape[1] * tile
    q_fp8, q_scale = quantize_query_rows(query)
    # Q's scale is dequantized by folding it into its head's weight, which is
    # the only place the kernel has for it. Elementwise, so the whole batch's
    # goes in one launch and a band takes its slice.
    scaled = scale_indexer_weights(
        weights.contiguous(), q_scale.view(rows, heads, 1), weights_scale
    )
    selected = torch.empty(rows, topk, dtype=torch.int32, device=query.device)
    chosen = (
        torch.empty(rows, candidate_count, dtype=torch.int32, device=query.device)
        if candidate_count
        else None
    )
    band = plane_rows(width)
    # One band's plane, reused; a short last band is a prefix of it.
    logits = torch.empty(
        min(rows, band), width, dtype=torch.float32, device=query.device
    )
    for start in range(0, rows, band):
        count = min(band, rows - start)
        span = slice(start, start + count)
        seen, scores = visible[span], logits[:count]
        deepgemm_fp8_paged_mqa_logits(
            q_fp8[span].view(count, 1, heads, q_fp8.shape[-1]),
            plane.unsqueeze(-2),
            scaled[span],
            scores,
            seen,
            tiles[span],
            width,
            KVBlockSize=tile,
            Preshuffle=True,
        )
        if candidate_count:
            pick_candidate_blocks(scores, seen, block_size, chosen[span])
        if candidates is not None:
            restrict_to_candidates(scores, candidates[span], seen, block_size)
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
    # Ascending already: `stable=True` is aiter's deterministic ascending,
    # smallest-index-first emit with `-1` for a short row, which is the order
    # the attention kernel sums its prefix in. Re-sorting it here was a kernel
    # that changed nothing, ties included.
    return selected, chosen
