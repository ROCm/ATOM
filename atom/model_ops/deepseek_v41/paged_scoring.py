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
    # Ascending already: `stable=True` is aiter's deterministic ascending,
    # smallest-index-first emit with `-1` for a short row, which is the order
    # the attention kernel sums its prefix in. Re-sorting it here was a kernel
    # that changed nothing, ties included.
    return selected, chosen
