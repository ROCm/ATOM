# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block scores over MiniMax-M3's MXFP4 index cache (``--index-cache-dtype fp4``).

aiter's FlyDSL scorers (``score_flydsl_fp4`` for decode, ``score_prefill_flydsl_fp4``
for prefill) read the cache as the two views of ``mxfp4_pages.page_views`` and write
the same ``[heads, rows, max_block]`` fp32 scores the Triton scorers do, so the
selection and the block-table emit after them are index_topk's own, unchanged.

The query is e4m3 as the fused writer emits it (cast as stored, no per-tensor
scale).

The decode scorer wants a per-step work map. It is built once per step in the
metadata (outside any graph) into a persistent buffer (``IndexScoreBuffers``);
its row count is the grid, so every call of a step -- and every graph replay --
must pass the same ``max_block`` the map was built against. The scores' -inf
sentinel is refilled the same way, once a step (``fresh_index_score``), not once a layer.
"""

from dataclasses import dataclass

import torch

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_TOKENS


def index_batch(scheduled_bs, running_bs, running_tokens, max_query_len):
    """Requests the MXFP4 decode scorer and selector run: they take their batch off
    the rows of ``idx_q``, which a padded step (a graph bucket, or an
    eager step padded for DP) runs more of than were scheduled -- the pad requests
    have length 0 and dispatch nothing -- and the selector's bounds are laid out
    head-major over those rows. ``running_tokens`` rows of ``max_query_len`` tokens
    are that many requests; anything that is not between the scheduled and the
    padded batch is a builder bug, refused rather than clamped."""
    batch, rem = divmod(running_tokens, max_query_len)
    if rem or not scheduled_bs <= batch <= running_bs:
        raise ValueError(
            f"index decode: {running_tokens} rows of {max_query_len} query "
            f"tokens is not {scheduled_bs}..{running_bs} requests"
        )
    return batch


def fresh_index_score(
    buf, rows, num_idx_heads, max_block, max_query_len, fill_blocks=None
):
    """``[H, rows, max_block]`` view of ``buf`` for the scorer's ``out=``,
    in the layout the scorer would allocate itself: feature-contiguous
    (``[max_block, rows, H]`` permuted) once S * H >= its TRANSPOSE_MIN_F, plain
    head-major below. Call it once a step, outside the graph.

    The first ``fill_blocks`` columns (every column, by default) are reset to the
    scorer's -inf sentinel: a page a row never reaches is never stored, so it would
    keep an earlier step's score. Columns past a bound no row's valid-block count
    exceeds are never read by either selector, so a caller may bound the fill by
    the batch's own block count."""
    from aiter.ops.flydsl.kernels.minimax_m3_index_score import (
        NEG_INF,
        TRANSPOSE_MIN_F,
    )

    n = num_idx_heads * rows * max_block
    if n > buf.numel():
        raise ValueError(f"{rows} rows x {max_block} blocks outgrow the score buffer")
    flat = buf[:n]
    if max_query_len * num_idx_heads >= TRANSPOSE_MIN_F:
        score = flat.view(max_block, rows, num_idx_heads).permute(2, 1, 0)
    else:
        score = flat.view(num_idx_heads, rows, max_block)
    score[..., : fill_blocks or max_block].fill_(NEG_INF)
    return score


@dataclass(frozen=True)
class IndexScoreStep:
    """What one decode step hands every layer's scorer and selector: the step's work
    map, the block bound it was built against, and the score storage view."""

    work_map: torch.Tensor
    max_block: int
    out: torch.Tensor | None  # None: the scorer allocates its own


class IndexScoreBuffers:
    """Persistent storage of the MXFP4 decode scorer, whose pointers a captured graph
    bakes: the work map and the scores. ``begin_step`` fills them for one step,
    outside any graph, and returns the step's ``IndexScoreStep``."""

    def __init__(self, work_map, scores, max_block, num_idx_heads):
        self.work_map = work_map
        self.scores = scores
        self.max_block = max_block
        self.num_idx_heads = num_idx_heads

    @classmethod
    def alloc(
        cls, max_batch, max_query_len, num_idx_heads, max_block, device, scores=None
    ):
        """Storage for up to ``max_batch`` requests of up to ``max_query_len`` tokens.
        ``scores`` is existing storage to use instead of a new allocation."""
        rows = 0
        if max_batch > 0:
            from aiter.ops.flydsl import minimax_m3_index_score_work_map_capacity

            rows = minimax_m3_index_score_work_map_capacity(
                max_batch, max_block, max_query_len, num_idx_heads
            )
        work_map = torch.empty((rows, 2), dtype=torch.int32, device=device)
        if scores is None:
            # flat, up to max_batch * max_query_len rows ``max_block`` wide: the
            # persistent ``out`` of every layer of a captured step
            scores = torch.empty(
                num_idx_heads * max_batch * max_query_len * max_block,
                dtype=torch.float32,
                device=device,
            )
        return cls(work_map, scores, max_block, num_idx_heads)

    def begin_step(self, seq_lens, max_query_len, max_seq_len, eager=False):
        """The step's map, built into the persistent buffer, and its score view,
        refilled with -inf. None on an empty batch (the scorer then builds its own).

        The bound the map (its grid) is built against is the buffers' own -- the model
        length, which a captured graph bakes and which must not move between replays.
        ``eager``: this step's kernels are not recorded in a graph, so the batch's own
        block count serves, and the scorer's grid and score width shrink with it."""
        if seq_lens.shape[0] <= 0:
            return None
        max_block = self.max_block
        if eager:
            max_block = min(max_block, max(1, -(-max_seq_len // PAGE_TOKENS)))
        from aiter.ops.flydsl import minimax_m3_index_score_work_map

        work_map = minimax_m3_index_score_work_map(
            seq_lens,
            max_block,
            max_query_len,
            self.num_idx_heads,
            out=self.work_map,
        )
        out = fresh_index_score(
            self.scores,
            seq_lens.shape[0] * max_query_len,
            self.num_idx_heads,
            max_block,
            max_query_len,
            fill_blocks=-(-max_seq_len // PAGE_TOKENS),
        )
        return IndexScoreStep(work_map, max_block, out)
