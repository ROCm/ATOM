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
metadata (outside any graph) into a persistent buffer (``IndexWorkMap``);
its row count is the grid, so every call of a step -- and every graph replay --
must pass the same ``max_block`` the map was built against. The scores themselves
are allocated per call, as the Triton scorers' are: the scorer, given no ``out``,
allocates them filled with its -inf sentinel (``alloc_score``).
"""

from dataclasses import dataclass

import torch

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_TOKENS


@dataclass(frozen=True)
class IndexWorkMapStep:
    """What one decode step hands every layer's scorer: the step's work map and the
    block bound it was built against."""

    work_map: torch.Tensor
    max_block: int


class IndexWorkMap:
    """Persistent storage of the MXFP4 decode scorer's work map, whose pointer a
    captured graph bakes. ``begin_step`` fills it for one step, outside any graph,
    and returns the step's ``IndexWorkMapStep``."""

    def __init__(self, work_map, max_block, num_idx_heads):
        self.work_map = work_map
        self.max_block = max_block
        self.num_idx_heads = num_idx_heads

    @classmethod
    def alloc(cls, max_batch, max_query_len, num_idx_heads, max_block, device):
        """Work map for up to ``max_batch`` requests of up to ``max_query_len`` tokens."""
        rows = 0
        if max_batch > 0:
            from aiter.ops.flydsl import minimax_m3_index_score_work_map_capacity

            rows = minimax_m3_index_score_work_map_capacity(
                max_batch, max_block, max_query_len, num_idx_heads
            )
        work_map = torch.empty((rows, 2), dtype=torch.int32, device=device)
        return cls(work_map, max_block, num_idx_heads)

    def begin_step(self, seq_lens, max_query_len, max_seq_len, eager=False):
        """The step's work map, built into the persistent buffer. None on an empty
        batch (the scorer then builds its own).

        The bound the map (its grid) is built against is the buffer's own -- the model
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
        return IndexWorkMapStep(work_map, max_block)
