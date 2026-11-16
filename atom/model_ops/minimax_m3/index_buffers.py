# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MXFP4 index scorer's persistent state as one record, and its allocation, in a
module that imports no aiter or triton at the top: the kernels' modules (`sparse_attn`,
`index_topk`) import aiter, and the non-GPU test suite has to be able to build the
record."""

from dataclasses import dataclass, replace

import torch

from atom.model_ops.minimax_m3.geometry import SPARSE_BLOCK_SIZE
from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace


@dataclass(frozen=True)
class IndexScoreBuffers:
    """The MXFP4 index scorer's persistent state, one immutable record.

    ``scores`` is the `LogitsWorkspace` every layer's scorer writes its fp32 block
    scores into, viewed contiguous ``[heads, rows, width]`` (what aiter's selector
    needs); never filled, since the selectors read a row only up to its valid
    blocks. ``work_map`` is the decode scorer's dispatch map and ``max_block`` the
    block count it was built at: checked for an exact match, and fixed across graph
    replays. A prefill step's record has neither.

    Under indexer CP (``cp_world > 1``) the decode scorer runs every index head over
    this rank's blocks ``b % cp_world == cp_rank`` only: the map is that shard's and
    ``max_block`` its bound, ``ceil(blocks / cp_world)``. A prefill stays TP.
    """

    scores: LogitsWorkspace
    work_map: torch.Tensor | None = None
    max_block: int = 0
    cp_world: int = 1
    cp_rank: int = 0

    @property
    def shard(self) -> dict:
        """The CP shard as aiter's work map and scorer take it."""
        return _shard(self.cp_world, self.cp_rank)


def _shard(cp_world: int, cp_rank: int) -> dict:
    # nothing unsharded, so that launch keeps the exact call it has without CP
    if cp_world == 1:
        return {}
    return {"cp_world": cp_world, "cp_rank": cp_rank}


def allocate_index_scorers(
    config,
    device,
    *,
    max_bs: int,
    max_qlen: int,
    num_idx_heads: int,
    cp: tuple[int, int] = (1, 0),
) -> dict[str, IndexScoreBuffers]:
    """The builder's records by ubatch prefix (``""`` the whole step, ``ub<i>_`` a decode
    micro-batch); empty unless the index cache is MXFP4, so ``"" in`` the result is the
    switch for "this model runs the MXFP4 index scorer".

    The scorer's grid is its work map's row count, so the block bound is the model
    length's and the map buffer is capacity (the exact grid is not monotonic in the
    batch). Every layer shares one `LogitsWorkspace` for the scores (a layer's scorer
    and selector run back to back on one stream), as wide as the map and tall enough
    for a decode step's rows or a prefill chunk's. Allocated in the builder's
    constructor, so the memory profile counts it.

    A decode TBO also gets a map per ubatch (both ubatches' metadata are alive at once)
    but shares the step's scores: both run on one compute stream and a layer's scorer
    and selector sit in one `index_topk` call. A ubatch is one token a request, so
    q == 1 for its map; `cut_decode_ubatches` cuts ``max_bs`` requests at
    ``max_bs // 2``.

    ``cp`` is this rank's indexer-CP ``(world, rank)``: the decode map and bound are
    the shard's, over ``num_idx_heads * world`` heads, and the workspace fits that
    too. A prefill chunk still scores ``num_idx_heads`` heads over every block.
    """
    if config.index_cache_dtype != "fp4":
        return {}
    # once, at construction: this module must stay importable without aiter
    from aiter.ops.flydsl import minimax_m3_index_score_work_map_capacity

    cp_world, cp_rank = cp
    max_block = -(-config.max_model_len // SPARSE_BLOCK_SIZE)
    decode_heads = num_idx_heads * cp_world
    decode_block = -(-max_block // cp_world)

    def work_map(batch: int, query_len: int) -> torch.Tensor:
        # allocated once: a captured graph bakes its pointer
        rows = 0
        if batch > 0:
            rows = minimax_m3_index_score_work_map_capacity(
                batch,
                decode_block,
                query_len,
                decode_heads,
                **_shard(cp_world, cp_rank),
            )
        return torch.empty((rows, 2), dtype=torch.int32, device=device)

    rows = max(max_bs * max_qlen, config.max_num_batched_tokens)
    capacity = max(
        num_idx_heads * rows * max_block,
        decode_heads * max_bs * max_qlen * decode_block,
    )
    scorers = {
        "": IndexScoreBuffers(
            scores=LogitsWorkspace(capacity, device),
            work_map=work_map(max_bs, max_qlen),
            max_block=decode_block,
            cp_world=cp_world,
            cp_rank=cp_rank,
        )
    }
    if config.enable_tbo and config.enable_tbo_decode:
        first = max_bs // 2
        for i, batch in enumerate((first, max_bs - first)):
            scorers[f"ub{i}_"] = replace(scorers[""], work_map=work_map(batch, 1))
    return scorers
