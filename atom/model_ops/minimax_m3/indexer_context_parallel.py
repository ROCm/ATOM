"""Opt-in M3 compute-only index context partitioning, independent of global DCP."""

import torch
import triton

from atom.model_ops.minimax_m3.index_topk import (
    _FP8_DTYPES,
    DECODE_TOPK_BLOCK_SIZE_K,
    DECODE_TOPK_NUM_WARPS,
    INDEX_CACHE_MUST_BE_FP8,
    PAGES_PER_SPARSE_BLOCK,
    _alloc_emit,
    _launch_select,
    _require_packable,
    decode_index_score,
)


def indexer_context_scores(
    idx_q,
    index_cache,
    block_table,
    seq_lens,
    max_seq_len,
    rank,
    world_size,
    max_query_len,
    sm_scale,
    work_map=None,
    max_block=0,
):
    """Return [heads,tokens,ceil(blocks/world)] with round-robin logical blocks.

    Index cache remains replicated. Caller supplies valid live lengths no greater
    than max_seq_len and in-bounds physical pages; no host reads of live metadata.

    ``work_map`` is the scorer's packed dispatch order, built once per decode
    step against this rank's LOCAL block bound ``max_block`` (omitted: see
    :func:`decode_index_score`). Under a cudagraph that bound is the model
    length, not ``max_seq_len``, so the shard may carry dead trailing blocks.

    Dead slots are left UNWRITTEN; `local_candidate_keys` masks them from
    (rank, world, global block count), so the result cannot be read raw.
    """
    if (
        world_size < 1
        or not 0 <= rank < world_size
        or max_seq_len < 1
        or max_query_len < 1
    ):
        raise ValueError("invalid context partition or query geometry")
    if (
        idx_q.ndim != 3
        or idx_q.shape[2] != 128
        or idx_q.dtype != torch.bfloat16
        or idx_q.stride(2) != 1
    ):
        raise ValueError(
            "index queries must be BF16 [tokens,heads,128] with contiguous D"
        )
    tokens, heads, _ = idx_q.shape
    if heads < 1 or seq_lens.ndim != 1 or tokens != seq_lens.numel() * max_query_len:
        raise ValueError("query rows must equal batch * max_query_len")
    if (
        index_cache.ndim != 3
        or index_cache.shape[1:] != (128, 128)
        or not index_cache.is_contiguous()
    ):
        raise ValueError("index cache must be contiguous [pages,128,128]")
    if index_cache.dtype not in _FP8_DTYPES:
        raise ValueError(f"{INDEX_CACHE_MUST_BE_FP8} Got {index_cache.dtype}.")
    blocks = triton.cdiv(max_seq_len, 128)
    if (
        block_table.ndim != 2
        or block_table.shape[0] != seq_lens.numel()
        or block_table.shape[1] < blocks
        or block_table.stride(1) != 1
    ):
        raise ValueError("block table does not cover the requested context")
    if (
        block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("block table and contiguous lengths must be int32")
    if any(
        not x.is_cuda or x.device != idx_q.device
        for x in (idx_q, index_cache, block_table, seq_lens)
    ):
        raise ValueError("all score inputs must be on the same GPU")
    local = triton.cdiv(blocks, world_size)
    if not tokens:
        return torch.empty((heads, 0, local), dtype=torch.float32, device=idx_q.device)
    # A LOCAL bound, never the global `blocks`; must match the map's bound.
    width = max_block or local
    # Contiguous on purpose: the scorer's own allocator can return a permuted
    # view, which `local_candidate_keys` rejects. Uninitialised is fine: the
    # trailing slot of an uneven shard is masked there (pinned by
    # `test_a_shard_that_runs_past_the_global_end_is_not_selectable`).
    out = torch.empty((heads, tokens, width), dtype=torch.float32, device=idx_q.device)
    return decode_index_score(
        idx_q, index_cache, block_table, seq_lens, width,
        max_query_len, heads, sm_scale, work_map,
        cp_world=world_size, cp_rank=rank, out=out,
    )  # fmt: skip


def select_global_blocks(
    scores, block_table, seq_lens, topk, init_blocks, local_blocks, max_query_len
):
    """Reuse native packed ordering/emission after restoring logical block order."""
    if scores.ndim != 3 or scores.dtype != torch.float32:
        raise ValueError("scores must be FP32 [local_heads,tokens,global_blocks]")
    heads, tokens, blocks = scores.shape
    if (
        topk < 1
        or topk > DECODE_TOPK_BLOCK_SIZE_K
        or min(init_blocks, local_blocks) < 0
    ):
        raise ValueError("unsupported top-k/forced-block counts")
    if max_query_len < 1 or tokens != seq_lens.numel() * max_query_len:
        raise ValueError("score rows must equal batch * max_query_len")
    if block_table.ndim != 2 or block_table.shape != (seq_lens.numel(), blocks):
        raise ValueError("block table and scores must have matching geometry")
    if (
        seq_lens.dtype != torch.int32
        or block_table.dtype != torch.int32
        or not seq_lens.is_contiguous()
        or block_table.stride(1) != 1
    ):
        raise ValueError(
            "selection metadata must be int32 with contiguous inner dimensions"
        )
    if any(
        not x.is_cuda or x.device != scores.device
        for x in (scores, block_table, seq_lens)
    ):
        raise ValueError("selection inputs must share a GPU")
    _require_packable(blocks)
    indices = torch.empty(
        (heads, tokens, topk), dtype=torch.int32, device=scores.device
    )
    output, args = _alloc_emit(tokens, heads, topk, block_table, True, scores.device)
    if tokens:
        _launch_select(
            scores,
            indices,
            seq_lens,
            seq_lens,
            seq_lens.numel(),
            topk,
            init_blocks,
            local_blocks,
            block_table,
            args,
            heads,
            max_query_len,
            max_query_len,
            DECODE_TOPK_BLOCK_SIZE_K,
            DECODE_TOPK_NUM_WARPS,
            True,
            # This path's cache has its blocks back to back, so the stride
            # equals the pages a block fills.
            block_page_stride=PAGES_PER_SPARSE_BLOCK,
        )
    return indices, *output
