"""Opt-in M3 compute-only index context partitioning, independent of global DCP."""

import functools

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx_runtime

from atom.model_ops.minimax_m3.index_topk import (
    DECODE_TOPK_BLOCK_SIZE_K,
    DECODE_TOPK_NUM_WARPS,
    PAGES_PER_SPARSE_BLOCK,
    _alloc_emit,
    _launch_select,
    _require_packable,
)

# A decode graph is captured for max_model_len, so a grid sized by the launch
# width is mostly programs past every request. The grid is sized by the batch
# instead, and each request's live blocks are split over its programs.
CP_SCORE_PROGRAMS = 1024
CP_SCORE_MIN_SPAN = 2  # blocks a program scores at least


@functools.cache
def _transposed_score() -> bool:
    # On gfx1250, use Q as WMMA's M operand and converted K as N. Same bits.
    # The architecture-specific K load policy is documented at the load.
    return get_gfx_runtime() == "gfx1250"


def _context_score_programs(batch: int, local_blocks: int) -> int:
    return max(
        1,
        min(
            CP_SCORE_PROGRAMS // max(1, batch),
            triton.cdiv(local_blocks, CP_SCORE_MIN_SPAN),
        ),
    )


@triton.jit(do_not_specialize=["LOCAL_BLOCKS", "GLOBAL_BLOCKS"])
def _context_score(
    Q,
    Cache,
    Table,
    Lengths,
    Scores,
    Q_TOKEN_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    LOCAL_BLOCKS,
    GLOBAL_BLOCKS,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    MIN_SPAN: tl.constexpr,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    request = tl.program_id(0)
    tokens = tl.num_programs(0) * QUERY_LEN
    program = tl.program_id(1)
    programs = tl.num_programs(1)
    n = tl.arange(0, N)
    token, head = n // HEADS, n % HEADS
    row = request * QUERY_LEN + token
    live = n < HEADS * QUERY_LEN
    length = tl.load(Lengths + request)
    cutoff = length - QUERY_LEN + token + 1
    s_row = Scores + (head * tokens + row) * LOCAL_BLOCKS
    # This rank's blocks of the request are local < live_end.
    request_blocks = tl.minimum(GLOBAL_BLOCKS, (length + 127) // 128)
    live_end = tl.minimum(
        LOCAL_BLOCKS, tl.maximum(0, (request_blocks - RANK + WORLD - 1) // WORLD)
    )
    span = tl.maximum(MIN_SPAN, (live_end + programs - 1) // programs)
    start = program * span
    end = tl.minimum(start + span, live_end)
    if start < end:
        d = tl.arange(0, 128)
        pos = tl.arange(0, 128)
        q_rows = Q + row * Q_TOKEN_STRIDE + head * Q_HEAD_STRIDE
        if TRANSPOSED:
            q = tl.load(q_rows[:, None] + d[None, :], mask=live[:, None], other=0)
        else:
            q = tl.load(q_rows[None, :] + d[:, None], mask=live[None, :], other=0)
        for local in tl.range(start, end):
            block = local * WORLD + RANK
            page = tl.load(Table + request * TABLE_STRIDE + block).to(tl.int64)
            k_page = Cache + page * 128 * 128
            if TRANSPOSED:
                # On CDNA5 `.cv` is a system-scoped volatile/last-use load, not
                # a streaming (`.cs`) policy. It is safe here because earlier
                # kernels finished producing K and no writer races this launch;
                # `.cs` has not been substituted without a separate benchmark.
                k = tl.load(
                    k_page + pos[None, :] * 128 + d[:, None], cache_modifier=".cv"
                )
                dot = tl.dot(q, k.to(q.dtype), out_dtype=tl.float32) * SCALE
                in_ctx = block * 128 + pos[None, :] < cutoff[:, None]
                score = tl.max(tl.where(in_ctx, dot, float("-inf")), 1)
            else:
                k = tl.load(k_page + pos[:, None] * 128 + d[None, :])
                dot = tl.dot(k.to(q.dtype), q, out_dtype=tl.float32) * SCALE
                in_ctx = block * 128 + pos[:, None] < cutoff[None, :]
                score = tl.max(tl.where(in_ctx, dot, float("-inf")), 0)
            tl.store(s_row + local, score, mask=live)


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
):
    """Return [heads,tokens,ceil(blocks/world)] with round-robin logical blocks.

    Index cache remains replicated. Caller supplies valid live lengths no greater
    than max_seq_len and in-bounds physical pages; no host reads of live metadata.
    Only each row's live shard prefix is initialized; selection masks the tail
    from ``seq_lens`` rather than reading replay leftovers.
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
    if index_cache.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        raise ValueError("unsupported index cache dtype")
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
    scores = torch.empty(
        (heads, tokens, local), dtype=torch.float32, device=idx_q.device
    )
    if tokens:
        programs = _context_score_programs(seq_lens.numel(), local)
        _context_score[(seq_lens.numel(), programs)](
            idx_q,
            index_cache,
            block_table,
            seq_lens,
            scores,
            Q_TOKEN_STRIDE=idx_q.stride(0),
            Q_HEAD_STRIDE=idx_q.stride(1),
            TABLE_STRIDE=block_table.stride(0),
            HEADS=heads,
            QUERY_LEN=max_query_len,
            LOCAL_BLOCKS=local,
            GLOBAL_BLOCKS=blocks,
            RANK=rank,
            WORLD=world_size,
            MIN_SPAN=CP_SCORE_MIN_SPAN,
            N=max(16, triton.next_power_of_2(heads * max_query_len)),
            SCALE=sm_scale * 1.4426950409,
            TRANSPOSED=_transposed_score(),
            num_stages=3,
        )
    return scores


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
