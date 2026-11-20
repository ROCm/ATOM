"""Opt-in M3 compute-only index context partitioning, independent of global DCP."""

import torch
import triton
import triton.language as tl

from atom.model_ops.minimax_m3.index_topk import (
    DECODE_TOPK_BLOCK_SIZE_K,
    DECODE_TOPK_NUM_WARPS,
    PAGES_PER_SPARSE_BLOCK,
    _alloc_emit,
    _decode_score_chunks,
    _launch_select,
    _require_packable,
)
from atom.model_ops.minimax_m3.nvfp4_kernels.nvfp4 import (
    NVFP4_SCALED_DOT,
    nvfp4_block_max,
    nvfp4_load_packed,
    nvfp4_scaled_dot,
    nvfp4_split_query,
)

# The FP4 index read holds its tiles in registers; deeper pipelining only holds
# more of them.
CP_SCORE_FP4_NUM_STAGES = 1


@triton.jit(do_not_specialize=["LOCAL_BLOCKS", "GLOBAL_BLOCKS", "CHUNK"])
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
    CHUNK,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    # FP4 index (`index_fp4`): Cache is the payload [pages, 128, 64] uint8 and
    # Scales its E4M3 scales [pages, 128, 8], pages PAGE_STRIDE / SCALE_PAGE_STRIDE
    # bytes apart, read by the scaled WMMA against the query in two E4M3 terms
    # (`nvfp4_scaled_dot`) -- the gfx1250 TP decode scorer's read, so the
    # selection is the TP path's. Left None otherwise: constexpr, so the FP8 /
    # BF16 build is the one before them.
    Scales=None,
    PAGE_STRIDE=None,
    SCALE_PAGE_STRIDE=None,
    IS_FP4: tl.constexpr = False,
):
    tl.static_assert(not IS_FP4 or NVFP4_SCALED_DOT)
    request = tl.program_id(0)
    tokens = tl.num_programs(0) * QUERY_LEN
    chunk = tl.program_id(1)
    n = tl.arange(0, N)
    token, head = n // HEADS, n % HEADS
    row = request * QUERY_LEN + token
    d = tl.arange(0, 128)
    q = tl.load(
        Q + row[None, :] * Q_TOKEN_STRIDE + head[None, :] * Q_HEAD_STRIDE + d[:, None],
        mask=n[None, :] < HEADS * QUERY_LEN,
        other=0,
    )
    if IS_FP4:
        q_hi, q_lo, q_lo_scale = nvfp4_split_query(tl.trans(q))
    length = tl.load(Lengths + request)
    cutoff = length - QUERY_LEN + token + 1
    pos = tl.arange(0, 128)
    for local in range(chunk * CHUNK, tl.minimum((chunk + 1) * CHUNK, LOCAL_BLOCKS)):
        block = local * WORLD + RANK
        valid = (block < GLOBAL_BLOCKS) & (block * 128 < length)
        # Every output slot is overwritten, including empty shards and replay padding.
        score = tl.full((N,), float("-inf"), tl.float32)
        if valid:
            page = tl.load(Table + request * TABLE_STRIDE + block).to(tl.int64)
            if IS_FP4:
                payload, kscale = nvfp4_load_packed(
                    Cache,
                    Scales,
                    page * PAGE_STRIDE,
                    page * SCALE_PAGE_STRIDE,
                    pos,
                    128,
                )
                score = nvfp4_block_max(
                    nvfp4_scaled_dot(q_hi, q_lo, q_lo_scale, payload, kscale),
                    block * 128,
                    pos,
                    cutoff,
                    # The first query row's cutoff, the smallest.
                    length - QUERY_LEN + 1,
                    SCALE,
                )
            else:
                k = tl.load(Cache + page * 128 * 128 + pos[:, None] * 128 + d[None, :])
                dot = tl.dot(k.to(q.dtype), q, out_dtype=tl.float32) * SCALE
                dot = tl.where(
                    block * 128 + pos[:, None] < cutoff[None, :], dot, float("-inf")
                )
                score = tl.max(dot, 0)
        tl.store(
            Scores + (head * tokens + row) * LOCAL_BLOCKS + local,
            score,
            mask=n < HEADS * QUERY_LEN,
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
    index_scale=None,
):
    """Return [heads,tokens,ceil(blocks/world)] with round-robin logical blocks.

    Index cache remains replicated. Caller supplies valid live lengths no greater
    than max_seq_len and in-bounds physical pages; no host reads of live metadata.

    ``index_scale``: the FP4 index cache (``--index-cache-dtype fp4``), whose
    payload ``index_cache`` is then uint8 ``[pages, 128, 64]`` (see index_fp4).
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
    is_fp4 = index_scale is not None
    if is_fp4:
        # The two planes' rows are contiguous; their pages may sit anywhere
        # (pool fields), 16 B apart at least.
        if (
            not NVFP4_SCALED_DOT
            or index_cache.dtype != torch.uint8
            or index_scale.dtype != torch.uint8
            or index_cache.ndim != 3
            or index_cache.shape[1:] != (128, 64)
            or index_cache.stride()[1:] != (64, 1)
            or index_scale.shape != (index_cache.shape[0], 128, 8)
            or index_scale.stride()[1:] != (8, 1)
            or index_cache.stride(0) % 16
            or index_scale.stride(0) % 16
        ):
            raise ValueError(
                "FP4 index cache must be uint8 [pages,128,64] with uint8 scales "
                "[pages,128,8], contiguous rows, on an arch with the scaled read"
            )
    elif (
        index_cache.ndim != 3
        or index_cache.shape[1:] != (128, 128)
        or not index_cache.is_contiguous()
    ):
        raise ValueError("index cache must be contiguous [pages,128,128]")
    elif index_cache.dtype not in (
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
        + ((index_scale,) if is_fp4 else ())
    ):
        raise ValueError("all score inputs must be on the same GPU")
    local = triton.cdiv(blocks, world_size)
    scores = torch.empty(
        (heads, tokens, local), dtype=torch.float32, device=idx_q.device
    )
    if tokens:
        chunks = min(local, _decode_score_chunks(seq_lens.numel(), local))
        _context_score[(seq_lens.numel(), chunks)](
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
            CHUNK=triton.cdiv(local, chunks),
            N=max(16, triton.next_power_of_2(heads * max_query_len)),
            SCALE=sm_scale * 1.4426950409,
            **(
                {
                    "Scales": index_scale,
                    "PAGE_STRIDE": index_cache.stride(0),
                    "SCALE_PAGE_STRIDE": index_scale.stride(0),
                    "IS_FP4": True,
                    "num_stages": CP_SCORE_FP4_NUM_STAGES,
                }
                if is_fp4
                else {"num_stages": 3}
            ),
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
