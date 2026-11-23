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
from atom.utils import envs
from atom.model_ops.minimax_m3.nvfp4_kernels.nvfp4 import (
    NVFP4_SCALED_DOT,
    nvfp4_block_max,
    nvfp4_load_packed,
    nvfp4_scaled_dot,
    nvfp4_split_query,
)

# A decode graph is captured for max_model_len, so a grid sized by the launch
# width is mostly programs past every request. The grid is sized by the batch
# instead, and each request's live blocks are split over its programs.
CP_SCORE_PROGRAMS = 1024
CP_SCORE_MIN_SPAN = 2  # blocks a program scores at least
CP_SCORE_PAD_TILE = 64  # -inf slots a store writes past the live blocks
CP_SCORE_FP4_NUM_STAGES = 1
# FP4 programs (all requests together), warps and query rows a program. A
# program walks its blocks one at a time and the scaled read's block max
# dominates each step, so FP4 wants many short programs and one warp. MI455X b64
# sweeps (us; FP8 CP at 1024 x 4 warps: q1 131K 19.7, q1 300K 46.4, q8 131K
# 24.9, q8 300K 51.9): q1 16384 x 1 warp 12.5 / 31.5; q8 with all 32 rows (4
# heads x 8) in one program, 16384 x 1 20.3 / 43.3 -- splitting them over two
# 16-row programs (grid axis 2) was slower, 22.0 / 52.9 at its best: the second
# read of each block costs more than the smaller tile saves. Rows beyond
# CP_SCORE_FP4_ROWS still split (MTP16 and up). One warp builds to 136 VGPRs at
# 16 rows and 290 at 32, none spilled.
CP_SCORE_FP4_PROGRAMS = 16384
CP_SCORE_FP4_NUM_WARPS = 1
CP_SCORE_FP4_ROWS = 32


@functools.cache
def _transposed_score() -> bool:
    # As the gfx1250 TP decode scorer (ROCm/ATOM#2982): Q as WMMA's M operand,
    # the converted K as its N operand, K loads streaming. Same bits.
    return get_gfx_runtime() == "gfx1250"


def _context_score_programs(
    batch: int, local_blocks: int, total: int = CP_SCORE_PROGRAMS
) -> int:
    return max(
        1,
        min(
            total // max(1, batch),
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
    PAD_TILE: tl.constexpr,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    # FP4 index (`index_fp4`): Cache is the payload [pages, 128, 64] uint8 and
    # Scales its E4M3 scales [pages, 128, 8], pages PAGE_STRIDE / SCALE_PAGE_STRIDE
    # bytes apart, read by the scaled WMMA against the query in two E4M3 terms
    # (`nvfp4_scaled_dot`) -- the gfx1250 TP decode scorer's read, so the
    # selection is the TP path's. Left
    # None otherwise: constexpr, so the FP8 / BF16 build is the one before them.
    Scales=None,
    PAGE_STRIDE=None,
    SCALE_PAGE_STRIDE=None,
    IS_FP4: tl.constexpr = False,
    # Query-row split (FP4): grid axis 2 has ROW_SPLIT programs, each scoring
    # rows [N * split, N * split + N) of the request's HEADS * QUERY_LEN.
    ROW_SPLIT: tl.constexpr = 1,
    # Whether the slots past a request's live blocks are written (-inf). Off for
    # a reader that stops at them itself (`_shard_winners`).
    FILL_DEAD_SLOTS: tl.constexpr = True,
):
    tl.static_assert(not IS_FP4 or (NVFP4_SCALED_DOT and TRANSPOSED))
    request = tl.program_id(0)
    tokens = tl.num_programs(0) * QUERY_LEN
    program = tl.program_id(1)
    programs = tl.num_programs(1)
    if ROW_SPLIT > 1:
        n = tl.arange(0, N) + tl.program_id(2) * N
    else:
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
        if IS_FP4:
            q_hi, q_lo, q_lo_scale = nvfp4_split_query(q)
        for local in tl.range(start, end):
            block = local * WORLD + RANK
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
                k_page = Cache + page * 128 * 128
                if TRANSPOSED:
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
    if FILL_DEAD_SLOTS:
        # Every output slot is overwritten, including empty shards and replay
        # padding.
        pad_span = (LOCAL_BLOCKS - live_end + programs - 1) // programs
        lo = live_end + program * pad_span
        hi = tl.minimum(lo + pad_span, LOCAL_BLOCKS)
        for t in range(lo, hi, PAD_TILE):
            off = t + tl.arange(0, PAD_TILE)
            tl.store(
                s_row[:, None] + off[None, :],
                tl.full((N, PAD_TILE), float("-inf"), tl.float32),
                mask=live[:, None] & (off[None, :] < hi),
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
    fill_dead_slots=True,
):
    """Return [heads,tokens,ceil(blocks/world)] with round-robin logical blocks.

    ``fill_dead_slots``: write -inf to the slots past each request's live
    blocks. A decode graph is captured for max_model_len, so those are most of
    the row (at a 1M bound and 167K-token requests, 84%), and filling them costs
    the 32-row FP4 scorer 40% on top of its scoring (MI455X, batch 24, verify of
    8: 17.7us against 12.7us at the batch's own length). False for a reader that
    bounds its reads by the same live count, as `_shard_winners` does: the dead
    slots are then left unwritten.

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
            or not _transposed_score()
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
                "[pages,128,8], contiguous rows, on an arch reading whole rows"
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
    if tokens and is_fp4 and not fill_dead_slots and envs.ATOM_M3_CP_SCORE_FLYDSL:
        # gfx1250 (`_transposed_score`, checked above): the FlyDSL scorer, the
        # same bits with the K stream overlapped (indexer_cp_score_flydsl).
        from atom.model_ops.minimax_m3 import indexer_cp_score_flydsl as fly

        if fly.supported(idx_q, index_cache, index_scale, max_query_len):
            return fly.cp_scores_fp4(
                idx_q,
                index_cache,
                index_scale,
                block_table,
                seq_lens,
                max_seq_len,
                rank,
                world_size,
                max_query_len,
                sm_scale,
                scores,
            )
    if tokens:
        programs = _context_score_programs(
            seq_lens.numel(),
            local,
            CP_SCORE_FP4_PROGRAMS if is_fp4 else CP_SCORE_PROGRAMS,
        )
        rows = heads * max_query_len
        rows_pow2 = max(16, triton.next_power_of_2(rows))
        # FP4: at most CP_SCORE_FP4_ROWS query rows a program, the rest on grid
        # axis 2 (FP8 / BF16 score every row in one program).
        row_tile = min(rows_pow2, CP_SCORE_FP4_ROWS) if is_fp4 else rows_pow2
        row_split = triton.cdiv(rows, row_tile)
        grid = (seq_lens.numel(), programs) + ((row_split,) if row_split > 1 else ())
        _context_score[grid](
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
            PAD_TILE=CP_SCORE_PAD_TILE,
            N=row_tile,
            SCALE=sm_scale * 1.4426950409,
            TRANSPOSED=_transposed_score(),
            FILL_DEAD_SLOTS=fill_dead_slots,
            **(
                {
                    "Scales": index_scale,
                    "PAGE_STRIDE": index_cache.stride(0),
                    "SCALE_PAGE_STRIDE": index_scale.stride(0),
                    "IS_FP4": True,
                    # One stage, one warp (see CP_SCORE_FP4_PROGRAMS).
                    "num_stages": CP_SCORE_FP4_NUM_STAGES,
                    "num_warps": CP_SCORE_FP4_NUM_WARPS,
                    "ROW_SPLIT": row_split,
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
