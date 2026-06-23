# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 lightning-indexer block scoring + top-k.

Index queries score each 128-token block of index keys (max over the block),
then the top-k blocks (plus forced init/local blocks) are selected per query
token. Ported from the sglang reference (minimax_sparse_ops), adapted to vLLM's
paged KV cache: the KV page size is forced to equal the sparse block size (128),
so one sparse block maps to exactly one page.

Index-K cache layout (vLLM): ``(num_blocks, 128, idx_head_dim)`` (single head).

Only the paths MiniMax M3 uses are implemented: score_type="max", index value
disabled (score-only indexer), single shared index head. The selected block ids
feed the block-sparse attention kernels in ``sparse_attn``.
"""

import torch

try:
    from vllm.triton_utils import tl, triton
except ModuleNotFoundError:
    import triton
    import triton.language as tl

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128
# Physical 16-pages per logical 128-block for the page-16 SHUFFLE ASM/gluon cache
# (must match sparse_attn.PAGES_PER_SPARSE_BLOCK). Used by the fused block-table
# emission in the topk kernels.
PAGES_PER_SPARSE_BLOCK = 8


# ---------------------------------------------------------------------------
# Bitonic top-k helpers (layout-agnostic; ported verbatim from sglang).
# ---------------------------------------------------------------------------
@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


# ---------------------------------------------------------------------------
# Index block-score kernel (paged). score[h, token, block] = max over the
# 128-token block of (idx_q . index_k), causal-masked. BLOCK_SIZE_K == 128 so
# each K-tile is exactly one page (BLOCKS_PER_K_BLOCK == 1).
# ---------------------------------------------------------------------------
@triton.jit
def _index_block_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads,
    head_dim: tl.constexpr,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    # Block table row for this request.
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Causal window: only blocks up to the last query token's position.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        # index-K for this page: [BLOCK_SIZE_D, BLOCK_SIZE_K] (transposed)
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk = tl.dot(q, k) * sm_scale_log2e
        # apply causal mask as needed
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        # one sparse block per K-tile -> max over the 128 positions
        score = tl.max(qk, axis=1)  # [BLOCK_SIZE_Q]
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)


# ---------------------------------------------------------------------------
# Top-k selection over per-token block scores (layout-agnostic). block_size_q
# is 1 for M3, so top-k is computed per query token.
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["BLOCK_SIZE_T"],
)
@triton.jit
def _topk_index_kernel(
    s_ptr,  # [num_heads, total_q, max_block]
    ti_ptr,  # [num_heads, total_q, topk]
    sample_interval: tl.constexpr,  # block_size_q (1 for M3)
    block_size: tl.constexpr,  # sparse block size (128)
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    # --- fused sparse block-table emission (ASM/gluon prefill path) ---
    block_table_ptr,  # [batch, max_blocks] int32 logical 128-granularity (or dummy)
    sparse_bt_ptr,  # out: [total_q, topk*pages_per_block] int32 (or dummy)
    sparse_ctx_ptr,  # out: [total_q] int32 (or dummy)
    stride_bt_b,
    stride_sbt_n,
    NUM_KV_HEADS: tl.constexpr,  # kv-head count folded into the emitted row + page id
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    pages_per_block: tl.constexpr,  # 16-pages per sparse block (8)
    EMIT_SPARSE_BT: tl.constexpr,  # fuse compaction (per-kv-head row + encoded page)
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    store_mask = off_t < topk
    valid_mask = off_t < valid_blocks
    topk_idx = tl.where(store_mask & valid_mask, topk_idx, -1)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=store_mask)

    # --- fused sparse block-table build (per-query-token causal compaction) ---
    # Mirrors _build_sparse_block_table_prefill_kernel over the in-register
    # selection. EVERY kv-head emits its own row (the ASM/gluon path collapses
    # (token, kv_head) into the row dim). Token absolute pos p = prefix_len + pid_q
    # (sample_interval == 1); causal self-block = p // block_size, length p + 1.
    # Page id is kv-head-encoded: (phys16_page)*NUM_KV_HEADS + pid_h; row is
    # (block_start + pid_q)*NUM_KV_HEADS + pid_h. NUM_KV_HEADS == 1 -> original.
    if EMIT_SPARSE_BT:
        p = prefix_len + pid_q * sample_interval
        self_blk = p // block_size
        causal_len = p + 1
        bt_blk = tl.where(off_t < topk, topk_idx, -1)
        # causal: drop any selected block above the self-block (defensive; the
        # indexer already caps selection at valid_blocks == self_blk + 1).
        bt_valid = (bt_blk >= 0) & (bt_blk <= self_blk)
        bt_is_tail = bt_valid & (bt_blk == self_blk)
        bt_is_full = bt_valid & (bt_blk < self_blk)
        bt_n_full = tl.sum(bt_is_full.to(tl.int32), axis=0)
        bt_n_valid = tl.sum(bt_valid.to(tl.int32), axis=0)
        bt_earlier_full = tl.cumsum(bt_is_full.to(tl.int32), axis=0) - bt_is_full.to(
            tl.int32
        )
        bt_slot = tl.where(bt_is_full, bt_earlier_full, bt_n_full)  # tail -> n_full

        bt_row = block_table_ptr + pid_b * stride_bt_b
        bt_logical_page = tl.load(bt_row + bt_blk, mask=bt_valid, other=0).to(tl.int32)
        bt_base_phys = bt_logical_page * pages_per_block * NUM_KV_HEADS + pid_h
        bt_dst_base = bt_slot * pages_per_block

        sbt_row = (
            sparse_bt_ptr
            + ((block_start + pid_q) * NUM_KV_HEADS + pid_h) * stride_sbt_n
        )
        for pj in range(pages_per_block):
            tl.store(
                sbt_row + bt_dst_base + pj,
                bt_base_phys + pj * NUM_KV_HEADS,
                mask=bt_valid,
            )
        bt_n_used = bt_n_valid * pages_per_block
        off_w = tl.arange(0, BLOCK_SIZE_T * pages_per_block)
        tl.store(sbt_row + off_w, tl.zeros_like(off_w), mask=off_w >= bt_n_used)

        bt_tail_tokens = causal_len - self_blk * block_size
        bt_has_tail = tl.sum(bt_is_tail.to(tl.int32), axis=0) > 0
        bt_ctx = bt_n_full * block_size + tl.where(bt_has_tail, bt_tail_tokens, 0)
        bt_ctx = tl.where(
            bt_has_tail, bt_ctx, tl.minimum(bt_n_valid * block_size, causal_len)
        )
        tl.store(
            sparse_ctx_ptr + ((block_start + pid_q) * NUM_KV_HEADS + pid_h), bt_ctx
        )


# ---------------------------------------------------------------------------
# Decode index-score kernel (split-K over seq blocks). Decode == one query
# token per request, so this parallelizes over the KV dimension instead of the
# query dimension. Chunk counts depend only on shape constants so the grid is
# fixed within a cuda graph. Base-2 (exp2/log2) softmax matches prefill.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [batch]
    num_idx_heads,
    total_q,  # batch * max_q (one row per query token)
    head_dim,
    init_blocks,
    local_blocks,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    MAX_Q: tl.constexpr,  # query tokens per request (num_spec + 1; 1 == plain decode)
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_KV_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_tc, pid_h = tl.program_id(0), tl.program_id(1)
    pid_t = pid_tc % total_q  # global query-token row
    pid_c = pid_tc // total_q
    pid_b = pid_t // MAX_Q  # request index
    tok = pid_t % MAX_Q  # token position within the request (0..MAX_Q-1)
    seq_len = tl.load(seq_lens + pid_b)
    # Per-token causal length: token `tok` sits at absolute position
    # (seq_len - MAX_Q + tok), so it attends (seq_len - MAX_Q + tok + 1) keys.
    # MAX_Q == 1 -> causal_len == seq_len (plain decode, unchanged).
    causal_len = seq_len - MAX_Q + tok + 1
    num_blocks = (causal_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    # block-aligned fixed-count split: grid independent of seq_len (cuda graph).
    chunk_size_blocks = (num_blocks + NUM_KV_CHUNKS - 1) // NUM_KV_CHUNKS
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if (causal_len <= 0) | (chunk_start_block >= chunk_end_block):
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)  # positions within a 128-block
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Force-select init (1e30) and local (1e29, higher priority) blocks.
    local_start = tl.maximum(0, num_blocks - local_blocks)
    # single query vector for this (token, index head)
    q = tl.load(
        q_ptr + pid_t * stride_q_n + pid_h * stride_q_h + off_d * stride_q_d,
        mask=d_mask,
        other=0.0,
    ).to(
        tl.float32
    )  # [D]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos < causal_len
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        ).to(
            tl.float32
        )  # [D, N]
        qk = tl.sum(q[:, None] * k, axis=0) * sm_scale_log2e  # [N]
        qk = tl.where(pos_mask, qk, float("-inf"))
        score = tl.max(qk, axis=0)  # one score for this 128-block
        is_init = blk < init_blocks
        is_local = (blk >= local_start) & (blk < num_blocks)
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr + pid_h * stride_s_h + pid_t * stride_s_n + blk * stride_s_k,
            score,
        )


# ---------------------------------------------------------------------------
# Decode top-k (split-K): per-chunk partial top-k + merge. Forced init/local
# blocks are already encoded in the scores. Ported from the sglang reference.
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["topk"],
)
@triton.jit
def _topk_index_partial_kernel(
    s_ptr,  # score: [num_idx_heads, total_q, max_block]
    ts_partial_ptr,  # partial scores out: [NUM_TOPK_CHUNKS, num_idx_heads, total_q, T]
    ti_partial_ptr,  # partial idx out (1-indexed global, 0=invalid): same shape
    seq_lens,  # [batch]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    chunk_blocks: tl.constexpr,  # how many score-blocks each chunk owns
    MAX_Q: tl.constexpr,  # query tokens per request (num_spec + 1; 1 == plain decode)
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    tl.static_assert(topk < BLOCK_SIZE_K)
    pid_t = tl.program_id(0)  # global query-token row
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    pid_b = pid_t // MAX_Q  # request index
    tok = pid_t % MAX_Q  # token position within the request (0..MAX_Q-1)
    seq_len = tl.load(seq_lens + pid_b)
    # Per-token causal length (MAX_Q == 1 -> causal_len == seq_len, unchanged).
    causal_len = seq_len - MAX_Q + tok + 1
    num_blocks = (causal_len + block_size - 1) // block_size

    # Slice this chunk owns within [0, num_blocks).
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)
    chunk_actual = tl.maximum(chunk_end - chunk_start, 0)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)

    s_ptrs = (
        s_ptr
        + pid_t * stride_s_b
        + pid_h * stride_s_h
        + (chunk_start + off_k) * stride_s_k
    )

    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2

    # Streaming top-K within this chunk. tl.range(0, 0) is a no-op so empty
    # chunks (chunk_actual == 0) skip the body and store sentinel -1e30 / 0.
    for i in tl.range(0, chunk_actual, BLOCK_SIZE_K):
        mask = off_k < chunk_actual - i
        score = tl.load(s_ptrs, mask=mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (
            tl.where(mask, chunk_start + i + off_k + 1, 0),  # 1-indexed global
            topk_idx,
        )
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )

    # Extract first BLOCK_SIZE_T entries (top-K of this chunk after the sort).
    topk_mask_extract = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    final_score = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_score, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    final_idx = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_idx, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    # Always write all BLOCK_SIZE_T slots — invalid slots carry -1e30 / 0
    # sentinels and lose to real scores in the merge stage.
    ts_ptrs = (
        ts_partial_ptr
        + pid_chunk * stride_ts_c
        + pid_t * stride_ts_b
        + pid_h * stride_ts_h
        + off_t * stride_ts_t
    )
    ti_ptrs = (
        ti_partial_ptr
        + pid_chunk * stride_ti_c
        + pid_t * stride_ti_b
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    tl.store(ts_ptrs, final_score)
    tl.store(ti_ptrs, final_idx)


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "BLOCK_SIZE_K": lambda args: triton.next_power_of_2(
            args["NUM_TOPK_CHUNKS"] * triton.next_power_of_2(args["topk"])
        ),
    }
)
@triton.jit
def _topk_index_merge_kernel(
    ts_partial_ptr,  # partial scores: [NUM_TOPK_CHUNKS, num_idx_heads, total_q, T]
    ti_partial_ptr,  # partial idx (1-indexed global, 0=invalid): same shape
    ti_final_ptr,  # final idx (0-indexed, -1=invalid): [num_idx_heads, total_q, topk]
    seq_lens,  # [batch]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    # --- fused sparse block-table emission (ASM/gluon decode path) ---
    block_table_ptr,  # [batch, max_blocks] int32 logical 128-granularity (or dummy)
    sparse_bt_ptr,  # out: [total_q, topk*pages_per_block] int32 (or dummy)
    sparse_ctx_ptr,  # out: [total_q] int32 (or dummy)
    stride_bt_b,
    stride_sbt_b,
    MAX_Q: tl.constexpr,  # query tokens per request (num_spec + 1; 1 == plain decode)
    NUM_KV_HEADS: tl.constexpr,  # kv-head count folded into the emitted row + page id
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    pages_per_block: tl.constexpr,  # 16-pages per sparse block (8)
    EMIT_SPARSE_BT: tl.constexpr,  # fuse compaction (per-kv-head row + encoded page)
):
    pid_t = tl.program_id(0)  # global query-token row
    pid_h = tl.program_id(1)

    pid_b = pid_t // MAX_Q  # request index
    tok = pid_t % MAX_Q  # token position within the request (0..MAX_Q-1)
    seq_len = tl.load(seq_lens + pid_b)
    # Per-token causal length (MAX_Q == 1 -> causal_len == seq_len, unchanged).
    causal_len = seq_len - MAX_Q + tok + 1
    num_blocks = (causal_len + block_size - 1) // block_size

    # Load NUM_TOPK_CHUNKS * BLOCK_SIZE_T candidates, padded to BLOCK_SIZE_K.
    # Candidate at flat position p comes from chunk = p // BLOCK_SIZE_T,
    # in_chunk = p % BLOCK_SIZE_T.
    off = tl.arange(0, BLOCK_SIZE_K)
    chunk_idx = off // BLOCK_SIZE_T
    in_chunk_idx = off % BLOCK_SIZE_T
    valid = chunk_idx < NUM_TOPK_CHUNKS

    score_offset = (
        chunk_idx * stride_ts_c
        + pid_h * stride_ts_h
        + pid_t * stride_ts_b
        + in_chunk_idx * stride_ts_t
    )
    idx_offset = (
        chunk_idx * stride_ti_c
        + pid_h * stride_ti_h
        + pid_t * stride_ti_b
        + in_chunk_idx * stride_ti_t
    )

    score = tl.load(ts_partial_ptr + score_offset, mask=valid, other=-1e30).to(
        tl.float32
    )
    score = tl.where(score != score, -1e30, score)
    idx = tl.load(ti_partial_ptr + idx_offset, mask=valid, other=0).to(tl.int32)

    # Full bitonic descending sort of BLOCK_SIZE_K items.
    n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
    for j in tl.static_range(1, n_dims):
        score, idx = _bitonic_merge(score, idx.to(tl.int32), j, 2, n_dims)
    score, idx = _bitonic_merge(score, idx.to(tl.int32), n_dims, True, n_dims)

    # Extract first BLOCK_SIZE_T positions — these are the global top-K.
    extract_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx_final = tl.sum(
        extract_mask[:, None]
        * tl.reshape(idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    off_t = tl.arange(0, BLOCK_SIZE_T)
    tif_ptrs = (
        ti_final_ptr
        + pid_h * stride_tif_h
        + pid_t * stride_tif_b
        + off_t * stride_tif_t
    )
    store_mask = off_t < topk
    topk_idx_final = tl.where(off_t < tl.minimum(topk, num_blocks), topk_idx_final, -1)
    tl.store(
        tif_ptrs, topk_idx_final.to(ti_final_ptr.dtype.element_ty), mask=store_mask
    )

    # --- fused sparse block-table build (per-(token, kv-head) compaction) ---
    # Mirrors _build_sparse_block_table_kernel over the in-register selection,
    # avoiding a second kernel launch + topk_idx HBM round-trip. EVERY kv-head
    # emits its own row: the ASM/gluon path collapses (token, kv_head) into the
    # row dim so it can run with num_kv_heads_view == 1. The physical page id is
    # encoded as (phys16_page)*NUM_KV_HEADS + kv_head, matching the collapsed KV
    # cache view [num_phys16*NUM_KV_HEADS, 1, ...]. NUM_KV_HEADS == 1 reduces to
    # the original per-token emit (row == pid_t, page == phys16).
    if EMIT_SPARSE_BT:
        # Per-token tail block: the 128-block containing this token's last causal
        # key (causal_len - 1). MAX_Q == 1 -> self_blk == (seq_len-1)//block_size.
        self_blk = (causal_len - 1) // block_size
        bt_blk = tl.where(off_t < topk, topk_idx_final, -1)
        bt_valid = bt_blk >= 0
        bt_is_tail = bt_valid & (bt_blk == self_blk)
        bt_is_full = bt_valid & (bt_blk != self_blk)
        bt_n_full = tl.sum(bt_is_full.to(tl.int32), axis=0)
        bt_n_valid = tl.sum(bt_valid.to(tl.int32), axis=0)
        bt_earlier_full = tl.cumsum(bt_is_full.to(tl.int32), axis=0) - bt_is_full.to(
            tl.int32
        )
        bt_slot = tl.where(bt_is_full, bt_earlier_full, bt_n_full)  # tail -> n_full

        bt_row = block_table_ptr + pid_b * stride_bt_b
        bt_logical_page = tl.load(bt_row + bt_blk, mask=bt_valid, other=0).to(tl.int32)
        # Encode kv-head into the page id. The 8 phys16 pages of one 128-block are
        # NUM_KV_HEADS apart in the collapsed cache view (block-major then kv-head),
        # so consecutive pj differ by NUM_KV_HEADS, not 1.
        bt_base_phys = bt_logical_page * pages_per_block * NUM_KV_HEADS + pid_h
        bt_dst_base = bt_slot * pages_per_block

        # Fold kv-head into the row: row = pid_t * NUM_KV_HEADS + pid_h.
        sbt_row = sparse_bt_ptr + (pid_t * NUM_KV_HEADS + pid_h) * stride_sbt_b
        # write valid slots -> their pages; unused tail -> 0 (in-bounds page id).
        for pj in range(pages_per_block):
            tl.store(
                sbt_row + bt_dst_base + pj,
                bt_base_phys + pj * NUM_KV_HEADS,
                mask=bt_valid,
            )
        bt_n_used = bt_n_valid * pages_per_block
        off_w = tl.arange(0, BLOCK_SIZE_T * pages_per_block)
        tl.store(sbt_row + off_w, tl.zeros_like(off_w), mask=off_w >= bt_n_used)

        bt_tail_tokens = causal_len - self_blk * block_size
        bt_has_tail = tl.sum(bt_is_tail.to(tl.int32), axis=0) > 0
        bt_ctx = bt_n_full * block_size + tl.where(bt_has_tail, bt_tail_tokens, 0)
        bt_ctx = tl.where(
            bt_has_tail, bt_ctx, tl.minimum(bt_n_valid * block_size, causal_len)
        )
        tl.store(sparse_ctx_ptr + (pid_t * NUM_KV_HEADS + pid_h), bt_ctx)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_index_topk(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    sm_scale: float,
    emit_sparse_block_table: bool = False,
):
    """Index block-score + top-k selection. block_size_q == 1 (per-token).

    Returns topk_idx [num_kv_heads, total_q, topk] of 0-indexed block ids
    (right-padded with -1). M3 has num_idx_heads == num_kv_heads, so the
    per-index-head top-k maps 1:1 to kv heads (no index-head reduction needed).

    When ``emit_sparse_block_table`` is True (requires num_idx_heads == 1), the
    topk kernel ALSO fuses the per-query-token page-16 SHUFFLE block-table
    compaction and returns ``(topk_idx, sparse_bt [total_q, topk*8], sparse_ctx
    [total_q])`` ready for the ASM prefill kernel -- saving a separate build
    launch + topk_idx HBM round-trip.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    score = torch.empty(
        (num_idx_heads, total_q, max_block),
        dtype=torch.float32,
        device=idx_q.device,
    )
    BLOCK_SIZE_Q = 64
    grid_score = (triton.cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads)
    _index_block_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        head_dim,
        sm_scale,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )

    topk_idx = torch.empty(
        (num_idx_heads, total_q, topk),
        dtype=torch.int32,
        device=idx_q.device,
    )
    # One emitted row per (query token, kv-head): the ASM/gluon path collapses
    # kv-head into the row dim, so sparse_bt/ctx are total_q*num_idx_heads rows
    # and the page ids are kv-head-encoded in the kernel. num_idx_heads == 1
    # reduces to the original per-token layout.
    emit = emit_sparse_block_table
    if emit:
        sparse_bt = torch.empty(
            (total_q * num_idx_heads, topk * PAGES_PER_SPARSE_BLOCK),
            dtype=torch.int32,
            device=idx_q.device,
        )
        sparse_ctx = torch.empty(
            (total_q * num_idx_heads,), dtype=torch.int32, device=idx_q.device
        )
        sbt_arg, sctx_arg = sparse_bt, sparse_ctx
        bt_stride0, sbt_stride0 = block_table.stride(0), sparse_bt.stride(0)
    else:
        sbt_arg = torch.empty(1, dtype=torch.int32, device=idx_q.device)
        sctx_arg = torch.empty(1, dtype=torch.int32, device=idx_q.device)
        bt_stride0, sbt_stride0 = 0, 0
    # block_size_q == 1 -> query blocks coincide with query tokens.
    grid_topk = (max_query_len, batch, num_idx_heads)
    _topk_index_kernel[grid_topk](
        score,
        topk_idx,
        1,  # sample_interval (block_size_q)
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table,
        sbt_arg,
        sctx_arg,
        bt_stride0,
        sbt_stride0,
        NUM_KV_HEADS=num_idx_heads,
        MASK_INIT=False,
        MASK_LOCAL=False,
        pages_per_block=PAGES_PER_SPARSE_BLOCK,
        EMIT_SPARSE_BT=emit,
    )
    if emit:
        return topk_idx, sparse_bt, sparse_ctx
    return topk_idx


@torch.no_grad()
def minimax_m3_index_topk_decode(
    idx_q: torch.Tensor,  # [total_q == batch*max_query_len, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    seq_lens: torch.Tensor,  # [batch] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    sm_scale: float,
    emit_sparse_block_table: bool = False,
    max_query_len: int = 1,  # query tokens per request (num_spec+1); 1 == plain decode
):
    """Decode index block-score + top-k, both split-K (cudagraph-safe).

    Returns topk_idx [num_kv_heads, total_q, topk] (0-indexed block ids, -1 pad).
    For spec-decode (``max_query_len = num_spec+1``) each of the ``max_query_len``
    query tokens of a request is an independent row with its own causal cutoff
    ``causal_len = seq_len - max_query_len + tok + 1``; ``max_query_len == 1`` is
    plain decode (one token per request) and reduces to the original behavior.

    When ``emit_sparse_block_table`` is True (requires num_idx_heads == 1), the
    merge kernel ALSO fuses the page-16 SHUFFLE block-table compaction and returns
    ``(topk_idx, sparse_bt [total_q, topk*8], sparse_ctx [total_q])`` ready for the
    ASM/gluon decode kernel -- saving a separate build launch + topk_idx HBM
    round-trip.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    assert (
        total_q % max_query_len == 0
    ), f"total_q {total_q} not divisible by max_query_len {max_query_len}"
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score = torch.empty(
        (num_idx_heads, total_q, max_block),
        dtype=torch.float32,
        device=idx_q.device,
    )
    # split-K over seq blocks; chunk count depends only on shape constants so
    # the grid is fixed within a cuda graph.
    TARGET_GRID = 4096
    MAX_NUM_KV_CHUNKS = 256
    target = max(
        1, min(MAX_NUM_KV_CHUNKS, TARGET_GRID // max(1, total_q * num_idx_heads))
    )
    num_kv_chunks = 1 << (target.bit_length() - 1)
    grid_score = (total_q * num_kv_chunks, num_idx_heads)
    _decode_index_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        total_q,
        head_dim,
        init_blocks,
        local_blocks,
        sm_scale,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        MAX_Q=max_query_len,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_KV_CHUNKS=num_kv_chunks,
    )

    topk_idx = torch.empty(
        (num_idx_heads, total_q, topk),
        dtype=torch.int32,
        device=idx_q.device,
    )
    # Chunk count is shape-constant (cudagraph-safe), capped so the merge sorts
    # pow2(num_topk_chunks * pow2(topk)) candidates.
    TOPK_TARGET_GRID = 64
    MAX_NUM_TOPK_CHUNKS = 16
    topk_target = max(
        1, min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, total_q * num_idx_heads))
    )
    num_topk_chunks = 1 << (topk_target.bit_length() - 1)
    block_size_t = triton.next_power_of_2(topk)
    chunk_blocks = (max_block + num_topk_chunks - 1) // num_topk_chunks
    topk_score_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        total_q,
        block_size_t,
        dtype=torch.float32,
        device=idx_q.device,
    )
    topk_idx_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        total_q,
        block_size_t,
        dtype=torch.int32,
        device=idx_q.device,
    )
    _topk_index_partial_kernel[(total_q, num_idx_heads, num_topk_chunks)](
        score,
        topk_score_partial,
        topk_idx_partial,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        chunk_blocks,
        max_query_len,  # MAX_Q
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
    )
    # The fused emit now produces one row per (token, kv-head): the ASM/gluon path
    # collapses kv-head into the row dim. sparse_bt/ctx are sized total_q*num_idx_heads
    # and the page ids are kv-head-encoded inside the kernel. num_idx_heads == 1
    # reduces to the original per-token layout.
    emit = emit_sparse_block_table
    if emit:
        sparse_bt = torch.empty(
            (total_q * num_idx_heads, topk * PAGES_PER_SPARSE_BLOCK),
            dtype=torch.int32,
            device=idx_q.device,
        )
        sparse_ctx = torch.empty(
            (total_q * num_idx_heads,), dtype=torch.int32, device=idx_q.device
        )
        sbt_arg, sctx_arg = sparse_bt, sparse_ctx
        bt_stride0, sbt_stride0 = block_table.stride(0), sparse_bt.stride(0)
    else:
        # dummy 1-elem tensors so the kernel always has valid pointers.
        sbt_arg = torch.empty(1, dtype=torch.int32, device=idx_q.device)
        sctx_arg = torch.empty(1, dtype=torch.int32, device=idx_q.device)
        bt_stride0, sbt_stride0 = 0, 0
    _topk_index_merge_kernel[(total_q, num_idx_heads)](
        topk_score_partial,
        topk_idx_partial,
        topk_idx,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table,
        sbt_arg,
        sctx_arg,
        bt_stride0,
        sbt_stride0,
        MAX_Q=max_query_len,
        NUM_KV_HEADS=num_idx_heads,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        pages_per_block=PAGES_PER_SPARSE_BLOCK,
        EMIT_SPARSE_BT=emit,
    )
    if emit:
        return topk_idx, sparse_bt, sparse_ctx
    return topk_idx
