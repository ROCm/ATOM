# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 lightning-indexer block scoring + top-k.

Index queries score each 128-token block of index keys (max over the block),
then the top-k blocks (plus forced init/local blocks) are selected per query
token. The KV page size is forced to equal the sparse block size (128), so one
sparse block maps to exactly one page.

The two phases score differently -- prefill over whole query tiles, decode over
one request's rows at a time -- and then share a selector.

Index-K cache layout: ``(num_blocks, 128, idx_head_dim)``. The indexer is MQA:
index_q carries num_idx_heads heads, index_k one, shared by all of them.

Only the paths MiniMax M3 uses are implemented: score_type="max", index value
disabled (score-only indexer). The selected block ids feed the block-sparse
attention kernels in ``sparse_attn``.
"""

import functools

import torch

# Adapted for ATOM: the vLLM original took these from `vllm.triton_utils`,
# which couples to vLLM internals. ATOM imports triton directly everywhere else.
import triton
import triton.language as tl

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128
# Query rows one prefill score program owns.
#
# Before the block axis was split, this and the grid were the same knob, so it
# had to be chosen per shape -- narrow wherever a wide one would drop the grid
# below a CTA per compute unit, which on an fp8 cache was everywhere. Chunks
# supply the parallelism now, so a flat tile is free to be the widest one that
# still amortizes: 128 was best or within noise on all 20 measured
# (dtype x heads x shape) points, while the old occupancy rule gives away 12x
# more than it does and no other flat tile comes close (64 gives away 25x, 256
# gives away 9x).
SCORE_BLOCK_SIZE_Q = 128
# Workgroups per compute unit the block-axis split aims for (_score_chunk_blocks).
SCORE_CHUNK_CTAS_PER_CU = 8
# Stages, both score kernels. 2 -- triton's default -- was the worst of 1/2/3
# nearly everywhere. Between 1 and 3, 3 wins on fp8 and ties on bf16; 1 wins the
# short-context rows, which is not where a step spends its time.
SCORE_NUM_STAGES = 3
DECODE_SCORE_NUM_STAGES = 3
# Grid the tiled decode score aims for, and the per-request chunk ceiling that
# keeps a large batch from multiplying into a pointless one (_decode_score_chunks).
DECODE_SCORE_TARGET_GRID = 2048
DECODE_SCORE_MAX_CHUNKS = 64
# Physical 16-pages per logical 128-block for the page-16 SHUFFLE ASM/gluon cache
# (must match sparse_attn.PAGES_PER_SPARSE_BLOCK). Used by the fused block-table
# emission in the topk kernels.
PAGES_PER_SPARSE_BLOCK = 8


# ---------------------------------------------------------------------------
# Fused sparse block-table emission, shared by the prefill and decode
# selectors. Mirrors _build_sparse_block_table_kernel over an in-register
# selection, avoiding a second kernel launch + a topk_idx HBM round-trip.
#
# EVERY kv-head emits its own row: the ASM/gluon path collapses (token, kv_head)
# into the row dim so it can run with num_kv_heads_view == 1. The physical page
# id is encoded as (phys16_page)*NUM_KV_HEADS + kv_head, matching the collapsed
# KV cache view [num_phys16*NUM_KV_HEADS, 1, ...], which is also why the 8 pages
# of one 128-block are NUM_KV_HEADS apart rather than adjacent. NUM_KV_HEADS ==
# 1 reduces to the original per-token emit.
# ---------------------------------------------------------------------------
@triton.jit
def _emit_sparse_block_table_row(
    topk_idx,  # [BLOCK_SIZE_T] selected 128-block ids, -1 for the pads
    bt_row,  # block_table_ptr already offset to this request
    sbt_row,  # sparse_bt_ptr already offset to this (token, kv-head) row
    sctx_ptr,  # sparse_ctx_ptr already offset to that same row
    causal_len,  # keys this query token may attend (its position + 1)
    topk,
    pid_h,
    block_size: tl.constexpr,
    pages_per_block: tl.constexpr,  # 16-pages per sparse block (8)
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    off_t = tl.arange(0, BLOCK_SIZE_T)
    # Tail block: the 128-block holding this token's last causal key. It is the
    # only selected block that can be partial, so it has to land last -- ctx
    # counts the leading full blocks and then however much of it is real.
    self_blk = (causal_len - 1) // block_size
    bt_blk = tl.where(off_t < topk, topk_idx, -1)
    # The selector already caps selection at self_blk; the upper bound is
    # defensive, and keeps an out-of-range id from indexing the block table.
    bt_valid = (bt_blk >= 0) & (bt_blk <= self_blk)
    bt_is_tail = bt_valid & (bt_blk == self_blk)
    bt_is_full = bt_valid & (bt_blk < self_blk)
    bt_n_full = tl.sum(bt_is_full.to(tl.int32), axis=0)
    bt_n_valid = tl.sum(bt_valid.to(tl.int32), axis=0)
    bt_earlier_full = tl.cumsum(bt_is_full.to(tl.int32), axis=0) - bt_is_full.to(
        tl.int32
    )
    bt_slot = tl.where(bt_is_full, bt_earlier_full, bt_n_full)  # tail -> n_full

    bt_logical_page = tl.load(bt_row + bt_blk, mask=bt_valid, other=0).to(tl.int32)
    bt_base_phys = bt_logical_page * pages_per_block * NUM_KV_HEADS + pid_h
    bt_dst_base = bt_slot * pages_per_block

    # Write valid slots -> their pages, then the unused tail -> 0 (an in-bounds
    # page id). The pages of one block are a contiguous run in the destination
    # but NUM_KV_HEADS apart in the source, so this scatters a [slot, page]
    # rectangle rather than looping the page axis: one masked store instead of
    # pages_per_block of them, on a path that runs per (query token, kv head).
    pj = tl.arange(0, pages_per_block)
    tl.store(
        sbt_row + bt_dst_base[:, None] + pj[None, :],
        bt_base_phys[:, None] + pj[None, :] * NUM_KV_HEADS,
        mask=bt_valid[:, None],
    )
    off_w = tl.arange(0, BLOCK_SIZE_T * pages_per_block)
    tl.store(
        sbt_row + off_w,
        tl.zeros_like(off_w),
        mask=off_w >= bt_n_valid * pages_per_block,
    )

    bt_tail_tokens = causal_len - self_blk * block_size
    bt_has_tail = tl.sum(bt_is_tail.to(tl.int32), axis=0) > 0
    bt_ctx = bt_n_full * block_size + tl.where(bt_has_tail, bt_tail_tokens, 0)
    bt_ctx = tl.where(
        bt_has_tail, bt_ctx, tl.minimum(bt_n_valid * block_size, causal_len)
    )
    tl.store(sctx_ptr, bt_ctx)


# ---------------------------------------------------------------------------
# Index block-score kernel (paged). score[h, token, block] = max over the
# 128-token block of (idx_q . index_k), causal-masked. BLOCK_SIZE_K == 128 so
# each K-tile is exactly one page (BLOCKS_PER_K_BLOCK == 1).
#
# Programs are (query tile, request x head, block chunk). The block chunk is
# what lets the query tile grow: a taller tile amortizes each loaded page over
# more query rows, but on its own it also halves the grid, and past one CTA per
# compute unit the idle half of the machine costs more than the re-read saves.
# Splitting the block axis puts the parallelism back somewhere that does not
# cost traffic, so the two knobs stop fighting. Chunk size is a launch-time
# constant derived from max_seq_len, never from the live lengths, so the grid
# does not move between steps.
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
    chunk_blocks,  # 128-blocks one chunk owns
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
    pid_chunk = tl.program_id(2)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    _q_ptrs_0 = (pid_q * BLOCK_SIZE_Q) + tl.arange(0, BLOCK_SIZE_Q)
    _q_ptrs_1 = (0) + tl.arange(0, head_dim)
    q = tl.load(
        q_ptr
        + seq_start * stride_q_n
        + pid_h * stride_q_h
        + _q_ptrs_0[:, None] * (stride_q_n)
        + _q_ptrs_1[None, :] * (stride_q_d),
        mask=(_q_ptrs_0[:, None] < (q_len)),
        other=0.0,
    )
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    # Block table row for this request.
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Causal window: only blocks up to the last query token's position, then
    # this chunk's slice of it. Early tiles see fewer blocks than late ones, so
    # their later chunks are empty and exit here.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    blk_end = tl.cdiv(hi, BLOCK_SIZE_K)
    blk_lo = pid_chunk * chunk_blocks
    if blk_lo >= blk_end:
        return
    blk_hi = min(blk_lo + chunk_blocks, blk_end)
    for blk in tl.range(blk_lo, blk_hi):
        i = blk * BLOCK_SIZE_K
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
        if k.dtype.is_fp8():
            qk = tl.dot(q.to(k.dtype), k, out_dtype=tl.float32) * sm_scale_log2e
        else:
            qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale_log2e
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
# is 1 for M3, so top-k is computed per query token. Both phases select here:
# prefill hands it the score its own kernel materialized, decode the score the
# tiled pass did.
#
# The score and the block id fold into one int64 so that tl.topk does the
# selection. The alternative this replaced carried them as two parallel vectors
# through a bitonic sort, which moves both on every compare-and-swap and fully
# sorts each tile (log2(K)^2/2 stages) to keep only the leading BLOCK_SIZE_T
# lanes; a bitonic top-k instead halves the live data on each stage past
# log2(T), so a 2048-wide tile costs ~38 stages of one vector rather than ~66
# of two. Measured 1.4x on prefill and 1.7x on decode over the deployed shapes.
# ---------------------------------------------------------------------------
@triton.jit
def _pack_score_key(score, index, valid):
    """One int64 that orders exactly like (score descending, block id descending).

    fp32 already compares like a sign-magnitude integer, so flipping the whole
    word for negatives and just the sign bit for positives yields an unsigned
    key with the same order. The 1-based block id rides in the low 16 bits, so
    equal scores resolve to the higher id. Bit 48 keeps every real candidate
    above the zero that masked-off lanes carry, so padding always loses.

    The tie rule is load-bearing, not cosmetic: block scores are a max over 128
    keys from a 3-mantissa-bit fp8 cache, so two blocks land on the same fp32
    score about once per 25k rows. Which one wins does not change the selection
    -- both are in it -- but it changes their order in sparse_bt, which is the
    order the attention accumulates them in. Neither order is more correct, and
    the selector this replaced had no statable rule at all (its order came from
    where an element sat in a sort network, id-monotone in neither direction).
    GSM8K cannot grade the choice: at that prompt length every block is selected
    anyway, and the benchmark moves further between reruns than between rules.

    Caller must keep ``index <= 0xFFFF``; the wrapper bounds max_block for that.
    """
    bits = score.to(tl.uint32, bitcast=True)
    # -0.0 and 0.0 are one score with two bit patterns; give them one key.
    bits = tl.where(bits == 0x80000000, 0, bits)
    ordered = bits ^ tl.where(bits >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    # A NaN bitcasts above +inf and would win every selection it entered. Send
    # it to the bottom -- below -inf, whose image is nonzero. Testing the bits
    # (exponent all ones, mantissa nonzero) rather than `x != x` keeps this in
    # the integer domain and leaves the infinities exactly where they belong.
    ordered = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0, ordered)
    key = (1 << 48) | (ordered.to(tl.int64) << 16) | index.to(tl.int64)
    return tl.where(valid, key, 0)


@triton.jit
def _score_tile_keys(
    s_row,
    offset,
    valid_blocks,
    local_start,
    stride_s_k,
    init_blocks: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One tile of block scores, forced blocks applied, packed into sort keys.

    Outside the causal window a block is invisible. A forced block is pinned
    above every real score -- init to 1e30, local to 1e29, local applied second
    so a block that is both ends up local. NaN needs no scrub here;
    _pack_score_key ranks it last whichever pin ran first.
    """
    idx = offset + tl.arange(0, BLOCK_SIZE_K)
    valid = idx < valid_blocks
    score = tl.load(s_row + idx * stride_s_k, mask=valid, other=-1e30).to(tl.float32)
    score = tl.where(valid & (idx < init_blocks), 1e30, score)
    score = tl.where(valid & (idx >= local_start), 1e29, score)
    return _pack_score_key(score, idx + 1, valid)


# Selector tile width and warp count. The width that wins tracks a row's
# candidate-block count; the phases get two pairs because only one of them can
# see that number in time.
#
# Prefill derives it, because it is eager: its grids already move with
# max_query_len and max_block, so a shape-dependent constexpr is free and
# bottoms out at four compiled variants. clamp(next_pow2(max_block), 128, 1024)
# is the per-shape best everywhere the selector costs anything -- at 800 and 896
# blocks nothing on a 32..2048 x 1..8 warp sweep beat it. The 128 floor is the
# one place it gives something up: at exactly 128 blocks a 64-wide tile wins
# 12%, because that shape is a first chunk whose rows span 1..128 blocks and the
# narrow tile lets the short majority finish in one pass. The floor stays
# because the next shape down (68 blocks) prefers 128 over 64 by 10%, so
# lowering it is not a one-line change; it is worth ~0.1ms of a prefill step.
#
# Decode cannot derive it, because it is captured: the wrapper's Python runs
# once, at capture, and capture is handed max_seqlen_k = max_model_len so the
# grid covers every replay. A derived width would therefore be the
# longest-context one on every replay -- (1024, 8) here, which is 1.4-1.7x off
# the per-shape best at 64 blocks. So decode takes a constant, chosen by total
# regret over the deployed shapes. That regret is real and mostly irreducible:
# the per-shape optima span (64, 1) to (1024, 8), and the best single pair,
# (256, 4), is only 0.6% better than this one over 16 shapes.
#
# Neither phase may autotune, and not for the usual reason: triton's key here
# would be BLOCK_SIZE_T == next_pow2(topk) == 16, a constant, so the first shape
# a process sees would fix the width for every later one. That lottery is not
# academic -- the widths differ by more than 4x across these shapes.
#
# Warps go opposite ways for the same reason the widths do. Prefill runs one
# program per query token (16k CTAs), so one warp per row is plenty and more
# only adds cross-lane work: 1 won all 28 cells of the sweep. Decode's selector
# grid is 50-200 CTAs, far too few to fill the machine on their own, so it wants
# the warps instead.
PREFILL_TOPK_MIN_BLOCK_SIZE_K = 128
PREFILL_TOPK_MAX_BLOCK_SIZE_K = 1024
PREFILL_TOPK_NUM_WARPS = 1
DECODE_TOPK_BLOCK_SIZE_K = 512
DECODE_TOPK_NUM_WARPS = 8


@functools.cache
def _compute_units(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _score_chunk_blocks(max_block: int, q_tiles: int, batch: int, heads: int, device):
    """128-blocks one prefill score program owns.

    The query tile decides how many rows share a loaded page; this decides how
    many programs there are. Keeping them separate is the point -- without it
    the only way to raise the grid is to shrink the tile, which is the same
    thing as reading every page more times.

    Derived from max_block, which is a launch-time bound, so a cuda graph
    replays the grid it captured. Chunks are sized rather than counted so that
    chunk z is the same pages for every query tile that reaches it.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    want = _compute_units(index) * SCORE_CHUNK_CTAS_PER_CU
    chunks = max(1, want // max(1, q_tiles * batch * heads))
    chunks = min(1 << (chunks.bit_length() - 1), max(1, max_block))
    return triton.cdiv(max_block, chunks)


def _decode_score_chunks(batch: int, max_block: int) -> int:
    """Chunks one decode score row is split across.

    A count, not a size: the grid is (request, chunk), so this IS the second
    grid dim and it must stay shape-constant for a cuda graph to replay it.
    Enough chunks that a long context is not serialized inside a handful of
    CTAs, capped so a large batch does not multiply into a pointless grid.

    The round trip through the size is not redundant. The caller turns this
    count back into a size with the same cdiv, and a count that does not divide
    the blocks overshoots: 800 blocks over 64 chunks is 13 blocks each, and
    13 * 64 covers 832, so the last two chunks start past the end and launch
    only to return. Deriving the count from the size it implies keeps every
    chunk non-empty, and both are still pure functions of launch-time bounds,
    so a cuda graph replays the grid it captured.
    """
    target = max(
        1, min(DECODE_SCORE_MAX_CHUNKS, DECODE_SCORE_TARGET_GRID // max(1, batch))
    )
    if max_block <= 0:
        # A grid dim still has to be positive. Capping `chunks` is not enough:
        # the round trip below divides by its own inner `cdiv`, zero here, and
        # the raise lands outside any launch -- one rank down while the others
        # wait in the collective that follows.
        return 1
    chunks = min(1 << (target.bit_length() - 1), max_block)
    return triton.cdiv(max_block, triton.cdiv(max_block, chunks))


def _require_packable(max_block: int) -> None:
    """The packed key spends its low 16 bits on the 1-based block id.

    Raised, not asserted: this is input validation (`max_block` follows from
    `--max-model-len` and `--block-size`), and under `python -O` an assertion
    would vanish and let the id wrap into the tie-break field -- a top-k that
    quietly picks the wrong blocks.
    """
    if max_block >= 0xFFFF:
        raise ValueError(
            f"packed top-k addresses at most {0xFFFF - 1} blocks, got "
            f"{max_block}; widen the tie-break field"
        )


def _alloc_emit(total_q, num_idx_heads, topk, block_table, emit, device):
    """Buffers for the fused page-16 emit, as (what to return, what to pass).

    One row per (query token, kv-head): the ASM/gluon path collapses kv-head
    into the row dim, so the rows are total_q * num_idx_heads and the page ids
    carry the head. num_idx_heads == 1 reduces to the plain per-token layout.
    Both phases allocate exactly this, which is why it is not inline in either.

    The first half is None when the emit is off, which is also what tells the
    wrapper whether it returns one tensor or three; the second half is always
    four valid kernel arguments, because the kernel dereferences them either
    way.
    """
    if not emit:
        dummy = torch.empty(1, dtype=torch.int32, device=device)
        return None, (dummy, dummy, 0, 0)
    rows = total_q * num_idx_heads
    sparse_bt = torch.empty(
        (rows, topk * PAGES_PER_SPARSE_BLOCK), dtype=torch.int32, device=device
    )
    sparse_ctx = torch.empty((rows,), dtype=torch.int32, device=device)
    args = (sparse_bt, sparse_ctx, block_table.stride(0), sparse_bt.stride(0))
    return (sparse_bt, sparse_ctx), args


def _launch_select(
    score,
    topk_idx,
    row_starts,
    row_prefix,
    batch,
    topk,
    init_blocks,
    local_blocks,
    block_table,
    emit_args,
    num_idx_heads,
    rows_per_req,
    decode_max_q,
    block_size_k,
    num_warps,
    emit,
):
    """The one selection pass, launched the same way by both phases.

    Five things differ between them, and they are all one decision: prefill's
    rows are ragged so it hands (query starts, keys behind each request) and
    picks its tile from the shape, decode's are dense so both index arrays
    collapse to seq_lens and the tile and warp count have to be constants.
    """
    sbt, sctx, bt_stride0, sbt_stride0 = emit_args
    _topk_index_packed_kernel[(rows_per_req, batch, num_idx_heads)](
        score,
        topk_idx,
        1,  # sample_interval (block_size_q)
        SPARSE_BLOCK_SIZE,
        row_starts,
        row_prefix,
        topk,
        init_blocks,
        local_blocks,
        *score.stride(),
        *topk_idx.stride(),
        block_table,
        sbt,
        sctx,
        bt_stride0,
        sbt_stride0,
        NUM_KV_HEADS=num_idx_heads,
        DECODE_MAX_Q=decode_max_q,
        BLOCK_SIZE_K=block_size_k,
        pages_per_block=PAGES_PER_SPARSE_BLOCK,
        EMIT_SPARSE_BT=emit,
        num_warps=num_warps,
    )


def _prefill_topk_block_size_k(max_block: int) -> int:
    """Widest tile the row can use, so the merge chain is as short as it gets."""
    return min(
        PREFILL_TOPK_MAX_BLOCK_SIZE_K,
        max(PREFILL_TOPK_MIN_BLOCK_SIZE_K, triton.next_power_of_2(max(1, max_block))),
    )


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit
def _topk_index_packed_kernel(
    s_ptr,  # [num_heads, total_q, max_block]
    ti_ptr,  # [num_heads, total_q, topk]
    sample_interval: tl.constexpr,  # block_size_q (1 for M3)
    block_size: tl.constexpr,  # sparse block size (128)
    # How a request's rows are laid out, in the two forms the phases have.
    # Prefill's rows are ragged, so it hands the query starts (cu_seqlens_q,
    # which is also cu_seqblocks_q at block_size_q == 1) and the keys already
    # behind each request. Decode's are a dense DECODE_MAX_Q per request ending
    # at seq_len, so both follow from seq_lens and it passes that twice.
    row_starts,
    row_prefix,
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
    DECODE_MAX_Q: tl.constexpr,  # 0 = prefill; else query rows per request
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    pages_per_block: tl.constexpr,  # 16-pages per sparse block (8)
    EMIT_SPARSE_BT: tl.constexpr,  # fuse compaction (per-kv-head row + encoded page)
):
    tl.static_assert(BLOCK_SIZE_K >= BLOCK_SIZE_T)
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    if DECODE_MAX_Q > 0:
        # Deriving the starts here rather than materializing them cost three
        # elementwise launches per decode step, ~5% of the op.
        seq_start = pid_b * DECODE_MAX_Q
        block_num = DECODE_MAX_Q
        prefix_len = tl.load(row_prefix + pid_b) - DECODE_MAX_Q
    else:
        seq_start = tl.load(row_starts + pid_b)
        block_num = tl.load(row_starts + pid_b + 1) - seq_start
        prefix_len = tl.load(row_prefix + pid_b)
    if pid_q >= block_num:
        return

    s_row = (
        s_ptr + (seq_start + pid_q * sample_interval) * stride_s_n + pid_h * stride_s_h
    )
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    local_start = tl.maximum(0, valid_blocks - local_blocks)

    # valid_blocks >= 1 always, so the first tile is unconditional and the loop
    # covers only the rows long enough to need a second one. Folding it into the
    # loop instead would cost the single-tile rows -- the common case -- one
    # extra cat and topk against an all-padding vector. (The two argument lists
    # are spelled out because triton's Python subset has no *args.)
    winners = tl.topk(
        _score_tile_keys(
            s_row, 0, valid_blocks, local_start, stride_s_k, init_blocks, BLOCK_SIZE_K
        ),
        BLOCK_SIZE_T,
    )
    for offset in tl.range(BLOCK_SIZE_K, valid_blocks, BLOCK_SIZE_K):
        tile = tl.topk(
            _score_tile_keys(
                s_row,
                offset,
                valid_blocks,
                local_start,
                stride_s_k,
                init_blocks,
                BLOCK_SIZE_K,
            ),
            BLOCK_SIZE_T,
        )
        winners = tl.topk(tl.cat(winners, tile, can_reorder=True), BLOCK_SIZE_T)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = (winners & 0xFFFF).to(tl.int32) - 1  # back to a 0-based block id
    topk_idx = tl.where(off_t < tl.minimum(topk, valid_blocks), topk_idx, -1)
    ti_ptrs = (
        ti_ptr
        + (seq_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=off_t < topk)

    if EMIT_SPARSE_BT:
        emit_row = (seq_start + pid_q) * NUM_KV_HEADS + pid_h
        _emit_sparse_block_table_row(
            topk_idx,
            block_table_ptr + pid_b * stride_bt_b,
            sparse_bt_ptr + emit_row * stride_sbt_n,
            sparse_ctx_ptr + emit_row,
            prefix_len + pid_q * sample_interval + 1,
            topk,
            pid_h,
            block_size,
            pages_per_block,
            NUM_KV_HEADS,
            BLOCK_SIZE_T,
        )


# ---------------------------------------------------------------------------
# Decode index-score kernel, tiled over (query row x index head).
#
# One program owns a (request, block chunk) and scores every (row, head) of
# that request against each block it loads: the columns become the N dim of a
# tl.dot, so a 128-token index-K block is read once per request. What this
# replaced scored one (row, head) at a time and re-read each block
# heads * query_rows times -- 8x at tp2 under MTP-4, which is why that
# configuration cost 5x what tp4 plain decode does on identical unique traffic.
#
# The block scores land in HBM instead of being consumed in-register, which
# costs a separate selection pass. That buffer is ~1.3 MB at the 100k/conc-50
# serving point against 1.3 GB of index K, so it is not the trade it looks like.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        # tl.dot wants a real N; a request with one row and one head would
        # otherwise ask for a 1-wide matmul.
        "BLOCK_SIZE_N": lambda args: max(
            16, triton.next_power_of_2(args["NUM_IDX_HEADS"] * args["MAX_Q"])
        ),
    }
)
@triton.jit
def _decode_index_score_tiled_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # out: [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [batch]
    head_dim,
    sm_scale,
    chunk_blocks,  # blocks this program's chunk owns
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_bt_b,
    NUM_IDX_HEADS: tl.constexpr,
    MAX_Q: tl.constexpr,  # query tokens per request (num_spec + 1; 1 == plain decode)
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    seq_len = tl.load(seq_lens + pid_b)
    # The request's last query row is the one that sees the most blocks.
    num_blocks = (seq_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)
    if chunk_start >= chunk_end:
        return

    off_n = tl.arange(0, BLOCK_SIZE_N)
    tok = off_n // NUM_IDX_HEADS
    head = off_n - tok * NUM_IDX_HEADS
    n_valid = off_n < NUM_IDX_HEADS * MAX_Q
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    off_k = tl.arange(0, BLOCK_SIZE_K)
    row = pid_b * MAX_Q + tok

    # [D, N]: one column per (query row, index head) of this request.
    q = tl.load(
        q_ptr
        + row[None, :] * stride_q_n
        + head[None, :] * stride_q_h
        + off_d[:, None] * stride_q_d,
        mask=d_mask[:, None] & n_valid[None, :],
        other=0.0,
    )
    # Row `tok` sits at absolute position seq_len - MAX_Q + tok, so its causal
    # cutoff is one past that. Every column has its own.
    causal_len = seq_len - MAX_Q + tok + 1
    bt_row = block_table_ptr + pid_b * stride_bt_b
    sm_scale_log2e = sm_scale * 1.4426950409

    for blk in tl.range(chunk_start, chunk_end):
        page = tl.load(bt_row + blk).to(tl.int64)
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[:, None] * stride_ik_pos
            + off_d[None, :] * stride_ik_d,
            mask=d_mask[None, :],
            other=0.0,
        )
        # Keep the query at its own precision and lift the key to meet it, the
        # way the gemv path did -- prefill instead rounds the query down to the
        # cache dtype, and the two phases are meant to keep scoring differently.
        qk = tl.dot(k.to(q.dtype), q, out_dtype=tl.float32) * sm_scale_log2e
        pos = blk * BLOCK_SIZE_K + off_k
        qk = tl.where(pos[:, None] < causal_len[None, :], qk, float("-inf"))
        score = tl.max(qk, axis=0)
        tl.store(
            score_ptr + head * stride_s_h + row * stride_s_b + blk * stride_s_k,
            score,
            mask=n_valid,
        )


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

    When ``emit_sparse_block_table`` is True the topk kernel ALSO fuses the
    per-query-token page-16 SHUFFLE block-table
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
    _require_packable(max_block)

    score = torch.empty(
        (num_idx_heads, total_q, max_block),
        dtype=torch.float32,
        device=idx_q.device,
    )
    q_tiles = triton.cdiv(max_query_len, SCORE_BLOCK_SIZE_Q)
    chunk_blocks = _score_chunk_blocks(
        max_block, q_tiles, batch, num_idx_heads, idx_q.device
    )
    grid_score = (
        q_tiles,
        batch * num_idx_heads,
        triton.cdiv(max_block, chunk_blocks),
    )
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
        chunk_blocks,
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
        BLOCK_SIZE_Q=SCORE_BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        num_stages=SCORE_NUM_STAGES,
    )

    topk_idx = torch.empty(
        (num_idx_heads, total_q, topk), dtype=torch.int32, device=idx_q.device
    )
    emit_out, emit_args = _alloc_emit(
        total_q, num_idx_heads, topk, block_table, emit_sparse_block_table, idx_q.device
    )
    _launch_select(
        score,
        topk_idx,
        cu_seqlens_q,  # ragged: query starts, one past the end for the count
        prefix_lens,
        batch,
        topk,
        init_blocks,
        local_blocks,
        block_table,
        emit_args,
        num_idx_heads,
        rows_per_req=max_query_len,
        decode_max_q=0,
        block_size_k=_prefill_topk_block_size_k(max_block),
        num_warps=PREFILL_TOPK_NUM_WARPS,
        emit=emit_sparse_block_table,
    )
    return (topk_idx, *emit_out) if emit_out else topk_idx


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
    plain decode (one token per request).

    When ``emit_sparse_block_table`` is True the selector ALSO fuses the page-16
    SHUFFLE block-table compaction and returns ``(topk_idx, sparse_bt [total_q,
    topk*8], sparse_ctx [total_q])`` ready for the ASM/gluon decode kernel --
    saving a separate build launch + topk_idx HBM round-trip.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    assert (
        total_q % max_query_len == 0
    ), f"total_q {total_q} not divisible by max_query_len {max_query_len}"
    batch = seq_lens.shape[0]
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    _require_packable(max_block)
    topk_idx = torch.empty(
        (num_idx_heads, total_q, topk), dtype=torch.int32, device=idx_q.device
    )
    emit_out, emit_args = _alloc_emit(
        total_q, num_idx_heads, topk, block_table, emit_sparse_block_table, idx_q.device
    )

    score = torch.empty(
        (num_idx_heads, total_q, max_block),
        dtype=torch.float32,
        device=idx_q.device,
    )
    num_score_chunks = _decode_score_chunks(batch, max_block)
    _decode_index_score_tiled_kernel[(batch, num_score_chunks)](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        head_dim,
        sm_scale,
        triton.cdiv(max_block, num_score_chunks),
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
        NUM_IDX_HEADS=num_idx_heads,
        MAX_Q=max_query_len,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        num_stages=DECODE_SCORE_NUM_STAGES,
    )

    _launch_select(
        score,
        topk_idx,
        seq_lens,  # dense: both the starts and the prefix follow from these
        seq_lens,
        batch,
        topk,
        init_blocks,
        local_blocks,
        block_table,
        emit_args,
        num_idx_heads,
        rows_per_req=max_query_len,
        decode_max_q=max_query_len,
        block_size_k=DECODE_TOPK_BLOCK_SIZE_K,
        num_warps=DECODE_TOPK_NUM_WARPS,
        emit=emit_sparse_block_table,
    )
    return (topk_idx, *emit_out) if emit_out else topk_idx
