"""Opt-in M3 candidate exchange: local top-k per shard, then a cross-rank merge.

The alternative to exchanging every block score. Each rank still scores all
index heads over its own 1/P of the context, but sends only its own top-k
candidates instead of the full score row, so the payload stops growing with
context. Correctness rests on one fact: a rank that keeps k candidates cannot
drop a global winner, because a block it discarded lost to k blocks it kept,
all of which are also in the global candidate set. Keeping k/P per rank would
NOT be safe -- the global top-k may lie entirely inside one shard.

Scores and ids travel as the packed int64 sort key the native selector already
uses, so the merge is another tl.topk over the same key space and the tie rule
survives the round trip unchanged.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.minimax_m3.index_topk import (
    DECODE_TOPK_NUM_WARPS,
    _alloc_emit,
    _emit_sparse_block_table_row,
    _pack_score_key,
    _require_packable,
)


@triton.jit
def _force(score, block, valid, local_start, INIT_BLOCKS: tl.constexpr):
    """Lift the always-selected blocks above every scored one.

    Two tiers so the sink blocks outrank the sliding window, matching the
    native selector's ordering; both sit above any real score.
    """
    score = tl.where(valid & (block < INIT_BLOCKS), 1e30, score)
    return tl.where(valid & (block >= local_start), 1e29, score)


# Retain bounded alignment specialization for the score row stride.
@triton.jit(do_not_specialize=["GLOBAL_BLOCKS"])
def _local_topk(
    Scores,
    Keys,
    Lengths,
    QUERY_LEN: tl.constexpr,
    LOCAL_BLOCKS,
    GLOBAL_BLOCKS,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    TOPK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_KEEP: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Pack this shard's [heads, tokens, local] scores down to its own top-k keys.

    The id inside each key is the GLOBAL block id, so the receiving merge never
    needs to know which rank a candidate came from.

    Forced blocks are pinned HERE as well as in the merge. Pinning only at the
    merge loses them: a forced block that lost its own shard's top-k never
    arrives to be pinned. Pinning here costs nothing, because the candidate it
    evicts could not have placed globally anyway -- that block lost to k-1 other
    candidates on this rank plus the forced one, and the global top-k is k wide.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    tokens = tl.num_programs(0)
    request = row // QUERY_LEN
    token = row % QUERY_LEN
    length = tl.load(Lengths + request)
    # Blocks this query token may attend, in global numbering.
    causal_blocks = (length - QUERY_LEN + token + 128) // 128
    local_start = tl.maximum(0, causal_blocks - LOCAL_KEEP)
    s_row = Scores + (head * tokens + row) * LOCAL_BLOCKS

    off = tl.arange(0, BLOCK_SIZE_K)
    local_valid = off < LOCAL_BLOCKS
    block = off * WORLD + RANK
    valid = local_valid & (block < GLOBAL_BLOCKS) & (block < causal_blocks)
    score = tl.load(s_row + off, mask=local_valid, other=-1e30).to(tl.float32)
    score = _force(score, block, valid, local_start, INIT_BLOCKS)
    winners = tl.topk(_pack_score_key(score, block + 1, valid), BLOCK_SIZE_T)
    for start in tl.range(BLOCK_SIZE_K, LOCAL_BLOCKS, BLOCK_SIZE_K):
        off = start + tl.arange(0, BLOCK_SIZE_K)
        local_valid = off < LOCAL_BLOCKS
        block = off * WORLD + RANK
        valid = local_valid & (block < GLOBAL_BLOCKS) & (block < causal_blocks)
        score = tl.load(s_row + off, mask=local_valid, other=-1e30).to(tl.float32)
        score = _force(score, block, valid, local_start, INIT_BLOCKS)
        tile = tl.topk(_pack_score_key(score, block + 1, valid), BLOCK_SIZE_T)
        winners = tl.topk(tl.cat(winners, tile, can_reorder=True), BLOCK_SIZE_T)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    tl.store(
        Keys + (head * tokens + row) * TOPK + off_t,
        winners,
        mask=off_t < TOPK,
    )


@triton.jit(do_not_specialize=["SRC_STRIDE"])
def _merge_topk(
    Keys,
    Indices,
    Lengths,
    Table,
    SparseBt,
    SparseCtx,
    TABLE_STRIDE: tl.constexpr,
    SBT_STRIDE: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_KEEP: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    KEYS_PER_SHARD: tl.constexpr,
    SRC_STRIDE,
    ROW_STRIDE: tl.constexpr,
    REAL_CANDIDATES: tl.constexpr,
    CANDIDATES: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    EMIT: tl.constexpr,
):
    """Merge WORLD*TOPK gathered candidates into the global top-k for one head.

    Forced blocks are re-pinned here even though the shard pass already pinned
    them, because the pin has to survive the score round trip: this is what
    keeps a forced block ahead of a real block that happens to score above
    1e29. It is not what makes forced blocks arrive -- see ``_local_topk``.
    """
    row = tl.program_id(0)
    request = row // QUERY_LEN
    token = row % QUERY_LEN
    length = tl.load(Lengths + request)
    causal_len = length - QUERY_LEN + token + 1
    valid_blocks = (causal_len + 127) // 128
    local_start = tl.maximum(0, valid_blocks - LOCAL_KEEP)

    off = tl.arange(0, CANDIDATES)
    # The all-to-all delivers [shard, token, topk]; reading it with a stride
    # costs two integer ops and saves the contiguous() copy that flattening the
    # shard axis into the token axis would otherwise need.
    # CANDIDATES is rounded up to a power of two; the pad lanes must read 0,
    # the same key an empty shard slot carries, so _pack_score_key ranks them
    # below every real candidate.
    src = (off // KEYS_PER_SHARD) * SRC_STRIDE + row * ROW_STRIDE + off % KEYS_PER_SHARD
    key = tl.load(Keys + src, mask=off < REAL_CANDIDATES, other=0)
    # Unpack the id, re-pin forced blocks, repack. Padding lanes carry key 0.
    block = (key & 0xFFFF).to(tl.int32) - 1
    real = key != 0
    score = ((key >> 16) & 0xFFFFFFFF).to(tl.uint32)
    # Inverse of the order-preserving image: a set top bit means the original
    # was positive (it was flipped in), a clear one means the whole word was.
    bits = score ^ tl.where(score >> 31 != 0, 0x80000000, 0xFFFFFFFF)
    value = bits.to(tl.float32, bitcast=True)
    value = _force(value, block, real, local_start, INIT_BLOCKS)
    winners = tl.topk(_pack_score_key(value, block + 1, real), BLOCK_SIZE_T)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = (winners & 0xFFFF).to(tl.int32) - 1
    topk_idx = tl.where(off_t < tl.minimum(TOPK, valid_blocks), topk_idx, -1)
    tl.store(Indices + row * TOPK + off_t, topk_idx, mask=off_t < TOPK)
    if EMIT:
        _emit_sparse_block_table_row(
            topk_idx,
            Table + request * TABLE_STRIDE,
            SparseBt + row * SBT_STRIDE,
            SparseCtx + row,
            causal_len,
            TOPK,
            0,
            128,
            PAGES_PER_BLOCK,
            NUM_KV_HEADS,
            BLOCK_SIZE_T,
        )


def local_candidate_keys(
    scores,
    seq_lens,
    topk,
    rank,
    world_size,
    max_query_len,
    global_blocks,
    init_blocks=0,
    local_blocks=0,
):
    """Reduce [heads,tokens,local] scores to [heads,tokens,topk] packed keys.

    Keys carry global block ids, so the exchange needs no rank metadata.

    The forced-block counts must match the ones the merge is given: a forced
    block has to be pinned on the rank that owns it or it never reaches the
    merge to be pinned there.
    """
    if scores.ndim != 3 or scores.dtype != torch.float32:
        raise ValueError("scores must be FP32 [heads,tokens,local_blocks]")
    heads, tokens, local = scores.shape
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid context partition")
    if max_query_len < 1 or tokens != seq_lens.numel() * max_query_len:
        raise ValueError("score rows must equal batch * max_query_len")
    if topk < 1 or topk > 512:
        raise ValueError("unsupported top-k")
    if min(init_blocks, local_blocks) < 0:
        raise ValueError("forced-block counts must be non-negative")
    _require_packable(global_blocks)
    if not scores.is_contiguous() or seq_lens.dtype != torch.int32:
        raise ValueError("scores must be contiguous and lengths int32")
    keys = torch.empty((heads, tokens, topk), dtype=torch.int64, device=scores.device)
    if tokens:
        # The tile must be at least as wide as the top-k it selects: the
        # kernel's FIRST `tl.topk(x[BLOCK_SIZE_K], BLOCK_SIZE_T)` runs before
        # any `tl.cat`, so BLOCK_SIZE_K < BLOCK_SIZE_T cannot compile. A small
        # shard reaches that -- e.g. 8 local blocks with topk 32 gives a 16-wide
        # tile and a 32-wide selection. M3 ships sparse_topk_blocks=16, which
        # the max(16, ...) floor already covers, so this is unreachable on the
        # shipped config and costs it exactly nothing; it is here so the
        # function is correct for the topk its own guard admits (<= 512), rather
        # than only for the one value production happens to pass.
        width = max(16, triton.next_power_of_2(local), triton.next_power_of_2(topk))
        # Select medium rows in one tile; smaller streaming tiles avoid
        # excessive padding work when a longer row crosses a tile boundary.
        if width > 2048:
            width = 1024
        _local_topk[(tokens, heads)](
            scores,
            keys,
            seq_lens,
            QUERY_LEN=max_query_len,
            LOCAL_BLOCKS=local,
            GLOBAL_BLOCKS=global_blocks,
            RANK=rank,
            WORLD=world_size,
            TOPK=topk,
            INIT_BLOCKS=init_blocks,
            LOCAL_KEEP=local_blocks,
            BLOCK_SIZE_K=width,
            BLOCK_SIZE_T=triton.next_power_of_2(topk),
            num_warps=DECODE_TOPK_NUM_WARPS,
        )
    return keys


def merge_candidate_keys(
    keys, block_table, seq_lens, topk, init_blocks, local_blocks, max_query_len
):
    """Select the global top-k from gathered candidates for this rank's head.

    ``keys`` holds every shard's candidates for the one head this rank owns
    after the exchange, either already flattened to [tokens, world*topk] or in
    the [world, tokens, topk] layout the all-to-all writes. The 3D form is read
    with a stride rather than copied.
    """
    if keys.dtype != torch.int64 or keys.ndim not in (2, 3):
        raise ValueError(
            "keys must be int64 [tokens, world*topk] or [world,tokens,topk]"
        )
    if keys.ndim == 3:
        shards, tokens, per_shard = keys.shape
        src_stride, row_stride, key_stride = keys.stride()
        candidates = shards * per_shard
    else:
        tokens, candidates = keys.shape
        per_shard, src_stride = candidates, 0
        row_stride, key_stride = keys.stride()
    if key_stride != 1:
        raise ValueError("the candidate axis must be contiguous")
    if max_query_len < 1 or tokens != seq_lens.numel() * max_query_len:
        raise ValueError("key rows must equal batch * max_query_len")
    if topk < 1 or topk > candidates:
        raise ValueError("top-k must not exceed the gathered candidate count")
    if min(init_blocks, local_blocks) < 0:
        raise ValueError("forced-block counts must be non-negative")
    if block_table.ndim != 2 or block_table.shape[0] != seq_lens.numel():
        raise ValueError("block table must have one row per request")
    if (
        block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
        or block_table.stride(1) != 1
    ):
        raise ValueError("metadata must be int32 with contiguous inner dimensions")
    if any(
        not x.is_cuda or x.device != keys.device for x in (keys, block_table, seq_lens)
    ):
        raise ValueError("merge inputs must share a GPU")
    indices = torch.empty((1, tokens, topk), dtype=torch.int32, device=keys.device)
    output, _ = _alloc_emit(tokens, 1, topk, block_table, True, keys.device)
    if tokens:
        _merge_topk[(tokens,)](
            keys,
            indices,
            seq_lens,
            block_table,
            output[0],
            output[1],
            TABLE_STRIDE=block_table.stride(0),
            SBT_STRIDE=output[0].stride(0),
            QUERY_LEN=max_query_len,
            TOPK=topk,
            INIT_BLOCKS=init_blocks,
            LOCAL_KEEP=local_blocks,
            NUM_KV_HEADS=1,
            PAGES_PER_BLOCK=8,
            KEYS_PER_SHARD=per_shard,
            SRC_STRIDE=src_stride,
            ROW_STRIDE=row_stride,
            REAL_CANDIDATES=candidates,
            CANDIDATES=triton.next_power_of_2(candidates),
            # Exactly next_pow2(topk), as the native selector uses: the fused
            # emit's zero-fill spans BLOCK_SIZE_T*pages_per_block unmasked, and
            # the sparse_bt row is only topk*pages_per_block wide. A padded
            # BLOCK_SIZE_T writes past the row into the next allocation.
            BLOCK_SIZE_T=triton.next_power_of_2(topk),
            EMIT=True,
            num_warps=DECODE_TOPK_NUM_WARPS,
        )
    return indices, *output
