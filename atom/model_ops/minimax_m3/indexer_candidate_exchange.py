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
    PAGES_PER_SPARSE_BLOCK,
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


@triton.jit
def _tile_winners(
    s_row,
    scratch,
    start,
    LOCAL_BLOCKS,
    GLOBAL_BLOCKS,
    causal_blocks,
    local_start,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    CANDIDATES: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    FILTER: tl.constexpr,
):
    """One tile's top-k keys, sorting only the keys that can place.

    The BLOCK_SIZE_T-th largest group maximum bounds the tile's own
    BLOCK_SIZE_T-th largest key from below -- that many groups each hold a key
    at least that large -- so every winner clears it, and usually only a few
    dozen keys do. Those are compacted into this row's scratch and sorted
    there; a tile where more clear it sorts whole. Keys are unique (the block
    id rides in them), so both ways select the same keys in the same order.
    """
    off = start + tl.arange(0, WIDTH)
    local_valid = off < LOCAL_BLOCKS
    block = off * WORLD + RANK
    valid = local_valid & (block < GLOBAL_BLOCKS) & (block < causal_blocks)
    score = tl.load(s_row + off, mask=local_valid, other=-1e30).to(tl.float32)
    score = _force(score, block, valid, local_start, INIT_BLOCKS)
    key = _pack_score_key(score, block + 1, valid)
    if FILTER:
        group_max = tl.max(
            tl.reshape(key, (GROUPS, WIDTH // GROUPS), can_reorder=True), axis=1
        )
        # Masked lanes carry key 0; the floor keeps them out of the scratch
        # when fewer than BLOCK_SIZE_T groups hold a real key.
        bound = tl.maximum(tl.min(tl.topk(group_max, BLOCK_SIZE_T)), 1)
        keep = key >= bound
        count = tl.sum(keep.to(tl.int32))
        if count <= CANDIDATES:
            # A long row's previous tile may still be reading this scratch row.
            tl.debug_barrier()
            tl.store(scratch + tl.cumsum(keep.to(tl.int32), 0) - 1, key, mask=keep)
            tl.debug_barrier()
            off_c = tl.arange(0, CANDIDATES)
            winners = tl.topk(
                tl.load(scratch + off_c, mask=off_c < count, other=0), BLOCK_SIZE_T
            )
        else:
            winners = tl.topk(key, BLOCK_SIZE_T)
    else:
        winners = tl.topk(key, BLOCK_SIZE_T)
    return winners


# Retain bounded alignment specialization for the score row stride.
@triton.jit(do_not_specialize=["GLOBAL_BLOCKS"])
def _local_topk(
    Scores,
    Keys,
    Scratch,
    Lengths,
    QUERY_LEN: tl.constexpr,
    LOCAL_BLOCKS,
    GLOBAL_BLOCKS,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    TOPK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_KEEP: tl.constexpr,
    NARROW: tl.constexpr,
    MID: tl.constexpr,
    WIDE: tl.constexpr,
    NARROW_GROUPS: tl.constexpr,
    MID_GROUPS: tl.constexpr,
    WIDE_GROUPS: tl.constexpr,
    CANDIDATES: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    FILTER_NARROW: tl.constexpr,
):
    """Pack this shard's [heads, tokens, local] scores down to its own top-k keys.

    The id inside each key is the GLOBAL block id, so the receiving merge never
    needs to know which rank a candidate came from.

    Forced blocks are pinned HERE as well as in the merge. Pinning only at the
    merge loses them: a forced block that lost its own shard's top-k never
    arrives to be pinned. Pinning here costs nothing, because the candidate it
    evicts could not have placed globally anyway -- that block lost to k-1 other
    candidates on this rank plus the forced one, and the global top-k is k wide.

    Each row reads its shard in ONE tile of the narrowest width that covers it.
    The step waits on its longest row, and a long row walking its shard tile by
    tile set the whole kernel's time. Only a shard wider than WIDE (past a 1M
    bound at four ranks) still walks.
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
    # This row's own shard length: past it every block is masked, so a short
    # row stops there instead of reading the step's widest shard.
    row_local = tl.minimum(
        (tl.maximum(causal_blocks - RANK, 0) + WORLD - 1) // WORLD, LOCAL_BLOCKS
    )
    s_row = Scores + (head * tokens + row) * LOCAL_BLOCKS
    scratch = Scratch + (head * tokens + row) * CANDIDATES
    if row_local <= NARROW:
        winners = _tile_winners(
            s_row, scratch, 0, LOCAL_BLOCKS, GLOBAL_BLOCKS, causal_blocks,
            local_start, RANK, WORLD, INIT_BLOCKS, NARROW, NARROW_GROUPS,
            CANDIDATES, BLOCK_SIZE_T, FILTER_NARROW,
        )  # fmt: skip
    elif row_local <= MID:
        winners = _tile_winners(
            s_row, scratch, 0, LOCAL_BLOCKS, GLOBAL_BLOCKS, causal_blocks,
            local_start, RANK, WORLD, INIT_BLOCKS, MID, MID_GROUPS,
            CANDIDATES, BLOCK_SIZE_T, True,
        )  # fmt: skip
    else:
        winners = _tile_winners(
            s_row, scratch, 0, LOCAL_BLOCKS, GLOBAL_BLOCKS, causal_blocks,
            local_start, RANK, WORLD, INIT_BLOCKS, WIDE, WIDE_GROUPS,
            CANDIDATES, BLOCK_SIZE_T, True,
        )  # fmt: skip
        for start in tl.range(WIDE, row_local, WIDE):
            tile = _tile_winners(
                s_row, scratch, start, LOCAL_BLOCKS, GLOBAL_BLOCKS, causal_blocks,
                local_start, RANK, WORLD, INIT_BLOCKS, WIDE, WIDE_GROUPS,
                CANDIDATES, BLOCK_SIZE_T, True,
            )  # fmt: skip
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
    BLOCK_PAGE_STRIDE: tl.constexpr,
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
            # Passed by keyword: the parameter sits between `pages_per_block`
            # and `NUM_KV_HEADS`, so a positional list silently shifts the two
            # constexprs after it.
            block_page_stride=BLOCK_PAGE_STRIDE,
            NUM_KV_HEADS=NUM_KV_HEADS,
            BLOCK_SIZE_T=BLOCK_SIZE_T,
        )


# Per layer in a graph on AgentX contexts (S=8, 1M bound, 2048-wide shards):
# 6.0-10.0 us at batch 24-60 against 8.8-15.8 us for 512-wide tiles walked in
# turn, 5.6-6.2 against 7.2-9.5 at batch 8-16 with a long row. A grid under
# about three programs a CU is latency-bound: four warps and an unfiltered
# narrow tile win there (3.3-3.4 against 4.1 us at batch 4-8, short rows),
# two warps and the filter above it.
_LOCAL_TOPK_WIDTHS = (512, 1024, 2048)
_LOCAL_TOPK_FULL_GRID = 768


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
        block_t = triton.next_power_of_2(topk)
        # Every tile must be at least as wide as the top-k it selects:
        # `tl.topk(x[WIDTH], BLOCK_SIZE_T)` cannot compile for WIDTH <
        # BLOCK_SIZE_T. A small shard reaches that -- e.g. 8 local blocks with
        # topk 32. M3 ships sparse_topk_blocks=16, which the 16 floor already
        # covers; the block_t term keeps the function correct for the topk its
        # own guard admits (<= 512).
        full = max(16, block_t, triton.next_power_of_2(local))
        narrow, mid, wide = (
            min(full, max(16, block_t, width)) for width in _LOCAL_TOPK_WIDTHS
        )
        candidates = max(64, 2 * block_t)
        scratch = torch.empty(
            heads * tokens * candidates, dtype=torch.int64, device=scores.device
        )
        full_grid = heads * tokens >= _LOCAL_TOPK_FULL_GRID
        _local_topk[(tokens, heads)](
            scores,
            keys,
            scratch,
            seq_lens,
            QUERY_LEN=max_query_len,
            LOCAL_BLOCKS=local,
            GLOBAL_BLOCKS=global_blocks,
            RANK=rank,
            WORLD=world_size,
            TOPK=topk,
            INIT_BLOCKS=init_blocks,
            LOCAL_KEEP=local_blocks,
            NARROW=narrow,
            MID=mid,
            WIDE=wide,
            # At least BLOCK_SIZE_T groups, or the bound is not a bound.
            NARROW_GROUPS=max(block_t, min(32, narrow)),
            MID_GROUPS=max(block_t, min(32, mid)),
            WIDE_GROUPS=max(block_t, min(64, wide)),
            CANDIDATES=candidates,
            BLOCK_SIZE_T=block_t,
            FILTER_NARROW=full_grid,
            num_warps=2 if full_grid else 4,
        )
    return keys


def merge_candidate_keys(
    keys,
    block_table,
    seq_lens,
    topk,
    init_blocks,
    local_blocks,
    max_query_len,
    pages_per_block=PAGES_PER_SPARSE_BLOCK,
    block_page_stride=PAGES_PER_SPARSE_BLOCK,
):
    """Select the global top-k from gathered candidates for this rank's head.

    ``keys`` holds every shard's candidates for the one head this rank owns
    after the exchange, either already flattened to [tokens, world*topk] or in
    the [world, tokens, topk] layout the all-to-all writes. The 3D form is read
    with a stride rather than copied.

    ``pages_per_block`` / ``block_page_stride`` shape the emitted sparse block
    table as for the native selector: the page-16 SHUFFLE cache by default, one
    page-128 page a block (1, 1) for the MXFP4 KV.
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
    output, _ = _alloc_emit(
        tokens, 1, topk, block_table, True, keys.device, pages_per_block
    )
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
            PAGES_PER_BLOCK=pages_per_block,
            BLOCK_PAGE_STRIDE=block_page_stride,
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
