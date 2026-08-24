# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block/token mapping for a non-DCP prefill node -> DCP decode node KV push.

The prefill (producer) side stores KV contiguously: global token ``g`` lives in
block ``g // block_size`` at offset ``g % block_size``. The decode (consumer)
side runs DCP, so one block-table entry is a *virtual block* covering
``block_size * dcp_size`` global tokens and each rank physically holds only its
own share of them. Pushing KV across therefore needs a relayout, and the shape
of that relayout differs per cache region:

``SHARDED`` (kv_cache)
    Tokens are interleave-sharded across ranks in groups of ``interleave_size``
    (S): global token ``g`` lives on rank ``(g // S) % W`` at local index
    ``(g // (S*W)) * S + g % S``. Rank ``r``'s virtual block ``b``, offset ``j``
    therefore pulls global position ``dcp_global_pos(b * block_size + j, r)``.
    S == block_size collapses to "dst block b <- src block b*W + r", one whole
    block per descriptor; S == 1 gives one descriptor per token.

``REPLICATED`` (index_cache, once the indexer stops being DCP-split)
    Every rank holds the full ``block_size * dcp_size`` tokens of a virtual
    block, laid out in plain sequential order, so virtual block ``v`` is the
    concatenation of source blocks ``[v*W, (v+1)*W)``. Independent of both the
    rank and the interleave size, and the destination ``unit_bytes`` is ``W``
    times the source's.

Both kinds produce the same descriptor form (a source token range copied to a
destination token range), so the caller turns either into addresses with

    src_addr = src_base + src_block_ids[sb] * src_unit_bytes + so * token_bytes
    dst_addr = dst_base + dst_block_ids[db] * dst_unit_bytes + dt * token_bytes
    size     = n * token_bytes

where ``token_bytes = src_unit_bytes // block_size``. Addressing a single token
is only possible when the region stores a token's bytes contiguously, which
holds for the MLA ``[num_slots, 1, 576]`` layout but not for the MHA
``[blocks, heads, head_dim/x, block_size, x]`` one; callers must gate
``interleave_size < block_size`` on the former.

Pure numpy, no torch/GPU dependency, so the mapping is unit-testable on CPU.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class DCPShardKind(enum.Enum):
    """How a cache region is distributed over the DCP ranks."""

    SHARDED = "sharded"
    REPLICATED = "replicated"


@dataclass(frozen=True)
class DCPBlockPlan:
    """Token ranges to copy, in source/destination *block-local* coordinates.

    ``src_block``/``dst_block`` index into the caller's block-id lists (they are
    ordinals, not physical block ids); ``src_token``/``dst_token`` are token
    offsets inside those blocks. Entry ``i`` copies ``num_tokens[i]`` tokens.
    """

    kind: DCPShardKind
    block_size: int
    dst_block_tokens: int
    src_block: np.ndarray
    src_token: np.ndarray
    dst_block: np.ndarray
    dst_token: np.ndarray
    num_tokens: np.ndarray

    def __len__(self) -> int:
        return int(self.src_block.shape[0])


def expected_dst_blocks(num_src_blocks: int, dcp_size: int) -> int:
    """Virtual blocks a DCP consumer needs for ``num_src_blocks`` source blocks.

    The consumer allocates ``ceil(num_tokens / (block_size * W))`` entries and
    the producer holds ``ceil(num_tokens / block_size)``, so the two counts are
    related by a plain ``ceil`` division regardless of where the sequence ends.
    """
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    return -(-int(num_src_blocks) // int(dcp_size))


def resolve_shard_kind(
    producer_unit_bytes: int, consumer_unit_bytes: int, dcp_size: int
) -> DCPShardKind:
    """Classify a region from the two sides' per-block byte counts.

    Equal widths mean the consumer stores one block's worth of tokens per entry
    (sharded); a ``dcp_size``-times wider consumer entry means it stores the
    whole virtual block (replicated). Anything else is a layout the planner
    cannot express, and silently guessing would corrupt the cache, so it raises.
    """
    if consumer_unit_bytes == producer_unit_bytes:
        return DCPShardKind.SHARDED
    if consumer_unit_bytes == producer_unit_bytes * dcp_size:
        return DCPShardKind.REPLICATED
    raise ValueError(
        "cannot classify DCP region: producer unit_bytes="
        f"{producer_unit_bytes}, consumer unit_bytes={consumer_unit_bytes}, "
        f"dcp_size={dcp_size} (expected equal or {dcp_size}x)"
    )


def _validate(block_size: int, dcp_size: int, dcp_rank: int) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank {dcp_rank} out of range for dcp_size {dcp_size}")


def plan_sharded(
    num_dst_blocks: int,
    num_src_blocks: int,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> DCPBlockPlan:
    """Plan the KV (interleave-sharded) regions for one DCP rank.

    Emits ``block_size // interleave_size`` runs per destination block. Runs
    whose source block is past the end of the producer's list are dropped: the
    block manager sizes every rank's table from rank 0's share, so ranks above
    it can own a trailing virtual block that has no source tokens at all.
    """
    _validate(block_size, dcp_size, dcp_rank)
    S = int(interleave_size)
    if S < 1 or block_size % S:
        raise ValueError(
            f"interleave_size {S} must be >= 1 and divide block_size {block_size}"
        )
    runs_per_block = block_size // S
    n_dst = max(int(num_dst_blocks), 0)

    dst_block = np.repeat(np.arange(n_dst, dtype=np.int64), runs_per_block)
    dst_token = np.tile(np.arange(0, block_size, S, dtype=np.int64), n_dst)
    # Every run starts at an S-group boundary, so dcp_global_pos reduces to the
    # group form and the whole run lands inside one source block.
    local = dst_block * block_size + dst_token
    g = ((local // S) * dcp_size + dcp_rank) * S
    src_block, src_token = np.divmod(g, block_size)

    keep = src_block < int(num_src_blocks)
    return DCPBlockPlan(
        kind=DCPShardKind.SHARDED,
        block_size=block_size,
        dst_block_tokens=block_size,
        src_block=src_block[keep],
        src_token=src_token[keep],
        dst_block=dst_block[keep],
        dst_token=dst_token[keep],
        num_tokens=np.full(int(keep.sum()), S, dtype=np.int64),
    )


def plan_replicated(
    num_dst_blocks: int,
    num_src_blocks: int,
    block_size: int,
    dcp_size: int,
) -> DCPBlockPlan:
    """Plan the replicated (full-copy) regions; identical on every DCP rank."""
    _validate(block_size, dcp_size, 0)
    n_dst = max(int(num_dst_blocks), 0)

    dst_block = np.repeat(np.arange(n_dst, dtype=np.int64), dcp_size)
    sub = np.tile(np.arange(dcp_size, dtype=np.int64), n_dst)
    src_block = dst_block * dcp_size + sub

    keep = src_block < int(num_src_blocks)
    return DCPBlockPlan(
        kind=DCPShardKind.REPLICATED,
        block_size=block_size,
        dst_block_tokens=block_size * dcp_size,
        src_block=src_block[keep],
        src_token=np.zeros(int(keep.sum()), dtype=np.int64),
        dst_block=dst_block[keep],
        dst_token=(sub * block_size)[keep],
        num_tokens=np.full(int(keep.sum()), block_size, dtype=np.int64),
    )


def build_token_runs(
    plan: DCPBlockPlan,
    src_block_ids,
    dst_block_ids,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve a plan against real block ids and coalesce contiguous runs.

    Returns ``(src_token_offset, dst_token_offset, length)`` in token units,
    flat over each side's whole region (block id * tokens-per-block + offset).
    Merging matters because both sides address by physical block id: a virtual
    block's ``dcp_size`` replicated sources, and consecutively allocated blocks
    in general, collapse into one descriptor whenever the allocator handed out
    consecutive ids, which is the common case and keeps the RDMA batch small.
    """
    src_ids = np.asarray(src_block_ids, dtype=np.int64)
    dst_ids = np.asarray(dst_block_ids, dtype=np.int64)
    if len(plan) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    if plan.src_block.max() >= src_ids.shape[0]:
        raise IndexError("plan references a source block beyond src_block_ids")
    if plan.dst_block.max() >= dst_ids.shape[0]:
        raise IndexError("plan references a destination block beyond dst_block_ids")

    src = src_ids[plan.src_block] * plan.block_size + plan.src_token
    dst = dst_ids[plan.dst_block] * plan.dst_block_tokens + plan.dst_token
    length = plan.num_tokens

    contiguous = (src[1:] == src[:-1] + length[:-1]) & (
        dst[1:] == dst[:-1] + length[:-1]
    )
    starts = np.concatenate(([True], ~contiguous))
    group = np.cumsum(starts) - 1
    merged_len = np.bincount(group, weights=length).astype(np.int64)
    return src[starts], dst[starts], merged_len
