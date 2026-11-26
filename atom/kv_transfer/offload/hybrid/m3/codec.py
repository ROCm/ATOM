# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 PAGE bytes with the index cache sharded by block id.

Under ``ATOM_M3_INDEX_SHARD`` a rank's index regions hold the blocks
``b % P == rank`` at unit ``b // P`` (`KVTransferRegion.shard_units`), while the
K/V, their scales and the draft's regions hold every block. A rank's LMCache
chunk of blocks ``ids`` is therefore

    [ len(ids) x the whole regions' bytes ][ own(ids) x the index regions' bytes ]

``own(ids)`` being the chunk's blocks this rank keeps the index of, in block
order (`chunk_nbytes`). A block keeps its lane wherever it is loaded (both ends
allocate logical block j in lane j % P), so a chunk owns the same blocks when
it is stored and when it is loaded, and its object is exactly its own size: at
the lazy policy's one-block chunk, 3 of 4 chunks carry no index at all.
LMCache's token-path store sizes a chunk by its token count alone, so it
allocates the most n consecutive blocks hold (`max_chunk_bytes`); the bytes
past `chunk_nbytes` are never read.

A staging group (several chunks) moves in two launches: every chunk's whole
regions, item-major, then every owned block's index rows. `staging_layout` says
which piece of the staging buffer belongs to which chunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
from atom.kv_transfer.offload.metadata import chunk_bytes


class M3ShardedPageCodec:
    """The `BlockByteCodec` of M3's PAGE regions when some are sharded."""

    def __init__(
        self,
        page_regions: Sequence[KVTransferRegion],
        *,
        num_blocks: int,
        device: torch.device | str,
    ) -> None:
        whole = [r for r in page_regions if r.shard_world == 1]
        sharded = [r for r in page_regions if r.shard_world > 1]
        shards = {(r.shard_rank, r.shard_world) for r in sharded}
        if len(shards) != 1 or not whole:
            raise ValueError(
                f"M3 sharded PAGE codec: {len(whole)} whole regions and shards "
                f"{sorted(shards)}; it needs whole regions and one shard"
            )
        ((self.shard_rank, self.shard_world),) = shards
        if num_blocks % self.shard_world:
            raise ValueError(
                f"{num_blocks} blocks do not split {self.shard_world} ways"
            )
        self.num_blocks = int(num_blocks)
        self._whole = DSV4PageSlotCodec(
            page_regions=whole,
            slot_regions=(),
            num_blocks=num_blocks,
            num_slots=0,
            device=device,
        )
        self._sharded = DSV4PageSlotCodec(
            page_regions=sharded,
            slot_regions=(),
            num_blocks=num_blocks // self.shard_world,
            num_slots=0,
            device=device,
        )
        self.device = self._whole.device

    @property
    def bytes_per_block(self) -> int:
        """What every block of a chunk carries: the whole regions'."""
        return self._whole.bytes_per_block

    @property
    def sharded_bytes_per_block(self) -> int:
        """What a block this rank keeps the index of adds."""
        return self._sharded.bytes_per_block

    def own(self, block_ids: Sequence[int]) -> list[int]:
        """The index units of the blocks of ``block_ids`` this rank keeps."""
        world, rank = self.shard_world, self.shard_rank
        return [int(b) // world for b in block_ids if int(b) % world == rank]

    def chunk_nbytes(self, block_ids: Sequence[int]) -> int:
        """Bytes of the chunk of ``block_ids``: what it stores and loads."""
        return (
            len(block_ids) * self.bytes_per_block
            + len(self.own(block_ids)) * self.sharded_bytes_per_block
        )

    def max_chunk_bytes(self, blocks: int) -> int:
        """The most a chunk of ``blocks`` consecutive logical blocks holds."""
        return chunk_bytes(
            blocks,
            self.bytes_per_block,
            self.sharded_bytes_per_block,
            self.shard_world,
        )

    @property
    def has_fused_chunk_major_staging(self) -> bool:
        return (
            self._whole.has_fused_chunk_major_staging
            and self._sharded.has_fused_chunk_major_staging
        )

    @staticmethod
    def _flat(group: Sequence[Sequence[int]]) -> list[int]:
        return [int(b) for chunk in group for b in chunk]

    def staging_layout(
        self, block_id_groups: Sequence[Sequence[int]]
    ) -> tuple[list[int], list[list[int]]]:
        """The staging buffer of a group of chunks as pieces: their sizes in
        buffer order, and per chunk the indices of its pieces in its own byte
        order (its whole regions' bytes, then its index rows if it owns any)."""
        whole, shard = self.bytes_per_block, self.sharded_bytes_per_block
        sizes = [len(ids) * whole for ids in block_id_groups]
        pieces = [[i] for i in range(len(sizes))]
        for chunk, ids in zip(pieces, block_id_groups):
            owned = len(self.own(ids))
            if owned:
                chunk.append(len(sizes))
                sizes.append(owned * shard)
        return sizes, pieces

    def _plans(self, ids: Sequence[int]):
        whole = self._whole.page_plan(ids)
        own = self.own(ids)
        sharded = (
            self._sharded.page_plan(own, buffer_offset=whole.required_buffer_bytes)
            if own
            else None
        )
        return whole, sharded

    def gpu_to_chunk_major_device_buffer(
        self,
        device_buf: torch.Tensor,
        block_id_groups: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        whole, sharded = self._plans(self._flat(block_id_groups))
        self._whole.gather(whole, device_buf, stream=stream)
        if sharded is not None:
            self._sharded.gather(sharded, device_buf, stream=stream)

    def chunk_major_device_buffer_to_gpu(
        self,
        device_buf: torch.Tensor,
        block_id_groups: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        whole, sharded = self._plans(self._flat(block_id_groups))
        self._whole.scatter(device_buf, whole, stream=stream)
        if sharded is not None:
            self._sharded.scatter(device_buf, sharded, stream=stream)

    # One block-id upload per transfer, as `DSV4PageSlotCodec`'s: each part's
    # own prepared owner, the sharded part read past the group's whole bytes.
    def prepare_block_id_groups(
        self,
        group_block_ids: Sequence[Sequence[Sequence[int]]],
        *,
        device: torch.device | str,
        stream: torch.cuda.Stream,
    ) -> _PreparedShardedGroups:
        flat = [self._flat(group) for group in group_block_ids]
        return _PreparedShardedGroups(
            self,
            self._whole.prepare_block_id_groups(flat, device=device, stream=stream),
            self._sharded.prepare_block_id_groups(
                [self.own(ids) for ids in flat], device=device, stream=stream
            ),
            tuple(len(ids) * self.bytes_per_block for ids in flat),
        )

    def _prepared(self, device_buf, owner, group_index, *, gather, stream):
        if not isinstance(owner, _PreparedShardedGroups) or owner.codec is not self:
            raise ValueError("prepared block IDs belong to a different codec")
        whole_bytes = owner.whole_bytes[group_index]
        if gather:
            self._whole.gpu_to_chunk_major_device_buffer_prepared(
                device_buf, owner.whole, group_index, stream=stream
            )
            self._sharded.gpu_to_chunk_major_device_buffer_prepared(
                device_buf[whole_bytes:], owner.sharded, group_index, stream=stream
            )
        else:
            self._whole.chunk_major_device_buffer_to_gpu_prepared(
                device_buf, owner.whole, group_index, stream=stream
            )
            self._sharded.chunk_major_device_buffer_to_gpu_prepared(
                device_buf[whole_bytes:], owner.sharded, group_index, stream=stream
            )

    def gpu_to_chunk_major_device_buffer_prepared(
        self, device_buf, owner, group_index, *, stream=None
    ) -> None:
        self._prepared(device_buf, owner, group_index, gather=True, stream=stream)

    def chunk_major_device_buffer_to_gpu_prepared(
        self, device_buf, owner, group_index, *, stream=None
    ) -> None:
        self._prepared(device_buf, owner, group_index, gather=False, stream=stream)


@dataclass(frozen=True)
class _PreparedShardedGroups:
    """`M3ShardedPageCodec.prepare_block_id_groups`' owner: both parts' owners
    and each group's whole-region bytes (where its index rows start)."""

    codec: M3ShardedPageCodec
    whole: object
    sharded: object
    whole_bytes: tuple[int, ...]

    @property
    def group_count(self) -> int:
        return len(self.whole_bytes)

    @property
    def upload_count(self) -> int:
        return int(self.whole.upload_count) + int(self.sharded.upload_count)


# ported from main (#2752): the codec id for M3's complete-PAGE bytes as
# `pure_page_codec` builds them (pool -> field -> layer, then the draft's
# `merge_pages`). Bump when that order or the region set changes.
M3_CODEC_ID = "m3-page-regions"
