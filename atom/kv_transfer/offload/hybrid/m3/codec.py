# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 PAGE bytes with the index cache sharded by block id.

Under ``ATOM_M3_INDEX_SHARD`` a rank's index regions hold the blocks
``b % P == rank`` at unit ``b // P`` (`KVTransferRegion.shard_units`), while the
K/V, their scales and the draft's regions hold every block. A rank's LMCache
chunk of n blocks is therefore

    [ n x the whole regions' bytes ][ ceil(n / P) x the index regions' bytes ]

the second part holding the chunk's own blocks' index rows in block order, the
slots past them unused. LMCache sizes a chunk by its token count alone, so it
reserves the most any n consecutive blocks hold (`metadata.chunk_bytes`): a
chunk of P blocks or more stores 1/P of the index, a 1-block chunk (the lazy
policy) as much as a replicated one. A block keeps its lane wherever it is
loaded (both ends allocate logical block j in lane j % P), so the chunk's own
blocks on load are the ones it stored.

The bytes move through two `DSV4PageSlotCodec`s, one per part, with the plans
`chunk_plans` lays out: two launches a chunk.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4CopyPlan, DSV4PageSlotCodec
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

    def chunk_bytes(self, blocks: int) -> int:
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

    def chunk_plans(
        self, block_id_groups: Sequence[Sequence[int]]
    ) -> list[tuple[DSV4PageSlotCodec, DSV4CopyPlan]]:
        """The copies of chunks ``block_id_groups`` (one list of block ids a
        chunk), laid out back to back as the connector slices them."""
        plans: list[tuple[DSV4PageSlotCodec, DSV4CopyPlan]] = []
        offset = 0
        world, rank = self.shard_world, self.shard_rank
        for group in block_id_groups:
            ids = [int(b) for b in group]
            plans.append(
                (self._whole, self._whole.page_plan(ids, buffer_offset=offset))
            )
            own = [b // world for b in ids if b % world == rank]
            if own:
                sharded_at = offset + len(ids) * self.bytes_per_block
                plans.append(
                    (
                        self._sharded,
                        self._sharded.page_plan(own, buffer_offset=sharded_at),
                    )
                )
            offset += self.chunk_bytes(len(ids))
        return plans

    def gpu_to_chunk_major_device_buffer(
        self,
        device_buf: torch.Tensor,
        block_id_groups: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        for codec, plan in self.chunk_plans(block_id_groups):
            codec.gather(plan, device_buf, stream=stream)

    def chunk_major_device_buffer_to_gpu(
        self,
        device_buf: torch.Tensor,
        block_id_groups: Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        for codec, plan in self.chunk_plans(block_id_groups):
            codec.scatter(device_buf, plan, stream=stream)
