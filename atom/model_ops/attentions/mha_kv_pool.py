# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The paged KV of a set of MHA layers: what an entry costs, and where it lives.

Separate from the attention backend because a backend also owns per-step
metadata, and that half is per-runner: asking for a second pool by building a
second builder would overwrite the first's `forward_vars`. So anything wanting
a pool of MHA layers can have one -- the model's own, and a draft's.

Two units, and the pool shapes by exactly one. `block_size` is the caller's
block, the unit every view is taken at; an entry is `blocks_per_entry` of them
laid end to end, and exists only because that is what `page_pool` charges per.
Nothing reads at an entry.

A field says what an entry costs; `kv_views` says how those bytes read, and is
the only place the element order may be written -- a second copy of it agrees
on every byte while disagreeing on a shape, which no size check can see.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from atom.model_ops.attentions.pool_layout.entry_arena import (
    EntryField,
    LayerMajorArena,
    carve_layer_major,
    entry_bytes_for,
)

# Bytes an MFMA tile is addressed in.
_TILE_BYTES = 16


def shuffle_pack(kv_dtype: torch.dtype) -> int:
    """Elements of `kv_dtype` per MFMA tile — the trailing `x` of both views."""
    return _TILE_BYTES // kv_dtype.itemsize


def mha_kv_fields(
    *,
    layers: int,
    entry_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: torch.dtype,
) -> list[EntryField]:
    """What K and V of one entry cost, K first as allocated.

    Bytes and not the SHUFFLE shape: the price is the same whichever way the
    tokens lie, and the layout is one expression in `kv_views`.
    """
    per_entry = num_kv_heads * head_dim * entry_tokens * kv_dtype.itemsize
    return [
        EntryField("k", layers, (per_entry,), torch.uint8),
        EntryField("v", layers, (per_entry,), torch.uint8),
    ]


def mha_kv_scale_fields(
    *, layers: int, entry_tokens: int, num_kv_heads: int, kv_dtype: torch.dtype
) -> list[EntryField]:
    """The fp32 dequant scales a quantized cache reads, one per token.

    Empty for a cache wide enough to hold its own values: only an fp8 binder
    hands these to a module and every reader is guarded on that, so a bf16
    entry used to buy two unreachable fp32 planes per layer -- 1/64 of the
    pool at head_dim 128, moved by a P/D transfer as well. The dtype decides
    it because the dtype is why they exist.

    A separate list because they are a separate region, after the cache.
    """
    if kv_dtype.itemsize > 1:
        return []
    per_entry = num_kv_heads * entry_tokens * torch.float32.itemsize
    return [
        EntryField("k_scale", layers, (per_entry,), torch.uint8),
        EntryField("v_scale", layers, (per_entry,), torch.uint8),
    ]


class MhaKvPool:
    """`layers` MHA layers' worth of paged KV, sized and addressed.

    Declared at construction, allocated later: sizing has to answer
    `entry_bytes` before an entry count exists, and that count is what the byte
    budget buys.

    `extra_fields` is how a model rides something else in the same entry --
    MiniMax-M3's indexer key cache, owned by only some of the layers. Declared
    by the caller, because what it holds is the caller's subject; charged and
    allocated by the same two lines as K and V, because where it lives is this
    one's.
    """

    def __init__(
        self,
        *,
        layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        kv_dtype: torch.dtype,
        blocks_per_entry: int = 1,
        extra_fields: Sequence[EntryField] = (),
    ):
        # `block_size` is the caller's block, the only token count the pool
        # shapes anything by. `blocks_per_entry` is how many of them the entry
        # class holds -- a count, not a second block size, and the only reason
        # the pool has it is that `entry_bytes` is what sizing charges. The
        # blocks are contiguous, so the entry is bookkeeping and nothing reads
        # at it.
        if blocks_per_entry < 1:
            raise ValueError(
                f"an entry holds at least one block, not {blocks_per_entry}"
            )
        self.block_size = block_size
        self.x = shuffle_pack(kv_dtype)
        if head_dim % self.x or block_size % self.x:
            raise ValueError(
                f"SHUFFLE packs {self.x} elements of {kv_dtype} per "
                f"{_TILE_BYTES}B tile, which has to divide both head_dim "
                f"{head_dim} and block_size {block_size}"
            )
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_dtype = kv_dtype
        # Every field is priced per entry, and an entry is `blocks_per_entry`
        # blocks. Charging per block instead would multiply `entry_bytes_for`'s
        # 256 B field alignment by the same factor -- 9% of the pool at 8
        # blocks an entry, almost all of it padding on the 64 B scale planes.
        entry_tokens = block_size * blocks_per_entry
        self.cache_fields = mha_kv_fields(
            layers=layers,
            entry_tokens=entry_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_dtype=kv_dtype,
        )
        self.scale_fields = mha_kv_scale_fields(
            layers=layers,
            entry_tokens=entry_tokens,
            num_kv_heads=num_kv_heads,
            kv_dtype=kv_dtype,
        )
        self.extra_fields = [f for f in extra_fields if f.layers]
        # The regions an entry is charged for, in layout order. One list, walked
        # by both the price and the allocation, so a fourth group cannot be
        # added to one and missed by the other. A group a model does not need
        # is empty rather than absent, which keeps the positions fixed.
        self.field_groups = [self.cache_fields, self.scale_fields, self.extra_fields]
        self.entry_bytes = sum(entry_bytes_for(g) for g in self.field_groups)
        self.cache: LayerMajorArena | None = None
        self.scale: LayerMajorArena | None = None
        self.extra: LayerMajorArena | None = None
        self._views: dict[str, torch.Tensor] = {}

    @classmethod
    def from_hf_config(
        cls,
        hf_config,
        *,
        world_size: int,
        block_size: int,
        blocks_per_entry: int = 1,
        kv_dtype: torch.dtype,
    ) -> MhaKvPool:
        """A pool for every attention layer a model config declares.

        Sharded by `ModelRunner._get_num_kv_heads`' rule -- one head per rank
        is the floor -- so a draft's layers divide the way the target's do.

        Raised and not asserted: this reads a model's config, so `python -O`
        would drop it and shard by a floor division instead, silently, at a
        head count nobody chose.
        """
        heads = hf_config.num_key_value_heads
        if heads >= world_size:
            remainder, per_rank = heads % world_size, heads // world_size
        else:
            # Fewer heads than ranks: each is replicated across `world_size //
            # heads` of them, so one head per rank is the floor.
            remainder, per_rank = world_size % heads, 1
        if remainder:
            raise ValueError(
                f"{heads} KV heads and {world_size} ranks do not divide either way"
            )
        return cls(
            layers=hf_config.num_hidden_layers,
            block_size=block_size,
            blocks_per_entry=blocks_per_entry,
            num_kv_heads=per_rank,
            head_dim=hf_config.head_dim,
            kv_dtype=kv_dtype,
        )

    def pool_bytes(self, entries: int) -> int:
        """Bytes the pool takes at `entries` entries -- the size of the region
        `allocate` wants, and what sizing charged for them. Entries and not
        blocks: in here a block is the caller's, and an entry holds several."""
        return self.entry_bytes * entries

    def allocate(self, entries: int, device, buf: torch.Tensor | None = None) -> None:
        """Back the declaration, in memory of its own or a region of the
        runner's paged allocation -- which is also how the decode side of a P/D
        pair reads the pool it was handed. Same groups, so the same layout.

        The per-field views are built once here rather than per bind: each is
        an `as_strided` over the same buffer, and a layer only ever indexes
        into them.
        """
        self.cache, self.scale, self.extra = carve_layer_major(
            self.field_groups, entries, device, buf
        )
        self._views = {
            field.name: arena.view(field.name)
            for arena in (self.cache, self.scale, self.extra)
            if arena is not None
            for field in arena.fields
        }

    def release(self) -> None:
        """Drop the backing, keep the declaration.

        The rollout sleep path frees the pool by dropping the runner's buffers,
        and these views would keep the allocation alive. `allocate` puts it
        back.
        """
        self.cache = self.scale = self.extra = None
        self._views = {}

    def field_view(self, name: str, layer: int, dtype, shape) -> torch.Tensor:
        """One layer's field, retyped and shaped by its owner.

        An entry's bytes are contiguous, so the blocks in it are just the
        leading dim shaped out -- `-1` counts them, and nothing has to know how
        many an entry holds. Public because an `extra_fields` owner reads its
        own field here, in the shape only it knows.
        """
        return self._views[name][layer].view(dtype).view(shape)

    def kv_views(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's `(k, v)`, aliasing the pool, in SHUFFLE.

        The one place the element order is written: the fused writer produces
        it and `cp_mha_gather_cache_kernel` reads it in place, and declaring V
        the other way round sends `_gather_prefix_and_concat_kv` down its
        densifying branch. K's tokens are one head-major span where V's are
        already split by the pack -- which fp8 hides, `x == block_size`
        collapsing V's outer half to one.
        """
        x, nh, hd, bs = self.x, self.num_kv_heads, self.head_dim, self.block_size
        return (
            self.field_view("k", layer, self.kv_dtype, (-1, nh, hd // x, bs, x)),
            self.field_view("v", layer, self.kv_dtype, (-1, nh, bs // x, hd, x)),
        )

    def scale_views(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's `(k_scale, v_scale)`, one fp32 per (kv head, token).

        An fp8 cache only, which is the only kind that declares them: a
        `KeyError` here is a binder asking for a scale this dtype never needed.
        """
        shape = (-1, self.num_kv_heads, self.block_size)
        return (
            self.field_view("k_scale", layer, torch.float32, shape),
            self.field_view("v_scale", layer, torch.float32, shape),
        )

    def region_tensors(self) -> list[tuple[str, torch.Tensor]]:
        """One `(role, tensor)` per (field, layer), in declared field order.

        The granularity a transfer registers, and it cannot be coarser while
        the pool is layer-major: an entry's bytes are `entries` apart, so no
        contiguous range is one entry.

        Named because position is otherwise a region's only identity to the far
        end, and reordering the fields keeps every count and every byte: a
        mismatched pair then transfers K into V and says nothing.
        """
        return [
            (f"{name}.layer_{layer}", view[layer])
            for name, view in self._views.items()
            for layer in range(len(view))
        ]
