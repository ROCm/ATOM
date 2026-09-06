# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The paged KV of a set of MHA layers: what a block costs, and where it lives.

Separate from the attention backend because a backend also owns per-step
metadata, and that half is per-runner: asking for a second pool by building a
second builder would overwrite the first's `forward_vars`. This half takes five
numbers and owns two arenas, so anything that needs a pool of MHA layers can
have one -- the model's own layers, and a draft's when that draft's flavor
resolves here.

The shapes are the interface. `EntryField.shape` is what one (layer, block)
pair holds, which for a KV cache *is* the element order, and the reader picks
its path off `v_cache.ndim`: a V declared non-transposed still agrees with
every byte count in the tree and quietly relays the whole pool out once per
chunked-prefill forward. That is what four hand-written copies of these shapes
cost, and why there is one.
"""

from __future__ import annotations

import torch

from atom.model_ops.attentions.pool_layout.entry_arena import (
    EntryField,
    LayerMajorArena,
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
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: torch.dtype,
) -> list[EntryField]:
    """K and V of one block, in the SHUFFLE order the fused writer produces
    and `cp_mha_gather_cache_kernel` reads in place. K first, as allocated."""
    x = shuffle_pack(kv_dtype)
    if head_dim % x or block_size % x:
        raise ValueError(
            f"SHUFFLE packs {x} elements of {kv_dtype} per {_TILE_BYTES}B tile, "
            f"which has to divide both head_dim {head_dim} and block_size "
            f"{block_size}"
        )
    return [
        EntryField("k", layers, (num_kv_heads, head_dim // x, block_size, x), kv_dtype),
        EntryField("v", layers, (num_kv_heads, block_size // x, head_dim, x), kv_dtype),
    ]


def mha_kv_scale_fields(
    *, layers: int, block_size: int, num_kv_heads: int
) -> list[EntryField]:
    """The fp32 dequantization scales an fp8 cache reads, one per token.

    A separate list because they are a separate allocation; the two merge when
    the allocations do. Read per (kv_head, token), not as tiles, so no `x`.
    """
    return [
        EntryField("k_scale", layers, (num_kv_heads, block_size), torch.float32),
        EntryField("v_scale", layers, (num_kv_heads, block_size), torch.float32),
    ]


class MhaKvPool:
    """`layers` MHA layers' worth of paged KV, sized and addressed.

    Declared at construction, allocated later: sizing has to answer
    `entry_bytes` before a block count exists, and the block count is what the
    byte budget buys.

    A sparse-attention model (MiniMax-M3) rides an indexer key cache in the
    same block, owned by only some of the layers. It is a field like the rest,
    so it is charged for and allocated by the same two lines -- which is the
    point: the two used to be a formula in sizing and a `torch.zeros` in
    allocation, reading *different* block-count attributes.
    """

    def __init__(
        self,
        *,
        layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        kv_dtype: torch.dtype,
        index_layers: int = 0,
        index_dim: int = 0,
        index_dtype: torch.dtype | None = None,
    ):
        self.block_size = block_size
        self.index_dim = index_dim
        self.cache_fields = mha_kv_fields(
            layers=layers,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_dtype=kv_dtype,
        )
        self.scale_fields = mha_kv_scale_fields(
            layers=layers, block_size=block_size, num_kv_heads=num_kv_heads
        )
        # One indexer row per token of the *scheduler* block, which is the
        # entry `page_pool` charges. Sizing it at the backend's page instead
        # undercharged by `block_ratio`.
        self.index_fields = (
            [EntryField("index", index_layers, (block_size, index_dim), index_dtype)]
            if index_layers
            else []
        )
        # A block pays for the scales whatever the cache dtype: they are a
        # second allocation, not a second configuration.
        self.entry_bytes = sum(
            entry_bytes_for(fields)
            for fields in (self.cache_fields, self.scale_fields, self.index_fields)
        )
        self.cache: LayerMajorArena | None = None
        self.scale: LayerMajorArena | None = None
        self.index: LayerMajorArena | None = None
        self._views: dict[str, torch.Tensor] = {}

    @classmethod
    def from_hf_config(
        cls, hf_config, *, world_size: int, block_size: int, kv_dtype: torch.dtype
    ) -> MhaKvPool:
        """A pool for every attention layer a model config declares.

        Sharded by `ModelRunner._get_num_kv_heads`' rule, asserts included, so
        a draft's layers divide exactly the way the target's do.
        """
        heads = hf_config.num_key_value_heads
        if heads >= world_size:
            assert heads % world_size == 0
            per_rank = heads // world_size
        else:
            assert world_size % heads == 0
            per_rank = 1
        return cls(
            layers=hf_config.num_hidden_layers,
            block_size=block_size,
            num_kv_heads=per_rank,
            head_dim=hf_config.head_dim,
            kv_dtype=kv_dtype,
        )

    def allocate(
        self, blocks: int, device, cache_buf=None, scale_buf=None, index_buf=None
    ) -> None:
        """Back the declaration, with a fresh allocation or an imported one.

        One expression for both: the decode side of a P/D pair receives the
        pool as an IPC handle and has to read it at the layout the prefill side
        wrote it at. Handing those buffers in is also the check -- an arena
        refuses one that does not fit its own declaration.

        The per-field views are built once here rather than per bind: each is
        an `as_strided` over the same buffer, and a layer only ever indexes
        into them.
        """
        self.cache = LayerMajorArena(self.cache_fields, blocks, device, buf=cache_buf)
        self.scale = LayerMajorArena(self.scale_fields, blocks, device, buf=scale_buf)
        self.index = (
            LayerMajorArena(self.index_fields, blocks, device, buf=index_buf)
            if self.index_fields
            else None
        )
        self._views = {
            field.name: arena.view(field.name)
            for arena in (self.cache, self.scale, self.index)
            if arena is not None
            for field in arena.fields
        }

    def release(self) -> None:
        """Drop the backing, keep the declaration.

        The rollout sleep path frees the pool by dropping the runner's buffers,
        and these views would keep the allocation alive. `allocate` puts it
        back.
        """
        self.cache = self.scale = self.index = None
        self._views = {}

    def kv_views(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's `(k, v)`, aliasing the pool."""
        return self._views["k"][layer], self._views["v"][layer]

    def scale_views(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's `(k_scale, v_scale)`. Allocated for every cache dtype,
        read only by an fp8 one."""
        return self._views["k_scale"][layer], self._views["v_scale"][layer]

    def index_view(self, layer: int) -> torch.Tensor:
        """One indexer layer's rows, `[blocks, block_size, index_dim]`.

        `layer` counts the layers that *own* an indexer, not the model's; the
        caller assigns those in bind order the way it does for every other
        compact per-layer axis. Left at the scheduler block's shape -- how many
        of the backend's own pages that is stays with the backend.
        """
        return self._views["index"][layer]

    def region_tensors(self) -> list[torch.Tensor]:
        """One tensor per (field, layer), in declared field order.

        The granularity a transfer registers, and it cannot be coarser while
        the pool is layer-major: a block's bytes are `blocks` apart, so no
        contiguous range is one block. That changes when the pool does.
        """
        return [
            view[layer] for view in self._views.values() for layer in range(len(view))
        ]
