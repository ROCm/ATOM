# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The paged KV of a set of MLA layers: what a block costs, and where it lives.

The MHA pool's sibling, and separate from the attention backend for the same
reason (`mha_kv_pool`): a backend also owns per-step metadata, and that half is
per-runner.

MLA packs k_c and k_pe into one row and absorbs V into the latent, so a block
is one tensor rather than four -- no split K/V, no dequantization scales. The
sparse variants add a second: an indexer key cache owned by only some layers,
holding one row per token or, where the indexer pools, fewer. Both are fields;
their differing layer counts and row counts are what `EntryField` already says.

An entry here is a *scheduler* block, which is the index space `page_pool`
charges. That is not always the backend's own page: MLA usually pages at size 1
(`ATOM_MLA_PAGE_SIZE`), so `block_ratio` rows of the allocation make up one
entry. Keeping the entry at the scheduler block is what lets a row of
`region_tensors` be the unit a transfer registers, with no `block_ratio` factor
applied after the fact.
"""

from __future__ import annotations

import torch

from atom.model_ops.attentions.pool_layout.entry_arena import (
    EntryField,
    LayerMajorArena,
    carve_layer_major,
    entry_bytes_for,
)
from atom.model_ops.indexer.layout import PRESHUFFLE
from atom.model_ops.sparse_indexer_fp4 import fp4_index_block_shapes

# One NVFP4 MLA page (64 tokens) in AITER's native layout: C codes and the
# RoPE / residual / scale aux plane. A scheduler block is one page.
NVFP4_PAGE_TOKENS = 64
NVFP4_NOPE_BYTES = 16384
NVFP4_AUX_BYTES = 8192


class MlaKvPool:
    """`layers` MLA layers' worth of paged KV, sized and addressed.

    Declared at construction, allocated later: sizing has to answer
    `entry_bytes` before a block count exists, and the block count is what the
    byte budget buys.

    The indexer cache is a second region when there is one at all, so a model
    without indexers declares no field for it and pays nothing.
    """

    def __init__(
        self,
        *,
        layers: int,
        block_size: int,
        entry_dim: int,
        kv_dtype: torch.dtype,
        index_layers: int = 0,
        index_rows_per_block: int = 0,
        index_dim: int = 0,
        index_dtype: torch.dtype | None = None,
        index_head_dim: int = 0,
        index_fp4: bool = False,
        index_layout: str = PRESHUFFLE,
        nvfp4: bool = False,
    ):
        self.layers = layers
        self.block_size = block_size
        self.entry_dim = entry_dim
        self.nvfp4 = nvfp4
        if nvfp4:
            if block_size != NVFP4_PAGE_TOKENS:
                raise ValueError(
                    f"NVFP4 MLA pages hold {NVFP4_PAGE_TOKENS} tokens; the "
                    f"scheduler block size is {block_size}."
                )
            # Two planes per layer instead of the packed latent row.
            self.cache_fields = [
                EntryField("kv_nope", layers, (NVFP4_NOPE_BYTES,), torch.uint8),
                EntryField("kv_aux", layers, (NVFP4_AUX_BYTES,), torch.uint8),
            ]
        else:
            self.cache_fields = [
                EntryField("kv", layers, (block_size, entry_dim), kv_dtype)
            ]
        self.index_fields: list[EntryField] = []
        if index_layers and index_fp4:
            data_shape, scale_shape = fp4_index_block_shapes(
                index_rows_per_block, index_head_dim, layout=index_layout
            )
            self.index_fields = [
                EntryField("index", index_layers, data_shape, torch.uint8),
                EntryField("index_scale", index_layers, scale_shape, torch.uint8),
            ]
        elif index_layers:
            self.index_fields = [
                EntryField(
                    "index",
                    index_layers,
                    (index_rows_per_block, index_dim),
                    index_dtype,
                )
            ]
        self.index_dim = index_dim
        # The regions a block is charged for, in layout order; one list for the
        # price and the allocation both.
        self.field_groups = [self.cache_fields, self.index_fields]
        self.entry_bytes = sum(entry_bytes_for(g) for g in self.field_groups)
        self.cache: LayerMajorArena | None = None
        self.index: LayerMajorArena | None = None
        self._views: dict[str, torch.Tensor] = {}

    def pool_bytes(self, entries: int) -> int:
        """Bytes the pool takes at `entries` entries -- the size of the region
        `allocate` wants, and what sizing charged for them."""
        return self.entry_bytes * entries

    def allocate(self, entries: int, device, buf: torch.Tensor | None = None) -> None:
        """Back the declaration, in memory of its own or a region of the
        runner's paged allocation -- the MHA pool's `allocate` exactly."""
        self.cache, self.index = carve_layer_major(
            self.field_groups, entries, device, buf
        )
        self._views = {
            field.name: arena.view(field.name)
            for arena in (self.cache, self.index)
            if arena is not None
            for field in arena.fields
        }

    def release(self) -> None:
        """Drop the backing, keep the declaration."""
        self.cache = self.index = None
        self._views = {}

    def layer(self, name: str, layer: int) -> torch.Tensor:
        """One layer's slice of a field, `[blocks, rows_per_block, dim]`.

        Left at that shape rather than the one the kernels bind: MLA addresses
        its cache by row, and how many rows a block holds is the backend's
        paging, not the pool's. The slice is contiguous, so the backend's
        reshape is a view.
        """
        return self._views[name][layer]

    def field(self, name: str) -> torch.Tensor:
        """Every layer of a field, `[layers, blocks, *shape]`."""
        return self._views[name]

    def region_tensors(self) -> list[tuple[str, torch.Tensor]]:
        """One `(role, tensor)` per (field, layer), in declared field order.

        A row of each is one scheduler block, which is the unit a transfer
        registers -- so `stride(0)` is already the bytes per block and needs no
        `block_ratio` applied. Named for `MhaKvPool.region_tensors`' reason.
        """
        return [
            (f"{name}.layer_{layer}", view[layer])
            for name, view in self._views.items()
            for layer in range(len(view))
        ]
