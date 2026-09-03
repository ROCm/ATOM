# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Where a checkpoint image's bytes live in the MLA paged pool.

The destination side of a K3 PAGE-copy checkpoint: given logical block (PAGE
unit) ids, it gives the addresses and tensor views of the KV pool a checkpoint
copy lands on. `_KimiMLAGDNCommon` mixes it in over `GDNStateMixin`.

Kept aiter/GPU-import-free and pure over `self.model_runner` so this geometry --
which decides where every checkpoint byte goes -- can be unit-tested on a
GPU-less runner (the K3 builder's own module is `importorskip`ped on CI).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class PageUnitGeometryMixin:
    """PAGE-unit address/view arithmetic for the MLA paged pool.

    Reads only `self.model_runner` (`kv_cache`, `block_size`, `state_runtime`)
    and caches the pool-derived regions on `self._page_unit_region_cache`. No
    aiter, no CUDA -- the concrete K3 builder supplies the pool.
    """

    def _page_unit_index_cache(self) -> torch.Tensor | None:
        """The sparse-indexer cache that rides this PAGE unit, or `None`.

        Plain MLA owns only its KV rows, so the generic geometry has no index
        cache. A sparse-indexer model (`runner.is_deepseek_v32`) overrides this
        to return `runner.index_cache`, whose layers `_page_unit_regions`
        appends after the MLA rows.
        """
        return None

    def _page_unit_regions(self) -> tuple[np.ndarray, np.ndarray]:
        """Base address and per-unit stride of every region a PAGE id owns.

        `kv_cache` is `(rows, physical_blocks, physical_block_size, entry)`: a
        block owns one contiguous region per row, rows a fixed stride apart.
        Affine in the block id and a pool property, so it is computed once.

        The units are the trap: `unit_ids` carries **logical** block ids while
        the tensor is shaped in **physical** blocks (K3's `block_ratio` is 128),
        so a region is `runner.block_size` tokens wide, not `physical_block_size`.
        The granularity assertion below is the one relation that cannot hold if
        the two are confused -- a startup error, not 127 blocks of scrambled state.
        """
        runner = self.model_runner
        cache = runner.kv_cache
        index_cache = self._page_unit_index_cache()
        # Key on the full layout, not `data_ptr()` alone. `model_runner.kv_cache`
        # is reassigned in two places -- the P/D IPC import
        # (`model_runner.py:4472`) and the rollout sleep/wake path
        # (`rollout/memory_manager.py` deletes it, then `_resume_kv_cache` ->
        # `allocate_kv_cache` with a possibly *reduced* block count). Address
        # equality does not imply geometry equality: a same-start-address
        # reallocation keeps `data_ptr()` but changes `stride(0)`, so `row_stride`
        # (and thus every base for row > 0) goes stale while `rows`/`region`
        # survive -- and the granularity assertion below could not catch it even
        # if the early return did not skip it, since it never involves
        # `row_stride`. Keying on `(data_ptr, shape, stride, element_size)` makes
        # any of those changing a cache miss that rebuilds. Mirrors the DSV4
        # sibling (`deepseek_v4_attn.py`), which keys on the same evidence.
        #
        # A sparse-indexer model (`runner.is_deepseek_v32`) rides its index cache
        # in the same PAGE unit, appended after the MLA rows. Its layout is part
        # of the identity too -- key on it (or `None`) so a reallocated or
        # resized index cache is a miss that rebuilds, exactly like the MLA pool.
        index_owner = (
            None
            if index_cache is None
            else (
                index_cache.data_ptr(),
                tuple(index_cache.shape),
                tuple(index_cache.stride()),
                index_cache.element_size(),
            )
        )
        owner = (
            cache.data_ptr(),
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.element_size(),
            index_owner,
        )
        cached = getattr(self, "_page_unit_region_cache", None)
        if cached is not None and cached[0] == owner:
            return cached[1]

        if not cache.is_contiguous():
            raise RuntimeError("the MLA pool must be contiguous to be copied")
        item = cache.element_size()
        entry = cache.shape[3]
        rows = cache.shape[0]
        # One logical block's bytes inside one row.
        region = runner.block_size * entry * item
        row_stride = cache.stride(0) * item

        base_addr = cache.data_ptr()
        bases = [base_addr + row * row_stride for row in range(rows)]
        sizes = [region] * rows

        index_layers = 0
        index_region = 0
        if index_cache is not None:
            if not index_cache.is_contiguous():
                raise RuntimeError("the index cache must be contiguous to be copied")
            index_item = index_cache.element_size()
            index_layers = index_cache.shape[0]
            layer_stride = index_cache.stride(0) * index_item
            index_region = index_cache.stride(1) * index_item
            bases += [
                index_cache.data_ptr() + layer * layer_stride
                for layer in range(index_layers)
            ]
            sizes += [index_region] * index_layers

        runtime = getattr(runner, "state_runtime", None)
        spec = None if runtime is None else runtime.checkpoint_spec
        owned = sum(sizes)
        page_unit_bytes = spec.page_unit_bytes if spec is not None else owned
        if owned != page_unit_bytes:
            gives = f"{rows} rows x {region} B"
            if index_cache is not None:
                gives += f" + {index_layers} index layers x {index_region} B"
            raise RuntimeError(
                f"a PAGE unit is {page_unit_bytes} B but this pool gives a "
                f"logical block {gives} = {owned} B; the two disagree about "
                "block granularity"
            )
        regions = (
            np.array(bases, dtype=np.int64),
            np.array(sizes, dtype=np.int64),
        )
        self._page_unit_region_cache = (owner, regions)
        return regions

    def _page_unit_bases(self, unit_ids: Sequence[Sequence[int]]) -> np.ndarray:
        """Start address of every destination segment, one row per image.

        `unit_ids` is `(images, units_per_checkpoint)`. A unit's regions are
        each at `base + id * stride`, so one image's worth is an outer product
        and a batch's is the same product with an image axis in front. Unit
        major, region minor -- the order `_checkpoint_copy_plan` built the
        destination stream in.
        """
        base, stride = self._page_unit_regions()
        ids = np.asarray(unit_ids, dtype=np.int64)
        return (base + ids[..., None] * stride).reshape(len(ids), -1)

    def _page_unit_stream_sizes(self, units: int) -> np.ndarray:
        """Bytes in each destination segment of an image of `units` units."""
        return np.tile(self._page_unit_regions()[1], units)

    def page_unit_views(self, unit_ids: Sequence[int]) -> list[torch.Tensor]:
        """Tensor-view counterpart of `_page_unit_regions`, for the CPU tier.

        `_page_unit_regions` gives raw addresses for the Triton descriptor; the
        LMCache packer takes `list[torch.Tensor]`, so the same bytes need naming
        as views. Order is the contract: unit major, row minor, the same ravel
        as `_page_unit_bases` (cross-build safety is `layout_id` in
        `StateByteCodec.key`, not this order).

        `unit_ids` carries *logical* block ids while `kv_cache` is shaped in
        *physical* ones (K3's ratio is 128); indexing by `unit * block_size` is
        that conversion, range-checked against the logical count.

        `_page_unit_regions` runs first only for its side effect (the contiguity
        and granularity checks). Its return is for the Triton descriptor, not
        this view path -- called bare, not unpacked.
        """
        self._page_unit_regions()
        cache = self.model_runner.kv_cache
        rows, entry = cache.shape[0], cache.shape[3]
        block = self.model_runner.block_size
        # Logical blocks the pool holds, which is what a unit id indexes.
        # Range-checked against THIS count, not the physical one: the physical
        # count is `block_ratio` times smaller, so checking it would pass ids
        # that address past the end of the tensor.
        logical_blocks = (cache.shape[1] * cache.shape[2]) // block
        planes = [cache[row].view(-1, entry) for row in range(rows)]
        views: list[torch.Tensor] = []
        for unit in unit_ids:
            unit = int(unit)
            if not 0 <= unit < logical_blocks:
                raise IndexError(
                    f"PAGE unit {unit} outside the pool's {logical_blocks} "
                    "logical blocks"
                )
            lo = unit * block
            views.extend(plane[lo : lo + block] for plane in planes)

        # ---- cut the padding tail off the final unit ----
        #
        # An image occupies `ceil(image_bytes / page_unit_bytes)` WHOLE units,
        # so the last one is mostly padding. The HBM copy already trims it
        # (`_checkpoint_copy_plan` passes `plan_segmented_copy` both the unit
        # stream and `spec.image_bytes`); gathering whole units for the CPU tier
        # skipped that and offered the packer the padding too, against a
        # MemoryObj `StateByteCodec.put` sizes at `entry_bytes` == `image_bytes`
        # -- so every store died with "MemoryObj tensor is too small".
        #
        # Byte-exact, not view-aligned: an image need not end on a row boundary,
        # so the boundary view is reinterpreted as `uint8` and sliced (the
        # packer sizes segments as `numel * element_size`). The load side needs
        # no counterpart -- it scatters into `state_entry_views`, which sums to
        # `entry_bytes`. Inline, not a helper: the tests drive this with a
        # `SimpleNamespace` `self`, and a helper would be one more stub each.
        runtime = getattr(self.model_runner, "state_runtime", None)
        spec = None if runtime is None else runtime.checkpoint_spec
        # `getattr`: a fork build or a test stand-in carries no `image_bytes` --
        # no image size to trim against, so the whole-unit stream is correct.
        budget = int(getattr(spec, "image_bytes", 0) or 0) if spec else 0
        if not budget:
            return views
        out: list[torch.Tensor] = []
        for view in views:
            nbytes = view.numel() * view.element_size()
            if budget >= nbytes:
                out.append(view)
                budget -= nbytes
                continue
            if budget > 0:
                out.append(view.reshape(-1).view(torch.uint8)[:budget])
                budget = 0
            break
        if budget:
            raise RuntimeError(
                f"a checkpoint image is {spec.image_bytes} B but its "
                f"{len(views)} unit views hold "
                f"{int(spec.image_bytes) - budget} B; "
                "the unit geometry and the image size disagree"
            )
        return out
