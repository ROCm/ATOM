# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Where a checkpoint image's bytes live in the MLA paged pool.

The destination side of a K3 PAGE-copy checkpoint: given logical block (PAGE
unit) ids, it gives the addresses and tensor views of the KV pool a checkpoint
copy lands on. `_KimiMLAGDNCommon` mixes it in over `GDNStateMixin`.

Pure over `self.model_runner`, and here rather than beside that builder for the
reason the package exists (see `pool_layout/__init__.py`): the geometry deciding
where every checkpoint byte goes is then testable on a GPU-less runner, which
the K3 builder's own module -- `importorskip`ped on CI -- is not.
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
        cache. A sparse-indexer model (`runner.has_mla_indexer`) overrides this
        to return the pool's index view, whose layers `_page_unit_regions`
        appends after the MLA rows.
        """
        return None

    def _page_unit_kv_cache(self):
        """The MLA rows a PAGE unit owns a region of.

        The pool's own view, `(rows, blocks, block_size, entry)`. A hook for the
        same reason `_page_unit_index_cache` is one: this module owns the
        arithmetic, not where the tensor comes from.
        """
        return self.kv_pool.cache.view("kv")

    def _page_unit_region_tensors(self) -> list[torch.Tensor]:
        """Every region a PAGE unit owns, each indexed by logical unit id: the MLA
        rows, then the index layers. Addresses and views are both taken from it."""
        cache = self._page_unit_kv_cache()
        block, entry = self.model_runner.block_size, cache.shape[3]
        logical = (cache.shape[1] * cache.shape[2]) // block
        regions = [
            row.view(-1, entry)[: logical * block].view(logical, block, entry)
            for row in cache.unbind(0)
        ]
        index_cache = self._page_unit_index_cache()
        if index_cache is not None:
            regions += index_cache.unbind(0)
        return regions

    def _page_unit_regions(self) -> tuple[np.ndarray, np.ndarray]:
        """Base address and per-unit stride of every region a PAGE id owns.

        The pool's KV view is `(rows, blocks, block_size, entry)`: a block owns
        one contiguous region per row, rows a fixed stride apart. Affine in the
        block id and a pool property, so it is computed once.

        The units were the trap here, and the view is what takes it away: it is
        shaped in the same **logical** blocks `unit_ids` carries, so a region is
        `runner.block_size` tokens wide by construction rather than by
        remembering that the allocation pages at 1 (K3's `block_ratio` is 128).
        The granularity assertion below still checks the relation.
        """
        runner = self.model_runner
        cache = self._page_unit_kv_cache()
        index_cache = self._page_unit_index_cache()
        # Key on the full layout, not `data_ptr()` alone. The pool is rebuilt in
        # two places -- the P/D IPC import (`_bind_kv_cache_to_modules`) and the
        # rollout sleep/wake path (`rollout/memory_manager.py` releases it, then
        # `_resume_kv_cache` -> `allocate_kv_cache` with a possibly *reduced*
        # block count). Address
        # equality does not imply geometry equality: a same-start-address
        # reallocation keeps `data_ptr()` but changes `stride(0)`, so the row stride
        # (and thus every base for row > 0) goes stale while `rows`/`region`
        # survive -- and the granularity assertion below could not catch it even
        # if the early return did not skip it, since it never involves
        # the row stride. Keying on `(data_ptr, shape, stride, element_size)` makes
        # any of those changing a cache miss that rebuilds. Mirrors the DSV4
        # sibling (`deepseek_v4_attn.py`), which keys on the same evidence.
        #
        # A sparse-indexer model (`runner.has_mla_indexer`) rides its index cache
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
        if index_cache is not None and not index_cache.is_contiguous():
            raise RuntimeError("the index cache must be contiguous to be copied")
        tensors = self._page_unit_region_tensors()
        bases = [t.data_ptr() for t in tensors]
        # Unit-major and contiguous: a unit's bytes in a region are its stride.
        sizes = [t.stride(0) * t.element_size() for t in tensors]
        rows = cache.shape[0]
        region = sizes[0]
        index_layers = len(tensors) - rows
        index_region = sizes[rows] if index_layers else 0

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
        as views. Order is the contract: unit major, region minor, the same ravel
        as `_page_unit_bases` (cross-build safety is `layout_id` in
        `StateByteCodec.key`, not this order).

        Each region is indexed by the logical unit id: `_page_unit_region_tensors`
        shapes the MLA rows in logical blocks, and is the same list
        `_page_unit_regions` takes its addresses from. Ids are range-checked
        against the logical count.

        `_page_unit_regions` runs first only for its side effect (the contiguity
        and granularity checks). Its return is for the Triton descriptor, not
        this view path -- called bare, not unpacked.
        """
        self._page_unit_regions()
        regions = self._page_unit_region_tensors()
        # A unit id names a logical block in every region.
        logical_blocks = min(region.shape[0] for region in regions)
        views: list[torch.Tensor] = []
        for unit in unit_ids:
            unit = int(unit)
            if not 0 <= unit < logical_blocks:
                raise IndexError(
                    f"PAGE unit {unit} outside the pool's {logical_blocks} "
                    "logical blocks"
                )
            views.extend(region[unit] for region in regions)

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
