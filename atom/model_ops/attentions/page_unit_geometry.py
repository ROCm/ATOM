# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Where a checkpoint image's bytes live in the MLA paged pool.

This is the destination side of a K3 PAGE-copy checkpoint: given logical block
(PAGE unit) ids, it says which addresses -- and which tensor views -- of the KV
pool a checkpoint copy lands on. `_KimiMLAGDNCommon` mixes it in over
`GDNStateMixin`, whose stubs raise `NotImplementedError` for a pool it does not
own.

Kept in its own module, free of any GPU/aiter import, for one reason: it is the
CPU tier's only seam onto PAGE units, and it is pure arithmetic over
`self.model_runner`. Living here it can be unit-tested on a runner with no GPU
(the K3 builder's own module imports aiter at load and is `importorskip`ped on
CI), so the geometry that decides where every checkpoint byte goes is actually
exercised there.
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

    def _page_unit_regions(self) -> tuple[np.ndarray, np.ndarray]:
        """Base address and per-unit stride of every region a PAGE id owns.

        The destination side of a checkpoint copy. `GDNStateMixin` knows where
        a state slot's bytes are; this knows where a KV block's are, because
        this class owns the MLA pool.

        `kv_cache` is `(rows, physical_blocks, physical_block_size, entry)`,
        so a block owns one contiguous region per row and the rows are a fixed
        stride apart. Affine in the block id, and a property of the pool rather
        than of any block, so it is worked out once.

        The units are the trap. `unit_ids` carries **logical** block ids -- what
        `BlockPool` hands out and what `sub_pool_specs` priced -- while the
        tensor is shaped in **physical** blocks, and K3's `block_ratio` is 128.
        So a region is `runner.block_size` tokens wide, not
        `physical_block_size`, and the two differ by exactly that ratio. The
        assertion below is what makes a mix-up a startup error rather than 127
        blocks of scrambled state: it is the one relation that cannot hold if
        the granularity is wrong.
        """
        runner = self.model_runner
        cache = runner.kv_cache
        owner = cache.data_ptr()
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

        runtime = getattr(runner, "state_runtime", None)
        spec = None if runtime is None else runtime.checkpoint_spec
        page_unit_bytes = spec.page_unit_bytes if spec is not None else rows * region
        if rows * region != page_unit_bytes:
            raise RuntimeError(
                f"a PAGE unit is {page_unit_bytes} B but this pool gives a "
                f"logical block {rows} rows x {region} B = {rows * region} B; "
                "the two disagree about block granularity"
            )
        base = np.array(
            [owner + row * row_stride for row in range(rows)], dtype=np.int64
        )
        regions = (base, np.full(rows, region, dtype=np.int64))
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
        as views. This is the tier's only new seam -- a load writes the Active
        Slot directly and reuses `state_entry_views`.

        Order is the contract: unit major, row minor, the same ravel as
        `_page_unit_bases`. Cross-build safety is `layout_id` in
        `StateByteCodec.key`, not this order.

        `unit_ids` carries *logical* block ids while `kv_cache` is shaped in
        *physical* ones (K3's ratio is 128). Flattening the two block axes and
        indexing by `unit * block_size` is that conversion -- the same
        arithmetic `_page_unit_bases` does in addresses, so the two cannot
        drift. Range-checked against the logical count for the same reason.

        `_page_unit_regions` runs first for its side effect: the contiguity and
        granularity checks live there. Its return (base/stride addresses) is for
        the Triton descriptor, not this view path -- called bare, not unpacked.
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
        # An image is `image_bytes`, but it occupies `ceil(image_bytes /
        # page_unit_bytes)` WHOLE units, so the last one is mostly padding: at
        # K3's 2,138,112 B unit and 58,079,232 B image that is 28 units =
        # 59,867,136 B, of which 1,787,904 belong to no image.
        #
        # `_checkpoint_copy_plan` already knows this -- it hands
        # `plan_segmented_copy` the unit stream *and* `spec.image_bytes`, so the
        # HBM copy writes only the leading `image_bytes` and leaves the tail
        # alone. Gathering whole units for the CPU tier skipped that argument
        # and asked the packer for the padding too, against a MemoryObj
        # `StateByteCodec.put` sizes at `entry_bytes` -- which IS `image_bytes`.
        # Every store died with "MemoryObj tensor is too small for 59867136
        # bytes; got 58079232".
        #
        # The trim is byte-exact rather than view-aligned because the image does
        # not end on one: 58,079,232 leaves 350,208 B of the last unit in use,
        # which is 20.96 rows of it. That view is reinterpreted as `uint8` and
        # sliced; the packer sizes segments as `numel * element_size`, so it
        # costs nothing to accept. The load side needs no counterpart -- it
        # scatters into `state_entry_views`, which sums to `entry_bytes`.
        #
        # Inline rather than a helper method: the unit tests drive this function
        # with a `SimpleNamespace` stand-in for `self`, so a second attribute
        # would have to be stubbed by every one of them.
        runtime = getattr(self.model_runner, "state_runtime", None)
        spec = None if runtime is None else runtime.checkpoint_spec
        # `getattr`: a fork build carries no spec, and the unit tests hand this
        # a stand-in that carries no `image_bytes`. Either way there is no image
        # size to trim against, and the whole-unit stream is the right answer.
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
