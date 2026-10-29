# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a checkpoint image's bytes live in a layer-major paged pool.

The layer-major counterpart of ``page_unit_geometry``. That module is a mixin
over an MLA cache ``(rows, blocks, block_size, entry)`` and only
``_KimiMLAGDNCommon`` mixes it in. A layer-major pool -- Qwen3-Next / Qwen3.5's
``MhaKvPool``, Qwen3.8's QSA ``LayerMajorArena`` -- publishes one contiguous
region per field per layer, with any alignment padding between fields left
out of the regions, so these are free functions over the pools'
``region_tensors`` rather than methods on ``self.model_runner``.

``GDNAttentionMetadataBuilder`` is the caller, the same role
``kimi_mla_gdn_attn`` plays for the MLA mixin: ``_page_unit_regions``,
``_page_unit_bases``, ``_page_unit_stream_sizes``, and ``page_unit_views``
there forward here. No aiter, so the byte arithmetic can be tested without a
GPU.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from atom.model_ops.attentions.pool_layout.entry_arena import published_bytes_for


def layer_major_published_page_bytes(pools: Iterable) -> int:
    """Bytes one scheduler block of an ``MhaKvPool`` publishes as PAGE regions.

    ``MhaKvPool.entry_bytes`` also counts alignment padding between fields.
    Those gaps are not part of any ``region_tensors`` row, and ``lmcache_mp``
    requires the checkpoint unit to equal the sum of the published regions.
    """

    return sum(
        published_bytes_for(group) for pool in pools for group in pool.field_groups
    )


def layer_major_page_unit_regions(pools: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Base address and per-block stride of every published PAGE region.

    Order is each pool's ``region_tensors`` (``MhaKvPool`` or
    ``LayerMajorArena``): all of one field, then the next, which is the order
    the builder's ``get_kv_transfer_tensors`` publishes. One region is one
    layer of one field, and its stride is one scheduler block.
    """

    bases: list[int] = []
    sizes: list[int] = []
    for pool in pools:
        for role, tensor in pool.region_tensors():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"PAGE region {role} is not a tensor")
            if not tensor.is_contiguous():
                raise RuntimeError(
                    "a PAGE region must be contiguous to hold a GDN "
                    f"checkpoint image; {role} stride {tensor.stride()} is not"
                )
            unit = tensor.stride(0) * tensor.element_size()
            if unit <= 0:
                raise RuntimeError(f"PAGE region {role} has no bytes")
            bases.append(tensor.data_ptr())
            sizes.append(unit)
    if not bases:
        raise RuntimeError("GDN PAGE checkpoint found no PAGE regions in its pools")
    return np.array(bases, dtype=np.int64), np.array(sizes, dtype=np.int64)


def layer_major_page_unit_bases(
    regions: tuple[np.ndarray, np.ndarray], unit_ids
) -> np.ndarray:
    """Start address of every destination segment, one row per image.

    Same product as ``PageUnitGeometryMixin._page_unit_bases``. ``unit_ids``
    is ``(images, units_per_checkpoint)``, unit major and region minor, which
    is the order the copy plan tiled the destination stream in.
    """

    base, stride = regions
    ids = np.asarray(unit_ids, dtype=np.int64)
    return (base + ids[..., None] * stride).reshape(len(ids), -1)


def layer_major_page_unit_stream_sizes(
    regions: tuple[np.ndarray, np.ndarray], units: int
) -> np.ndarray:
    """Bytes in each destination segment of an image of ``units`` units.

    Same tile as ``PageUnitGeometryMixin._page_unit_stream_sizes``.
    """

    return np.tile(regions[1], units)


def layer_major_page_unit_segments(
    regions: tuple[np.ndarray, np.ndarray],
    unit_ids,
    *,
    entries: int,
    image_bytes: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Addresses and byte sizes of the stream ``layer_major_page_unit_views`` names.

    Same order and the same ``image_bytes`` trim, without one tensor view per
    (unit, region). An image is ~10k segments, and building those views held
    the store thread's GIL for tens of milliseconds per checkpoint, which the
    scheduler thread then waited out.
    """

    ids = np.asarray(unit_ids, dtype=np.int64).reshape(-1)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= entries):
        raise IndexError(f"PAGE unit outside the pool's {entries} scheduler blocks")
    ptrs = layer_major_page_unit_bases(regions, ids[None, :])[0]
    sizes = layer_major_page_unit_stream_sizes(regions, ids.size)
    budget = int(image_bytes or 0)
    if budget <= 0:
        return ptrs, sizes
    ends = np.cumsum(sizes)
    if not ends.size or ends[-1] < budget:
        held = int(ends[-1]) if ends.size else 0
        raise RuntimeError(
            f"a checkpoint image is {budget} B but its {sizes.size} unit "
            f"segments hold {held} B"
        )
    last = int(np.searchsorted(ends, budget))
    sizes = sizes[: last + 1].copy()
    sizes[-1] -= int(ends[last]) - budget
    return ptrs[: last + 1], sizes


def layer_major_page_unit_views(
    pools: Iterable, unit_ids, *, image_bytes: int = 0
) -> list[torch.Tensor]:
    """Tensor views of one checkpoint image, unit major and region minor.

    The layer-major counterpart of ``PageUnitGeometryMixin.page_unit_views``. The
    in-process state tier packs these and later unpacks the same byte stream
    into an Active Slot. Order matches ``layer_major_page_unit_bases``. A whole unit
    is ``ceil(image / unit)`` scheduler blocks, so the tail past
    ``image_bytes`` is padding and must not be offered to the packer.
    """

    rows: list[tuple[str, torch.Tensor]] = []
    for pool in pools:
        for role, tensor in pool.region_tensors():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"PAGE region {role} is not a tensor")
            if tensor.ndim < 1 or not tensor.is_contiguous():
                raise RuntimeError(
                    "a PAGE region must be contiguous to hold a GDN "
                    f"checkpoint image; {role} stride {tensor.stride()} is not"
                )
            rows.append((role, tensor))
    if not rows:
        raise RuntimeError("GDN PAGE checkpoint found no PAGE regions in its pools")
    entries = int(rows[0][1].shape[0])
    views: list[torch.Tensor] = []
    for unit in unit_ids:
        unit = int(unit)
        if not 0 <= unit < entries:
            raise IndexError(
                f"PAGE unit {unit} outside the pool's {entries} scheduler blocks"
            )
        for role, tensor in rows:
            row = tensor[unit]
            if not row.is_contiguous():
                raise RuntimeError(f"PAGE region {role} block {unit} is not contiguous")
            views.append(row)
    budget = int(image_bytes or 0)
    if budget <= 0:
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
            f"a checkpoint image is {image_bytes} B but its {len(views)} "
            f"unit views hold {image_bytes - budget} B"
        )
    return out
