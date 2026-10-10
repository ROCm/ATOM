# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a checkpoint image's bytes live in an MHA paged pool.

The MHA counterpart of ``page_unit_geometry``. That module is a mixin over an
MLA cache ``(rows, blocks, block_size, entry)`` and only ``_KimiMLAGDNCommon``
mixes it in. An MHA pool publishes one contiguous region per field per layer,
with alignment padding between fields that is not a region, so these are free
functions over the pool's ``region_tensors`` rather than methods on
``self.model_runner``.

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


def mha_published_page_bytes(pools: Iterable) -> int:
    """Bytes one scheduler block publishes as MHA PAGE regions.

    ``MhaKvPool.entry_bytes`` also counts alignment padding between fields.
    Those gaps are not part of any ``region_tensors`` row, and ``lmcache_mp``
    requires the checkpoint unit to equal the sum of the published regions.
    """

    total = 0
    for pool in pools:
        for group in pool.field_groups:
            for field in group:
                total += field.bytes_per_entry
    return total


def mha_page_unit_regions(pools: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Base address and per-block stride of every published MHA region.

    Order is ``MhaKvPool.region_tensors``: all of one field, then the next,
    which is the order ``AiterAttentionMetadataBuilder.get_kv_transfer_tensors``
    publishes. One region is one layer of one field, and its stride is one
    scheduler block.
    """

    bases: list[int] = []
    sizes: list[int] = []
    for pool in pools:
        for role, tensor in pool.region_tensors():
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"MHA PAGE region {role} is not a tensor")
            if not tensor.is_contiguous():
                raise RuntimeError(
                    "an MHA PAGE region must be contiguous to hold a GDN "
                    f"checkpoint image; {role} stride {tensor.stride()} is not"
                )
            unit = tensor.stride(0) * tensor.element_size()
            if unit <= 0:
                raise RuntimeError(f"MHA PAGE region {role} has no bytes")
            bases.append(tensor.data_ptr())
            sizes.append(unit)
    if not bases:
        raise RuntimeError("GDN PAGE checkpoint found no MHA regions")
    return np.array(bases, dtype=np.int64), np.array(sizes, dtype=np.int64)


def mha_page_unit_bases(regions: tuple[np.ndarray, np.ndarray], unit_ids) -> np.ndarray:
    """Start address of every destination segment, one row per image.

    Same product as ``PageUnitGeometryMixin._page_unit_bases``. ``unit_ids``
    is ``(images, units_per_checkpoint)``, unit major and region minor, which
    is the order the copy plan tiled the destination stream in.
    """

    base, stride = regions
    ids = np.asarray(unit_ids, dtype=np.int64)
    return (base + ids[..., None] * stride).reshape(len(ids), -1)


def mha_page_unit_stream_sizes(
    regions: tuple[np.ndarray, np.ndarray], units: int
) -> np.ndarray:
    """Bytes in each destination segment of an image of ``units`` units.

    Same tile as ``PageUnitGeometryMixin._page_unit_stream_sizes``.
    """

    return np.tile(regions[1], units)


def mha_page_unit_views(
    pools: Iterable, unit_ids, *, image_bytes: int = 0
) -> list[torch.Tensor]:
    """Tensor views of one checkpoint image, unit major and region minor.

    The MHA counterpart of ``PageUnitGeometryMixin.page_unit_views``. The
    in-process state tier packs these and later unpacks the same byte stream
    into an Active Slot. Order matches ``mha_page_unit_bases``. A whole unit
    is ``ceil(image / unit)`` scheduler blocks, so the tail past
    ``image_bytes`` is padding and must not be offered to the packer.
    """

    rows: list[tuple[str, torch.Tensor]] = []
    for pool in pools:
        for role, tensor in pool.region_tensors():
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"MHA PAGE region {role} is not a tensor")
            if tensor.ndim < 1 or not tensor.is_contiguous():
                raise RuntimeError(
                    "an MHA PAGE region must be contiguous to hold a GDN "
                    f"checkpoint image; {role} stride {tensor.stride()} is not"
                )
            rows.append((role, tensor))
    if not rows:
        raise RuntimeError("GDN PAGE checkpoint found no MHA regions")
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
                raise RuntimeError(
                    f"MHA PAGE region {role} block {unit} is not contiguous"
                )
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
