# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The one check that backend-published PAGE views are what they claim to be.

Both LMCache MP registrations -- PAGE-only and native state -- hand the same
block-major views to the server, so they share this validation rather than
keeping two copies that drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PageView:
    """One validated block-major PAGE view and the region it aliases."""

    index: int
    view: torch.Tensor
    region: Any
    unit_bytes: int


def validate_page_views(
    transfer_tensors: Any,
    *,
    num_blocks: int,
    block_size: int | None = None,
) -> list[PageView]:
    """Validate ``block_tensor_views`` against ``block_regions``.

    Each view must be ``[num_blocks, physical_slots, opaque_width]``, tightly
    block-major (one unit per block, contiguous inside it), exactly cover and
    start at its region, be forward-indexed, and share one device with the
    rest. ``block_size``, when given, also requires the physical slots of a
    block to divide it.
    """
    if transfer_tensors is None:
        raise ValueError("lmcache_mp requires KVTransferTensors")
    if type(num_blocks) is not int or num_blocks <= 0:
        raise ValueError("lmcache_mp num_blocks must be a positive integer")
    regions = list(getattr(transfer_tensors, "block_regions", None) or [])
    views = list(getattr(transfer_tensors, "block_tensor_views", None) or [])
    if not regions or len(views) != len(regions):
        raise ValueError(
            "lmcache_mp requires one block_tensor_view per block region: "
            f"views={len(views)} regions={len(regions)}"
        )

    validated: list[PageView] = []
    devices: set[torch.device] = set()
    for index, (view, region) in enumerate(zip(views, regions, strict=True)):
        name = f"LMCache MP PAGE view {index}"
        if not isinstance(view, torch.Tensor):
            raise TypeError(f"{name} is not a Tensor")
        if view.ndim != 3 or int(view.shape[0]) != num_blocks:
            raise ValueError(
                f"{name} has shape={tuple(view.shape)}, "
                f"expected [{num_blocks}, physical_slots, opaque_width]"
            )
        if view.numel() == 0 or not view[0].is_contiguous():
            raise ValueError(f"{name} must be non-empty and contiguous")
        if block_size is not None and block_size % int(view.shape[1]):
            raise ValueError(f"{name} physical slots must divide block size")
        unit_bytes = int(region.unit_bytes)
        actual_unit_bytes = view[0].numel() * view.element_size()
        if (
            unit_bytes <= 0
            or actual_unit_bytes != unit_bytes
            or view.stride(0) * view.element_size() != unit_bytes
            or int(region.total_bytes) != num_blocks * unit_bytes
        ):
            raise ValueError(
                f"{name} byte geometry mismatch: unit={actual_unit_bytes}/"
                f"{unit_bytes} stride={view.stride(0) * view.element_size()} "
                f"total={region.total_bytes}/{num_blocks * unit_bytes}"
            )
        if view.data_ptr() != int(region.base_addr):
            raise ValueError(f"{name} does not alias its declared region")
        if bool(getattr(region, "reverse_indexed", False)):
            raise ValueError("lmcache_mp PAGE regions cannot be reverse-indexed")
        devices.add(view.device)
        validated.append(PageView(index, view, region, unit_bytes))
    if len(devices) != 1:
        raise ValueError("lmcache_mp PAGE views must share one device")
    return validated


@dataclass(frozen=True)
class _CacheViews:
    """Validated block-major tensors and their copy-kernel group indices."""

    tensors: dict[str, torch.Tensor]
    layer_groups: tuple[tuple[int, ...], ...]
    bytes_per_block: int


def _build_cache_views(
    transfer_tensors: Any,
    *,
    num_blocks: int,
) -> _CacheViews:
    """Validate backend-published PAGE views without inspecting model internals."""

    if transfer_tensors is None:
        raise ValueError("lmcache_mp requires KVTransferTensors")

    stateful_fields = {
        "num_slots": getattr(transfer_tensors, "num_slots", 0),
        "slot_regions": getattr(transfer_tensors, "slot_regions", None),
        "swa_block_regions": getattr(transfer_tensors, "swa_block_regions", None),
        "staging_region": getattr(transfer_tensors, "staging_region", None),
        "gather_slot": getattr(transfer_tensors, "gather_slot", None),
        "scatter_slot": getattr(transfer_tensors, "scatter_slot", None),
        "expected_full_slot_region_count": getattr(
            transfer_tensors, "expected_full_slot_region_count", None
        ),
    }
    populated_stateful_fields = [
        name for name, value in stateful_fields.items() if value
    ]
    if populated_stateful_fields:
        raise NotImplementedError(
            "lmcache_mp supports PAGE-only layouts; stateful SLOT data was "
            f"published through {', '.join(populated_stateful_fields)}"
        )

    tensors: dict[str, torch.Tensor] = {}
    indices_by_layout: dict[tuple[torch.dtype, tuple[int, ...]], list[int]] = {}
    bytes_per_block = 0
    for page in validate_page_views(transfer_tensors, num_blocks=num_blocks):
        index, view, region = page.index, page.view, page.region
        # LMCache receives these tensors as opaque PAGE storage, not numerical
        # values.  Publish a zero-copy byte view so every transfer path copies
        # the exact bit pattern.  This is especially important on ROCm, where
        # LMCache's Python raw-pointer fallback cannot express FP8 through the
        # CUDA array interface and would otherwise reconstruct the destination
        # as uint8 while keeping the staging object as FP8.
        byte_view = view.view(torch.uint8)
        role = str(getattr(region, "semantic_role", None) or f"plane_{index}")
        tensors[f"page.{index}.{role}"] = byte_view
        layout = (byte_view.dtype, tuple(int(dim) for dim in byte_view.shape[1:]))
        indices_by_layout.setdefault(layout, []).append(index)
        bytes_per_block += page.unit_bytes

    return _CacheViews(
        tensors=tensors,
        layer_groups=tuple(tuple(indices) for indices in indices_by_layout.values()),
        bytes_per_block=bytes_per_block,
    )


__all__ = ["PageView", "validate_page_views"]
