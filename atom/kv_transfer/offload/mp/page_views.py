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


__all__ = ["PageView", "validate_page_views"]
