# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Owned tensor aliases for generic PAGE-backed native checkpoint images.

The native codec streams PAGE units in ordinal order, with regions inside
each unit. Every ordinal gets its own engine block-id list while aliasing the
original PAGE allocation. No Active SLOT or extra state allocation is exposed.
LMCache imports stay behind ``engine_group_infos`` so layout validation and
copy-plan tests run without a GPU or LMCache's native extension.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class NativeStateMPKernelGroup:
    """One actual dtype/shape/engine-address-space registration identity."""

    tensor_indices: tuple[int, ...]
    engine_group_id: int
    tokens_per_block: int
    sw_size_tokens: int = -1
    recurrent_state: bool = False
    extra_object_group_tag: int = 0


@dataclass(frozen=True)
class NativeStateMPLayout:
    tensors: tuple[torch.Tensor, ...]
    kernel_groups: tuple[NativeStateMPKernelGroup, ...]
    checkpoint_spec: Any

    def engine_group_infos(self) -> list[Any]:
        """Convert the neutral plan at the actual LMCache registration boundary."""
        from lmcache.v1.multiprocess.group_view import EngineGroupInfo

        return [
            EngineGroupInfo(
                engine_group_id=group.engine_group_id,
                layer_indices=group.tensor_indices,
                tokens_per_block=group.tokens_per_block,
                sw_size_tokens=group.sw_size_tokens,
                recurrent_state=group.recurrent_state,
                extra_object_group_tag=group.extra_object_group_tag,
            )
            for group in self.kernel_groups
        ]

    def validate_unit_ids(self, unit_ids: Sequence[int]) -> tuple[int, ...]:
        """Validate the PAGE IDs addressed by one native checkpoint image."""
        ids = tuple(unit_ids)
        units_per_checkpoint = int(self.checkpoint_spec.units_per_checkpoint)
        if len(ids) != units_per_checkpoint:
            raise ValueError(
                f"native image needs {units_per_checkpoint} unit IDs, got {len(ids)}"
            )
        num_blocks = self.tensors[0].shape[0]
        for unit_id in ids:
            if type(unit_id) is not int or not 0 <= unit_id < num_blocks:
                raise ValueError(f"invalid native checkpoint unit ID: {unit_id!r}")
        if len(set(ids)) != len(ids):
            raise ValueError("native checkpoint unit IDs must be distinct")
        return ids


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_native_state_mp_layout(
    transfer_tensors: Any,
    *,
    block_size: int,
    chunk_size: int,
    num_blocks: int | None = None,
) -> NativeStateMPLayout:
    """Validate and alias PAGE plus compact native STATE for LMCache MP.

    PAGE uses engine group zero. STATE ordinal ``j`` uses group ``1+j`` and one
    logical chunk per block. The MP server's global null block ID must be
    configured as ``-1``. The final ordinal's final region ends exactly at
    ``image_bytes``; its dim-0 stride continues to address the original full
    PAGE region.
    """
    block_size = _positive_int("native PAGE block size", block_size)
    chunk_size = _positive_int("native checkpoint chunk size", chunk_size)
    if chunk_size % block_size:
        raise ValueError("native checkpoint chunk size must be divisible by block size")
    if transfer_tensors is None:
        raise ValueError("native-state LMCache MP requires KVTransferTensors")
    spec = getattr(transfer_tensors, "paged_state_checkpoint_spec", None)
    if spec is None:
        raise ValueError("native-state LMCache MP requires paged checkpoint geometry")
    layout_id = str(getattr(spec, "layout_id", "")).strip()
    if not layout_id:
        raise ValueError("native checkpoint layout_id must be non-empty")
    page_bytes = _positive_int("native PAGE unit bytes", spec.page_unit_bytes)
    image_bytes = _positive_int("native checkpoint image bytes", spec.image_bytes)
    slot_bytes = _positive_int("native Active SLOT bytes", spec.slot_bytes)
    if image_bytes > slot_bytes:
        raise ValueError("native image bytes exceed Active SLOT bytes")
    units = _positive_int("native checkpoint units", spec.units_per_checkpoint)
    if units != (image_bytes + page_bytes - 1) // page_bytes:
        raise ValueError("native checkpoint units disagree with image geometry")
    if not callable(getattr(transfer_tensors, "execute_paged_state_copies", None)):
        raise TypeError("native-state LMCache MP requires execute_paged_state_copies")
    if num_blocks is None:
        num_blocks = getattr(transfer_tensors, "num_blocks", None)
    num_blocks = _positive_int("native scheduler block count", num_blocks)
    published_blocks = getattr(transfer_tensors, "num_blocks", 0)
    if published_blocks not in (0, num_blocks):
        raise ValueError("native published block count disagrees with scheduler")

    regions = list(getattr(transfer_tensors, "block_regions", None) or [])
    page_views = list(getattr(transfer_tensors, "block_tensor_views", None) or [])
    if not regions or len(regions) != len(page_views):
        raise ValueError(
            "native-state LMCache MP needs one owned tensor view per PAGE region"
        )
    devices = set()
    actual_page_bytes = 0
    for index, (view, region) in enumerate(zip(page_views, regions, strict=True)):
        if not isinstance(view, torch.Tensor):
            raise TypeError(f"native PAGE view {index} must be a Tensor")
        if view.ndim != 3 or view.shape[0] != num_blocks or view.numel() == 0:
            raise ValueError(f"native PAGE view {index} has invalid block geometry")
        if not view[0].is_contiguous():
            raise ValueError(f"native PAGE view {index} has non-contiguous inner rows")
        if block_size % view.shape[1]:
            raise ValueError(
                f"native PAGE view {index} physical slots must divide block size"
            )
        unit_bytes = _positive_int(
            f"native PAGE region {index} bytes", region.unit_bytes
        )
        if (
            view[0].numel() * view.element_size() != unit_bytes
            or view.stride(0) * view.element_size() != unit_bytes
            or region.total_bytes != num_blocks * unit_bytes
        ):
            raise ValueError(f"native PAGE view {index} byte geometry mismatch")
        if view.data_ptr() != region.base_addr:
            raise ValueError(f"native PAGE view {index} does not alias its region")
        if region.reverse_indexed:
            raise ValueError("native PAGE regions cannot be reverse-indexed")
        devices.add(view.device)
        actual_page_bytes += unit_bytes
    if len(devices) != 1:
        raise ValueError("native PAGE views must share one device")
    if actual_page_bytes != page_bytes:
        raise ValueError("PAGE regions do not cover the native PAGE unit")

    tensors = list(page_views)
    engine_ids = [0] * len(tensors)
    image_offset = 0
    for ordinal in range(units):
        for page_view, region in zip(page_views, regions, strict=True):
            nbytes = min(region.unit_bytes, image_bytes - image_offset)
            if not nbytes:
                break
            # Retyping and reshaping contiguous inner dimensions retain the
            # allocation owner. The explicit shape retains full PAGE stride,
            # including for a partial final region.
            byte_view = page_view.view(torch.uint8).view(num_blocks, 1, -1)
            # Give the singleton physical-slot dimension its canonical tight
            # stride too: LMCache's padded-layout validation checks stride(1)
            # even for a size-one dimension. A plain final-axis slice retains
            # the full region width there and is rejected at registration.
            alias = byte_view.as_strided(
                (num_blocks, 1, nbytes), (byte_view.stride(0), nbytes, 1)
            )
            tensors.append(alias)
            engine_ids.append(1 + ordinal)
            image_offset += nbytes
    if image_offset != image_bytes:
        raise ValueError("PAGE aliases do not cover the native image")

    by_identity: dict[tuple, list[int]] = {}
    for index, (tensor, engine_id) in enumerate(zip(tensors, engine_ids, strict=True)):
        identity = tensor.dtype, tuple(tensor.shape[1:]), engine_id
        members = by_identity.setdefault(identity, [])
        if members and tensors[members[0]].stride(0) != tensor.stride(0):
            # LMCache currently stamps one representative stride per identity.
            # Splitting metadata cannot fix this: it re-coalesces the identity
            # during registration. Reject instead of addressing another PAGE.
            raise ValueError(
                "native-state LMCache MP equal-shape regions have different "
                "block strides; "
                "LMCache requires one stride per physical kernel identity"
            )
        members.append(index)
    groups = tuple(
        NativeStateMPKernelGroup(
            tensor_indices=tuple(indices),
            engine_group_id=identity[2],
            tokens_per_block=chunk_size if identity[2] else block_size,
            sw_size_tokens=chunk_size if identity[2] else -1,
            recurrent_state=bool(identity[2]),
        )
        for identity, indices in by_identity.items()
    )
    return NativeStateMPLayout(
        tensors=tuple(tensors),
        kernel_groups=groups,
        checkpoint_spec=spec,
    )


__all__ = [
    "NativeStateMPKernelGroup",
    "NativeStateMPLayout",
    "build_native_state_mp_layout",
]
