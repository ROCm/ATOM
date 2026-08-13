# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 raw-byte PAGE/SLOT region gather and scatter kernels.

The kernel understands only the forward/reverse region ABI.  C4, C128,
compressor state, and SWA remain encoded by the registered region order and
unit sizes; no model shape is hard-coded here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import operator

import torch
import triton
import triton.language as tl

TILE_BYTES = 1024


@dataclass(frozen=True)
class _DeviceRegionPlan:
    region_base: torch.Tensor
    region_total_bytes: torch.Tensor
    region_unit_bytes: torch.Tensor
    tile_region: torch.Tensor
    tile_unit_offset: torch.Tensor
    tile_output_offset: torch.Tensor
    tile_valid_bytes: torch.Tensor
    bytes_per_item: int
    item_count: int
    device: torch.device
    reverse: bool

    @property
    def tiles_per_item(self) -> int:
        return int(self.tile_region.numel())


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _stream_context(stream: torch.cuda.Stream | None):
    return torch.cuda.stream(stream) if stream is not None else _NullCtx()


def _device_i64(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(tuple(values), dtype=torch.int64, device=device)


def _static_device_i64(values: Sequence[int], device: torch.device) -> torch.Tensor:
    # Static plans are published to other worker streams after construction.
    # Keep this one-time copy blocking; an async version would require a
    # readiness event and wait_event on every first consumer stream.
    return torch.tensor(tuple(values), dtype=torch.int64).to(
        device=device, non_blocking=False
    )


def _static_device_i32(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(tuple(values), dtype=torch.int32).to(
        device=device, non_blocking=False
    )


def build_region_plan(
    regions: Sequence[object],
    *,
    item_count: int,
    device: torch.device | str,
    reverse: bool,
    stream: torch.cuda.Stream | None = None,
) -> _DeviceRegionPlan:
    """Compile immutable region/tile metadata once for repeated launches."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("DSV4 region plans require a CUDA/HIP device")
    if device.index is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    if not regions:
        raise ValueError("DSV4 region plan requires at least one region")
    try:
        item_count = operator.index(item_count)
    except TypeError as exc:
        raise ValueError("DSV4 region plan item_count must be an integer") from exc
    if item_count <= 0:
        raise ValueError("DSV4 region plan item_count must be positive")

    bases: list[int] = []
    totals: list[int] = []
    units: list[int] = []
    tile_regions: list[int] = []
    tile_unit_offsets: list[int] = []
    tile_output_offsets: list[int] = []
    tile_valid_bytes: list[int] = []
    output_prefix = 0
    for region_index, region in enumerate(regions):
        try:
            base = operator.index(region.base_addr)
            total = operator.index(region.total_bytes)
            unit = operator.index(region.unit_bytes)
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "regions must expose integer base_addr, total_bytes, and unit_bytes"
            ) from exc
        if base <= 0 or total <= 0 or unit <= 0:
            raise ValueError("region addresses and sizes must be positive")
        required = item_count * unit
        if total < required:
            raise ValueError(
                f"region {region_index} has {total} bytes, need {required} "
                f"for {item_count} items"
            )
        bases.append(base)
        totals.append(total)
        units.append(unit)
        for tile_offset in range(0, unit, TILE_BYTES):
            tile_regions.append(region_index)
            tile_unit_offsets.append(tile_offset)
            tile_output_offsets.append(output_prefix + tile_offset)
            tile_valid_bytes.append(min(TILE_BYTES, unit - tile_offset))
        output_prefix += unit

    with _stream_context(stream):
        return _DeviceRegionPlan(
            region_base=_static_device_i64(bases, device),
            region_total_bytes=_static_device_i64(totals, device),
            region_unit_bytes=_static_device_i64(units, device),
            tile_region=_static_device_i32(tile_regions, device),
            tile_unit_offset=_static_device_i64(tile_unit_offsets, device),
            tile_output_offset=_static_device_i64(tile_output_offsets, device),
            tile_valid_bytes=_static_device_i32(tile_valid_bytes, device),
            bytes_per_item=output_prefix,
            item_count=item_count,
            device=device,
            reverse=bool(reverse),
        )


@triton.jit
def _gather_region_items_kernel(
    output,
    item_ids,
    region_base,
    region_total_bytes,
    region_unit_bytes,
    tile_region,
    tile_unit_offset,
    tile_output_offset,
    tile_valid_bytes,
    output_buffer_offset,
    bytes_per_item,
    REVERSE: tl.constexpr,
    TILE: tl.constexpr,
):
    item_pos = tl.program_id(0)
    tile_id = tl.program_id(1)
    item_id = tl.load(item_ids + item_pos).to(tl.int64)
    region = tl.load(tile_region + tile_id).to(tl.int64)
    unit_bytes = tl.load(region_unit_bytes + region).to(tl.int64)
    base = tl.load(region_base + region).to(tl.int64)
    if REVERSE:
        total = tl.load(region_total_bytes + region).to(tl.int64)
        unit_base = base + total - (item_id + 1) * unit_bytes
    else:
        unit_base = base + item_id * unit_bytes

    lanes = tl.arange(0, TILE).to(tl.int64)
    valid = tl.load(tile_valid_bytes + tile_id).to(tl.int64)
    mask = lanes < valid
    source = (unit_base + tl.load(tile_unit_offset + tile_id).to(tl.int64) + lanes).to(
        tl.pointer_type(tl.uint8)
    )
    target = (
        output
        + output_buffer_offset
        + item_pos.to(tl.int64) * bytes_per_item
        + tl.load(tile_output_offset + tile_id).to(tl.int64)
        + lanes
    )
    tl.store(target, tl.load(source, mask=mask), mask=mask)


@triton.jit
def _scatter_region_items_kernel(
    input,
    item_ids,
    region_base,
    region_total_bytes,
    region_unit_bytes,
    tile_region,
    tile_unit_offset,
    tile_output_offset,
    tile_valid_bytes,
    input_buffer_offset,
    bytes_per_item,
    REVERSE: tl.constexpr,
    TILE: tl.constexpr,
):
    item_pos = tl.program_id(0)
    tile_id = tl.program_id(1)
    item_id = tl.load(item_ids + item_pos).to(tl.int64)
    region = tl.load(tile_region + tile_id).to(tl.int64)
    unit_bytes = tl.load(region_unit_bytes + region).to(tl.int64)
    base = tl.load(region_base + region).to(tl.int64)
    if REVERSE:
        total = tl.load(region_total_bytes + region).to(tl.int64)
        unit_base = base + total - (item_id + 1) * unit_bytes
    else:
        unit_base = base + item_id * unit_bytes

    lanes = tl.arange(0, TILE).to(tl.int64)
    valid = tl.load(tile_valid_bytes + tile_id).to(tl.int64)
    mask = lanes < valid
    source = (
        input
        + input_buffer_offset
        + item_pos.to(tl.int64) * bytes_per_item
        + tl.load(tile_output_offset + tile_id).to(tl.int64)
        + lanes
    )
    target = (unit_base + tl.load(tile_unit_offset + tile_id).to(tl.int64) + lanes).to(
        tl.pointer_type(tl.uint8)
    )
    tl.store(target, tl.load(source, mask=mask), mask=mask)


def _validate_launch(
    plan: _DeviceRegionPlan,
    item_ids: Sequence[int],
    buffer: torch.Tensor,
    *,
    buffer_offset: int,
    stream: torch.cuda.Stream | None,
) -> tuple[tuple[int, ...], int]:
    if not isinstance(plan, _DeviceRegionPlan):
        raise TypeError("plan must be a DSV4 device region plan")
    if not isinstance(buffer, torch.Tensor) or not buffer.is_cuda:
        raise TypeError("buffer must be a CUDA/HIP tensor")
    if buffer.dtype is not torch.uint8:
        raise TypeError("buffer must have dtype torch.uint8")
    if not buffer.is_contiguous():
        raise ValueError("buffer must be contiguous")
    if buffer.device != plan.device:
        raise ValueError(
            f"buffer device {buffer.device} does not match plan device {plan.device}"
        )
    try:
        ids = tuple(operator.index(item_id) for item_id in item_ids)
        offset = operator.index(buffer_offset)
    except TypeError as exc:
        raise ValueError("item ids and buffer_offset must be integers") from exc
    if offset < 0:
        raise ValueError("buffer_offset must not be negative")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item ids are not supported")
    if any(item_id < 0 or item_id >= plan.item_count for item_id in ids):
        raise ValueError(f"item id outside plan pool [0, {plan.item_count})")
    required = offset + len(ids) * plan.bytes_per_item
    if required > int(buffer.numel()):
        raise ValueError(
            f"buffer is too small; need {required} bytes, got {int(buffer.numel())}"
        )
    if stream is not None and torch.device(stream.device) != buffer.device:
        raise ValueError("stream device does not match buffer device")
    return ids, offset


def gather_region_items(
    plan: _DeviceRegionPlan,
    item_ids: Sequence[int],
    dst: torch.Tensor,
    *,
    buffer_offset: int = 0,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Enqueue raw region gather; synchronization remains caller-owned."""

    ids, offset = _validate_launch(
        plan, item_ids, dst, buffer_offset=buffer_offset, stream=stream
    )
    if not ids:
        return
    with _stream_context(stream):
        item_ids_d = _device_i64(ids, dst.device)
        grid = (len(ids), plan.tiles_per_item)
        _gather_region_items_kernel[grid](
            dst,
            item_ids_d,
            plan.region_base,
            plan.region_total_bytes,
            plan.region_unit_bytes,
            plan.tile_region,
            plan.tile_unit_offset,
            plan.tile_output_offset,
            plan.tile_valid_bytes,
            offset,
            plan.bytes_per_item,
            REVERSE=plan.reverse,
            TILE=TILE_BYTES,
            num_warps=8,
        )


def scatter_region_items(
    plan: _DeviceRegionPlan,
    item_ids: Sequence[int],
    src: torch.Tensor,
    *,
    buffer_offset: int = 0,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Enqueue raw region scatter; synchronization remains caller-owned."""

    ids, offset = _validate_launch(
        plan, item_ids, src, buffer_offset=buffer_offset, stream=stream
    )
    if not ids:
        return
    with _stream_context(stream):
        item_ids_d = _device_i64(ids, src.device)
        grid = (len(ids), plan.tiles_per_item)
        _scatter_region_items_kernel[grid](
            src,
            item_ids_d,
            plan.region_base,
            plan.region_total_bytes,
            plan.region_unit_bytes,
            plan.tile_region,
            plan.tile_unit_offset,
            plan.tile_output_offset,
            plan.tile_valid_bytes,
            offset,
            plan.bytes_per_item,
            REVERSE=plan.reverse,
            TILE=TILE_BYTES,
            num_warps=8,
        )


__all__ = [
    "TILE_BYTES",
    "build_region_plan",
    "gather_region_items",
    "scatter_region_items",
]
