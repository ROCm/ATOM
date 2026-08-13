# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Page-major byte codec for block-indexed KV transfer regions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
import operator

import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion

logger = logging.getLogger("atom")


@dataclass(frozen=True)
class PageCopy:
    """One contiguous PAGE-region unit and its staging-buffer destination."""

    block_id: int
    region_index: int
    src_addr: int
    dst_offset: int
    nbytes: int


@dataclass(frozen=True)
class CopyTile:
    """One valid Triton job from a possibly larger copy-plan entry."""

    src_addr: int
    dst_offset: int
    nbytes: int


def build_copy_tiles(
    plan: Sequence[object],
    *,
    tile_bytes: int,
) -> list[CopyTile]:
    """Expand copy ranges into only the tiles that contain valid bytes."""
    try:
        tile_bytes = operator.index(tile_bytes)
    except TypeError as exc:
        raise ValueError("copy tile bytes must be an integer") from exc
    if tile_bytes <= 0:
        raise ValueError(f"copy tile bytes must be > 0, got {tile_bytes}")

    tiles: list[CopyTile] = []
    for copy_index, copy in enumerate(plan):
        try:
            src_addr = operator.index(copy.src_addr)
            dst_offset = operator.index(copy.dst_offset)
            nbytes = operator.index(copy.nbytes)
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "copy plan entries must define integer src_addr, "
                "dst_offset, and nbytes"
            ) from exc
        if src_addr <= 0:
            raise ValueError(
                f"copy plan entry {copy_index} has invalid src_addr={src_addr}"
            )
        if dst_offset < 0:
            raise ValueError(
                f"copy plan entry {copy_index} has negative dst_offset={dst_offset}"
            )
        if nbytes <= 0:
            raise ValueError(
                f"copy plan entry {copy_index} has nonpositive nbytes={nbytes}"
            )
        for tile_offset in range(0, nbytes, tile_bytes):
            tiles.append(
                CopyTile(
                    src_addr=src_addr + tile_offset,
                    dst_offset=dst_offset + tile_offset,
                    nbytes=min(tile_bytes, nbytes - tile_offset),
                )
            )
    return tiles


@dataclass(frozen=True)
class _PageRegionSnapshot:
    """Validated immutable address geometry for one PAGE region."""

    base_addr: int
    total_bytes: int
    unit_bytes: int
    reverse_indexed: bool = False

    def unit_addr(self, index: int) -> int:
        return self.base_addr + index * self.unit_bytes


class ATOMPageRegionCodec:
    """Move block-indexed transfer regions through page-major staging buffers."""

    def __init__(
        self,
        regions: Sequence[KVTransferRegion],
        *,
        num_blocks: int,
        device: torch.device | str,
    ) -> None:
        raw_regions = tuple(regions)
        if not raw_regions:
            raise ValueError(
                "ATOMPageRegionCodec: at least one PAGE region is required"
            )

        try:
            self.num_blocks = operator.index(num_blocks)
        except TypeError as exc:
            raise ValueError(
                "ATOMPageRegionCodec: num_blocks must be an integer"
            ) from exc
        if self.num_blocks <= 0:
            raise ValueError(
                f"ATOMPageRegionCodec: num_blocks must be > 0, got {self.num_blocks}"
            )

        region_snapshots: list[_PageRegionSnapshot] = []
        region_unit_bytes: list[int] = []
        for region_index, region in enumerate(raw_regions):
            try:
                base_addr = operator.index(region.base_addr)
            except TypeError as exc:
                raise ValueError(
                    "ATOMPageRegionCodec: PAGE region "
                    f"{region_index} base_addr must be an integer"
                ) from exc
            if base_addr <= 0:
                raise ValueError(
                    "ATOMPageRegionCodec: PAGE region "
                    f"{region_index} base_addr must be > 0, got {base_addr}"
                )

            try:
                unit_bytes = operator.index(region.unit_bytes)
            except TypeError as exc:
                raise ValueError(
                    "ATOMPageRegionCodec: PAGE region "
                    f"{region_index} unit_bytes must be an integer"
                ) from exc
            if unit_bytes <= 0:
                raise ValueError(
                    "ATOMPageRegionCodec: PAGE region "
                    f"{region_index} unit_bytes must be > 0, got {unit_bytes}"
                )

            try:
                total_bytes = operator.index(region.total_bytes)
            except TypeError as exc:
                raise ValueError(
                    "ATOMPageRegionCodec: PAGE region "
                    f"{region_index} total_bytes must be an integer"
                ) from exc
            if total_bytes <= 0:
                raise ValueError(
                    "ATOMPageRegionCodec: PAGE region "
                    f"{region_index} total_bytes must be > 0, got {total_bytes}"
                )
            if region.reverse_indexed:
                raise ValueError(
                    "ATOMPageRegionCodec: reverse_indexed=True is invalid for "
                    f"PAGE region {region_index}"
                )
            required_bytes = self.num_blocks * unit_bytes
            if total_bytes < required_bytes:
                raise ValueError(
                    "ATOMPageRegionCodec: PAGE region "
                    f"{region_index} total_bytes is too small; "
                    f"got {total_bytes}, need {required_bytes}"
                )
            region_snapshots.append(
                _PageRegionSnapshot(
                    base_addr=base_addr,
                    total_bytes=total_bytes,
                    unit_bytes=unit_bytes,
                )
            )
            region_unit_bytes.append(unit_bytes)

        self.regions = tuple(region_snapshots)
        self._region_unit_bytes = tuple(region_unit_bytes)
        self.bytes_per_block = sum(self._region_unit_bytes)
        self.device = torch.device(device)
        if (
            self.device.type == "cuda"
            and self.device.index is None
            and torch.cuda.is_available()
        ):
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._fused_kv_staging = None
        if self.device.type == "cuda":
            try:
                from atom.kv_transfer.offload import triton_kv_staging

                self._fused_kv_staging = triton_kv_staging
            except Exception:
                logger.warning(
                    "ATOMPageRegionCodec: Triton copy-plan staging unavailable",
                    exc_info=True,
                )

    @property
    def has_fused_chunk_major_staging(self) -> bool:
        return self._fused_kv_staging is not None

    def _matches_device(self, device: torch.device) -> bool:
        device = torch.device(device)
        return self.device.type == device.type and (
            self.device.index is None or self.device.index == device.index
        )

    @staticmethod
    def _reject_duplicate_block_ids(block_ids: Sequence[int]) -> None:
        if len(set(block_ids)) != len(block_ids):
            raise ValueError(
                "ATOMPageRegionCodec: duplicate block ids are not supported"
            )

    def _normalize_block_ids(self, block_ids: Sequence[int]) -> list[int]:
        try:
            normalized = [operator.index(block_id) for block_id in block_ids]
        except TypeError as exc:
            raise ValueError("ATOMPageRegionCodec: block_ids must be integers") from exc
        for block_id in normalized:
            if not 0 <= block_id < self.num_blocks:
                raise ValueError(
                    "ATOMPageRegionCodec: block id "
                    f"{block_id} outside pool [0, {self.num_blocks})"
                )
        self._reject_duplicate_block_ids(normalized)
        return normalized

    def _flatten_block_ids(
        self,
        block_ids: Sequence[int] | Sequence[Sequence[int]],
    ) -> list[int]:
        values = list(block_ids)
        if not values:
            return []

        normalized: list[int] = []
        all_scalar = True
        for value in values:
            try:
                normalized.append(operator.index(value))
            except TypeError:
                all_scalar = False
                break
        if all_scalar:
            return self._normalize_block_ids(normalized)

        flattened: list[int] = []
        for group in values:
            try:
                group_values = list(group)
            except TypeError as exc:
                raise ValueError(
                    "ATOMPageRegionCodec: block_ids must be integers or "
                    "groups of integers"
                ) from exc
            flattened.extend(self._normalize_block_ids(group_values))
        self._reject_duplicate_block_ids(flattened)
        return flattened

    def copy_plan(self, block_ids: Sequence[int]) -> list[PageCopy]:
        """Return page-major copies for ``block_ids`` in caller order."""
        normalized = self._normalize_block_ids(block_ids)
        copies: list[PageCopy] = []
        dst_offset = 0
        for block_id in normalized:
            for region_index, (region, nbytes) in enumerate(
                zip(self.regions, self._region_unit_bytes, strict=True)
            ):
                copies.append(
                    PageCopy(
                        block_id=block_id,
                        region_index=region_index,
                        src_addr=int(region.unit_addr(block_id)),
                        dst_offset=dst_offset,
                        nbytes=nbytes,
                    )
                )
                dst_offset += nbytes
        return copies

    def _validate_device_buffer(
        self,
        buffer: torch.Tensor,
        *,
        nblocks: int,
        name: str,
    ) -> None:
        if not isinstance(buffer, torch.Tensor):
            raise TypeError(f"ATOMPageRegionCodec: {name} must be a torch.Tensor")
        if buffer.dtype != torch.uint8:
            raise TypeError(
                f"ATOMPageRegionCodec: {name} must be a uint8 tensor, "
                f"got {buffer.dtype}"
            )
        if not self._matches_device(buffer.device):
            raise TypeError(
                f"ATOMPageRegionCodec: {name} must be on the PAGE cache device "
                f"{self.device}, got {buffer.device}"
            )
        if not buffer.is_contiguous():
            raise ValueError(f"ATOMPageRegionCodec: {name} must be contiguous")
        required = int(nblocks) * self.bytes_per_block
        if int(buffer.numel()) < required:
            raise ValueError(
                f"ATOMPageRegionCodec: {name} is too small for {nblocks} blocks; "
                f"need {required} bytes, got {int(buffer.numel())}"
            )

    def _device_ctx(self):
        if self.device.type == "cuda" and self.device.index is not None:
            return torch.cuda.device(self.device)
        return _NullCtx()

    def gpu_to_chunk_major_device_buffer(
        self,
        block_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
        dst: torch.Tensor | Sequence[int] | Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Gather PAGE units into a page-major device staging buffer.

        The canonical call is ``(block_ids, dst)``. The existing LMCache
        connector's ``(dst, block_id_groups)`` call shape is also accepted.
        """
        if isinstance(block_ids, torch.Tensor):
            device_buffer = block_ids
            caller_block_ids = dst
        else:
            device_buffer = dst
            caller_block_ids = block_ids
        if not isinstance(device_buffer, torch.Tensor):
            raise TypeError("ATOMPageRegionCodec: dst must be a torch.Tensor")

        flattened = self._flatten_block_ids(caller_block_ids)
        self._validate_device_buffer(
            device_buffer,
            nblocks=len(flattened),
            name="dst",
        )
        if not flattened:
            return
        if self._fused_kv_staging is None:
            raise RuntimeError(
                "ATOMPageRegionCodec requires Triton fused copy-plan staging"
            )
        plan = self.copy_plan(flattened)
        with self._device_ctx():
            self._fused_kv_staging.gather_copy_plan(
                plan,
                device_buffer,
                stream=stream,
            )

    def chunk_major_device_buffer_to_gpu(
        self,
        src: torch.Tensor,
        block_ids: Sequence[int] | Sequence[Sequence[int]],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Scatter a page-major device staging buffer into PAGE regions."""
        flattened = self._flatten_block_ids(block_ids)
        self._validate_device_buffer(src, nblocks=len(flattened), name="src")
        if not flattened:
            return
        if self._fused_kv_staging is None:
            raise RuntimeError(
                "ATOMPageRegionCodec requires Triton fused copy-plan staging"
            )
        plan = self.copy_plan(flattened)
        with self._device_ctx():
            self._fused_kv_staging.scatter_copy_plan(
                src,
                plan,
                stream=stream,
            )


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
