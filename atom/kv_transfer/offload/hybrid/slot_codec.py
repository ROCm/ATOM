# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full-slot GPU snapshot staging for reverse-indexed SLOT regions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from numbers import Integral

import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.copy_plan import PageCopy

logger = logging.getLogger("atom")


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class _SlotRegionSnapshot:
    """Immutable address geometry for one validated SLOT region."""

    base_addr: int
    total_bytes: int
    unit_bytes: int
    reverse_indexed: bool = True

    def unit_addr(self, group: int) -> int:
        return self.base_addr + self.total_bytes - (group + 1) * self.unit_bytes


class ATOMSlotSidecarCodec:
    """Gather and restore one complete logical SLOT through bounded GPU rows."""

    def __init__(
        self,
        slot_regions: Sequence[KVTransferRegion],
        *,
        num_slots: int,
        device: torch.device | str,
        staging_slots: int = 1,
    ) -> None:
        raw_regions = tuple(slot_regions)
        if not raw_regions:
            raise ValueError(
                "ATOMSlotSidecarCodec: at least one SLOT region is required"
            )

        self.num_slots = _integer("num_slots", num_slots)
        if self.num_slots <= 0:
            raise ValueError(
                f"ATOMSlotSidecarCodec: num_slots must be > 0, got {self.num_slots}"
            )
        self.staging_slots = _integer("staging_slots", staging_slots)
        if self.staging_slots <= 0:
            raise ValueError(
                "ATOMSlotSidecarCodec: staging_slots must be > 0, "
                f"got {self.staging_slots}"
            )

        snapshots: list[_SlotRegionSnapshot] = []
        for region_index, region in enumerate(raw_regions):
            prefix = f"ATOMSlotSidecarCodec: SLOT region {region_index}"
            base_addr = _integer(f"{prefix} base_addr", region.base_addr)
            if base_addr <= 0:
                raise ValueError(f"{prefix} base_addr must be > 0, got {base_addr}")

            unit_bytes = _integer(f"{prefix} unit_bytes", region.unit_bytes)
            if unit_bytes <= 0:
                raise ValueError(f"{prefix} unit_bytes must be > 0, got {unit_bytes}")

            total_bytes = _integer(f"{prefix} total_bytes", region.total_bytes)
            if total_bytes <= 0:
                raise ValueError(f"{prefix} total_bytes must be > 0, got {total_bytes}")
            if region.reverse_indexed is not True:
                raise ValueError(f"{prefix} must have reverse_indexed=True")

            required_bytes = self.num_slots * unit_bytes
            if total_bytes < required_bytes:
                raise ValueError(
                    f"{prefix} total_bytes is too small; "
                    f"got {total_bytes}, need {required_bytes}"
                )
            snapshots.append(
                _SlotRegionSnapshot(
                    base_addr=base_addr,
                    total_bytes=total_bytes,
                    unit_bytes=unit_bytes,
                )
            )

        self.regions = tuple(snapshots)
        self.payload_bytes = sum(region.unit_bytes for region in self.regions)
        self.device = torch.device(device)
        if (
            self.device.type == "cuda"
            and self.device.index is None
            and torch.cuda.is_available()
        ):
            self.device = torch.device("cuda", torch.cuda.current_device())

        self.staging = torch.empty(
            (self.staging_slots, self.payload_bytes),
            dtype=torch.uint8,
            device=self.device,
        )

        self._fused_kv_staging = None
        if self.device.type == "cuda":
            try:
                from atom.kv_transfer.offload import triton_kv_staging

                self._fused_kv_staging = triton_kv_staging
            except Exception:
                logger.warning(
                    "ATOMSlotSidecarCodec: Triton copy-plan staging unavailable",
                    exc_info=True,
                )

    @property
    def has_fused_copy_plan_staging(self) -> bool:
        return self._fused_kv_staging is not None

    def _validate_group(self, group: int) -> int:
        normalized = _integer("group id", group)
        if not 0 <= normalized < self.num_slots:
            raise ValueError(
                "ATOMSlotSidecarCodec: group id "
                f"{normalized} outside pool [0, {self.num_slots})"
            )
        return normalized

    def _validate_staging_id(self, staging_id: int) -> int:
        normalized = _integer("staging id", staging_id)
        if not 0 <= normalized < self.staging_slots:
            raise ValueError(
                "ATOMSlotSidecarCodec: staging id "
                f"{normalized} outside pool [0, {self.staging_slots})"
            )
        return normalized

    def _copy_plan(self, group: int) -> list[PageCopy]:
        copies: list[PageCopy] = []
        dst_offset = 0
        for region_index, region in enumerate(self.regions):
            copies.append(
                PageCopy(
                    block_id=group,
                    region_index=region_index,
                    # Task 1's descriptor calls this ``src_addr``.  It is the
                    # raw source for gather and the raw target for scatter.
                    src_addr=region.unit_addr(group),
                    dst_offset=dst_offset,
                    nbytes=region.unit_bytes,
                )
            )
            dst_offset += region.unit_bytes
        return copies

    def copy_plan(self, group: int) -> list[PageCopy]:
        """Return one CPU-only region-order plan for a logical SLOT group."""
        return self._copy_plan(self._validate_group(group))

    def staging_view(self, staging_id: int) -> torch.Tensor:
        """Return the exact contiguous staging row owned by ``staging_id``."""
        return self.staging[self._validate_staging_id(staging_id)]

    def _require_staging_support(self) -> None:
        if self._fused_kv_staging is None:
            raise RuntimeError(
                "ATOMSlotSidecarCodec requires Triton fused copy-plan staging"
            )

    def _validate_stream(self, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            return
        try:
            stream_device = torch.device(stream.device)
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise ValueError(
                "ATOMSlotSidecarCodec: stream must expose a CUDA/HIP device"
            ) from exc
        if stream_device.type != "cuda" or stream_device != self.device:
            raise ValueError(
                "ATOMSlotSidecarCodec: stream device "
                f"{stream_device} does not match codec device {self.device}"
            )

    def _device_ctx(self):
        if self.device.type == "cuda" and self.device.index is not None:
            return torch.cuda.device(self.device)
        return _NullCtx()

    def snapshot_to_staging(
        self,
        group: int,
        staging_id: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Gather every region of one logical SLOT into a staging row."""
        normalized_group = self._validate_group(group)
        staging = self.staging_view(staging_id)
        self._validate_stream(stream)
        self._require_staging_support()
        with self._device_ctx():
            self._fused_kv_staging.gather_copy_plan(
                self._copy_plan(normalized_group),
                staging,
                stream=stream,
            )

    def restore_from_staging(
        self,
        group: int,
        staging_id: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Scatter one staging row back into every region of a logical SLOT."""
        normalized_group = self._validate_group(group)
        staging = self.staging_view(staging_id)
        self._validate_stream(stream)
        self._require_staging_support()
        with self._device_ctx():
            self._fused_kv_staging.scatter_copy_plan(
                staging,
                self._copy_plan(normalized_group),
                stream=stream,
            )


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
