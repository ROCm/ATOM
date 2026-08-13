# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility SLOT-staging adapter over the unified DSV4 codec."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from contextlib import nullcontext

import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.copy_plan import PageCopy
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec


def _integer(name: str, value: object) -> int:
    value_type = type(value)
    if isinstance(value, (bool, torch.Tensor)) or (
        value_type.__module__.split(".", 1)[0] == "numpy"
        and value_type.__name__ in {"bool", "bool_"}
    ):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


class ATOMSlotSidecarCodec(DSV4PageSlotCodec):
    """Deprecated adapter retaining only the old caller-owned-row facade."""

    def __init__(
        self,
        slot_regions: Sequence[KVTransferRegion],
        *,
        num_slots: int,
        device: torch.device | str,
        staging_slots: int = 1,
    ) -> None:
        num_slots = _integer("num_slots", num_slots)
        staging_slots = _integer("staging_slots", staging_slots)
        if num_slots <= 0:
            raise ValueError(f"num_slots must be > 0, got {num_slots}")
        if staging_slots <= 0:
            raise ValueError(f"staging_slots must be > 0, got {staging_slots}")
        super().__init__(
            (),
            slot_regions,
            num_blocks=0,
            num_slots=num_slots,
            device=device,
        )
        self.staging_slots = staging_slots
        self.staging = torch.empty(
            (staging_slots, self.slot_bytes),
            dtype=torch.uint8,
            device=self.device,
        )
        self._fused_kv_staging = None

    @property
    def regions(self):
        return self.slot_regions

    @property
    def has_fused_copy_plan_staging(self) -> bool:
        return self._fused_kv_staging is not None or self.device.type == "cuda"

    def _validate_group(self, group: int) -> int:
        group = _integer("group id", group)
        if not 0 <= group < self.num_slots:
            raise ValueError(f"group id {group} outside pool [0, {self.num_slots})")
        return group

    def _validate_staging_id(self, staging_id: int) -> int:
        staging_id = _integer("staging id", staging_id)
        if not 0 <= staging_id < self.staging_slots:
            raise ValueError(
                f"staging id {staging_id} outside pool [0, {self.staging_slots})"
            )
        return staging_id

    def copy_plan(self, group: int) -> list[PageCopy]:
        plan = self.slot_plan(self._validate_group(group))
        return [
            PageCopy(
                block_id=span.item_id,
                region_index=span.region_index,
                src_addr=span.device_addr,
                dst_offset=span.buffer_offset,
                nbytes=span.nbytes,
            )
            for span in self.iter_reference_spans(plan)
        ]

    def staging_view(self, staging_id: int) -> torch.Tensor:
        return self.staging[self._validate_staging_id(staging_id)]

    def _validate_stream(self, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            return
        try:
            stream_device = torch.device(stream.device)
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise ValueError("stream must expose a CUDA/HIP device") from exc
        if stream_device.type != "cuda" or stream_device != self.device:
            raise ValueError(
                f"stream device {stream_device} does not match codec device "
                f"{self.device}"
            )

    def _device_ctx(self):
        if self.device.type == "cuda" and self.device.index is not None:
            return torch.cuda.device(self.device)
        return nullcontext()

    def snapshot_to_staging(
        self,
        group: int,
        staging_id: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        group = self._validate_group(group)
        row = self.staging_view(staging_id)
        self._validate_stream(stream)
        if self._fused_kv_staging is not None:
            with self._device_ctx():
                self._fused_kv_staging.gather_copy_plan(
                    self.copy_plan(group), row, stream=stream
                )
            return
        self.gather_slot(group, row, stream=stream)

    def restore_from_staging(
        self,
        group: int,
        staging_id: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        group = self._validate_group(group)
        row = self.staging_view(staging_id)
        self._validate_stream(stream)
        if self._fused_kv_staging is not None:
            with self._device_ctx():
                self._fused_kv_staging.scatter_copy_plan(
                    row, self.copy_plan(group), stream=stream
                )
            return
        self.scatter_slot(row, group, stream=stream)


__all__ = ["ATOMSlotSidecarCodec"]
