# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility adapter for :class:`dsv4.codec.DSV4PageSlotCodec`."""

from __future__ import annotations

from collections.abc import Sequence
import operator

import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.copy_plan import CopyTile, PageCopy, build_copy_tiles
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec


class ATOMPageRegionCodec(DSV4PageSlotCodec):
    """Deprecated PAGE-only view over the unified DSV4 codec."""

    def __init__(
        self,
        regions: Sequence[KVTransferRegion],
        *,
        num_blocks: int,
        device: torch.device | str,
    ) -> None:
        try:
            normalized_blocks = operator.index(num_blocks)
        except TypeError as exc:
            raise ValueError("num_blocks must be an integer") from exc
        if normalized_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {normalized_blocks}")
        super().__init__(
            regions,
            (),
            num_blocks=normalized_blocks,
            num_slots=0,
            device=device,
        )
        # Legacy unit tests inject the former copy-plan module here.  Production
        # calls leave this unset and use the DSV4 region kernel.
        self._fused_kv_staging = None

    @property
    def has_fused_chunk_major_staging(self) -> bool:
        return self._fused_kv_staging is not None or self.device.type == "cuda"

    def copy_plan(self, block_ids: Sequence[int]) -> list[PageCopy]:
        return [
            PageCopy(
                block_id=span.item_id,
                region_index=span.region_index,
                src_addr=span.device_addr,
                dst_offset=span.buffer_offset,
                nbytes=span.nbytes,
            )
            for span in self.iter_reference_spans(self.page_plan(block_ids))
        ]

    def _validate_legacy_buffer(
        self, buffer: torch.Tensor, *, nblocks: int, name: str
    ) -> None:
        if not isinstance(buffer, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if buffer.dtype is not torch.uint8:
            raise TypeError(f"{name} must be a uint8 tensor")
        if not self._matches_device(buffer.device):
            raise TypeError(
                f"{name} must be on the PAGE cache device {self.device}, "
                f"got {buffer.device}"
            )
        if not buffer.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        required = nblocks * self.bytes_per_block
        if int(buffer.numel()) < required:
            raise ValueError(
                f"{name} is too small; need {required} bytes, got {int(buffer.numel())}"
            )

    def gpu_to_chunk_major_device_buffer(
        self,
        device_buf,
        block_id_groups,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if isinstance(device_buf, torch.Tensor):
            buffer, values = device_buf, block_id_groups
        else:
            buffer, values = block_id_groups, device_buf
        if not isinstance(buffer, torch.Tensor):
            raise TypeError("dst must be a torch.Tensor")
        ids = self._flatten_block_ids(values)
        self._validate_legacy_buffer(buffer, nblocks=len(ids), name="dst")
        if not ids:
            return
        if self._fused_kv_staging is not None:
            self._fused_kv_staging.gather_copy_plan(
                self.copy_plan(ids), buffer, stream=stream
            )
            return
        super().gpu_to_chunk_major_device_buffer(buffer, ids, stream=stream)

    def chunk_major_device_buffer_to_gpu(
        self,
        src: torch.Tensor,
        block_ids,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        ids = self._flatten_block_ids(block_ids)
        self._validate_legacy_buffer(src, nblocks=len(ids), name="src")
        if not ids:
            return
        if self._fused_kv_staging is not None:
            self._fused_kv_staging.scatter_copy_plan(
                src, self.copy_plan(ids), stream=stream
            )
            return
        super().chunk_major_device_buffer_to_gpu(src, ids, stream=stream)


__all__ = ["ATOMPageRegionCodec", "CopyTile", "PageCopy", "build_copy_tiles"]
