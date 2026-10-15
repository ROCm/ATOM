# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Producer-side GPU repack of MLA pages onto one DCP consumer rank's shard.

An unsharded producer owns no page of a DCP consumer whole once the consumer
interleaves below a page: every ``interleave_size`` tokens belong to the next
rank. Written in place, that is one RDMA descriptor per run per layer -- over
a million 576-byte writes for a 120K-token Kimi-K3 prompt on one PP stage.
Gathering the rank's tokens into staging pages first leaves one descriptor per
destination page.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from atom.kv_transfer.disaggregation.sharded_transfer import DCPShardPlan
from atom.kv_transfer.disaggregation.types import (
    MLA_KV_ROLE,
    KVTransferRegion,
    PageRegion,
)

MLA_STAGING_ROLE = "mla.kv_staging"
_SLOT_ALIGN = 4096


@dataclass(frozen=True)
class StagedShard:
    """One consumer rank's producer token rows, in its destination order."""

    rows: torch.Tensor
    tokens: int
    block_size: int

    @property
    def pages(self) -> int:
        return -(-self.tokens // self.block_size)

    def page_tokens(self, start: int, stop: int) -> np.ndarray:
        """Tokens this shard puts in destination pages ``[start, stop)``."""
        first = np.arange(start, stop, dtype=np.int64) * self.block_size
        return np.clip(self.tokens - first, 0, self.block_size)


def _row_view(view: torch.Tensor, block_size: int) -> torch.Tensor:
    """A page view as one row per token, in the widest word that tiles a row."""
    flat = view.reshape(-1)
    row_bytes = flat.numel() // (view.shape[0] * block_size)
    for dtype in (torch.int64, torch.int32, torch.int16):
        width = dtype.itemsize
        if row_bytes % width == 0 and flat.data_ptr() % width == 0:
            return flat.view(dtype).view(-1, row_bytes // width)
    return flat.view(-1, row_bytes)


class MLAShardStaging:
    """A pool of staging slots plus the token-row views of the pages they read.

    One slot per send worker, held for one request's push to one consumer
    rank, so a slot is never shared and a pushed chunk is never overwritten
    before its RDMA write returns.
    """

    def __init__(
        self,
        sources: dict[int, torch.Tensor],
        *,
        block_size: int,
        pool_size: int,
        chunk_pages: int,
        device: torch.device,
    ):
        if pool_size <= 0 or chunk_pages <= 0:
            raise ValueError(
                f"MLA staging needs positive slots and pages, got "
                f"{pool_size=} {chunk_pages=}"
            )
        self.block_size = block_size
        self.chunk_pages = chunk_pages
        self._sources = sources
        row_bytes = max(
            rows[0].numel() * rows.element_size() for rows in sources.values()
        )
        slot_bytes = chunk_pages * block_size * row_bytes
        self.slot_bytes = -(-slot_bytes // _SLOT_ALIGN) * _SLOT_ALIGN
        self._pool = torch.empty(
            (pool_size, self.slot_bytes), dtype=torch.uint8, device=device
        )
        self._free = list(range(pool_size))
        self._free_cv = threading.Condition()
        self._stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )

    @classmethod
    def build(
        cls,
        pages: Sequence[PageRegion],
        *,
        block_size: int,
        pool_size: int,
        chunk_pages: int,
        device: torch.device,
    ) -> MLAShardStaging | None:
        """Staging over every token-contiguous MLA page, or None if none is."""
        sources = {}
        for idx, page in enumerate(pages):
            region, view = page.region, page.view
            if (
                region.semantic_role != MLA_KV_ROLE
                or region.reverse_indexed
                or view is None
                or view.device != device
                or region.unit_bytes % block_size
            ):
                continue
            sources[idx] = _row_view(view, block_size)
        if not sources:
            return None
        return cls(
            sources,
            block_size=block_size,
            pool_size=pool_size,
            chunk_pages=chunk_pages,
            device=device,
        )

    @property
    def region(self) -> KVTransferRegion:
        return KVTransferRegion(
            base_addr=self._pool.data_ptr(),
            total_bytes=self._pool.numel(),
            unit_bytes=self.slot_bytes,
            semantic_role=MLA_STAGING_ROLE,
        )

    @property
    def pool_size(self) -> int:
        return self._pool.shape[0]

    def stages(self, region_idx: int) -> bool:
        return region_idx in self._sources

    def acquire(self) -> int:
        with self._free_cv:
            self._free_cv.wait_for(lambda: bool(self._free))
            return self._free.pop()

    def release(self, slot: int) -> None:
        with self._free_cv:
            self._free.append(slot)
            self._free_cv.notify()

    def _on_stream(self):
        if self._stream is None:
            return contextlib.nullcontext()
        return torch.cuda.stream(self._stream)

    def prepare(self, plan: DCPShardPlan, ready_event=None) -> StagedShard:
        """Upload the plan's source token rows; later gathers wait on ``ready_event``."""
        if plan.block_size != self.block_size:
            raise ValueError(
                f"DCP shard plan block size {plan.block_size} is not the "
                f"staged pages' {self.block_size}"
            )
        runs = int(np.count_nonzero(plan.valid))
        # Gathering by position assumes the consumer's tokens are a prefix of
        # its pages; the direct path never needed that, so check it here.
        if not plan.valid[:runs].all():
            raise RuntimeError(
                "DCP shard plan's valid runs are not a prefix of the consumer's "
                "tokens; staged MLA pages would be misplaced"
            )
        src = plan.src_block_id_per_run[:runs] * plan.block_size + plan.src_token[:runs]
        if plan.interleave_size > 1:
            src = (src[:, None] + np.arange(plan.interleave_size)).reshape(-1)
        src = np.ascontiguousarray(src, dtype=np.int64)
        with self._on_stream():
            if ready_event is not None and self._stream is not None:
                self._stream.wait_event(ready_event)
            rows = torch.from_numpy(src).to(self._pool.device)
        return StagedShard(rows=rows, tokens=src.size, block_size=self.block_size)

    def gather(
        self, region_idx: int, shard: StagedShard, start: int, stop: int, slot: int
    ) -> tuple[int, int]:
        """Pack pages ``[start, stop)`` of ``shard`` into ``slot``.

        Returns the slot's address and the region's page bytes. The bytes are
        in place when this returns, so the slot can be read by the NIC.
        """
        if not 0 <= start < stop <= min(start + self.chunk_pages, shard.pages):
            raise ValueError(
                f"Staged page range [{start}, {stop}) is outside the shard's "
                f"{shard.pages} pages or a {self.chunk_pages}-page slot"
            )
        source = self._sources[region_idx]
        first = start * self.block_size
        last = min(stop * self.block_size, shard.tokens)
        out = self._pool[slot].view(source.dtype)[: (last - first) * source.shape[1]]
        done = None
        with self._on_stream():
            torch.index_select(
                source, 0, shard.rows[first:last], out=out.view(last - first, -1)
            )
            if self._stream is not None:
                done = torch.cuda.Event()
                done.record(self._stream)
        if done is not None:
            done.synchronize()
        page_bytes = self.block_size * source.shape[1] * source.element_size()
        return self._pool[slot].data_ptr(), page_bytes


def mla_staging_for(
    staging: MLAShardStaging | None, plan: DCPShardPlan | None
) -> MLAShardStaging | None:
    """``staging`` when ``plan`` hands its rank runs shorter than a page."""
    if staging is None or plan is None or plan.interleave_size >= staging.block_size:
        return None
    return staging


__all__ = ["MLA_STAGING_ROLE", "MLAShardStaging", "StagedShard", "mla_staging_for"]
