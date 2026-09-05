# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Shared DCP page-relayout planning for KV disaggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def coalesce_contiguous(
    src: np.ndarray, dst: np.ndarray, length: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge adjacent runs that are contiguous on both sides."""

    if src.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    contiguous = (src[1:] == src[:-1] + length[:-1]) & (
        dst[1:] == dst[:-1] + length[:-1]
    )
    starts = np.concatenate(([True], ~contiguous))
    start_indices = np.flatnonzero(starts)
    merged_length = np.add.reduceat(length, start_indices)
    return src[starts], dst[starts], merged_length


@dataclass(frozen=True)
class DCPShardPlan:
    """Canonical producer-run mapping for one DCP consumer rank.

    Each entry is one contiguous ``interleave_size`` token run and names its
    physical producer block plus source/destination token offsets. Direct RDMA
    consumes these runs as-is. Layout-specific staging projects them to gather
    indices; the current preshuffled index layout requires one-token runs.
    """

    block_size: int
    interleave_size: int
    dst_pages: int
    src_block_id_per_run: np.ndarray
    src_token: np.ndarray
    dst_page: np.ndarray
    dst_token: np.ndarray
    run_length: np.ndarray
    valid: np.ndarray

    def slice_pages(self, start: int, stop: int) -> DCPShardPlan:
        """Return a destination-page slice rebased to page zero."""

        if not 0 <= start <= stop <= self.dst_pages:
            raise ValueError(
                f"Invalid DCP shard page slice [{start}, {stop}) for "
                f"{self.dst_pages} pages"
            )
        runs_per_page = self.block_size // self.interleave_size
        row_start = start * runs_per_page
        row_stop = stop * runs_per_page
        return DCPShardPlan(
            block_size=self.block_size,
            interleave_size=self.interleave_size,
            dst_pages=stop - start,
            src_block_id_per_run=self.src_block_id_per_run[row_start:row_stop],
            src_token=self.src_token[row_start:row_stop],
            dst_page=self.dst_page[row_start:row_stop] - start,
            dst_token=self.dst_token[row_start:row_stop],
            run_length=self.run_length[row_start:row_stop],
            valid=self.valid[row_start:row_stop],
        )

    def token_runs(
        self, dst_block_ids: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Materialize source/destination offsets in token units.

        Runs are not merged. The sharded-transfer caller uses this for
        ``dcp_size > 1``; consecutive runs are then ``interleave_size *
        dcp_size`` tokens apart at the source but only ``interleave_size``
        apart at the destination, so they cannot be contiguous at both ends.
        """

        dst_ids = np.asarray(dst_block_ids, dtype=np.int64)
        if dst_ids.size != self.dst_pages:
            raise ValueError(
                f"DCP shard plan has {self.dst_pages} destination pages, got "
                f"{dst_ids.size} block ids"
            )
        keep = self.valid
        src = self.src_block_id_per_run[keep] * self.block_size + self.src_token[keep]
        dst = dst_ids[self.dst_page[keep]] * self.block_size + self.dst_token[keep]
        return src, dst, self.run_length[keep]


def build_dcp_shard_plan(
    src_block_ids: Sequence[int],
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
    dst_pages: int | None = None,
) -> DCPShardPlan:
    """Build the shared token-ownership plan for one DCP consumer rank."""

    block_size = int(block_size)
    dcp_size = int(dcp_size)
    dcp_rank = int(dcp_rank)
    interleave_size = int(interleave_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank={dcp_rank} is outside [0, {dcp_size})")
    if interleave_size <= 0 or block_size % interleave_size:
        raise ValueError(
            f"interleave_size={interleave_size} must divide block_size={block_size}"
        )

    src_ids = np.asarray(src_block_ids, dtype=np.int64)
    if src_ids.ndim != 1:
        raise ValueError("src_block_ids must be one-dimensional")
    if dst_pages is None:
        dst_pages = (src_ids.size + dcp_size - 1) // dcp_size
    dst_pages = int(dst_pages)
    if dst_pages < 0:
        raise ValueError(f"dst_pages must be nonnegative, got {dst_pages}")

    local_token = np.arange(0, dst_pages * block_size, interleave_size, dtype=np.int64)
    dst_page, dst_token = np.divmod(local_token, block_size)
    global_token = (
        (local_token // interleave_size) * dcp_size + dcp_rank
    ) * interleave_size
    src_ordinal, src_token = np.divmod(global_token, block_size)
    valid = src_ordinal < src_ids.size

    src_block_id_per_run = np.zeros(local_token.size, dtype=np.int64)
    if src_ids.size:
        safe_src_ordinal = np.minimum(src_ordinal, src_ids.size - 1)
        src_block_id_per_run[:] = src_ids[safe_src_ordinal]

    return DCPShardPlan(
        block_size=block_size,
        interleave_size=interleave_size,
        dst_pages=dst_pages,
        src_block_id_per_run=src_block_id_per_run,
        src_token=src_token,
        dst_page=dst_page,
        dst_token=dst_token,
        run_length=np.full(local_token.size, interleave_size, dtype=np.int64),
        valid=valid,
    )
