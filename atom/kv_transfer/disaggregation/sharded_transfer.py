# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Shared DCP page-relayout planning for KV disaggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from atom.distributed.dcp_layout import dcp_global_pos


def coalesce_contiguous(
    src: np.ndarray,
    dst: np.ndarray,
    length: np.ndarray,
    *,
    src_mr: tuple[int, int] | None = None,
    dst_mr: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge adjacent runs, optionally splitting at either side's MR boundaries.

    Each MR specification is (region_base, chunk_bytes), matching registration's
    regular chunk spacing. Input runs must already lie inside each region;
    the final MR's alignment remainder may be split conservatively. Splitting
    also handles an individual oversized page.
    """
    for mr in (src_mr, dst_mr):
        if mr is not None and mr[1] <= 0:
            raise ValueError("MR chunk bytes must be positive")

    if src.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    contiguous = (src[1:] == src[:-1] + length[:-1]) & (
        dst[1:] == dst[:-1] + length[:-1]
    )
    starts = np.concatenate(([True], ~contiguous))
    start_indices = np.flatnonzero(starts)
    merged_length = np.add.reduceat(length, start_indices)
    merged_src, merged_dst = src[starts], dst[starts]
    if src_mr is None and dst_mr is None:
        return merged_src, merged_dst, merged_length

    merged_src = np.asarray(merged_src, dtype=np.int64)
    merged_dst = np.asarray(merged_dst, dtype=np.int64)
    merged_length = np.asarray(merged_length, dtype=np.int64)
    # A contiguous page batch often merges to one run. Avoid repeat/index
    # expansion for this case: its cut positions are simple MR progressions.
    if merged_length.size == 1:
        size = int(merged_length[0])
        if size == 0:
            return merged_src[:0], merged_dst[:0], merged_length[:0]
        first_src = (
            size
            if src_mr is None
            else src_mr[1] - (int(merged_src[0]) - src_mr[0]) % src_mr[1]
        )
        first_dst = (
            size
            if dst_mr is None
            else dst_mr[1] - (int(merged_dst[0]) - dst_mr[0]) % dst_mr[1]
        )
        if size <= min(first_src, first_dst):
            return merged_src, merged_dst, merged_length
        cuts = [np.array([0, size], dtype=np.int64)]
        if src_mr is not None and first_src < size:
            cuts.append(np.arange(first_src, size, src_mr[1], dtype=np.int64))
        if dst_mr is not None and first_dst < size:
            cuts.append(np.arange(first_dst, size, dst_mr[1], dtype=np.int64))
        offsets = np.unique(np.concatenate(cuts))
        return (
            merged_src[0] + offsets[:-1],
            merged_dst[0] + offsets[:-1],
            np.diff(offsets),
        )

    # Split one side at a time: destination splits preserve all source MR
    # boundaries. The two passes operate on whole arrays, never on Python
    # address lists or individual runs. No padded runs-by-boundaries matrix
    # is needed; repeat allocates only the actual output descriptors.
    for side, mr in enumerate((src_mr, dst_mr)):
        if mr is None:
            continue
        base, chunk = mr
        addresses = merged_src if side == 0 else merged_dst
        first = chunk - (addresses - base) % chunk
        if np.all((merged_length <= first) & (merged_length != 0)):
            continue
        # Count only boundaries strictly inside a run; ending exactly at an
        # MR boundary must not produce an extra, zero-length descriptor.
        counts = 1 + np.maximum(0, (merged_length - first - 1) // chunk + 1)
        counts[merged_length == 0] = 0

        run_ids = np.repeat(np.arange(counts.size), counts)
        group_starts = np.cumsum(counts) - counts
        piece_ids = np.arange(run_ids.size) - group_starts[run_ids]
        first_piece = piece_ids == 0
        offsets = np.where(first_piece, 0, first[run_ids] + (piece_ids - 1) * chunk)
        limits = np.where(first_piece, first[run_ids], chunk)
        merged_src = merged_src[run_ids] + offsets
        merged_dst = merged_dst[run_ids] + offsets
        merged_length = np.minimum(merged_length[run_ids] - offsets, limits)
    return merged_src, merged_dst, merged_length


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
    # Sampled at S-aligned run starts, so local_token % S is 0 and the extra
    # term in dcp_global_pos is unused -- still call the canonical inverse so
    # a layout change in attention cannot silently skip the RDMA plan.
    global_token = dcp_global_pos(local_token, dcp_rank, dcp_size, interleave_size)
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
