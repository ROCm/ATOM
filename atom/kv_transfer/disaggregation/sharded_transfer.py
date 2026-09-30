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
    split_before: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge adjacent runs that are contiguous on both sides.

    A run whose ``split_before`` entry is True always starts a new merged run,
    e.g. where the destination crosses into another registered memory region.
    """

    if src.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    contiguous = (src[1:] == src[:-1] + length[:-1]) & (
        dst[1:] == dst[:-1] + length[:-1]
    )
    if split_before is not None:
        contiguous &= ~split_before[1:]
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

    def landing_source_tokens(self) -> np.ndarray:
        """Producer token of every row a landing slot carries, in row order.

        Rows are the valid destination tokens in destination order, unpadded:
        row ``j`` lands at token ``j % block_size`` of destination page
        ``j // block_size``, because the valid tokens are a page prefix (see
        ``staged_page_runs``). The decode side relies on exactly that mapping
        to scatter the rows (``landing_scatter.landing_segments``).
        """

        if self.valid.size and (self.valid[1:] > self.valid[:-1]).any():
            raise ValueError("DCP shard plan valid tokens are not a prefix")
        keep = self.valid
        run_start = self.src_block_id_per_run[keep] * self.block_size
        run_start = run_start + self.src_token[keep]
        run_length = self.run_length[keep]
        if (run_length == 1).all():
            return run_start
        return np.repeat(run_start - np.cumsum(run_length) + run_length, run_length) + (
            np.arange(int(run_length.sum()), dtype=np.int64)
        )

    def source_token_per_dst_token(self) -> np.ndarray:
        """Source token index for every destination token, page-major.

        Row ``p * block_size + t`` names the producer token (``block_id *
        block_size + token``) that lands at token ``t`` of destination page
        ``p``. Tokens past the source end read token 0; ``staged_page_runs``
        never sends them.
        """

        run_start = self.src_block_id_per_run * self.block_size + self.src_token
        run_start = np.where(self.valid, run_start, 0)
        offsets = np.arange(self.interleave_size, dtype=np.int64)
        valid_offsets = np.where(self.valid[:, None], offsets, 0)
        return (run_start[:, None] + valid_offsets).reshape(-1)

    def staged_page_runs(
        self,
        dst_block_ids: Sequence[int],
        staging_addr: int,
        dst_base: int,
        token_bytes: int,
        dst_pages_per_mr: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Byte descriptors for pages staged page-major at ``staging_addr``.

        The staging buffer holds ``dst_pages`` pages in destination token order
        (see ``source_token_per_dst_token``). Each page sends only its valid
        tokens, which the plan keeps as a page prefix because a rank's global
        positions grow with its local ones, so the bytes written are exactly
        those of ``token_runs``. Runs adjacent on both ends are merged, except
        across a destination page that is a multiple of ``dst_pages_per_mr``:
        the consumer registers its region as memory regions of that many
        pages, and one RDMA op must stay inside one of them.
        """

        dst_ids = np.asarray(dst_block_ids, dtype=np.int64)
        if dst_ids.size != self.dst_pages:
            raise ValueError(
                f"DCP shard plan has {self.dst_pages} destination pages, got "
                f"{dst_ids.size} block ids"
            )
        if self.valid.size and (self.valid[1:] > self.valid[:-1]).any():
            raise ValueError("DCP shard plan valid tokens are not a prefix")
        runs_per_page = self.block_size // self.interleave_size
        valid_tokens = (
            self.valid.reshape(self.dst_pages, runs_per_page).sum(axis=1)
            * self.interleave_size
        )
        page_bytes = self.block_size * token_bytes
        keep = valid_tokens > 0
        src = staging_addr + np.flatnonzero(keep).astype(np.int64) * page_bytes
        dst = dst_base + dst_ids[keep] * page_bytes
        return coalesce_contiguous(
            src,
            dst,
            valid_tokens[keep] * token_bytes,
            dst_ids[keep] % dst_pages_per_mr == 0,
        )


def pack_landing_rows(
    row_bytes: Sequence[int], num_rows: int, slot_bytes: int, page_rows: int
) -> list[list[tuple[int, int, int, int]]]:
    """Pack every region's ``num_rows`` landing rows into fixed-size slots.

    Regions are packed in order, each split into row ranges that fill the
    current slot before a new one opens. Returns one list per slot of
    ``(region_pos, row_start, row_stop, slot_offset)`` items, where the rows
    occupy ``[slot_offset, slot_offset + (row_stop - row_start) * width)``.
    Unlike ``pack_staging_slots`` there is no page padding: the rows are the
    unpadded ``DCPShardPlan.landing_source_tokens`` order. A region is only
    split at a multiple of ``page_rows``, so the rows a sender has not landed
    yet always start on a destination page and can go out as whole pages.
    """

    if slot_bytes <= 0:
        raise ValueError(f"slot_bytes must be positive, got {slot_bytes}")
    slots: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = []
    used = 0
    for region_pos, width in enumerate(row_bytes):
        if width * page_rows > slot_bytes:
            raise ValueError(
                f"Landing slot of {slot_bytes} bytes cannot hold one "
                f"{page_rows}-row page of {width}-byte rows"
            )
        row = 0
        while row < num_rows:
            capacity = (slot_bytes - used) // width
            if row + capacity < num_rows:
                capacity -= capacity % page_rows
            if capacity == 0:
                slots.append(current)
                current, used = [], 0
                continue
            stop = min(num_rows, row + capacity)
            current.append((region_pos, row, stop, used))
            used += (stop - row) * width
            row = stop
    if current:
        slots.append(current)
    return slots


def pack_staging_slots(
    page_bytes: Sequence[int],
    dst_pages: int,
    slot_bytes: int,
    first_pages: Sequence[int] | None = None,
) -> list[list[tuple[int, int, int, int]]]:
    """Pack every (region, destination page) into fixed-size staging slots.

    Regions are packed in order, each split into page ranges that fill the
    current slot before a new one opens. Returns one list per slot of
    ``(region_pos, page_start, page_stop, slot_offset)`` items, where
    ``region_pos`` indexes ``page_bytes`` and the pages occupy
    ``[slot_offset, slot_offset + (page_stop - page_start) * width)``.
    ``first_pages`` optionally starts each region at a later page (its
    earlier pages were already sent).
    """

    if slot_bytes <= 0:
        raise ValueError(f"slot_bytes must be positive, got {slot_bytes}")
    slots: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = []
    used = 0
    for region_pos, width in enumerate(page_bytes):
        if width > slot_bytes:
            raise ValueError(
                f"Staging slot of {slot_bytes} bytes cannot hold one "
                f"{width}-byte page"
            )
        page = 0 if first_pages is None else first_pages[region_pos]
        while page < dst_pages:
            capacity = (slot_bytes - used) // width
            if capacity == 0:
                slots.append(current)
                current, used = [], 0
                continue
            stop = min(dst_pages, page + capacity)
            current.append((region_pos, page, stop, used))
            used += (stop - page) * width
            page = stop
    if current:
        slots.append(current)
    return slots


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
