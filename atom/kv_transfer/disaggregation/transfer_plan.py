# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Map P/D rank ownership to physical gather units and destination PAGEs.

Topology selects token ownership; a PAGE plan takes the head cuts its caller
already validated. Region metadata selects byte addressing. The transport
consumes TransferPlan without knowing whether the model uses MLA, MHA, or
another attention family.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from atom.kv_transfer.disaggregation.gather import GatherLayout
from atom.kv_transfer.disaggregation.landing_scatter import landing_segments
from atom.kv_transfer.disaggregation.sharded_transfer import (
    DCPShardPlan,
    build_dcp_shard_plan,
    coalesce_contiguous,
    pack_slots,
)

TOKEN_UNITS = "token_rows_v1"
PAGE_UNITS = "page_units_v1"


@dataclass(frozen=True)
class TransferTopology:
    target_dcp: int = 1
    target_dcp_rank: int = 0
    interleave: int = 1


@dataclass(frozen=True)
class RegionLayout:
    source_region: int
    target_region: int
    target_base: int
    source_page_bytes: int
    target_page_bytes: int


@dataclass(frozen=True)
class PlannedRegion:
    source_region: int
    target_region: int
    target_base: int
    gather: GatherLayout


@dataclass(frozen=True)
class TransferPlan:
    regions: tuple[PlannedRegion, ...]
    source_units: np.ndarray
    destination_pages: np.ndarray
    units_per_page: int
    wire_layout: str

    def pack(self, slot_bytes: int) -> list[list[tuple[int, int, int, int]]]:
        return pack_slots(
            [r.gather.unit_bytes for r in self.regions],
            self.source_units.size,
            slot_bytes,
            self.units_per_page,
        )

    def ready_items(
        self, items: Sequence[tuple[int, int, int, int]], page_offset: int
    ) -> list[list[int]]:
        return [
            [
                self.regions[pos].target_region,
                page_offset * self.units_per_page + start,
                stop - start,
                offset,
            ]
            for pos, start, stop, offset in items
        ]

    def direct_descriptors(
        self,
        items: Sequence[tuple[int, int, int, int]],
        staging_base: int,
        pages_per_mr: Callable[[int], int],
    ) -> tuple[list[int], list[int], list[int]]:
        """Coalesce destination-adjacent PAGEs within one region and one MR."""
        runs = []
        for pos, start, stop, offset in items:
            region = self.regions[pos]
            page_bytes = region.gather.unit_bytes * self.units_per_page
            src, dst, size = landing_segments(
                np.array([start]),
                np.array([stop - start]),
                np.array([staging_base + offset]),
                np.array([region.target_base]),
                np.array([page_bytes]),
                self.destination_pages,
                self.units_per_page,
                0,
            )
            mr = (dst - region.target_base) // (pages_per_mr(page_bytes) * page_bytes)
            boundary = np.concatenate(([True], mr[1:] != mr[:-1]))
            runs.append(coalesce_contiguous(src, dst, size, boundary))
        return tuple(np.concatenate(part).tolist() for part in zip(*runs))


def page_transfer_plan(
    regions: Sequence[PlannedRegion],
    source_pages: Sequence[int],
    destination_pages: Sequence[int],
) -> TransferPlan:
    """Plan one destination PAGE per source PAGE.

    Each region's gather already holds the consumer rank's cut of a source
    PAGE (``GatherLayout.from_slices`` over the connector's validated head
    cuts), so the plan only pairs the two block lists.
    """
    return TransferPlan(
        tuple(regions),
        np.asarray(source_pages, dtype=np.int64),
        np.asarray(destination_pages, dtype=np.int64),
        1,
        PAGE_UNITS,
    )


def build_transfer_plan(
    topology: TransferTopology,
    layouts: Sequence[RegionLayout],
    source_pages: Sequence[int],
    destination_pages: Sequence[int],
    block_size: int,
    *,
    token_ownership: DCPShardPlan | None = None,
) -> TransferPlan:
    """Plan a non-DCP producer's PAGEs onto one DCP consumer rank.

    PP layer ownership is already expressed by each region's source/target map.
    DCP ownership is calculated before addressing bytes; a TP-replicated plane
    is owned by token alone, whatever the two TP sizes. Each region's source
    and target PAGEs must be equally wide and hold whole token rows.
    token_ownership may reuse the same topology's plan from the direct/index
    path, so a mixed-region transfer does not calculate DCP ownership twice.
    """
    t = topology
    destination_pages = np.asarray(destination_pages, dtype=np.int64)
    ownership = (
        token_ownership
        if token_ownership is not None
        else build_dcp_shard_plan(
            source_pages,
            block_size=block_size,
            dcp_size=t.target_dcp,
            dcp_rank=t.target_dcp_rank,
            interleave_size=t.interleave,
            dst_pages=destination_pages.size,
        )
    )
    indices = ownership.source_token_per_dst_token()[: ownership.landing_row_count()]
    regions = []
    for layout in layouts:
        width = layout.source_page_bytes // block_size
        regions.append(
            PlannedRegion(
                layout.source_region,
                layout.target_region,
                layout.target_base,
                GatherLayout(width, width, ((0, width),)),
            )
        )
    return TransferPlan(
        tuple(regions), indices, destination_pages, block_size, TOKEN_UNITS
    )
