# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Map a DSV4 terminal boundary ``B`` onto offload-unit source regions.

This is the DSV4-specific glue between the builder's ``KVTransferTensors``
(``get_kv_transfer_tensors()``) and the geometry-agnostic offload-unit connector.
It answers three questions for a request at terminal boundary ``B``:

* **admission** — is ``B`` a valid checkpoint boundary? (``B % 128 == 0``; a full
  window exists). See ``dsv4-lmcache-bundle-plan.md`` Phase 0.
* **which physical blocks** — the compressed prefix ``[0, B)`` maps to
  ``seq.block_table[:ceil(B/block_size)]`` (all compressed regions share this id
  list, same as the RDMA path); the SWA terminal tail maps to the live (``>= 0``)
  entries of ``seq.swa_block_table``; the CSA overlap state is one ``v4_state_pool``
  slot (populated by ``gather_slot``).
* **how big** — the resulting offload-unit byte size, for LMCache metadata.

Address semantics mirror the Mooncake/NIXL producer path
(``src_base + block_id * unit_bytes``), so a saved unit is byte-compatible with
how attention reads its own layout back.
"""

from __future__ import annotations

import logging
from typing import Sequence

from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.kv_transfer.offload.dsv4.gpu_connector import SourceRegion
from atom.kv_transfer.offload.dsv4.unit import DSV4OffloadUnitLayout

logger = logging.getLogger("atom")

# DSV4 SWA window / HCA compression block; checkpoint boundaries must align here.
CHECKPOINT_ALIGN = 128


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


class DSV4OffloadSources:
    """Builds offload-unit source regions from a builder ``KVTransferTensors``."""

    def __init__(
        self,
        transfer_tensors: KVTransferTensors,
        *,
        block_size: int,
        window_size: int = CHECKPOINT_ALIGN,
    ) -> None:
        tt = transfer_tensors
        self.block_size = int(block_size)
        self.window_size = int(window_size)
        if self.block_size <= 0:
            raise ValueError("DSV4OffloadSources: block_size must be > 0")
        # (base_addr, unit_bytes) per region, mirroring the RDMA registration.
        self._block_regions = [(r.base_addr, int(r.unit_bytes)) for r in tt.block_regions]
        self._swa_regions = [
            (r.base_addr, int(r.unit_bytes)) for r in tt.swa_block_regions
        ]
        self._staging_base = (
            tt.staging_region.base_addr if tt.staging_region is not None else 0
        )
        self._staging_slot_bytes = (
            int(tt.staging_region.unit_bytes) if tt.staging_region is not None else 0
        )
        self.staging_pool_size = int(tt.staging_pool_size)
        self.gather_slot = tt.gather_slot
        self.scatter_slot = tt.scatter_slot
        if not self._block_regions:
            raise ValueError("DSV4OffloadSources: no compressed block regions")

    # -- admission (Phase 0) ---------------------------------------------
    def is_checkpoint_boundary(self, B: int) -> bool:
        """A valid terminal DSV4 checkpoint boundary: 128-aligned, >= one window."""
        B = int(B)
        return B >= self.window_size and B % CHECKPOINT_ALIGN == 0

    # -- id lists ---------------------------------------------------------
    def num_compressed_blocks(self, B: int) -> int:
        return _cdiv(B, self.block_size)

    def _compressed_ids(self, block_table: Sequence[int], B: int) -> list[int]:
        n = self.num_compressed_blocks(B)
        ids = [int(b) for b in list(block_table)[:n]]
        if len(ids) != n:
            raise ValueError(
                "DSV4OffloadSources: block_table too short for B="
                f"{B}: need {n} blocks, have {len(block_table)}"
            )
        return ids

    @staticmethod
    def _live_swa_ids(swa_block_table: Sequence[int]) -> list[int]:
        # Window-freeing leaves only the live tail as non-(-1) entries.
        return [int(b) for b in swa_block_table if int(b) >= 0]

    # -- source-region builders ------------------------------------------
    def compressed_sources(
        self, block_table: Sequence[int], B: int
    ) -> list[SourceRegion]:
        ids = self._compressed_ids(block_table, B)
        return [SourceRegion(base, bpb, ids) for base, bpb in self._block_regions]

    def swa_sources(self, swa_block_table: Sequence[int]) -> list[SourceRegion]:
        ids = self._live_swa_ids(swa_block_table)
        return [SourceRegion(base, bpb, ids) for base, bpb in self._swa_regions]

    def csa_state_sources(self, pool_idx: int) -> list[SourceRegion]:
        if self._staging_slot_bytes <= 0:
            return []
        if not (0 <= int(pool_idx) < self.staging_pool_size):
            raise ValueError(
                f"DSV4OffloadSources: pool_idx {pool_idx} out of range "
                f"[0, {self.staging_pool_size})"
            )
        return [SourceRegion(self._staging_base, self._staging_slot_bytes, [int(pool_idx)])]

    def build_save_sources(
        self,
        *,
        block_table: Sequence[int],
        swa_block_table: Sequence[int],
        state_pool_idx: int,
        B: int,
    ) -> tuple[list[SourceRegion], list[SourceRegion], list[SourceRegion]]:
        """(compressed, swa, csa_state) SourceRegion lists for a terminal B.

        Pure region descriptors — the caller must have already run
        ``gather_slot(compute_slot, state_pool_idx)`` so the ``v4_state_pool``
        slot holds the @B snapshot before the save reads it.
        """
        if not self.is_checkpoint_boundary(B):
            raise ValueError(
                f"DSV4OffloadSources: B={B} is not a 128-aligned checkpoint boundary"
            )
        return (
            self.compressed_sources(block_table, B),
            self.swa_sources(swa_block_table),
            self.csa_state_sources(state_pool_idx),
        )

    # -- sizing (for LMCache metadata) -----------------------------------
    def component_bytes(
        self, *, block_table: Sequence[int], swa_block_table: Sequence[int], B: int
    ) -> tuple[int, int, int]:
        n = self.num_compressed_blocks(B)
        compressed = sum(bpb * n for _, bpb in self._block_regions)
        n_swa = len(self._live_swa_ids(swa_block_table))
        swa = sum(bpb * n_swa for _, bpb in self._swa_regions)
        state = self._staging_slot_bytes
        return compressed, swa, state

    def unit_bytes(
        self, *, block_table: Sequence[int], swa_block_table: Sequence[int], B: int
    ) -> int:
        c, s, st = self.component_bytes(
            block_table=block_table, swa_block_table=swa_block_table, B=B
        )
        return DSV4OffloadUnitLayout.build(
            compressed_bytes=c, swa_bytes=s, csa_state_bytes=st
        ).total_bytes

    # -- startup logging (Phase 0) ---------------------------------------
    def log_sizing(self, *, rank: int, example_B: int, example_swa_blocks: int) -> None:
        n = self.num_compressed_blocks(example_B)
        compressed = sum(bpb * n for _, bpb in self._block_regions)
        swa = sum(bpb * example_swa_blocks for _, bpb in self._swa_regions)
        state = self._staging_slot_bytes
        layout = DSV4OffloadUnitLayout.build(
            compressed_bytes=compressed, swa_bytes=swa, csa_state_bytes=state
        )
        bpt = layout.total_bytes / example_B if example_B else 0.0
        logger.info(
            "DSV4 offload sizing rank=%d @B=%d: compressed=%d SWA=%d CSA-state=%d "
            "unit=%d bytes (%.2f bytes/token); compressed_regions=%d swa_regions=%d "
            "state_pool=%d slot_bytes=%d",
            rank,
            example_B,
            compressed,
            swa,
            state,
            layout.total_bytes,
            bpt,
            len(self._block_regions),
            len(self._swa_regions),
            self.staging_pool_size,
            self._staging_slot_bytes,
        )
