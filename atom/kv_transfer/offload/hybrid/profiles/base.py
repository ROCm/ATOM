# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Profile primitives + generic ``BundleSources`` for the hybrid layout family.

A :class:`HybridProfile` declares the ordered unit components and the region
kind each is sourced from. :class:`BundleSources` turns a profile + a builder
``KVTransferTensors`` into the ordered ``(name, [SourceRegion])`` component lists
consumed by ``BundleGPUConnector.save_bundle / load_bundle``.

Region kinds (how physical ids are chosen for a component):

* ``BLOCK``   — ``seq.block_table[:ceil(B / block_size)]`` (compressed / full KV)
* ``SWA``     — live (``>= 0``) entries of ``seq.swa_block_table`` (window tail)
* ``SLOT``    — one per-request slot id (e.g. GDN conv/temporal recurrent state)
* ``STAGING`` — one ``staging_region`` pool slot, populated by ``gather_slot``
                before save and drained by ``scatter_slot`` after load

DSV4 profile = (compressed=BLOCK, swa=SWA, csa_state=STAGING). A hybrid-KV model
(Qwen3-Next) would be (full_kv=BLOCK, gdn_state=SLOT).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.kv_transfer.offload.hybrid.gpu_connector import SourceRegion
from atom.kv_transfer.offload.hybrid.kv_bundle import BundleGeometry, BundleLayout


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


class RegionKind(Enum):
    BLOCK = "block"
    SWA = "swa"
    SLOT = "slot"
    STAGING = "staging"


@dataclass(frozen=True)
class ComponentSpec:
    """One ordered unit component and the region kind it is sourced from."""

    name: str
    kind: RegionKind


@dataclass(frozen=True)
class SaveCadence:
    """When a terminal checkpoint boundary ``B`` is valid for save/resume."""

    align: int
    min_len: int


@dataclass(frozen=True)
class HybridProfile:
    """Per-model description consumed by the hybrid layout / connector."""

    model_tag: str
    components: tuple[ComponentSpec, ...]
    cadence: SaveCadence
    block_size: int
    build_geometry: Callable[..., BundleGeometry]

    def component_names(self) -> list[str]:
        return [c.name for c in self.components]

    def is_checkpoint_boundary(self, B: int) -> bool:
        B = int(B)
        return B >= self.cadence.min_len and B % self.cadence.align == 0


class BundleSources:
    """Builds ordered unit component sources from a profile + transfer tensors."""

    def __init__(
        self,
        profile: HybridProfile,
        transfer_tensors: KVTransferTensors,
    ) -> None:
        tt = transfer_tensors
        self.profile = profile
        self.block_size = int(profile.block_size)
        if self.block_size <= 0:
            raise ValueError("BundleSources: block_size must be > 0")
        self._by_kind: dict[RegionKind, list[tuple[int, int]]] = {
            RegionKind.BLOCK: [(r.base_addr, int(r.unit_bytes)) for r in tt.block_regions],
            RegionKind.SWA: [
                (r.base_addr, int(r.unit_bytes)) for r in tt.swa_block_regions
            ],
            RegionKind.SLOT: [(r.base_addr, int(r.unit_bytes)) for r in tt.slot_regions],
        }
        self._staging_base = (
            tt.staging_region.base_addr if tt.staging_region is not None else 0
        )
        self._staging_slot_bytes = (
            int(tt.staging_region.unit_bytes) if tt.staging_region is not None else 0
        )
        self.staging_pool_size = int(tt.staging_pool_size)
        self.gather_slot = tt.gather_slot
        self.scatter_slot = tt.scatter_slot

    # -- id selection per kind -------------------------------------------
    def _block_ids(self, block_table: Sequence[int], B: int) -> list[int]:
        n = _cdiv(B, self.block_size)
        ids = [int(b) for b in list(block_table)[:n]]
        if len(ids) != n:
            raise ValueError(
                f"BundleSources: block_table too short for B={B}: need {n}, have "
                f"{len(block_table)}"
            )
        return ids

    @staticmethod
    def _live_swa_ids(swa_block_table: Sequence[int]) -> list[int]:
        return [int(b) for b in swa_block_table if int(b) >= 0]

    def _regions_for(
        self,
        spec: ComponentSpec,
        *,
        block_table: Sequence[int],
        swa_block_table: Sequence[int],
        B: int,
        slot_id: int,
        pool_idx: int,
    ) -> list[SourceRegion]:
        if spec.kind is RegionKind.STAGING:
            if self._staging_slot_bytes <= 0:
                return []
            if not (0 <= int(pool_idx) < self.staging_pool_size):
                raise ValueError(
                    f"BundleSources: pool_idx {pool_idx} out of range "
                    f"[0, {self.staging_pool_size})"
                )
            return [SourceRegion(self._staging_base, self._staging_slot_bytes, [int(pool_idx)])]

        if spec.kind is RegionKind.BLOCK:
            ids = self._block_ids(block_table, B)
        elif spec.kind is RegionKind.SWA:
            ids = self._live_swa_ids(swa_block_table)
        elif spec.kind is RegionKind.SLOT:
            if int(slot_id) < 0:
                raise ValueError(f"BundleSources: SLOT component {spec.name!r} needs slot_id")
            ids = [int(slot_id)]
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"BundleSources: unknown region kind {spec.kind}")

        return [SourceRegion(base, bpb, ids) for base, bpb in self._by_kind[spec.kind]]

    # -- component builders ----------------------------------------------
    def build_components(
        self,
        *,
        block_table: Sequence[int],
        swa_block_table: Sequence[int],
        B: int,
        slot_id: int = -1,
        pool_idx: int = -1,
    ) -> list[tuple[str, list[SourceRegion]]]:
        """Ordered ``(name, [SourceRegion])`` list, one entry per profile component."""
        return [
            (
                spec.name,
                self._regions_for(
                    spec,
                    block_table=block_table,
                    swa_block_table=swa_block_table,
                    B=B,
                    slot_id=slot_id,
                    pool_idx=pool_idx,
                ),
            )
            for spec in self.profile.components
        ]

    def unit_bytes(
        self,
        *,
        block_table: Sequence[int],
        swa_block_table: Sequence[int],
        B: int,
        slot_id: int = -1,
        pool_idx: int = -1,
    ) -> int:
        comps = self.build_components(
            block_table=block_table,
            swa_block_table=swa_block_table,
            B=B,
            slot_id=slot_id,
            pool_idx=pool_idx,
        )
        sizes = [
            (name, sum(r.nbytes() for r in regs)) for name, regs in comps
        ]
        return BundleLayout.build(sizes).total_bytes
