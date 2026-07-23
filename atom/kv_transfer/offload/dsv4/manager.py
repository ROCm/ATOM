# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Worker-side save/load orchestration for DSV4 terminal checkpoints.

Ties together the pieces built in the other modules into the two operations the
worker performs, mirroring the Mooncake/NIXL producer + consumer lifecycle but
against a CPU/NVMe :class:`UnitStore` instead of RDMA:

**save (best-effort, never blocks the request)** — at a 128-aligned prefill end:
  1. ``try_admit`` a state-pool slot + in-flight credit; skip on exhaustion.
  2. ``gather_slot(compute_slot, pool_idx)`` snapshots the CSA state @B into the
     ``v4_state_pool`` slot (so decode may immediately overwrite the compute slot).
  3. pin the live SWA pages so the allocator can't reuse them mid-gather.
  4. gather compressed + SWA tail + CSA-state into one offload unit and ``put`` it.
  5. on completion, unpin SWA and release the slot/credit.

**load (fail-closed)** — for a resumable terminal boundary ``B``:
  1. ``get`` the unit; miss => recompute.
  2. ``load_unit`` validates (magic/version/fingerprint/size/CRC/boundary) and
     scatters compressed + SWA tail into the freshly-allocated dest pages and the
     CSA state into a load pool slot; any failure => fail closed => recompute.
  3. ``scatter_slot(compute_slot, pool_idx)`` moves the CSA state pool->compute.

The real LMCache CPU/NVMe engine plugs in as a :class:`UnitStore`; the in-memory
store here backs unit tests and smoke runs.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

import torch

from atom.kv_transfer.offload.dsv4.gpu_connector import (
    DSV4OffloadUnitGPUConnector,
)
from atom.kv_transfer.offload.dsv4.admission import (
    DSV4CheckpointAdmission,
    DSV4SwaIoPins,
)
from atom.kv_transfer.offload.dsv4.sources import DSV4OffloadSources
from atom.kv_transfer.offload.dsv4.unit_codec import DSV4OffloadUnitError

logger = logging.getLogger("atom")


class UnitStore(Protocol):
    """A keyed store of opaque offload units (LMCache CPU/NVMe tier, or in-mem)."""

    def put(self, key: str, unit_host: torch.Tensor) -> None: ...

    def get(self, key: str) -> "torch.Tensor | None": ...

    def contains(self, key: str) -> bool: ...


class InMemoryUnitStore:
    """Trivial dict-backed :class:`UnitStore` for tests / single-process smoke."""

    def __init__(self) -> None:
        self._d: dict[str, torch.Tensor] = {}

    def put(self, key: str, unit_host: torch.Tensor) -> None:
        self._d[str(key)] = unit_host.clone()

    def get(self, key: str) -> "torch.Tensor | None":
        v = self._d.get(str(key))
        return None if v is None else v.clone()

    def contains(self, key: str) -> bool:
        return str(key) in self._d


def _default_alloc(nbytes: int) -> torch.Tensor:
    return torch.empty(int(nbytes), dtype=torch.uint8)


class DSV4OffloadManager:
    """Orchestrates DSV4 checkpoint save/load on the worker."""

    def __init__(
        self,
        *,
        connector: DSV4OffloadUnitGPUConnector,
        sources: DSV4OffloadSources,
        store: UnitStore,
        admission: DSV4CheckpointAdmission,
        swa_pins: DSV4SwaIoPins,
        alloc_host_unit: Callable[[int], torch.Tensor] | None = None,
    ) -> None:
        self.connector = connector
        self.sources = sources
        self.store = store
        self.admission = admission
        self.swa_pins = swa_pins
        self._alloc = alloc_host_unit or _default_alloc

    # -- save (best-effort, non-blocking) --------------------------------
    def try_save_checkpoint(
        self,
        *,
        key: str,
        block_table,
        swa_block_table,
        compute_slot: int,
        B: int,
    ) -> bool:
        """Attempt to capture + store a terminal checkpoint. Never blocks.

        Returns True if a unit was stored (or already present), False if the
        boundary is ineligible, resources are exhausted, or the save errored.
        """
        if not self.sources.is_checkpoint_boundary(B):
            return False
        if self.store.contains(key):
            return True  # already offloaded; nothing to do
        pool_idx = self.admission.try_admit()
        if pool_idx is None:
            logger.debug("DSV4 offload: admission skip (resources busy) key=%s B=%d", key, B)
            return False

        live_swa = [int(b) for b in swa_block_table if int(b) >= 0]
        pinned = False
        try:
            # Snapshot CSA state @B into the pool slot; decode may then advance.
            if self.sources.gather_slot is not None:
                self.sources.gather_slot(int(compute_slot), int(pool_idx))
            self.swa_pins.pin(live_swa)
            pinned = True

            compressed, swa, csa_state = self.sources.build_save_sources(
                block_table=block_table,
                swa_block_table=swa_block_table,
                state_pool_idx=pool_idx,
                B=B,
            )
            nbytes = self.sources.unit_bytes(
                block_table=block_table, swa_block_table=swa_block_table, B=B
            )
            host = self._alloc(nbytes)
            self.connector.save_unit(
                memory_obj=host,
                compressed=compressed,
                swa=swa,
                csa_state=csa_state,
                boundary_B=B,
            )
            self.store.put(key, host)
            return True
        except Exception:  # best-effort: a failed save just loses the opportunity
            logger.exception("DSV4 offload: save failed key=%s B=%d", key, B)
            return False
        finally:
            if pinned:
                self.swa_pins.unpin(live_swa)
            self.admission.complete(pool_idx)

    # -- load (fail-closed) ----------------------------------------------
    def load_checkpoint(
        self,
        *,
        key: str,
        block_table,
        swa_block_table,
        compute_slot: int,
        state_pool_idx: int,
        B: int,
    ) -> bool:
        """Load + restore a terminal checkpoint. Returns False (=> recompute) on
        miss or any validation failure. The caller must have allocated the dest
        compressed/SWA pages and reserved ``state_pool_idx``."""
        if not self.sources.is_checkpoint_boundary(B):
            raise ValueError(f"DSV4 offload load: B={B} not a checkpoint boundary")
        host = self.store.get(key)
        if host is None:
            return False
        compressed = self.sources.compressed_sources(block_table, B)
        swa = self.sources.swa_sources(swa_block_table)
        csa_state = self.sources.csa_state_sources(state_pool_idx)
        try:
            self.connector.load_unit(
                memory_obj=host,
                compressed=compressed,
                swa=swa,
                csa_state=csa_state,
                expect_boundary_B=B,
            )
        except DSV4OffloadUnitError:
            logger.warning("DSV4 offload: load fail-closed key=%s B=%d", key, B)
            return False
        # Move the restored CSA state from the pool slot into the compute slot.
        if csa_state and self.sources.scatter_slot is not None:
            self.sources.scatter_slot(int(compute_slot), int(state_pool_idx))
        return True
