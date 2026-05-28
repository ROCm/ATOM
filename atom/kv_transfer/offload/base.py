# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Abstract base classes for KV-cache offload connectors.

OFFLOAD connectors plug into the same engine hooks as PD-disaggregation
connectors (KVConnectorFactory registry, scheduler-side metadata, worker-
side per-step dispatch) but represent a *local* L2/L3 store (CPU DRAM,
NVMe) instead of a remote prefill/decode peer. See the package README for
the full data flow.

Worker-side (one per TP rank): owns the external store, issues D2H copies
on save and H2D copies on load.

Scheduler-side: looks up external-store hits during prefill admission,
queues blocks for save after BlockManager publishes new hashes, packages
both into per-step metadata.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorRole,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.offload.types import OffloadConnectorMetadata

if TYPE_CHECKING:
    from atom.model_engine.sequence import Sequence


class OffloadConnectorBase(KVConnectorBase):
    """Worker-side interface for a local KV-offload backend."""

    role = KVConnectorRole.OFFLOAD
    # OFFLOAD connectors are neither PD producer nor consumer; this field
    # is kept for back-compat with code paths still gating on the binary,
    # but the scheduler routes via `role` for offload-specific behavior.
    is_producer = False

    @abstractmethod
    def start_load_kv(self, metadata: OffloadConnectorMetadata) -> None:
        """Issue async H2D copies for pending loads, async D2H for pending
        saves. Called once per engine step on each worker. The single entry
        point matches the existing PD dispatch (``process_kvconnector_output``
        in EngineCore) so no new IPC channel is required.
        """
        ...

    @abstractmethod
    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return ``(done_saving, done_loading)`` request IDs since the last
        call. ``done_loading`` IDs propagate back to the scheduler so seqs
        waiting in ``WAITING_FOR_REMOTE_KVS`` can resume.
        """
        ...

    def get_failed_load(self) -> set[str]:
        """Return request IDs whose load attempt could not be satisfied
        since the last call (e.g. the optimistic scheduler-side mirror
        said HIT but the worker's pool no longer has the block).

        Default: empty set. Backends that may evict between scheduler
        lookup and worker load (FIFO/LRU pools) MUST override and surface
        the failed req_ids here, otherwise the scheduler will leave the
        request in ``WAITING_FOR_REMOTE_KVS`` with uninitialized GPU
        blocks and attention will read garbage KV.
        """
        return set()


class OffloadConnectorSchedulerBase(KVConnectorSchedulerBase):
    """Scheduler-side interface for a local KV-offload backend."""

    role = KVConnectorRole.OFFLOAD
    is_producer = False

    def bind_block_manager(self, block_manager: Any) -> None:
        """Give the connector a reference to the engine's BlockManager.

        Default impl stashes it as ``self.block_manager``. Override only if
        a connector needs to register listeners or precompute geometry.
        Called by ``Scheduler.__init__`` after both objects exist.
        """
        self.block_manager = block_manager

    @abstractmethod
    def lookup_external_hits(self, seq: "Sequence", block_hashes: list[int]) -> int:
        """Return the number of leading blocks of ``block_hashes`` that are
        present in the external store. ``block_hashes`` is the
        ``BlockManager.compute_hash`` chain for ``seq``'s prompt blocks.

        The scheduler calls this from ``get_num_new_matched_tokens`` after
        the in-HBM prefix-cache check has run; a positive return value
        causes the seq to be marked ``WAITING_FOR_REMOTE_KVS`` until the
        worker's load completes.
        """
        ...

    @abstractmethod
    def queue_save(self, request_id: str, published: list[tuple[int, int]]) -> None:
        """Enqueue freshly-finalized blocks for D2H save.

        Called by the scheduler from ``postprocess`` after
        ``BlockManager.hash_blocks`` returns the list of
        ``(block_id, hash)`` pairs newly published this step.
        Implementations may skip blocks already known to the external store
        (no-op if the connector tracks its own index).
        """
        ...

    @abstractmethod
    def queue_load(
        self, request_id: str, block_ids: list[int], block_hashes: list[int]
    ) -> None:
        """Enqueue blocks to be hydrated from the external store into the
        given GPU paged-block slots. Called by the scheduler from
        ``update_state_after_alloc`` once BlockManager has assigned
        destination slot ids.
        """
        ...

    @abstractmethod
    def build_connector_meta(self) -> OffloadConnectorMetadata | None:
        """Drain accumulated save/load queues into a per-step metadata
        snapshot for dispatch to workers. Returning ``None`` (or an empty
        metadata) signals no work for this step.
        """
        ...

    def handle_failed_load(self, request_id: str) -> list[int]:
        """Called by the scheduler when the worker reports a load failure
        for ``request_id`` (pool eviction between the optimistic mirror
        lookup and the actual H2D copy). The scheduler-side connector
        must drop any mirror entries for the request's external hashes
        so a future lookup does not re-hit the same stale entries, and
        must clear any per-request stash that ``update_state_after_alloc``
        populated.

        Returns the list of hashes that were evicted from the mirror —
        for logging only.

        Default: no-op (PD and OFFLOAD-with-no-eviction backends).
        """
        return []
