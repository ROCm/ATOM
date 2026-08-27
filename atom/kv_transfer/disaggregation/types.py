# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Shared type definitions for the KV cache disaggregation subsystem.

This module is the single source of truth for data structures exchanged
between the scheduler, engine core, worker-side connectors, and the
KV output aggregator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EngineId = str
ReqId = str | int
TransferId = int


def kv_config_has_producer(kv_config: object) -> bool:
    """Whether a KV config contains a P/D producer, including ``multi``.

    Both the scheduler and the model runner branch on this: a producer
    prefills and hands the request off after one token, so it neither defers
    output nor speculates.
    """
    if not isinstance(kv_config, dict):
        return False
    if kv_config.get("kv_role") == "kv_producer":
        return True
    return any(kv_config_has_producer(sub) for sub in kv_config.get("connectors", []))

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class KVTransferRegion:
    """One RDMA-registerable tensor region."""

    base_addr: int
    total_bytes: int
    unit_bytes: int  # bytes per block (block-indexed) or per slot (slot-indexed)
    # Unit `i` sits at `base_addr + total_bytes - (i+1) * unit_bytes` instead of
    # `base_addr + i * unit_bytes`. A pool that numbers its units back from its
    # end does so to keep adding one from relocating the rest; the region map
    # has to know, because both ends compute an address from the same id.
    reverse_indexed: bool = False

    def unit_addr(self, index: int) -> int:
        if self.reverse_indexed:
            return self.base_addr + self.total_bytes - (index + 1) * self.unit_bytes
        return self.base_addr + index * self.unit_bytes


@dataclass
class KVTransferTensors:
    block_regions: list[KVTransferRegion]
    slot_regions: list[KVTransferRegion]
    num_blocks: int
    num_slots: int = 0
    # The sliding window is a per-request ring, not part of the compressed
    # block_table, so it gets its own regions keyed by the request's state
    # slot. `unit_bytes` is one whole ring.
    swa_block_regions: list[KVTransferRegion] = field(default_factory=list)
    staging_region: KVTransferRegion | None = None
    staging_pool_size: int = 0
    gather_slot: Callable[[int, int], None] | None = None
    scatter_slot: Callable[[int, int], None] | None = None
    # The attention metadata builder, for `state_entry_views(group)`. Not a
    # region: the state offload tier reads a group's state as a list of
    # contiguous views rather than one base+stride range, because GDN's group
    # is strided across layers and has no single range. Carried here because
    # `register_kv_caches` is where the tier is built and this struct is the
    # only channel that already runs from the runner to every connector.
    # `None` on every backend that owns no per-request state.
    state_backend: Any | None = None


@dataclass
class KVConnectorOutput:
    """Per-worker snapshot of finished KV cache transfers.

    Each TP worker produces one of these per scheduler step.  The
    :class:`KVOutputAggregator` combines them to determine which
    request IDs have finished on *all* workers.

    Attributes:
        finished_sending: Request IDs whose KV send completed on this worker.
        finished_recving: Request IDs whose KV receive completed on this worker.
        failed_recving: Request IDs whose KV receive failed on this worker.
        finished_saving: Request IDs whose local fire-and-forget save completed.
        finished_loading: Request IDs whose local/offload KV load completed.
        failed_loading: Request IDs whose local/offload KV load failed.
        state_staging_released: State offload staging slots this worker is done
            reading out of.
        state_indexed: State checkpoint hashes this worker stored successfully.
        state_index_failed: State checkpoint hashes this worker failed to store.
            The aggregator takes quorum on ``state_indexed | state_index_failed``
            so a partial-store (some ranks succeed, some fail) resolves instead
            of pinning a key forever.  Hashes that appear here are never emitted
            on ``state_indexed`` — a load reads all shards back, so a partially-
            stored hash is unloadable.
        expected_finished_count: How many finished notifications should be
            expected per request (used by the aggregator).
    """

    finished_sending: set[ReqId] = field(default_factory=set)
    finished_recving: set[ReqId] = field(default_factory=set)
    failed_recving: set[ReqId] = field(default_factory=set)
    finished_saving: set[ReqId] = field(default_factory=set)
    finished_loading: set[ReqId] = field(default_factory=set)
    failed_loading: set[ReqId] = field(default_factory=set)
    # The state offload tier's three reports. Ints, not ReqIds: a staging slot
    # and a content hash have no request identity -- the request that was
    # evicted is long gone by the time its bytes land. Additive with empty
    # defaults, so the four connectors that never set them are unaffected.
    state_staging_released: set[int] = field(default_factory=set)
    state_indexed: set[int] = field(default_factory=set)
    state_index_failed: set[int] = field(default_factory=set)
    expected_finished_count: int = 0

    def is_empty(self) -> bool:
        """Return True if no transfers finished on this worker."""
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.failed_recving
            and not self.finished_saving
            and not self.finished_loading
            and not self.failed_loading
            and not self.state_staging_released
            and not self.state_indexed
            and not self.state_index_failed
        )

    def __repr__(self) -> str:
        return (
            f"KVConnectorOutput(sending={self.finished_sending}, "
            f"recving={self.finished_recving}, "
            f"failed_recving={self.failed_recving}, "
            f"finished_saving={self.finished_saving}, "
            f"loading={self.finished_loading}, "
            f"failed_loading={self.failed_loading}, "
            f"state_staging_released={self.state_staging_released}, "
            f"state_indexed={self.state_indexed}, "
            f"state_index_failed={self.state_index_failed})"
        )


@dataclass
class ReqMeta:
    """Per-request metadata needed for KV cache transfer.

    Captures both local and remote block locations together with
    networking information to reach the remote engine.
    """

    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_handshake_port: int
    remote_engine_id: str
    tp_size: int
    remote_dp_size: int
    remote_dp_rank: int = 0
    remote_pp_size: int = 1
    remote_tp_size: int = 0
    transfer_id: int = 0
    local_slot_index: int = -1
    # Producer's final committed state slot. This is distinct from
    # `local_slot_index`, which is the consumer destination slot. Prefix
    # checkpointing may move the producer request after admission.
    remote_slot_index: int = -1
    # The request's SWA ring slot, as a one-element list so it zips with the
    # region loop like block ids do. Empty for backends with no SWA state.
    local_swa_block_ids: list[int] = field(default_factory=list)
    remote_swa_block_ids: list[int] = field(default_factory=list)


@dataclass
class RemoteAllocInfo:
    """Allocation information received from the remote (decode) side."""

    block_ids: list[int] = field(default_factory=list)
    writes_done: int = 0
    decode_dp_rank: int = 0
    transfer_offset: tuple[list[int], list[int], list[int]] | None = None


@dataclass
class RemoteMeta:
    """Minimal metadata describing a remote block allocation."""

    block_ids: list[int]
    host: str
    port: int
    engine_id: str
    request_id: str


class ConnectorMetadata:
    """Snapshot of pending KV transfer requests, passed from scheduler to workers.

    The scheduler populates this each step with new receive / save requests,
    and the worker-side connector consumes it in ``start_load_kv``.
    """

    def __init__(self) -> None:
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()
        self.request_id_to_transfer_id: dict[ReqId, int] = {}

    @staticmethod
    def _build_req_meta(
        req_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        local_swa_block_ids: list[int] | None = None,
    ) -> ReqMeta:
        """Construct a :class:`ReqMeta` from raw transfer parameters."""
        return ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.get("remote_block_ids"),
            local_swa_block_ids=local_swa_block_ids or [],
            remote_swa_block_ids=kv_transfer_params.get("remote_swa_block_ids", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id"),
            remote_host=kv_transfer_params.get("remote_host"),
            remote_port=kv_transfer_params.get("remote_port"),
            remote_handshake_port=kv_transfer_params.get("remote_handshake_port"),
            remote_dp_size=kv_transfer_params.get("remote_dp_size", 1),
            remote_dp_rank=kv_transfer_params.get("remote_dp_rank", 0),
            remote_pp_size=kv_transfer_params.get("remote_pp_size", 1),
            remote_tp_size=kv_transfer_params.get("remote_tp_size", 0),
            tp_size=(
                kv_transfer_params.get("tp_size")
                if "tp_size" in kv_transfer_params
                else kv_transfer_params.get("remote_tp_size", 1)
            ),
            transfer_id=kv_transfer_params.get("transfer_id", 0),
            local_slot_index=kv_transfer_params.get("local_slot_index", -1),
            remote_slot_index=kv_transfer_params.get("remote_slot_index", -1),
        )

    def add_new_req_to_save(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        local_swa_block_ids: list[int] | None = None,
    ) -> None:
        self.reqs_to_save[request_id] = self._build_req_meta(
            request_id, local_block_ids, kv_transfer_params, local_swa_block_ids
        )

    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        local_swa_block_ids: list[int] | None = None,
    ) -> None:
        self.reqs_to_recv[request_id] = self._build_req_meta(
            request_id, local_block_ids, kv_transfer_params, local_swa_block_ids
        )
