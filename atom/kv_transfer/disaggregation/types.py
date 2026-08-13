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


@dataclass(frozen=True)
class SendOperationId:
    """Exact identity of one producer-side send lifecycle."""

    req_id: ReqId
    generation: int

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("send operation generation must be nonnegative")


SendCompletionId = ReqId | SendOperationId


@dataclass(frozen=True)
class SaveOperationId:
    """Exact identity of one scheduler-issued PAGE/SLOT save generation.

    A request can emit several overlapping asynchronous saves, and a request ID
    can later be reused. The scheduler-lifetime ``generation`` prevents delayed
    or duplicated TP-worker completions for one save from completing another.
    """

    req_id: ReqId
    generation: int

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("save operation generation must be nonnegative")


SaveCompletionId = ReqId | SaveOperationId


@dataclass(frozen=True)
class LoadOperationId:
    """Exact identity of one scheduler-issued PAGE/SLOT load generation."""

    req_id: ReqId
    generation: int

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("load operation generation must be nonnegative")


LoadCompletionId = ReqId | LoadOperationId

# Connector-owned completion channels are opaque to composite connectors and
# the TP output aggregator.  This one channel belongs to the model-engine
# checkpoint lease protocol, so the scheduler consumes it directly after TP
# aggregation.  Backend-specific channels stay in their backend package.
STATE_CHECKPOINT_STAGING_CHANNEL = "atom.state_checkpoint.staging"

ConnectorCompletionId = (
    ReqId | SendOperationId | SaveOperationId | LoadOperationId
)
ConnectorCompletionKey = tuple[str, ConnectorCompletionId]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectorCompletion:
    """One terminal event emitted on a connector-owned completion channel.

    ``channel`` names the protocol owner without teaching generic transport
    layers what the event means.  ``operation_id`` correlates the same event
    across TP workers, and ``succeeded=False`` is failure-dominant when the TP
    aggregator combines worker reports.
    """

    channel: str
    operation_id: ConnectorCompletionId
    succeeded: bool

    def __post_init__(self) -> None:
        if not self.channel:
            raise ValueError("connector completion channel must be non-empty")
        if type(self.succeeded) is not bool:
            raise TypeError("connector completion succeeded must be bool")
        try:
            hash(self.operation_id)
        except TypeError as exc:
            raise TypeError(
                "connector completion operation_id must be hashable"
            ) from exc

    @property
    def key(self) -> ConnectorCompletionKey:
        return self.channel, self.operation_id


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
    """Physical PAGE, SLOT, and compressor-only staging region contract.

    ``block_regions`` contain forward-indexed, block-indexed PAGE units.
    ``swa_block_regions`` is a legacy field name for complete reverse-indexed
    per-request SLOT units, including both compressor state and SWA.
    ``expected_full_slot_region_count`` makes a stateful layout's complete
    plane count explicit so registration can reject a missing plane.
    ``staging_region`` plus ``gather_slot``/``scatter_slot`` cover only the
    compressor-state PD staging pool and are invalid as sidecar SLOT sources.
    """

    # Block-indexed PAGE regions, indexed forward by physical block id.
    block_regions: list[KVTransferRegion]
    slot_regions: list[KVTransferRegion]
    num_blocks: int
    num_slots: int = 0
    # Legacy field name: full per-request SLOT regions keyed by pool group.
    # `unit_bytes` includes compressor state and SWA, not just one ring.
    swa_block_regions: list[KVTransferRegion] = field(default_factory=list)
    # Compressor-only PD staging; never a complete sidecar SLOT source.
    staging_region: KVTransferRegion | None = None
    staging_pool_size: int = 0
    gather_slot: Callable[[int, int], None] | None = None
    scatter_slot: Callable[[int, int], None] | None = None
    # Appended for positional compatibility with existing generic descriptors.
    expected_full_slot_region_count: int | None = None


@dataclass
class KVConnectorOutput:
    """Per-worker snapshot of finished KV cache transfers.

    Each TP worker produces one of these per scheduler step.  The
    :class:`KVOutputAggregator` combines them to determine which
    request IDs have finished on *all* workers.

    Attributes:
        finished_sending: Exact producer send generations whose KV transfer
            completed (legacy connectors may still report request IDs).
        finished_recving: Request IDs whose KV receive completed on this worker.
        failed_recving: Request IDs whose KV receive failed on this worker.
        finished_saving: Exact save generations whose local fire-and-forget
            PAGE work completed (legacy connectors may still report request IDs).
        finished_loading: Exact offload load generations that completed (legacy
            connectors may still report request IDs).
        failed_loading: Exact offload load generations that failed (legacy
            connectors may still report request IDs).
        connector_completions: Terminal events on connector-owned channels.
            Generic composite/aggregation layers transport these opaquely;
            channel owners interpret them after TP aggregation.
        expected_finished_count: How many finished notifications should be
            expected per request (used by the aggregator).
    """

    finished_sending: set[SendCompletionId] = field(default_factory=set)
    finished_recving: set[ReqId] = field(default_factory=set)
    failed_recving: set[ReqId] = field(default_factory=set)
    finished_saving: set[SaveCompletionId] = field(default_factory=set)
    finished_loading: set[LoadCompletionId] = field(default_factory=set)
    failed_loading: set[LoadCompletionId] = field(default_factory=set)
    expected_finished_count: int = 0
    connector_completions: set[ConnectorCompletion] = field(default_factory=set)

    def is_empty(self) -> bool:
        """Return True if no transfers finished on this worker."""
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.failed_recving
            and not self.finished_saving
            and not self.finished_loading
            and not self.failed_loading
            and not self.connector_completions
        )

    def __repr__(self) -> str:
        return (
            f"KVConnectorOutput(sending={self.finished_sending}, "
            f"recving={self.finished_recving}, "
            f"failed_recving={self.failed_recving}, "
            f"finished_saving={self.finished_saving}, "
            f"loading={self.finished_loading}, "
            f"failed_loading={self.failed_loading}, "
            f"connector_completions={self.connector_completions})"
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

    def iter_async_save_operations(
        self,
    ) -> tuple[tuple[ReqId, SaveCompletionId], ...]:
        """Return saves that a composite connector must lifetime-pair.

        Most transfer metadata does not participate in ``finished_saving``.
        An asynchronous storage backend opts in by overriding this method;
        composite connectors then remain independent of backend request specs.
        Each item is ``(request_id, completion_id)``.
        """

        return ()

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


def connector_metadata_has_work(metadata: object | None) -> bool:
    """Return whether connector metadata contains dispatchable work."""
    if metadata is None:
        return False
    metas = getattr(metadata, "metas", None)
    if metas is not None and any(connector_metadata_has_work(meta) for meta in metas):
        return True
    return any(
        bool(getattr(metadata, name, None))
        for name in (
            "requests",
            "lookup_requests_in_step",
            "reqs_to_recv",
            "reqs_to_save",
            "reqs_to_send",
            "reqs_in_batch",
            "reqs_not_processed",
        )
    )
