# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Abstract base classes for KV cache connectors.

These interfaces decouple the engine from any specific transfer backend
(e.g. MoRIIO/RDMA, NCCL P2P).  Concrete implementations live in
separate modules and are registered via :class:`KVConnectorFactory`.

Two roles are defined:

- **Worker-side** (:class:`KVConnectorBase`): runs inside each TP rank,
  handles RDMA / network I/O, and reports transfer completion.
- **Scheduler-side** (:class:`KVConnectorSchedulerBase`): runs in the
  scheduler process, manages transfer lifecycle and metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from atom.kv_transfer.disaggregation.types import ConnectorMetadata, KVConnectorOutput


class KVConnectorBase(ABC):
    """Worker-side KV connector interface (one instance per TP rank)."""

    is_producer: bool

    @abstractmethod
    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        """Register local KV cache tensors for remote access.

        Called once after model loading and KV cache allocation. ``num_blocks``
        is the scheduler-visible block count (used by the offload connector to
        byte-slice MLA's token-major latent cache); connectors that don't need
        it may ignore it.
        """
        ...

    @abstractmethod
    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Initiate async KV loads for pending receive requests.

        Called by the worker each engine step.
        """
        ...

    @abstractmethod
    def get_finished(self) -> tuple[set, set] | KVConnectorOutput:
        """Return transfer completion status.

        Older connectors may return ``(done_sending, done_recving)``. Connectors
        that need richer semantics can return :class:`KVConnectorOutput`.

        Called by the worker each engine step to report transfer status.
        """
        ...

    def get_finished_recv_blocks(self) -> list[int]:
        """Return block IDs from recently completed receives for GPU memory fence.

        RDMA writes to HBM may not be immediately visible to GPU compute
        kernels. Connectors using RDMA should override this to return
        blocks that need a GPU-side read-write cycle to ensure coherence.
        """
        return []


class KVConnectorSchedulerBase(ABC):
    """Scheduler-side KV connector interface.

    Every backend implements the retention hooks explicitly, including no-ops
    for lifecycle events it does not own.
    """

    is_producer: bool

    @abstractmethod
    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        """Check if *seq* needs remote KV prefill.

        Returns:
            ``(num_tokens, needs_async_load)``
        """
        ...

    @abstractmethod
    def build_connector_meta(self) -> ConnectorMetadata:
        """Build a metadata snapshot of pending transfer requests."""
        ...

    @abstractmethod
    def update_state_after_alloc(self, seq: Any) -> None:
        """Update internal state after the scheduler allocates blocks."""
        ...

    @abstractmethod
    def request_finished(self, seq: Any) -> None:
        """Populate KV transfer output metadata when a request completes."""
        ...

    @abstractmethod
    def should_defer_free(self, seq: Any) -> bool:
        """Whether this connector still owns the request's source blocks.

        A pure predicate used for preemption and final block release. Producers
        retain advertised blocks until their send claim is retired.
        """
        ...

    @abstractmethod
    def send_finished(self, req_id: Any) -> None:
        """Retire a P/D send claim; explicit no-op for backends without sends."""
        ...

    @abstractmethod
    def source_blocks_released(self, seq: Any) -> None:
        """The scheduler has returned this request's source blocks to the pool.

        The terminal half of `request_finished` for a connector that deferred
        the free: at `request_finished` time `should_defer_free` is still True,
        so any state whose lifetime is the *blocks* rather than the *request*
        cannot be dropped yet. This is the call that says it can.

        Deliberately not `request_finished` called a second time: that one also
        takes the P/D send claim, and re-invoking it would re-arm the very claim
        the release just cleared. Backends without block-lifetime cleanup
        implement an explicit no-op.
        """
        ...
