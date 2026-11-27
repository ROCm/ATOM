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
from typing import TYPE_CHECKING, Any

from atom.kv_transfer.disaggregation.types import ConnectorMetadata, KVConnectorOutput

if TYPE_CHECKING:
    from atom.model_engine.kv_block_pool import KVBlockPool, KVPoolMemory


class KVConnectorBase(ABC):
    """Worker-side KV connector interface (one instance per TP rank)."""

    is_producer: bool

    @classmethod
    def kv_budget_reserve_bytes(cls, config: Any) -> int:
        """GPU bytes this backend allocates once the KV cache is sized.

        The model runner holds them back from the KV cache budget (see
        :meth:`KVConnectorFactory.kv_budget_reserve_bytes`). It asks before
        any connector is built, so the answer comes from ``config`` alone.
        Backends that allocate nothing in ``register_kv_caches`` keep 0.
        """
        return 0

    @classmethod
    def allocate_kv_pool(cls, config: Any, nbytes: int, device: Any) -> Any:
        """A zeroed uint8 tensor of ``nbytes`` to back the paged KV pool, or None.

        For a backend whose peer maps this rank's KV cache, so the pool must
        live in exportable memory. Asked before any connector is built (see
        :meth:`KVConnectorFactory.allocate_kv_pool`); None, the default, lets
        the model runner allocate it.
        """
        return None

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

    def publish_prefill_chunks(self, batch: Any) -> None:
        """Publish completed prefill chunks; no-op without chunked P/D transfer."""

    def get_finished_recv_blocks(self) -> list[int]:
        """Return block IDs from recently completed receives for GPU memory fence.

        RDMA writes to HBM may not be immediately visible to GPU compute
        kernels. Connectors using RDMA should override this to return
        blocks that need a GPU-side read-write cycle to ensure coherence.
        """
        return []

    def kv_pool_memory(self, config: Any, rank: int) -> KVPoolMemory | None:
        """Provide the paged KV pool's memory, or None to leave it to the runner.

        A connector that takes the pool over (`atom/kv_transfer/pool`) returns a
        `KVPoolMemory`: where the bytes come from, how they divide into blocks
        and when the device has finished writing them. Asked once, before the
        pool is allocated. Transfer and offload connectors keep None.
        """
        return None


class KVConnectorSchedulerBase(ABC):
    """Scheduler-side KV connector interface.

    Every backend implements the retention hooks explicitly, including no-ops
    for lifecycle events it does not own.
    """

    is_producer: bool
    # A tier holding what HBM evicted rather than a copy of it is asked only
    # past the HBM hit, which the scheduler then probes before every lookup
    # (`Sequence.offload_hbm_probe_tokens`).
    needs_hbm_probe_before_lookup: bool = False

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

    def make_block_pool(
        self,
        config: Any,
        num_blocks: int,
        *,
        on_evict: Any = None,
        on_remap: Any = None,
        cache_policy: str = "lru",
        protected_ratio: float = 0.5,
        reserve_null_block: bool = False,
    ) -> KVBlockPool | None:
        """Provide the scheduler's block pool, or None for ATOM's own.

        The counterpart of `KVConnectorBase.kv_pool_memory`: a `KVBlockPool`
        deciding which block a request gets and what is evicted, over memory
        the worker side provided. Asked once, while `BlockManager` is built.
        """
        return None

    def process_pd_completions(self, output: KVConnectorOutput) -> KVConnectorOutput:
        """Preserve completions unless the backend consumes chunked P/D metadata."""
        return output

    def admission_ready(self, seq: Any) -> bool:
        """Whether the scheduler may allocate this request and start prefill."""
        return True

    def abort_send(self, seq: Any) -> None:
        """Retire an unsent abort; streaming senders must wait for reader drain."""
        self.send_finished(seq.id)

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
