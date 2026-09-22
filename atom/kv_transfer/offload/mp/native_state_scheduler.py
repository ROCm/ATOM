# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded transfers of native READY checkpoint images over LMCache MP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadOperationId,
    SaveOperationId,
    SaveSourceGroupId,
    StateStoreOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import (
    max_pending_saves,
    validated_kv_role,
)
from atom.kv_transfer.offload.metadata import NativeStateTransfer
from atom.kv_transfer.offload.mp.backend import (
    LMCacheMPConnectorScheduler,
    _extra_config,
    _validate_mp_config,
)
from atom.kv_transfer.offload.mp.native_state_worker import (
    NATIVE_STATE_MP_SOURCE_SAFE_CHANNEL,
    NATIVE_STATE_MP_STORE_CHANNEL,
    require_native_state_server,
)
from atom.model_engine.page_unit_checkpoint import SuspendedCheckpointRestore

_MAX_SAVE_ATTEMPTS = 3


@dataclass
class _NativeSave:
    seq: Any
    source: StateStoreOperationId
    saved_before: int
    boundary: int
    source_safe: bool = False


@dataclass
class _NativeLoad:
    seq: Any
    transfer: NativeStateTransfer
    hbm_cached_tokens: int
    local_restore: SuspendedCheckpointRestore | None = None
    dispatched: bool = False


class NativeStateLMCacheMPConnectorScheduler(LMCacheMPConnectorScheduler):
    """Pair PAGE KV with an exact native state image under one operation ID.

    Native sources are acquired only after save admission. Loads reserve raw
    PAGE units after request allocation and before parking; their release waits
    for the worker's terminal transfer-and-restore report. Neither side recycles
    these units in response to elapsed time.
    """

    _supports_early_block_release = True

    def __init__(self, config: Any) -> None:
        # The layout is known only after BlockManager builds the native pool.
        # Defer connecting so scheduler and workers use the same namespace.
        _validate_mp_config(config)
        self._config = config
        self.kv_role = validated_kv_role(
            getattr(config, "kv_transfer_config", {}) or {}
        )
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        self._block_manager = None

    def bind_block_manager(self, block_manager: Any) -> None:
        if self._block_manager is block_manager:
            return
        if self._block_manager is not None:
            raise RuntimeError(
                "native-state LMCache MP scheduler is already bound to a block manager"
            )
        coordinator = getattr(block_manager, "paged_state_checkpoints", None)
        if coordinator is None:
            raise ValueError(
                "native-state LMCache MP requires PAGE checkpoint geometry"
            )
        super().__init__(self._config, checkpoint_spec=coordinator.store.spec)
        try:
            require_native_state_server(self._mp_adapter, self._config)
            self._hash_block_size = int(block_manager.hash_block_size)
            if self.chunk_size % self._hash_block_size:
                raise ValueError(
                    "native-state LMCache MP chunk size must align to native "
                    "hash blocks"
                )
            self._max_pending_saves = max_pending_saves(
                getattr(self._config, "kv_transfer_config", {}) or {},
                int(os.environ.get("OFFLOAD_COPY_WORKERS", "1") or 1),
            )
            self._image_reservation_bytes = (
                coordinator.store.spec.units_per_checkpoint
                * coordinator.store.spec.page_unit_bytes
            )
            self._max_pinned_state_bytes = offcfg._strict_integer(
                "lmcache.mp.max_pinned_state_bytes",
                _extra_config(self._config).get(
                    "lmcache.mp.max_pinned_state_bytes",
                    self._max_pending_saves * self._image_reservation_bytes,
                ),
                minimum=self._image_reservation_bytes,
            )
            # The parent constructor only initializes the PAGE budget fields.
            # Bind through the parent as well so ratio/absolute PAGE limits are
            # enforced for native PAGE+STATE saves.
            super().bind_block_manager(block_manager)
        except Exception:
            shutdown = getattr(self._mp_adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise
        self._checkpoints = coordinator
        self._pinned_state_bytes = 0
        self._native_saves: dict[SaveOperationId, _NativeSave] = {}
        self._native_loads: dict[LoadOperationId, _NativeLoad] = {}
        self._native_load_operations: dict[str, LoadOperationId] = {}
        # One retry record per tracked request: a newer boundary can retry the
        # complete unsaved prefix, but a permanently failing boundary cannot
        # hold a finished request forever.
        self._save_failures: dict[str, tuple[Any, int, int]] = {}
        self._retired_requests: dict[str, Any] = {}

    def _boundary_hash(self, seq: Any, boundary: int) -> int:
        count = boundary // self._hash_block_size
        if boundary <= 0 or boundary % self._hash_block_size:
            raise ValueError("native checkpoint boundary must align to hash blocks")
        chain = getattr(seq, "block_hashes", ())
        if len(chain) >= count:
            return int(chain[count - 1])
        # Some checkpoint producers do not reserve midstep checkpoints, so
        # BlockManager may leave Sequence.block_hashes empty. Use its exact
        # public hashing algorithm and token slices, caching only this request's
        # immutable prompt chain.
        chain = getattr(seq, "_mp_checkpoint_hashes", None)
        if chain is None:
            chain = []
            seq._mp_checkpoint_hashes = chain
        prefix = chain[-1] if chain else -1
        for index in range(len(chain), count):
            start = index * self._hash_block_size
            prefix = self._block_manager.compute_hash(
                seq.token_ids[start : start + self._hash_block_size], prefix
            )
            chain.append(prefix)
        return int(chain[count - 1])

    def _lookup_token_ids(self, seq: Any) -> list[int]:
        # Truncate the QUERY, not its answer: MP recurrent/window keys and read
        # locks belong to the requested endpoint. A full-prompt query followed
        # by rounding the result would pair the earlier KV with later state.
        boundary = self._chunk_floor(max(0, int(seq.num_prompt_tokens) - 1))
        return list(seq.token_ids[:boundary])

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        if not getattr(seq, "has_per_req_cache", False):
            return 0, False
        previous = self._native_load_operations.get(str(seq.id))
        if previous is not None:
            return 0, False
        return super().get_num_new_matched_tokens(seq)

    def _has_state_budget(self) -> bool:
        return (
            self._pinned_state_bytes + self._image_reservation_bytes
            <= self._max_pinned_state_bytes
        )

    def _save_frontier(self, seq: Any) -> int:
        if not getattr(seq, "has_per_req_cache", False):
            return 0
        frontier = super()._save_frontier(seq)
        failed = self._save_failures.get(str(seq.id))
        exhausted = (
            failed[1]
            if failed is not None
            and failed[0] is seq
            and failed[2] >= _MAX_SAVE_ATTEMPTS
            else 0
        )
        for boundary in range(frontier, exhausted, -self.chunk_size):
            if self._checkpoints.contains(self._boundary_hash(seq, boundary)):
                return boundary
        return 0

    def _late_save_frontier(self, seq: Any, saved: int, available: int) -> int:
        available = self._chunk_floor(available)
        for boundary in range(available, saved, -self.chunk_size):
            if self._checkpoints.contains(self._boundary_hash(seq, boundary)):
                return boundary
        return saved

    def _may_emit_save(self) -> bool:
        return (
            len(self._save_inflight) < self._max_pending_saves
            and self._has_state_budget()
        )

    def _build_save_request(
        self, seq, saved, aligned, operation, block_ids, is_last_prefill
    ):
        if not self._may_emit_save():
            return None
        prefix_hash = self._boundary_hash(seq, aligned)
        # Reserve admission bytes before taking any source pin. All mutations
        # run on the scheduler thread, and a miss rolls this reservation back.
        self._pinned_state_bytes += self._image_reservation_bytes
        source = self._checkpoints.acquire_checkpoint_source(
            prefix_hash, max_inflight=self._max_pending_saves
        )
        if source is None:
            self._pinned_state_bytes -= self._image_reservation_bytes
            return None
        state_operation, unit_ids = source
        request = super()._build_save_request(
            seq, saved, aligned, operation, block_ids, is_last_prefill
        )
        request.native_state = NativeStateTransfer(unit_ids, aligned, prefix_hash)
        self._native_saves[operation] = _NativeSave(
            seq, state_operation, saved, aligned
        )
        return request

    def _complete_native_save(
        self, operation: SaveOperationId, *, succeeded: bool
    ) -> None:
        lease = self._native_saves.pop(operation, None)
        if lease is None:
            return
        # A terminal MP event is a backstop source fence even if an older
        # server did not provide per-chunk milestones, or those notifications
        # were coalesced/lost before reaching the scheduler.
        self._source_group_finished(
            SaveSourceGroupId(operation, ((lease.saved_before, lease.boundary),))
        )
        if not lease.source_safe:
            self._checkpoints.release_offload_store_source(lease.source)
        self._checkpoints.settle_offload_store(lease.source)
        self._pinned_state_bytes -= self._image_reservation_bytes
        sid = str(operation.req_id)
        if succeeded:
            self._save_failures.pop(sid, None)
        else:
            failed = self._save_failures.get(sid)
            attempts = (
                failed[2] + 1
                if failed is not None and failed[:2] == (lease.seq, lease.boundary)
                else 1
            )
            self._save_failures[sid] = (lease.seq, lease.boundary, attempts)
            entry = self._save_tracker.get(sid)
            if entry is not None and entry[0] is lease.seq:
                entry[1] = min(int(entry[1]), lease.saved_before)
        self._store_finished(operation, succeeded=succeeded)

    def save_finished(self, req_id: Any) -> None:
        if isinstance(req_id, SaveOperationId):
            self._complete_native_save(req_id, succeeded=True)

    def connector_completion(self, completion: ConnectorCompletion) -> bool | None:
        if completion.channel == NATIVE_STATE_MP_SOURCE_SAFE_CHANNEL:
            operation = completion.operation_id
            if not isinstance(operation, SaveOperationId):
                return False
            lease = self._native_saves.get(operation)
            if lease is not None and not lease.source_safe:
                self._checkpoints.release_offload_store_source(lease.source)
                lease.source_safe = True
            return None
        if completion.channel != NATIVE_STATE_MP_STORE_CHANNEL:
            return super().connector_completion(completion)
        if not isinstance(completion.operation_id, SaveOperationId):
            return False
        self._complete_native_save(
            completion.operation_id, succeeded=completion.succeeded
        )
        return True

    def _decide_load_after_alloc(self, seq: Any, load_spec):
        decision = super()._decide_load_after_alloc(seq, load_spec)
        should_load, _reason, hbm, lmc, need, chunk = decision
        if not should_load:
            return decision
        if not getattr(seq, "has_per_req_cache", False) or seq.state_slot < 0:
            return False, "native_destination_missing", hbm, lmc, need, chunk
        if lmc % chunk or lmc >= int(seq.num_prompt_tokens):
            return False, "native_boundary_unaligned", hbm, lmc, need, chunk
        sid = str(seq.id)
        operation = self._native_load_operations.get(sid)
        if operation is not None:
            lease = self._native_loads[operation]
            if lease.seq is seq:
                return decision
            return False, "native_load_id_busy", hbm, lmc, need, chunk
        if not self._has_state_budget():
            return False, "native_state_budget", hbm, lmc, need, chunk
        operation = LoadOperationId(seq.id, self._load_nonce)
        self._load_nonce += 1
        self._pinned_state_bytes += self._image_reservation_bytes
        units = self._checkpoints.reserve_transfer_units(operation)
        if units is None:
            self._pinned_state_bytes -= self._image_reservation_bytes
            return False, "native_state_units", hbm, lmc, need, chunk
        local_restore = self._checkpoints.suspend_queued_restore(int(seq.state_slot))
        self._native_loads[operation] = _NativeLoad(
            seq,
            NativeStateTransfer(
                units,
                lmc,
                self._boundary_hash(seq, lmc),
                destination_slot=int(seq.state_slot),
            ),
            hbm,
            local_restore,
        )
        self._native_load_operations[sid] = operation
        self._active_load_operations[sid] = (seq, operation)
        seq._load_operation = operation
        return decision

    def _new_load_operation(self, seq: Any) -> LoadOperationId:
        operation = self._native_load_operations[str(seq.id)]
        self._native_loads[operation].dispatched = True
        # The scheduler publishes the loaded PAGE prefix only after terminal
        # success. Its hash chain must exist before suffix prefill checkpoints.
        seq.offload_load_start_tokens = self._native_loads[operation].hbm_cached_tokens
        return operation

    def build_connector_meta(self):
        metadata = super().build_connector_meta()
        for request in metadata.requests:
            if request.load_operation is not None:
                request.native_state = self._native_loads[
                    request.load_operation
                ].transfer
        return metadata

    def _release_native_load(
        self, operation: LoadOperationId, *, release_units: bool = True
    ) -> _NativeLoad | None:
        lease = self._native_loads.pop(operation, None)
        if lease is not None:
            if release_units:
                self._checkpoints.release_transfer_units(operation)
            self._pinned_state_bytes -= self._image_reservation_bytes
            sid = str(operation.req_id)
            if self._native_load_operations.get(sid) == operation:
                del self._native_load_operations[sid]
        return lease

    def _clear_pending_load(self, sid: str) -> None:
        operation = self._native_load_operations.get(sid)
        lease = self._native_loads.get(operation)
        if lease is not None and not lease.dispatched:
            if lease.local_restore is not None:
                self._checkpoints.resume_suspended_restore(lease.local_restore)
                lease.local_restore = None
            self._release_native_load(operation)
            if self._active_load_operations.get(sid) == (lease.seq, operation):
                self._active_load_operations.pop(sid, None)
            if getattr(lease.seq, "_load_operation", None) == operation:
                delattr(lease.seq, "_load_operation")
        super()._clear_pending_load(sid)

    def _finish_native_load(self, operation: Any, *, succeeded: bool) -> bool:
        if (
            not isinstance(operation, LoadOperationId)
            or operation not in self._native_loads
        ):
            return False
        lease = self._native_loads[operation]
        if succeeded:
            self._checkpoints.adopt_transfer_units(
                operation, lease.transfer.prefix_hash
            )
            if lease.local_restore is not None:
                self._checkpoints.release_suspended_restore(lease.local_restore)
                lease.local_restore = None
            self._release_native_load(operation, release_units=False)
        else:
            if lease.local_restore is not None:
                self._checkpoints.resume_suspended_restore(lease.local_restore)
                lease.local_restore = None
            self._release_native_load(operation)
        finished = (
            super().load_finished(operation)
            if succeeded
            else super().load_failed(operation)
        )
        if self._retired_requests.get(str(operation.req_id)) is lease.seq:
            self.request_finished(lease.seq)
        return finished

    def load_finished(self, req_id: Any) -> bool:
        return self._finish_native_load(req_id, succeeded=True)

    def load_failed(self, req_id: Any) -> bool:
        return self._finish_native_load(req_id, succeeded=False)

    def _has_active_load(self, seq: Any) -> bool:
        # The native lease also retains a cancelled/reused request lifecycle
        # whose generic request-ID entry may have been replaced.
        return any(
            lease.seq is seq for lease in self._native_loads.values()
        ) or super()._has_active_load(seq)

    def cancel_pending_load(self, seq: Any) -> None:
        operation = self._native_load_operations.get(str(seq.id))
        lease = self._native_loads.get(operation)
        if lease is not None and lease.seq is seq and lease.dispatched:
            return  # A request cancellation does not cancel an MP DMA.
        super().cancel_pending_load(seq)

    def request_finished(self, seq: Any) -> None:
        sid = str(seq.id)
        self._retired_requests[sid] = seq
        operation = self._native_load_operations.get(sid)
        lease = self._native_loads.get(operation)
        if lease is not None and lease.seq is seq and lease.dispatched:
            return
        super().request_finished(seq)
        self._finish_retired_request(sid)

    def _finish_retired_request(self, sid: str) -> None:
        super()._finish_retired_request(sid)
        seq = self._retired_requests.get(sid)
        if seq is not None and not self.should_defer_free(seq):
            entry = self._save_tracker.get(sid)
            if entry is not None and entry[0] is seq:
                self._save_tracker.pop(sid, None)
            self._save_failures.pop(sid, None)
            self._retired_requests.pop(sid, None)

    def has_pending_work(self) -> bool:
        return (
            bool(self._native_loads)
            or bool(self._retired_requests)
            or super().has_pending_work()
        )

    def process_completions(self, output):
        output = super().process_completions(output)
        # A retired request may have waited behind admission with an unpinned
        # READY candidate. If eviction spends it before dispatch, no worker
        # completion exists to wake deferred-free cleanup. Report that local
        # terminal condition only for this same request lifecycle, after all
        # actual transfers and every remaining candidate are gone.
        for sid, seq in list(self._retired_requests.items()):
            entry = self._save_tracker.get(sid)
            lifecycle = self._load_lifecycles.get(sid)
            if (entry is not None and entry[0] is not seq) or (
                lifecycle is not None and lifecycle is not seq
            ):
                continue
            if not self.should_defer_free(seq):
                output.finished_saving.add(seq.id)
                self._finish_retired_request(sid)
        return output

    def abandon_save(self, req_id: Any) -> None:
        # This legacy callback carries no proof that the MP reader stopped.
        # Explicit terminal failures already settle the exact operation above.
        return None

    def reclaim_stale_leases(self, timeout_s: float) -> list[frozenset]:
        return []


__all__ = ["NativeStateLMCacheMPConnectorScheduler"]
