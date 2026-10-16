# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native PAGE-backed checkpoint transport over standalone LMCache."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import max_pending_saves
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, NativeStateTransfer
from atom.kv_transfer.offload.mp.deployment import (
    _make_worker_adapter,
    _mp_session_id,
    _published_tp_replication_factor,
    _storage_kv_transfer_config,
    _tp_replication_factor,
    _validate_mp_config,
)
from atom.kv_transfer.offload.mp.native_state_layout import (
    build_native_state_mp_layout,
)
from atom.kv_transfer.offload.mp.transfer import (
    _chunk_ranges,
    _enforce_transfer_deadline,
    _remember_operation_tombstone,
    _source_safe_completions,
    _terminal_future_result,
    _transfer_operation_id,
    _UnprovableSubmission,
)
from atom.kv_transfer.offload.mp.worker import LMCacheMPConnector
from atom.model_engine.page_unit_checkpoint import CheckpointRestoreOp
from atom.utils import envs

logger = logging.getLogger("atom")
NATIVE_STATE_MP_STORE_CHANNEL = "native_state_mp_store"


def require_native_state_server(adapter: Any, config: Any) -> None:
    """Validate native transfer geometry shared with the LMCache server."""
    configured_chunk = int(
        offcfg.build_lmcache_config(_storage_kv_transfer_config(config)).chunk_size
    )
    if configured_chunk != int(adapter.lmcache_tokens_per_chunk):
        raise ValueError(
            "Native-state LMCache configured chunk size must match the MP server: "
            f"configured={configured_chunk}, server={adapter.lmcache_tokens_per_chunk}"
        )


@dataclass
class _NativePending:
    request: LMCacheReqMeta
    future: Any
    restore_event: Any = None
    restore_succeeded: bool = False
    descriptor_slot: int | None = None
    immediate_success: bool = False
    started_at: float = field(default_factory=lambda: time.monotonic())


class NativeStateLMCacheMPConnector(LMCacheMPConnector):
    """Transfer pinned native images; never read a request's active SLOT to save."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._native_saves: dict[str, _NativePending] = {}
        self._native_loads: dict[str, _NativePending] = {}
        self._native_layout = None
        self._native_copy = None
        self._restore_stream = None
        self._restore_descriptor_slots: list[int] = []
        self._max_pending_saves = max_pending_saves(
            getattr(config, "kv_transfer_config", {}) or {},
            envs.OFFLOAD_COPY_WORKERS,
        )

    def register_kv_caches(
        self,
        _kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        from aiter.dist.parallel_state import get_tp_group

        spec = getattr(transfer_tensors, "paged_state_checkpoint_spec", None)
        native_copy = getattr(transfer_tensors, "execute_paged_state_copies", None)
        if spec is None or not callable(native_copy):
            raise ValueError(
                "native-state LMCache MP needs checkpoint geometry and a copy callback"
            )
        tp_size, _ = _validate_mp_config(self._config)
        rank = int(get_tp_group().rank_in_group)
        requested_replication = _tp_replication_factor(self._config)
        published_page_replication = _published_tp_replication_factor(
            transfer_tensors, tp_size=tp_size
        )
        published_state_replication = _published_tp_replication_factor(
            transfer_tensors, tp_size=tp_size, native_state=True
        )
        if requested_replication > min(
            published_page_replication, published_state_replication
        ):
            raise ValueError(
                "LMCache MP native TP rank collapse was requested, but the "
                "attention backend did not declare both PAGE and STATE fully "
                "replicated"
            )
        self._is_kv_writer = rank % requested_replication == 0
        adapter = _make_worker_adapter(self._config, rank, checkpoint_spec=spec)
        try:
            require_native_state_server(adapter, self._config)
            chunk_size = int(adapter.lmcache_tokens_per_chunk)
            layout = build_native_state_mp_layout(
                transfer_tensors,
                block_size=self.block_size,
                chunk_size=chunk_size,
                num_blocks=num_blocks,
            )
            adapter.register_kv_caches(
                {
                    f"native_state.{i}": tensor
                    for i, tensor in enumerate(layout.tensors)
                },
                engine_group_infos=layout.engine_group_infos(),
            )
        except Exception:
            adapter.shutdown()
            raise
        self._adapter = adapter
        self.chunk_size = chunk_size
        self._native_layout = layout
        self._native_copy = native_copy
        self._restore_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self._restore_descriptor_slots = list(
            range(1, max(2, self._max_pending_saves) + 1)
        )
        # Pinned staging allocation synchronizes the device. Reserve every
        # restore slot now, at registration, rather than on the connector
        # thread the first time a restore reaches it mid-serving.
        builder = getattr(transfer_tensors, "state_backend", None)
        reserve = getattr(builder, "reserve_checkpoint_descriptors", None)
        if callable(reserve):
            reserve(self._restore_descriptor_slots)
        logger.info(
            "LMCache MP native state registered rank=%d native_image=%d "
            "units=%d groups=%d chunk=%d",
            rank,
            spec.image_bytes,
            spec.units_per_checkpoint,
            len(layout.kernel_groups),
            chunk_size,
        )

    def _native_block_ids(
        self,
        req: LMCacheReqMeta,
        start: int,
        end: int,
        *,
        loading: bool,
    ) -> list[list[int]]:
        state = req.native_state
        if state is None:
            raise ValueError(
                "native-state LMCache MP transfers require an exact checkpoint"
            )
        if start % self.chunk_size or end % self.chunk_size or start >= end:
            raise ValueError(
                "native-state LMCache MP requires a nonempty chunk-aligned range"
            )
        if state.boundary_tokens != end or len(req.token_ids) != end:
            raise ValueError("native STATE and PAGE endpoints must match")
        if loading and state.destination_slot is None:
            raise ValueError("native-state restore requires a destination SLOT")
        if not loading and state.destination_slot is not None:
            raise ValueError("native-state save must refer to immutable PAGE units")
        unit_ids = self._native_layout.validate_unit_ids(state.unit_ids)
        pages = self._block_slice(req, start, end)
        if set(pages) & set(unit_ids):
            raise ValueError("checkpoint PAGE units must not overlap KV PAGE blocks")
        count = (end - start) // self.chunk_size
        # Earlier chunks have PAGE only. A boundary image is indivisible: all
        # its ordinal groups are present at exactly the same final chunk.
        return [pages] + [[-1] * (count - 1) + [unit] for unit in unit_ids]

    def _submit_native(self, req: LMCacheReqMeta, event: Any, *, loading: bool) -> None:
        from lmcache.integration.atom import AtomMPTransferSpec

        completion = req.load_operation if loading else req.save_operation
        if completion is None:
            raise ValueError(
                "native-state LMCache MP transfers require exact operation generations"
            )
        operation_id = _transfer_operation_id("load" if loading else "save", completion)
        pending = self._native_loads if loading else self._native_saves
        completed = (
            self._completed_load_operations
            if loading
            else self._completed_save_operations
        )
        with self._lock:
            if operation_id in pending or operation_id in completed:
                raise RuntimeError(
                    f"duplicate native-state LMCache MP operation {operation_id!r}"
                )
            if not loading and not self._is_kv_writer:
                pending[operation_id] = _NativePending(
                    req, None, immediate_success=True
                )
                return
            if not loading and len(pending) >= self._max_pending_saves:
                # Unreachable while the scheduler enforces the same bound: every
                # save held here is still in its `_save_inflight`. Refuse before
                # transport; a terminal False returns the admission credit.
                logger.error(
                    "Native-state LMCache MP save %s exceeds the scheduler's "
                    "pending bound %d; refusing it",
                    operation_id,
                    self._max_pending_saves,
                )
                pending[operation_id] = _NativePending(req, None)
                return
            end = len(req.token_ids)
            start = (
                req.load_spec.hbm_cached_tokens
                if loading
                else req.save_spec.skip_leading_tokens
            )
            try:
                groups = self._native_block_ids(req, start, end, loading=loading)
                spec = AtomMPTransferSpec(
                    token_ids=list(req.token_ids),
                    block_ids=groups,
                    start=start,
                    end=end,
                )
            except Exception:
                logger.exception(
                    "Invalid native-state LMCache MP descriptor %s", operation_id
                )
                pending[operation_id] = _NativePending(req, None)
                return
            # Installed before the transport call: if it raises, the server may
            # still have taken the request, so the lease stays until the
            # transfer deadline.
            entry = _NativePending(req, _UnprovableSubmission())
            pending[operation_id] = entry
        submit = (
            self._adapter.submit_retrieve_request
            if loading
            else getattr(
                self._adapter,
                "submit_store_request_with_chunk_events",
                self._adapter.submit_store_request,
            )
        )
        try:
            future = submit(_mp_session_id(self._config, req.req_id), spec, event)
        except Exception:
            # Retain the exact source/destination lease. The server might have
            # received the request before the connection raised an exception.
            logger.exception(
                "Native-state LMCache MP submission unprovable; retaining lease %s",
                operation_id,
            )
            return
        with self._lock:
            entry.future = future

    def _submit_load(self, req: LMCacheReqMeta, event: Any) -> None:
        self._submit_native(req, event, loading=True)

    def _submit_save(self, req: LMCacheReqMeta, event: Any) -> None:
        self._submit_native(req, event, loading=False)

    def _begin_restore(self, pending: _NativePending) -> bool:
        if not self._restore_descriptor_slots:
            return False
        state: NativeStateTransfer = pending.request.native_state
        spec = self._native_layout.checkpoint_spec
        event = torch.cuda.Event()
        descriptor_slot = self._restore_descriptor_slots.pop()
        pending.restore_event = event
        pending.descriptor_slot = descriptor_slot
        # Write-after-read: the destination SLOT's previous occupant may still
        # have compute work enqueued. Order the restore after everything the
        # compute stream has been given so far.
        self._restore_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._restore_stream):
            try:
                self._native_copy(
                    (),
                    (
                        CheckpointRestoreOp(
                            dst_slot=state.destination_slot,
                            unit_ids=state.unit_ids,
                            total_bytes=spec.image_bytes,
                            layout_id=spec.layout_id,
                        ),
                    ),
                    descriptor_slot=descriptor_slot,
                )
                pending.restore_succeeded = True
            except Exception:
                logger.exception("LMCache MP native restore failed")
            finally:
                event.record(self._restore_stream)
        return True

    def _emit_page_source_safe(
        self,
        output: KVConnectorOutput,
        pending: _NativePending,
        ranges: tuple[tuple[int, int], ...],
    ) -> None:
        # PAGE only. A chunk milestone says nothing about when the server has
        # read the STATE groups, so the image pin waits for the STORE terminal.
        operation = pending.request.save_operation
        if operation is not None and ranges:
            output.connector_completions |= _source_safe_completions(operation, ranges)

    def start_load_kv(self, metadata: Any) -> None:
        # Read-after-write: this step's compute (including SLOT relocations in
        # `build()`) must not touch a SLOT while its native restore is still
        # writing. The host only polls the event, so fence the compute stream
        # on the GPU instead of blocking here.
        with self._lock:
            restores = [
                pending.restore_event
                for pending in self._native_loads.values()
                if pending.restore_event is not None
            ]
        if restores:
            compute = torch.cuda.current_stream()
            for event in restores:
                compute.wait_event(event)
        super().start_load_kv(metadata)

    def get_finished(self) -> KVConnectorOutput:
        output = KVConnectorOutput()
        with self._lock:
            for operation_id, pending in list(self._native_saves.items()):
                self._poll_native_save(output, operation_id, pending)
            for operation_id, pending in list(self._native_loads.items()):
                self._poll_native_load(output, operation_id, pending)
        return output

    def _poll_native_save(
        self, output: KVConnectorOutput, operation_id: str, pending: _NativePending
    ) -> None:
        take_ranges = getattr(pending.future, "take_completed_ranges", None)
        if callable(take_ranges):
            try:
                self._emit_page_source_safe(output, pending, tuple(take_ranges()))
            except Exception:
                logger.warning(
                    "LMCache MP source-safe event polling failed", exc_info=True
                )
        if pending.immediate_success:
            terminal, result = True, True
        else:
            terminal, result = _terminal_future_result(pending.future)
        if not terminal:
            _enforce_transfer_deadline(
                operation_id, pending.started_at, self._transfer_deadline_s
            )
            return
        save_spec = pending.request.save_spec
        start = 0 if save_spec is None else int(save_spec.skip_leading_tokens)
        end = len(pending.request.token_ids)
        self._emit_page_source_safe(
            output, pending, _chunk_ranges(start, end, self.chunk_size)
        )
        output.connector_completions.add(
            ConnectorCompletion(
                NATIVE_STATE_MP_STORE_CHANNEL,
                pending.request.save_operation,
                succeeded=result is True,
            )
        )
        del self._native_saves[operation_id]
        _remember_operation_tombstone(
            operation_id,
            self._completed_save_operations,
            self._completed_save_operation_order,
        )

    def _poll_native_load(
        self, output: KVConnectorOutput, operation_id: str, pending: _NativePending
    ) -> None:
        if not self._advance_native_load(operation_id, pending):
            return
        if pending.descriptor_slot is not None:
            self._restore_descriptor_slots.append(pending.descriptor_slot)
            pending.descriptor_slot = None
        target = (
            output.finished_loading
            if pending.restore_succeeded
            else output.failed_loading
        )
        target.add(pending.request.load_operation)
        del self._native_loads[operation_id]
        _remember_operation_tombstone(
            operation_id,
            self._completed_load_operations,
            self._completed_load_operation_order,
        )

    def _advance_native_load(self, operation_id: str, pending: _NativePending) -> bool:
        """Move a load toward its outcome; True once that outcome is final.

        A retrieve that fails is final at once. One that succeeds starts the
        native restore, and is final only once the restore's event is done.
        """
        if pending.restore_event is None:
            terminal, result = _terminal_future_result(pending.future)
            if not terminal:
                _enforce_transfer_deadline(
                    operation_id, pending.started_at, self._transfer_deadline_s
                )
                return False
            if result is not True:
                return True
            try:
                if not self._begin_restore(pending):
                    return False  # every descriptor slot is busy; retry
            except Exception:
                logger.exception(
                    "LMCache MP native restore safety unknown; retaining lease"
                )
                self._hold_unprovable_restore(pending)
                return False
        try:
            restored = pending.restore_event.query()
        except Exception:
            logger.exception(
                "LMCache MP native restore event safety unknown; retaining lease"
            )
            self._hold_unprovable_restore(pending)
            return False
        if not restored:
            _enforce_transfer_deadline(
                operation_id, pending.started_at, self._transfer_deadline_s
            )
        return restored

    @staticmethod
    def _hold_unprovable_restore(pending: _NativePending) -> None:
        """A restore whose GPU progress cannot be proven keeps its SLOT, lease
        and descriptor slot (a queued copy may still read its staging); only
        the transfer deadline ends it."""
        pending.future = _UnprovableSubmission()
        pending.restore_event = None
        pending.restore_succeeded = False


__all__ = [
    "NATIVE_STATE_MP_STORE_CHANNEL",
    "NativeStateLMCacheMPConnector",
    "require_native_state_server",
]
