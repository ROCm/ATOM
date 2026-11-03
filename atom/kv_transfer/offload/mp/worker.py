# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker-side LMCache MP connector for backend-published PAGE views.

Attention backends publish block-major tensor views through
``KVTransferTensors``. LMCache MP can attach several physical kernel groups to
the same engine block-id space, so this connector registers those opaque views
without knowing which model or attention implementation produced them.

Transfers are submitted and polled on the step thread for native ATOM, or on a
drain thread for the vLLM plugin (`drain.py`), wait on a step event from a
reusable ring (`step_events.py`), and leave their finished futures to be
destroyed on the step thread (`ipc_lifetime.py`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from atom.kv_transfer.disaggregation.base import KVConnectorBase
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    LoadCompletionId,
    SaveCompletionId,
    SaveOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata, LMCacheReqMeta
from atom.kv_transfer.offload.mp.deployment import (
    _make_worker_adapter,
    _mp_session_id,
    _null_block_id,
    _published_tp_replication_factor,
    _tp_replication_factor,
    _validate_mp_config,
)
from atom.kv_transfer.offload.mp.drain import InlineTransfers, TransferDrain
from atom.kv_transfer.offload.mp.ipc_lifetime import OwnedFutures
from atom.kv_transfer.offload.mp.native_state_layout import (
    build_native_state_mp_layout,
    require_native_state_server,
    stores_state_images,
)
from atom.kv_transfer.offload.mp.page_views import _build_cache_views
from atom.kv_transfer.offload.mp.step_events import (
    PendingStepEvent,
    RecordedStepEvent,
    StepEventRing,
)
from atom.kv_transfer.offload.mp.transfer import (
    _chunk_ranges,
    _enforce_transfer_deadline,
    _PendingLoad,
    _PendingSave,
    _remember_operation_tombstone,
    _source_safe_completions,
    _terminal_future_result,
    _transfer_deadline_s,
    _transfer_operation_id,
    _UnprovableSubmission,
)
from atom.kv_transfer.offload.mp.transfer_thread import (
    _ProducerFences,
    _TransferThread,
)
from atom.utils import envs

logger = logging.getLogger("atom")

# How long `get_finished` waits, while a load is in flight, for a drain round
# that is a pure poll: a load that lands during the forward and misses it
# keeps its request in WAITING_FOR_REMOTE_KVS a step longer. Without it, the
# plugin's c16 TTFT p50 rose from 302 to 324 ms.
_LAST_LOOK_WAIT_S = 0.001
# How long `close` lets the drain finish its last round.
_CLOSE_TIMEOUT_S = 30.0
# How long worker teardown waits for queued transfer submissions.
_TRANSFER_DRAIN_TIMEOUT_S = 60.0


def _req_id_of(completion: Any) -> str:
    """The request id inside a completion id, which may be bare or structured."""
    return str(getattr(completion, "req_id", completion))


@dataclass(frozen=True)
class _QueuedTransfer:
    """One submit, as `_transfers.put` takes it."""

    kind: str
    operation_id: str
    completion: Any
    request_id: str
    spec: Any
    event: Any
    start: int
    end: int
    queued_at: float


class LMCacheMPConnector(KVConnectorBase):
    """Worker-side LMCache MP connector for backend-published PAGE views."""

    is_producer = False

    def __init__(self, config: Any, *, drain_thread: bool = False) -> None:
        """``drain_thread`` submits and polls on a drain thread (the vLLM
        plugin); without it they run on the step thread (native ATOM)."""
        _validate_mp_config(config)
        self._config = config
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = validated_kv_role(kvc)
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        self.block_size = offcfg._strict_integer(
            "LMCache MP block size",
            config.kv_cache_block_size,
            minimum=1,
        )
        self.chunk_size: int | None = None
        # What a native-state layout names for every chunk before the one
        # holding its image (`build_native_state_mp_layout`).
        self._null_block_id = _null_block_id(config)
        # Set only when the backend publishes PAGE-backed checkpoint images
        # (`_register_native_state_layout`); every transfer then carries one.
        self._native_layout: Any = None
        self._adapter: Any = None
        self._is_kv_writer = True
        self._lock = threading.Lock()
        self._drain_thread = drain_thread
        self._transfers = (TransferDrain if drain_thread else InlineTransfers)(
            self._lock, self._perform_submit, self._poll_round
        )
        self._step_events = StepEventRing()
        # The ring index each queued or pending transfer holds.
        self._step_event_of: dict[str, int] = {}
        self._owned_futures = OwnedFutures(self._lock, hold=drain_thread)
        # With OFFLOAD_MP_TRANSFER_THREAD (native ATOM): the submitting thread
        # and the reused producer fences it needs (see `transfer_thread`).
        self._transfer: _TransferThread | None = None
        self._fences: _ProducerFences | None = None
        # Same-step saves of the current step, held until their forward is
        # enqueued (`after_forward`) -- and, inline, until it has finished.
        self._after_forward_saves: list[LMCacheReqMeta] = []
        self._after_forward_ran = False
        self._late_flush_logged = False
        self._pending_saves: dict[str, _PendingSave] = {}
        self._pending_loads: dict[str, _PendingLoad] = {}
        self._submitting_saves: set[str] = set()
        self._submitting_loads: set[str] = set()
        # The request each queued or submitting operation reads or writes the
        # blocks of. `fence` covers these too: on a drain thread they reach
        # the server after the step that queued them.
        self._submitting_owner: dict[str, str] = {}
        # What a round settled since the last `get_finished`.
        self._drained_done_save: set[SaveCompletionId] = set()
        self._drained_completions: set[ConnectorCompletion] = set()
        self._drained_done_load: set[LoadCompletionId] = set()
        self._drained_failed_load: set[LoadCompletionId] = set()
        # Keep in-flight IDs in the unbounded pending/submitting collections.
        # Only completed IDs enter these bounded replay tombstones.
        self._completed_save_operations: set[str] = set()
        self._completed_load_operations: set[str] = set()
        self._completed_save_operation_order: deque[str] = deque()
        self._completed_load_operation_order: deque[str] = deque()
        # Immediate successes include collapsed-TP non-writers. Keep their
        # logical token range so they can report PAGE source-safety too: the TP
        # aggregator must receive the same chunk completion from every rank
        # before releasing the writer's source blocks early.
        self._immediate_saves: dict[SaveCompletionId, tuple[int, int] | None] = {}
        self._transfer_deadline_s = _transfer_deadline_s(config)
        self._immediate_load_failures: set[LoadCompletionId] = set()

    def register_kv_caches(
        self,
        _kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        if getattr(transfer_tensors, "paged_state_checkpoint_spec", None) is not None:
            # PAGE units that also hold state checkpoint images. The connector
            # shell hands such a layout to `NativeStateLMCacheMPConnector`,
            # which restores a loaded image into an Active SLOT. A backend that
            # restores its checkpoints from the PAGE units themselves registers
            # the same layout here, and its transfers name no SLOT.
            self._register_native_state_layout(transfer_tensors, num_blocks)
            return
        if num_blocks is None:
            num_blocks = getattr(transfer_tensors, "num_blocks", None)
        if num_blocks is None:
            raise ValueError("lmcache_mp requires the scheduler block count")
        normalized_num_blocks = offcfg._strict_integer(
            "LMCache MP block count",
            num_blocks,
            minimum=1,
        )

        from aiter.dist.parallel_state import get_tp_group
        from lmcache.v1.multiprocess.group_view import EngineGroupInfo

        tp = get_tp_group()
        rank = int(tp.rank_in_group)
        tp_size, _ = _validate_mp_config(self._config)
        requested_replication = _tp_replication_factor(self._config)
        published_replication = _published_tp_replication_factor(
            transfer_tensors,
            tp_size=tp_size,
        )
        if requested_replication > published_replication:
            raise ValueError(
                "LMCache MP TP rank collapse was requested, but the attention "
                "backend did not declare the complete PAGE layout fully "
                f"replicated (published factor={published_replication}, "
                f"TP size={tp_size})"
            )
        self._is_kv_writer = rank % requested_replication == 0
        views = _build_cache_views(
            transfer_tensors,
            num_blocks=normalized_num_blocks,
        )
        block_regions = getattr(transfer_tensors, "block_regions", None) or []
        expected = sum(int(region.unit_bytes) for region in block_regions)
        if expected != views.bytes_per_block:
            raise ValueError(
                "lmcache_mp block geometry mismatch: "
                f"views={views.bytes_per_block} transfer_regions={expected}"
            )

        adapter = _make_worker_adapter(self._config, rank)
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=layer_indices,
                tokens_per_block=self.block_size,
            )
            for layer_indices in views.layer_groups
        ]
        try:
            adapter.register_kv_caches(views.tensors, engine_group_infos=groups)
            chunk_size = offcfg._strict_integer(
                "LMCache MP chunk size",
                adapter.lmcache_tokens_per_chunk,
                minimum=1,
            )
            if chunk_size % self.block_size:
                raise ValueError(
                    f"LMCache MP chunk size {chunk_size} must be divisible by "
                    f"ATOM block size {self.block_size}"
                )
        except Exception:
            shutdown = getattr(adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise
        self._adapter = adapter
        self.chunk_size = chunk_size
        self._start_transfer_thread(rank)
        logger.info(
            "LMCache MP registered rank=%d tensors=%d groups=%d "
            "bytes_per_block=%d chunk=%d tp_replication=%d writer=%s "
            "save=%s load=%s",
            rank,
            len(views.tensors),
            len(views.layer_groups),
            views.bytes_per_block,
            self.chunk_size,
            requested_replication,
            self._is_kv_writer,
            self._do_save,
            self._do_load,
        )

    def _register_native_state_layout(
        self, transfer_tensors: Any, num_blocks: int | None
    ) -> None:
        """Register PAGE plus its checkpoint image groups, once for every worker.

        The layout is `build_native_state_mp_layout`'s, under the namespace
        its checkpoint spec and `stores_state_images` name, so a scheduler
        connected with the same spec finds what these workers store. TP rank
        collapse needs both PAGE and the images declared replicated.
        """
        from aiter.dist.parallel_state import get_tp_group

        spec = transfer_tensors.paged_state_checkpoint_spec
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
        images = stores_state_images(
            int(getattr(self._config, "state_replay_tokens", 0) or 0)
        )
        adapter = _make_worker_adapter(
            self._config, rank, checkpoint_spec=spec, images=images
        )
        try:
            require_native_state_server(adapter, self._config)
            chunk_size = int(adapter.lmcache_tokens_per_chunk)
            layout = build_native_state_mp_layout(
                transfer_tensors,
                block_size=self.block_size,
                chunk_size=chunk_size,
                num_blocks=num_blocks,
                images=images,
                null_block_id=self._null_block_id,
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
        self._start_transfer_thread(rank)
        self._native_layout = layout
        logger.info(
            "LMCache MP native state registered rank=%d images=%s native_image=%d "
            "units=%d groups=%d chunk=%d",
            rank,
            images,
            spec.image_bytes,
            spec.units_per_checkpoint,
            len(layout.kernel_groups),
            chunk_size,
        )

    def close(self) -> None:
        """Unregister this rank's cache views from the MP server.

        `ModelRunner.exit` calls this before the KV pool is freed. The server
        maps the pool over GPU IPC, and without an unregister it keeps that
        mapping -- and the memory -- until its worker reaper expires the
        instance (`--worker-reap-timeout-seconds`, 120 s by default), so an
        engine restarted on the same GPUs finds the memory still taken.
        `AtomMPWorkerAdapter.shutdown` drains in-flight operations, stops the
        heartbeat and unregisters.
        """
        if not self._transfers.close(timeout=_CLOSE_TIMEOUT_S):
            # A submit is still blocked (the client's own timeout is longer),
            # and the client is not thread-safe: shutting it down under the
            # drain thread is worse than leaving the instance to the server's
            # reaper. The drain refuses further work from here.
            logger.error(
                "LMCache MP transfer drain did not stop within %.0f s; "
                "leaving its client to the server's reaper",
                _CLOSE_TIMEOUT_S,
            )
            return
        transfer, self._transfer = self._transfer, None
        if transfer is not None:
            # Queued submissions go out first: shutdown drains only what the
            # adapter already holds.
            transfer.close(_TRANSFER_DRAIN_TIMEOUT_S)
        adapter, self._adapter = self._adapter, None
        if adapter is None:
            return
        try:
            adapter.shutdown()
        except Exception:
            # Teardown continues; the server's reaper is the backstop. A
            # transfer may still wait on a step event, so the events stay.
            logger.warning("LMCache MP worker shutdown failed", exc_info=True)
            return
        # Only now: `shutdown` drained the in-flight operations, so no
        # transfer waits on a step event any more.
        with self._lock:
            self._step_events.clear()

    def start_load_kv(self, metadata: Any) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        if self._adapter is None or self.chunk_size is None:
            raise RuntimeError("lmcache_mp KV caches are not registered")

        # Every step, so a backlog left when transfers stop still drains.
        self._owned_futures.release()
        self._settle_transfer_thread()
        # Saves still held from the previous step: this fence follows its
        # forward too.
        self._flush_after_forward()
        self._after_forward_ran = False
        event = None
        for req in metadata.requests:
            load = req.load_spec is not None and self._do_load
            save = req.save_spec is not None and self._do_save
            if save and req.after_forward:
                self._after_forward_saves.append(req)
                save = False
            if not (load or save):
                continue
            if event is None:
                event = self._step_event()
            if load:
                self._dispatch(self._submit_load, req, event)
            if save:
                self._dispatch(self._submit_save, req, event)

    def _step_event(self) -> PendingStepEvent | RecordedStepEvent:
        """The step's producer event, as `_queue_transfer` holds it.

        The plugin's drain takes one from the ring, recorded when a transfer
        first holds it (`step_events.py`). Native ATOM records
        `_producer_fence` now, on the forward thread.
        """
        if self._drain_thread:
            return PendingStepEvent(self._step_events)
        return RecordedStepEvent(self._producer_fence())

    def after_forward(self) -> None:
        """Fence this step's same-step saves behind its forward.

        Called by the model runner right after the step's forward is enqueued
        (before sampling). The fence is recorded here, but a save goes out only
        once that fence has completed: the server waits for a producer event on
        the cache's one copy stream and stages block ids with a host-blocking
        copy, so a save that reached it mid-forward would hold every transfer
        queued behind it -- loads included -- until the forward ended. With the
        transfer thread the save is submitted as soon as the fence completes;
        inline, `get_finished` submits it after the forward has returned.
        """
        self._after_forward_ran = True
        transfer = self._transfer
        if transfer is None or not self._after_forward_saves:
            return
        saves, self._after_forward_saves = self._after_forward_saves, []
        fence = self._producer_fence()
        event = RecordedStepEvent(fence)
        for req in saves:
            transfer.submit(self._submit_save, req, event, ready=fence.query)

    def _flush_after_forward(self) -> None:
        """Submit held same-step saves; their forward is enqueued or done."""
        saves, self._after_forward_saves = self._after_forward_saves, []
        if not saves:
            return
        if (
            self._transfer is not None or not self._after_forward_ran
        ) and not self._late_flush_logged:
            self._late_flush_logged = True
            logger.warning(
                "LMCache MP: same-step saves were not submitted after their "
                "forward; submitting them at the next connector call"
            )
        event = self._step_event()
        for req in saves:
            self._dispatch(self._submit_save, req, event)

    # -- submission thread ---------------------------------------------------

    def _start_transfer_thread(self, rank: int) -> None:
        if not envs.OFFLOAD_MP_TRANSFER_THREAD or self._drain_thread:
            # Off, or the drain thread already submits off the step thread.
            return
        adapter = self._adapter
        if getattr(adapter, "_producer_events", None) is not None and not (
            self._adapter_takes_recorded_events()
        ):
            # It would record its own pooled event on the submitting thread's
            # stream, not behind the forward.
            logger.warning(
                "LMCache MP adapter %s records producer events at submission; "
                "transfers stay on the forward thread",
                type(adapter).__name__,
            )
            return
        self._fences = _ProducerFences()
        self._transfer = _TransferThread(f"mp-transfer-{rank}")
        logger.info("LMCache MP transfer thread started rank=%d", rank)

    def _adapter_takes_recorded_events(self) -> bool:
        return bool(
            getattr(type(self._adapter), "ACCEPTS_RECORDED_PRODUCER_EVENTS", False)
        )

    def _submit_kwargs(self) -> dict[str, Any]:
        """Tell an adapter that can hear it the fence is recorded and reused."""
        if self._fences is not None and self._adapter_takes_recorded_events():
            return {"producer_event_recorded": True}
        return {}

    def _producer_fence(self) -> Any:
        """An interprocess event recorded at the current stream position."""
        if self._fences is not None:
            return self._fences.record()
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        return event

    def _dispatch(self, submit: Any, req: LMCacheReqMeta, event: Any) -> None:
        if self._transfer is None:
            submit(req, event)
        else:
            self._transfer.submit(submit, req, event)

    def _settle_transfer_thread(self) -> None:
        """Forward-thread housekeeping for the submission thread.

        Drops the events finished submissions referenced (an interprocess
        event must not be destroyed on that thread) and surfaces a
        submission that raised.
        """
        transfer = self._transfer
        if transfer is None:
            return
        transfer.take_spent()
        transfer.raise_error()

    def _queue_transfer(
        self,
        kind: str,
        operation_id: str,
        completion: Any,
        req: LMCacheReqMeta,
        spec: Any,
        event: PendingStepEvent | RecordedStepEvent,
        start: int = 0,
        end: int = 0,
    ) -> None:
        """Hand one submit to `_transfers`, holding the step's event.

        The caller has already put `operation_id` in its submitting set;
        `_perform_submit` moves it to the pending map once the submit lands,
        so a duplicate operation id is still refused while the submit is in
        flight.
        """
        with self._lock:
            index, step_event = event.hold_locked()
            if index is not None:
                self._step_event_of[operation_id] = index
            self._submitting_owner[operation_id] = str(req.req_id)
        self._transfers.put(
            _QueuedTransfer(
                kind,
                operation_id,
                completion,
                _mp_session_id(self._config, req.req_id),
                spec,
                step_event,
                start,
                end,
                time.monotonic(),
            )
        )

    def _release_step_event(self, operation_id: str) -> None:
        """The transfer is terminal: the server no longer waits on its event.

        Called with `_lock` held.
        """
        index = self._step_event_of.pop(operation_id, None)
        if index is not None:
            self._step_events.release(index)

    def _block_slice(self, req: LMCacheReqMeta, start: int, end: int) -> list[int]:
        if start < 0 or end < start:
            raise ValueError(f"invalid LMCache MP token range [{start}, {end})")
        if start % self.block_size or end % self.block_size:
            raise ValueError(
                f"LMCache MP token range [{start}, {end}) must align to "
                f"block size {self.block_size}"
            )
        block_ids = list(
            req.block_ids[start // self.block_size : end // self.block_size]
        )
        expected = (end - start) // self.block_size
        if len(block_ids) != expected:
            raise ValueError(
                f"LMCache MP request {req.req_id} needs {expected} blocks for "
                f"[{start}, {end}), got {len(block_ids)}"
            )
        return block_ids

    def _transfer_block_ids(
        self, req: LMCacheReqMeta, start: int, end: int, *, loading: bool
    ) -> list[list[int]]:
        """The block-id list per engine group that one transfer names."""
        if self._native_layout is not None:
            return self._native_block_ids(req, start, end, loading=loading)
        if req.native_state is not None:
            raise ValueError(
                f"LMCache MP request {req.req_id} carries a state image, but "
                "no image group was registered"
            )
        return [self._block_slice(req, start, end)]

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
        count = (end - start) // self.chunk_size
        if state.kv_only != (not self._native_layout.images):
            # PAGE alone exactly when the layout registered no image group: the
            # scheduler and this worker asked `stores_state_images` the same.
            raise ValueError(
                "a KV-only native transfer needs a layout without images, and "
                "only it"
            )
        if state.kv_only:
            if state.unit_ids or state.destination_slot is not None:
                raise ValueError("a KV-only native transfer moves no image")
            return [self._block_slice(req, start, end)]
        self._check_state_destination(state, loading=loading)
        unit_ids = self._native_layout.validate_unit_ids(state.unit_ids)
        pages = self._block_slice(req, start, end)
        if set(pages) & set(unit_ids):
            raise ValueError("checkpoint PAGE units must not overlap KV PAGE blocks")
        return self._native_layout.transfer_block_ids(pages, unit_ids, count)

    def _check_state_destination(self, state: Any, *, loading: bool) -> None:
        """Refuse an image transfer that names an Active SLOT.

        This worker restores no SLOT: a load leaves the image in the PAGE units
        it names, and the backend restores it from there.
        """
        del loading
        if state.destination_slot is not None:
            raise ValueError(
                "this LMCache MP worker restores no Active SLOT; a native-state "
                "transfer must not name one"
            )

    def _submit_load(
        self, req: LMCacheReqMeta, event: PendingStepEvent | RecordedStepEvent
    ) -> None:
        from lmcache.integration.atom import AtomMPTransferSpec

        assert req.load_spec is not None
        completion = req.load_operation or req.req_id
        operation_id = _transfer_operation_id("load", completion)
        with self._lock:
            if (
                operation_id in self._completed_load_operations
                or operation_id in self._pending_loads
                or operation_id in self._submitting_loads
            ):
                raise RuntimeError(
                    f"duplicate LMCache MP load operation {operation_id!r}"
                )
            self._submitting_loads.add(operation_id)
        start = int(req.load_spec.hbm_cached_tokens)
        end = (
            int(req.load_spec.lmcache_cached_tokens)
            if req.load_spec.transfer_end_tokens is None
            else int(req.load_spec.transfer_end_tokens)
        )
        try:
            if start % self.chunk_size or end % self.chunk_size:
                raise ValueError(
                    f"load range [{start}, {end}) is not LMCache chunk aligned "
                    f"({self.chunk_size})"
                )
            spec = AtomMPTransferSpec(
                token_ids=list(req.token_ids),
                block_ids=self._transfer_block_ids(req, start, end, loading=True),
                start=start,
                end=end,
            )
        except Exception:
            # Nothing was sent: the failure is provable and terminal.
            logger.exception(
                "Invalid LMCache MP load descriptor for %s",
                req.req_id,
            )
            with self._lock:
                self._submitting_loads.discard(operation_id)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_load_operations,
                    self._completed_load_operation_order,
                )
                self._immediate_load_failures.add(completion)
            return
        self._queue_transfer("load", operation_id, completion, req, spec, event)

    def _submit_save(
        self, req: LMCacheReqMeta, event: PendingStepEvent | RecordedStepEvent
    ) -> None:
        from lmcache.integration.atom import AtomMPTransferSpec

        assert req.save_spec is not None
        completion = req.save_operation or req.req_id
        operation_id = _transfer_operation_id("save", completion)
        with self._lock:
            if (
                operation_id in self._completed_save_operations
                or operation_id in self._pending_saves
                or operation_id in self._submitting_saves
            ):
                raise RuntimeError(
                    f"duplicate LMCache MP save operation {operation_id!r}"
                )
            self._submitting_saves.add(operation_id)
        end = (len(req.token_ids) // self.chunk_size) * self.chunk_size
        start = (
            int(req.save_spec.skip_leading_tokens) // self.chunk_size
        ) * self.chunk_size
        if not self._is_kv_writer or start >= end:
            with self._lock:
                self._submitting_saves.discard(operation_id)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_save_operations,
                    self._completed_save_operation_order,
                )
                self._immediate_saves[completion] = (
                    (start, end) if start < end else None
                )
            return
        try:
            spec = AtomMPTransferSpec(
                token_ids=list(req.token_ids),
                block_ids=self._transfer_block_ids(req, start, end, loading=False),
                start=start,
                end=end,
            )
        except Exception:
            # Nothing was sent: report a terminal failure so the save settles.
            logger.exception(
                "Invalid LMCache MP save descriptor for %s",
                req.req_id,
            )
            with self._lock:
                self._submitting_saves.discard(operation_id)
                self._pending_saves[operation_id] = _PendingSave(
                    completion=completion,
                    future=None,
                    start=start,
                    end=end,
                )
            # Nothing was queued: a round is the only place this terminal
            # entry is ever reported from.
            self._transfers.ensure_running()
            return
        self._queue_transfer(
            "save", operation_id, completion, req, spec, event, start, end
        )

    def _perform_submit(self, item: _QueuedTransfer) -> None:
        """Submit one transfer, on whichever thread `_transfers` runs it."""
        loading = item.kind == "load"
        try:
            if loading:
                submit = self._adapter.submit_retrieve_request
            else:
                submit = getattr(
                    self._adapter,
                    "submit_store_request_with_chunk_events",
                    self._adapter.submit_store_request,
                )
            transfer = submit(
                item.request_id, item.spec, item.event, **self._submit_kwargs()
            )
        except Exception:
            # The server may have taken the request before the connection
            # raised, and may still read or write the blocks, so they stay
            # leased until a terminal report -- which the transfer deadline
            # turns into a fail-stop if it never comes.
            logger.exception(
                "LMCache MP %s submission unprovable for %s",
                item.kind,
                item.request_id,
            )
            transfer = _UnprovableSubmission()
        with self._lock:
            self._submitting_owner.pop(item.operation_id, None)
            if loading:
                self._submitting_loads.discard(item.operation_id)
                self._pending_loads[item.operation_id] = _PendingLoad(
                    completion=item.completion,
                    future=transfer,
                    started_at=item.queued_at,
                )
            else:
                self._submitting_saves.discard(item.operation_id)
                self._pending_saves[item.operation_id] = _PendingSave(
                    completion=item.completion,
                    future=transfer,
                    started_at=item.queued_at,
                    start=item.start,
                    end=item.end,
                )

    def _poll_round(self) -> None:
        """Advance every pending transfer once: one round of `_transfers`."""
        self._poll_pending_loads()
        self._poll_pending_saves()

    def _poll_pending_loads(self) -> None:
        """Advance every pending load.

        The polls run without `_lock` held: a terminal poll imports the peer's
        IPC event and costs milliseconds, and holding the lock across it would
        just move the step's stall from the poll to the lock. Only this thread
        removes entries, so an entry the snapshot missed is picked up next
        round.
        """
        with self._lock:
            pending = list(self._pending_loads.items())
        terminal_ops: list[tuple[str, _PendingLoad, Any]] = []
        for operation_id, entry in pending:
            terminal, result = _terminal_future_result(entry.future)
            if not terminal:
                _enforce_transfer_deadline(
                    operation_id, entry.started_at, self._transfer_deadline_s
                )
                continue
            terminal_ops.append((operation_id, entry, result))
        if not terminal_ops:
            return
        with self._lock:
            for operation_id, entry, result in terminal_ops:
                self._pending_loads.pop(operation_id, None)
                self._release_step_event(operation_id)
                self._owned_futures.add_locked(entry.future)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_load_operations,
                    self._completed_load_operation_order,
                )
                if result is not True:
                    self._drained_failed_load.add(entry.completion)
                else:
                    self._drained_done_load.add(entry.completion)

    def _poll_pending_saves(self) -> None:
        """Advance every pending save; see `_poll_pending_loads`."""
        with self._lock:
            pending = list(self._pending_saves.items())
        if not pending:
            return
        done_save: set[SaveCompletionId] = set()
        completions: set[ConnectorCompletion] = set()
        terminal_ops: list[tuple[str, _PendingSave, Any]] = []
        for operation_id, entry in pending:
            take_ranges = getattr(entry.future, "take_completed_ranges", None)
            if callable(take_ranges) and isinstance(entry.completion, SaveOperationId):
                try:
                    completions |= _source_safe_completions(
                        entry.completion, take_ranges()
                    )
                except Exception:
                    logger.warning(
                        "LMCache MP source-safe event polling failed", exc_info=True
                    )
            terminal, result = _terminal_future_result(entry.future)
            if not terminal:
                _enforce_transfer_deadline(
                    operation_id, entry.started_at, self._transfer_deadline_s
                )
                continue
            terminal_ops.append((operation_id, entry, result))
        with self._lock:
            for operation_id, entry, result in terminal_ops:
                self._pending_saves.pop(operation_id, None)
                self._release_step_event(operation_id)
                self._owned_futures.add_locked(entry.future)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_save_operations,
                    self._completed_save_operation_order,
                )
                done_save.add(entry.completion)
                if isinstance(entry.completion, SaveOperationId):
                    completions |= _source_safe_completions(
                        entry.completion,
                        _chunk_ranges(entry.start, entry.end, int(self.chunk_size)),
                    )
                    completions.add(
                        ConnectorCompletion(
                            DENSE_PAGE_STORE_CHANNEL,
                            entry.completion,
                            result is True,
                        )
                    )
            self._drained_done_save |= done_save
            self._drained_completions |= completions

    def _transfers_in_flight_locked(self) -> bool:
        return bool(
            self._pending_loads or self._pending_saves or self._submitting_owner
        )

    def _outstanding_locked(self, wanted: set[str]) -> list[str]:
        """Transfers that still read or write the blocks of the `wanted` requests.

        Raises for one whose submit raised: the server may have the request
        and write its blocks however long the caller waits.
        """
        outstanding = [
            operation_id
            for operation_id, owner in self._submitting_owner.items()
            if owner in wanted
        ]
        for pending in (self._pending_saves, self._pending_loads):
            for operation_id, entry in pending.items():
                if _req_id_of(entry.completion) not in wanted:
                    continue
                if isinstance(entry.future, _UnprovableSubmission):
                    raise RuntimeError(  # noqa: TRY004 - a state, not a bad argument
                        f"LMCache MP: cannot release {sorted(wanted)}: the submit "
                        f"of {operation_id} raised, so the server may have the "
                        "request and still use its blocks"
                    )
                outstanding.append(operation_id)
        return outstanding

    def fence(self, req_ids) -> None:
        """Block until no transfer for these requests still uses their blocks.

        For a frontend that hands a preempted request's blocks to the next
        allocation at once (the vLLM plugin): a save still reading them would
        store the next occupant's bytes under the preempted request's tokens.
        Waits on the drain thread's rounds, up to the transfer deadline, and
        fail-stops past it. The completions stay for `get_finished`, which
        releases their leases.
        """
        wanted = {str(req_id) for req_id in req_ids}
        if not wanted:
            return
        self._transfers.fence(
            lambda: self._outstanding_locked(wanted),
            self._transfer_deadline_s,
            f"requests {sorted(wanted)}",
        )

    def _finished_housekeeping(self) -> None:
        """Start of `get_finished`, which runs after the step's forward.

        The forward is done, so destroying replaced fences no longer waits on
        it, and same-step saves held for it can go out.
        """
        self._flush_after_forward()
        self._settle_transfer_thread()
        if self._fences is not None:
            self._fences.release_retired()

    def get_finished(self) -> KVConnectorOutput:
        if self._adapter is None:
            return KVConnectorOutput()
        self._finished_housekeeping()
        # The drain thread cannot stop the engine; the step can.
        self._transfers.raise_if_failed()
        with self._lock:
            in_flight = self._transfers_in_flight_locked()
            wait_s = _LAST_LOOK_WAIT_S if self._pending_loads else 0.0
        if in_flight:
            # Inline, the step's own poll, reported below. On the drain thread,
            # a wake -- a step that queues nothing (an idle engine waiting on a
            # load) raises none of its own -- and a last look at a load.
            self._transfers.report_round(wait_s)
        output = KVConnectorOutput()
        with self._lock:
            # Heartbeat health is a control-plane signal, not proof that GPU
            # work has quiesced, so a save is only reported once its device
            # event is terminal -- that check lives in `_poll_pending_saves`.
            output.finished_saving |= self._drained_done_save
            output.connector_completions |= self._drained_completions
            output.finished_loading |= self._drained_done_load
            output.failed_loading |= self._drained_failed_load
            self._drained_done_save = set()
            self._drained_completions = set()
            self._drained_done_load = set()
            self._drained_failed_load = set()
            output.finished_saving.update(self._immediate_saves)
            for completion, token_range in self._immediate_saves.items():
                if isinstance(completion, SaveOperationId):
                    if token_range is not None:
                        output.connector_completions |= _source_safe_completions(
                            completion,
                            _chunk_ranges(*token_range, int(self.chunk_size)),
                        )
                    output.connector_completions.add(
                        ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, completion, True)
                    )
            output.failed_loading.update(self._immediate_load_failures)
            self._immediate_saves.clear()
            self._immediate_load_failures.clear()
        return output


__all__ = ["LMCacheMPConnector"]
