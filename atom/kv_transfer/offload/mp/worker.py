# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker-side LMCache MP connector for backend-published PAGE views.

Attention backends publish block-major tensor views through
``KVTransferTensors``. LMCache MP can attach several physical kernel groups to
the same engine block-id space, so this connector registers those opaque views
without knowing which model or attention implementation produced them.

Transfers are submitted and polled on the forward thread. Under
OFFLOAD_MP_TRANSFER_THREAD the transfer thread submits them. A frontend whose
forward thread cannot afford either (the vLLM plugin) builds the worker with
``transfers_off_step``: the transfer thread submits and polls, and `fence`
waits on its rounds.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import torch

from atom.kv_transfer.disaggregation.base import KVConnectorBase
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    LoadCompletionId,
    SaveCompletionId,
    SaveOperationId,
    req_id_of,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata
from atom.kv_transfer.offload.mp import deployment
from atom.kv_transfer.offload.mp.deployment import (
    _make_worker_adapter,
    _mp_session_id,
    _published_tp_replication_factor,
    _tp_replication_factor,
    _validate_mp_config,
    require_native_state_server,
    transfer_spec,
)
from atom.kv_transfer.offload.mp.metadata import MPReqMeta
from atom.kv_transfer.offload.mp.native_state_layout import (
    build_native_state_mp_layout,
    stores_state_images,
)
from atom.kv_transfer.offload.mp.page_views import _build_cache_views
from atom.kv_transfer.offload.mp.token_shipping import WorkerTokens, token_list
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

# How long worker teardown waits for queued transfer submissions.
_TRANSFER_DRAIN_TIMEOUT_S = 60.0
# How long `get_finished` waits, while a load is in flight, for a polling
# transfer thread's round that is a pure poll: a load that lands during the
# forward and misses it keeps its request in WAITING_FOR_REMOTE_KVS a step
# longer.
_LAST_LOOK_WAIT_S = 0.001
# A handed-back future holds the server's imported IPC event, which is
# destroyed only on the forward thread at a step boundary, while its stream is
# idle. A stream that is never idle (async scheduling) drops a batch per call
# once this many are held.
_RETURNED_FUTURES_CAP = 8192
_RETURNED_FUTURES_BATCH = 64


def _takes_recorded_events(adapter: Any) -> bool:
    return bool(getattr(type(adapter), "ACCEPTS_RECORDED_PRODUCER_EVENTS", False))


def _records_its_own_events(adapter: Any) -> bool:
    """It would record its pooled event on the submitting thread's stream, not
    behind the forward."""
    return getattr(adapter, "_producer_events", None) is not None and not (
        _takes_recorded_events(adapter)
    )


class LMCacheMPConnector(KVConnectorBase):
    """Worker-side LMCache MP connector for backend-published PAGE views."""

    is_producer = False
    # Whether a checkpoint-image load restores into an Active SLOT, which every
    # load then names and no save may. Here the image stays in the PAGE units a
    # load fills, and the backend restores it from there.
    restores_into_slot = False

    def __init__(self, config: Any, *, transfers_off_step: bool = False) -> None:
        _validate_mp_config(config)
        self._config = config
        # Each request's token prefix, rebuilt from incremental shipments
        # (`token_shipping.py`). Here, in the worker every frontend reaches (the
        # shell, the native-state worker, the vLLM plugin's adapter), not in a
        # shell only some of them go through.
        self._tokens = WorkerTokens()
        # Submit and poll on the transfer thread, whatever
        # OFFLOAD_MP_TRANSFER_THREAD says: for a frontend whose forward thread
        # cannot afford either (the vLLM plugin).
        self._transfers_off_step = transfers_off_step
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
        # Set only when the backend publishes PAGE-backed checkpoint images
        # (`_register_native_state_layout`); every transfer then carries one.
        self._native_layout: Any = None
        self._adapter: Any = None
        self._is_kv_writer = True
        self._pending_saves: dict[str, _PendingSave] = {}
        self._pending_loads: dict[str, _PendingLoad] = {}
        self._submitting_saves: set[str] = set()
        self._submitting_loads: set[str] = set()
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
        self._lock = threading.Lock()
        # With OFFLOAD_MP_TRANSFER_THREAD: the submitting thread and the
        # reused producer fences it needs (see `transfer_thread`).
        self._transfer: _TransferThread | None = None
        self._fences: _ProducerFences | None = None
        # Same-step saves of the current step, held until their forward is
        # enqueued (`after_forward`) -- and, inline, until it has finished.
        self._after_forward_saves: list[MPReqMeta] = []
        self._after_forward_ran = False
        self._late_flush_logged = False
        # Polling transfer thread only. Submissions it has queued, by request:
        # `fence` waits for them too, since they reach the server after the
        # step that queued them. And what its rounds settled since the last
        # `get_finished`, which swaps it out.
        self._queued_requests: dict[str, int] = {}
        self._polled = KVConnectorOutput()

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
            self._refuse_unfenceable_adapter(adapter)
            adapter.register_kv_caches(views.tensors, engine_group_infos=groups)
            chunk_size = offcfg._strict_integer(
                "LMCache MP chunk size",
                adapter.lmcache_tokens_per_chunk,
                minimum=1,
            )
            deployment.require_chunk_matches_server(adapter, self._config)
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
            self._refuse_unfenceable_adapter(adapter)
            require_native_state_server(adapter, self._config)
            deployment.require_chunk_matches_server(adapter, self._config)
            chunk_size = int(adapter.lmcache_tokens_per_chunk)
            layout = build_native_state_mp_layout(
                transfer_tensors,
                block_size=self.block_size,
                chunk_size=chunk_size,
                num_blocks=num_blocks,
                images=images,
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
        transfer = self._transfer
        if transfer is not None:
            # Queued submissions go out first: shutdown drains only what the
            # adapter already holds.
            stopped = transfer.close(_TRANSFER_DRAIN_TIMEOUT_S)
            self._transfer = None
            if not stopped:
                # Still inside a submit or a poll on this adapter's futures:
                # leave it registered for the server's reaper rather than tear
                # it down under a live thread.
                logger.error(
                    "LMCache MP transfer thread did not stop within %.0fs; "
                    "skipping the adapter shutdown",
                    _TRANSFER_DRAIN_TIMEOUT_S,
                )
                return
        adapter, self._adapter = self._adapter, None
        if adapter is None:
            return
        try:
            adapter.shutdown()
        except Exception:
            # Teardown continues; the server's reaper is the backstop.
            logger.warning("LMCache MP worker shutdown failed", exc_info=True)

    def start_load_kv(self, metadata: Any) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        self._tokens.resolve(metadata)
        if self._adapter is None or self.chunk_size is None:
            raise RuntimeError("lmcache_mp KV caches are not registered")

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
                event = self._producer_fence()
            if load:
                self._dispatch(self._submit_load, req, event)
            if save:
                self._dispatch(self._submit_save, req, event)

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
        event = self._producer_fence()
        for req in saves:
            self._dispatch(self._submit_save, req, event, ready=event.query)

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
        event = self._producer_fence()
        for req in saves:
            self._dispatch(self._submit_save, req, event)

    # -- submission thread ---------------------------------------------------

    def _start_transfer_thread(self, rank: int) -> None:
        off_step = self._transfers_off_step
        if not off_step and not envs.OFFLOAD_MP_TRANSFER_THREAD:
            return
        if _records_its_own_events(self._adapter):
            logger.warning(
                "LMCache MP adapter %s records producer events at submission; "
                "transfers stay on the forward thread",
                type(self._adapter).__name__,
            )
            return
        self._fences = _ProducerFences()
        self._transfer = _TransferThread(
            f"mp-transfer-{rank}", poll=self._poll_round if off_step else None
        )
        logger.info(
            "LMCache MP transfer thread started rank=%d polls=%s", rank, off_step
        )

    def _refuse_unfenceable_adapter(self, adapter: Any) -> None:
        """Before registration: a worker built to run transfers off the step
        cannot take an adapter that records its own producer events, which
        keeps them on the forward thread, where nothing could fence them."""
        if self._transfers_off_step and _records_its_own_events(adapter):
            raise RuntimeError(
                f"LMCache MP adapter {type(adapter).__name__} records producer "
                "events at submission, so its transfers cannot leave the "
                "forward thread, and this frontend's preemption fence needs "
                "them to"
            )

    def _submit_kwargs(self) -> dict[str, Any]:
        """Tell an adapter that can hear it the fence is recorded and reused."""
        if self._fences is not None and _takes_recorded_events(self._adapter):
            return {"producer_event_recorded": True}
        return {}

    def _producer_fence(self) -> Any:
        """An interprocess event recorded at the current stream position."""
        if self._fences is not None:
            return self._fences.record()
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        return event

    def _dispatch(
        self,
        submit: Any,
        req: MPReqMeta,
        event: Any,
        ready: Callable[[], bool] | None = None,
    ) -> None:
        transfer = self._transfer
        if transfer is None:
            submit(req, event)
            return
        if not transfer.polls:
            transfer.submit(submit, req, event, ready=ready)
            return
        owner = str(req.req_id)
        with self._lock:
            self._queued_requests[owner] = self._queued_requests.get(owner, 0) + 1
        transfer.submit(self._submit_queued, submit, req, event, ready=ready)

    def _submit_queued(self, submit: Any, req: MPReqMeta, event: Any) -> None:
        """One queued submission, on a polling transfer thread."""
        try:
            submit(req, event)
        finally:
            owner = str(req.req_id)
            with self._lock:
                left = self._queued_requests.get(owner, 1) - 1
                if left:
                    self._queued_requests[owner] = left
                else:
                    self._queued_requests.pop(owner, None)

    def _settle_transfer_thread(self) -> None:
        """Forward-thread housekeeping for the submission thread.

        Drops the events finished submissions referenced (an interprocess
        event must not be destroyed on that thread) and the futures its polls
        finished, and surfaces a submission that raised.
        """
        transfer = self._transfer
        if transfer is None:
            return
        transfer.take_spent()
        if transfer.returned:
            if torch.cuda.current_stream().query():
                transfer.take_returned()
            elif transfer.returned > _RETURNED_FUTURES_CAP:
                transfer.take_returned(_RETURNED_FUTURES_BATCH)
        transfer.raise_error()

    def _block_slice(self, req: MPReqMeta, start: int, end: int) -> list[int]:
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
        self, req: MPReqMeta, start: int, end: int, *, loading: bool
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
        req: MPReqMeta,
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
        if self.restores_into_slot:
            if loading and state.destination_slot is None:
                raise ValueError("native-state restore requires a destination SLOT")
            if not loading and state.destination_slot is not None:
                raise ValueError("native-state save must refer to immutable PAGE units")
        elif state.destination_slot is not None:
            raise ValueError(
                "this LMCache MP worker restores no Active SLOT; a native-state "
                "transfer must not name one"
            )
        unit_ids = self._native_layout.validate_unit_ids(state.unit_ids)
        pages = self._block_slice(req, start, end)
        if set(pages) & set(unit_ids):
            raise ValueError("checkpoint PAGE units must not overlap KV PAGE blocks")
        return self._native_layout.transfer_block_ids(pages, unit_ids, count)

    def _submit_load(self, req: MPReqMeta, event: Any) -> None:
        assert req.load_spec is not None
        completion = req.load_operation or req.req_id
        request_id = _mp_session_id(self._config, req.req_id)
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
            op = transfer_spec(
                token_ids=token_list(req.token_ids),
                block_ids=self._transfer_block_ids(req, start, end, loading=True),
                start=start,
                end=end,
                cache_salt=req.cache_salt,
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
        try:
            transfer = self._adapter.submit_retrieve_request(
                request_id,
                op,
                event,
                **self._submit_kwargs(),
            )
        except Exception:
            # The server may have taken the request before the connection
            # raised, and may still write the destination blocks.
            logger.exception(
                "LMCache MP load submission unprovable for %s",
                req.req_id,
            )
            transfer = _UnprovableSubmission()
        with self._lock:
            self._submitting_loads.discard(operation_id)
            self._pending_loads[operation_id] = _PendingLoad(
                completion=completion,
                future=transfer,
            )

    def _submit_save(self, req: MPReqMeta, event: Any) -> None:
        assert req.save_spec is not None
        completion = req.save_operation or req.req_id
        request_id = _mp_session_id(self._config, req.req_id)
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
        if not self._is_kv_writer:
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
        if start >= end:
            with self._lock:
                self._submitting_saves.discard(operation_id)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_save_operations,
                    self._completed_save_operation_order,
                )
                self._immediate_saves[completion] = None
            return
        try:
            op = transfer_spec(
                token_ids=token_list(req.token_ids),
                block_ids=self._transfer_block_ids(req, start, end, loading=False),
                start=start,
                end=end,
                cache_salt=req.cache_salt,
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
            return
        try:
            submit = getattr(
                self._adapter,
                "submit_store_request_with_chunk_events",
                self._adapter.submit_store_request,
            )
            transfer = submit(
                request_id,
                op,
                event,
                **self._submit_kwargs(),
            )
        except Exception:
            # The server may have received the request before the connection
            # raised: keep the source leased until a terminal report, which
            # the transfer deadline turns into a fail-stop if it never comes.
            logger.exception(
                "LMCache MP save submission unprovable for %s",
                req.req_id,
            )
            transfer = _UnprovableSubmission()
        with self._lock:
            self._submitting_saves.discard(operation_id)
            self._pending_saves[operation_id] = _PendingSave(
                completion=completion,
                future=transfer,
                start=start,
                end=end,
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

    def _poll_entries(
        self,
        saves: list[tuple[str, _PendingSave]],
        loads: list[tuple[str, _PendingLoad]],
        into: KVConnectorOutput,
    ) -> tuple[list[tuple[str, Any, Any]], list[tuple[str, Any, Any]]]:
        """Poll ``saves``, then ``loads``, once; return the terminal ones.

        Source-safe ranges go into ``into`` as they arrive. Changes no worker
        state, so a polling transfer thread runs it without `_lock`: a terminal
        poll imports the peer's IPC event and costs milliseconds. Raises at a
        transfer past its deadline.
        """
        # Heartbeat health is a control-plane signal, not proof that GPU work
        # submitted before the failure has quiesced. Keep every real future
        # until its device event is terminal. A pre-submit drop is represented
        # by None, while LMCache's missing-registration path returns an
        # event-free terminal False future.
        terminal_saves = []
        for operation_id, pending in saves:
            take_ranges = getattr(pending.future, "take_completed_ranges", None)
            if callable(take_ranges) and isinstance(
                pending.completion, SaveOperationId
            ):
                try:
                    into.connector_completions |= _source_safe_completions(
                        pending.completion, take_ranges()
                    )
                except Exception:
                    logger.warning(
                        "LMCache MP source-safe event polling failed", exc_info=True
                    )
            terminal, result = self._poll_entry(operation_id, pending)
            if terminal:
                terminal_saves.append((operation_id, pending, result))
        terminal_loads = []
        for operation_id, pending in loads:
            terminal, result = self._poll_entry(operation_id, pending)
            if terminal:
                terminal_loads.append((operation_id, pending, result))
        return terminal_saves, terminal_loads

    def _poll_entry(self, operation_id: str, pending: Any) -> tuple[bool, Any]:
        """Poll one transfer: ``(terminal, result)``, or a fail-stop when it
        is past its deadline and not terminal."""
        terminal, result = _terminal_future_result(pending.future)
        if not terminal:
            _enforce_transfer_deadline(
                operation_id, pending.started_at, self._transfer_deadline_s
            )
        return terminal, result

    def _settle_terminal_locked(
        self,
        terminal_saves: list[tuple[str, Any, Any]],
        terminal_loads: list[tuple[str, Any, Any]],
        into: KVConnectorOutput,
        retire: Callable[[Any], None] | None = None,
    ) -> None:
        """Report what `_poll_entries` found terminal; called with `_lock` held.

        ``retire`` takes each finished future, for a polling transfer thread
        to hand back; without it the caller, the forward thread, drops them.
        """
        for operation_id, pending, result in terminal_saves:
            self._pending_saves.pop(operation_id, None)
            _remember_operation_tombstone(
                operation_id,
                self._completed_save_operations,
                self._completed_save_operation_order,
            )
            into.finished_saving.add(pending.completion)
            if isinstance(pending.completion, SaveOperationId):
                into.connector_completions |= _source_safe_completions(
                    pending.completion,
                    _chunk_ranges(pending.start, pending.end, int(self.chunk_size)),
                )
                into.connector_completions.add(
                    ConnectorCompletion(
                        DENSE_PAGE_STORE_CHANNEL,
                        pending.completion,
                        result is True,
                    )
                )
            if retire is not None:
                retire(pending.future)
        for operation_id, pending, result in terminal_loads:
            self._pending_loads.pop(operation_id, None)
            _remember_operation_tombstone(
                operation_id,
                self._completed_load_operations,
                self._completed_load_operation_order,
            )
            if result is not True:
                into.failed_loading.add(pending.completion)
            else:
                into.finished_loading.add(pending.completion)
            if retire is not None:
                retire(pending.future)

    def _poll_round(self, transfer: _TransferThread) -> bool:
        """One round on the polling transfer thread ``transfer``.

        Polls without `_lock` and settles what finished into `_polled` for
        `get_finished`, handing each finished future back to ``transfer``
        itself: the round may still run while `close` shuts that thread down.
        Returns whether transfers are still in flight.
        """
        with self._lock:
            saves = list(self._pending_saves.items())
            loads = list(self._pending_loads.items())
        polled = KVConnectorOutput()
        terminal_saves, terminal_loads = self._poll_entries(saves, loads, polled)
        with self._lock:
            self._polled.connector_completions |= polled.connector_completions
            self._settle_terminal_locked(
                terminal_saves, terminal_loads, self._polled, transfer.hand_back
            )
            return self._transfers_in_flight_locked()

    def _transfers_in_flight_locked(self) -> bool:
        return bool(self._pending_loads or self._pending_saves or self._queued_requests)

    def _outstanding_locked(self, wanted: set[str]) -> list[str]:
        """Transfers that still read or write the blocks of the `wanted` requests.

        Raises for one whose submit raised: the server may have the request
        and write its blocks however long the caller waits.
        """
        outstanding = [
            f"queued:{owner}" for owner in self._queued_requests if owner in wanted
        ]
        for pending in (self._pending_saves, self._pending_loads):
            for operation_id, entry in pending.items():
                if req_id_of(entry.completion) not in wanted:
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
        Waits on the transfer thread's rounds, up to the transfer deadline, and
        fail-stops past it. The completions stay for `get_finished`, which
        releases their leases.
        """
        wanted = {str(req_id) for req_id in req_ids}
        if not wanted:
            return
        transfer = self._transfer
        if transfer is None or not transfer.polls:
            raise NotImplementedError(
                "LMCache MP fence: only a worker built with transfers_off_step "
                "fences"
            )

        def outstanding() -> list[str]:
            with self._lock:
                return self._outstanding_locked(wanted)

        transfer.fence(
            outstanding, self._transfer_deadline_s, f"requests {sorted(wanted)}"
        )

    def get_finished(self) -> KVConnectorOutput:
        if self._adapter is None:
            return KVConnectorOutput()
        self._finished_housekeeping()
        transfer = self._transfer
        if transfer is not None and transfer.polls:
            return self._finished_off_step(transfer)
        output = KVConnectorOutput()
        with self._lock:
            terminal_saves, terminal_loads = self._poll_entries(
                list(self._pending_saves.items()),
                list(self._pending_loads.items()),
                output,
            )
            self._settle_terminal_locked(terminal_saves, terminal_loads, output)
            self._take_immediate_locked(output)
        return output

    def _finished_off_step(self, transfer: _TransferThread) -> KVConnectorOutput:
        """What the polling transfer thread's rounds settled, plus the
        immediate completions."""
        with self._lock:
            in_flight = self._transfers_in_flight_locked()
            wait_s = _LAST_LOOK_WAIT_S if self._pending_loads else 0.0
        if in_flight:
            # A wake -- a step that queues nothing (an idle engine waiting on a
            # load) raises none of its own -- and a last look at a load.
            transfer.look(wait_s)
        with self._lock:
            output, self._polled = self._polled, KVConnectorOutput()
            self._take_immediate_locked(output)
        return output

    def _take_immediate_locked(self, into: KVConnectorOutput) -> None:
        """Report what settled without the server; called with `_lock` held."""
        into.finished_saving.update(self._immediate_saves)
        for completion, token_range in self._immediate_saves.items():
            if isinstance(completion, SaveOperationId):
                if token_range is not None:
                    into.connector_completions |= _source_safe_completions(
                        completion,
                        _chunk_ranges(*token_range, int(self.chunk_size)),
                    )
                into.connector_completions.add(
                    ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, completion, True)
                )
        into.failed_loading.update(self._immediate_load_failures)
        self._immediate_saves.clear()
        self._immediate_load_failures.clear()


__all__ = ["LMCacheMPConnector"]
