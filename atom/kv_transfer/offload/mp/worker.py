# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker-side LMCache MP connector for backend-published PAGE views.

Attention backends publish block-major tensor views through
``KVTransferTensors``. LMCache MP can attach several physical kernel groups to
the same engine block-id space, so this connector registers those opaque views
without knowing which model or attention implementation produced them.

Transfers are submitted and polled on the forward thread. Under
OFFLOAD_MP_TRANSFER_THREAD the transfer thread submits them; a frontend whose
forward thread cannot afford the polls either (the vLLM plugin) has it poll
too (`run_transfers_off_step`).
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
    _published_tp_replication_factor,
    _tp_replication_factor,
    _validate_mp_config,
)
from atom.kv_transfer.offload.mp.native_state_layout import (
    build_native_state_mp_layout,
    require_native_state_server,
    stores_state_images,
)
from atom.kv_transfer.offload.mp.page_views import _build_cache_views
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
# longer. Without it, the plugin's c16 TTFT p50 rose from 302 to 324 ms.
_LAST_LOOK_WAIT_S = 0.001


def _req_id_of(completion: Any) -> str:
    """The request id inside a completion id, which may be bare or structured."""
    return str(getattr(completion, "req_id", completion))


class LMCacheMPConnector(KVConnectorBase):
    """Worker-side LMCache MP connector for backend-published PAGE views."""

    is_producer = False

    def __init__(self, config: Any) -> None:
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
        # Set only when the backend publishes PAGE-backed checkpoint images
        # (`_register_native_state_layout`); every transfer then carries one.
        self._native_layout: Any = None
        self._adapter: Any = None
        self._rank: int | None = None
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
        self._after_forward_saves: list[LMCacheReqMeta] = []
        self._after_forward_ran = False
        self._late_flush_logged = False
        # Submissions queued on the transfer thread, by request: `fence` waits
        # for them too, since they reach the server after the step that queued
        # them.
        self._queued_requests: dict[str, int] = {}
        # What a round on a polling transfer thread settled since the last
        # `get_finished`.
        self._polled_done_save: set[SaveCompletionId] = set()
        self._polled_completions: set[ConnectorCompletion] = set()
        self._polled_done_load: set[LoadCompletionId] = set()
        self._polled_failed_load: set[LoadCompletionId] = set()

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
        self._rank = rank
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
        self._rank = rank
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
            # Teardown continues; the server's reaper is the backstop.
            logger.warning("LMCache MP worker shutdown failed", exc_info=True)

    def start_load_kv(self, metadata: Any) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
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

    def run_transfers_off_step(self) -> None:
        """Submit and poll on the transfer thread from here on.

        For a frontend whose forward thread cannot afford either: the vLLM
        plugin, where a submit cost 18.7 ms and a terminal poll 13.8 ms under
        its launch pressure. Called once the caches are registered.
        """
        if self._transfer is not None or self._rank is None:
            raise RuntimeError(
                "LMCache MP: register the caches first, and start the transfer "
                "thread once"
            )
        if self._adapter_records_its_own_events():
            # Inline, transfers would be safe but nothing could fence them.
            raise RuntimeError(
                f"LMCache MP adapter {type(self._adapter).__name__} records "
                "producer events at submission, so its transfers cannot leave "
                "the forward thread, and this frontend's preemption fence needs "
                "them to"
            )
        self._start_transfer_thread(self._rank, poll=self._poll_round)

    def _start_transfer_thread(
        self, rank: int, *, poll: Callable[[], None] | None = None
    ) -> None:
        if poll is None and not envs.OFFLOAD_MP_TRANSFER_THREAD:
            return
        if self._adapter_records_its_own_events():
            logger.warning(
                "LMCache MP adapter %s records producer events at submission; "
                "transfers stay on the forward thread",
                type(self._adapter).__name__,
            )
            return
        self._fences = _ProducerFences()
        self._transfer = _TransferThread(f"mp-transfer-{rank}", poll=poll)
        logger.info(
            "LMCache MP transfer thread started rank=%d polls=%s",
            rank,
            poll is not None,
        )

    def _adapter_records_its_own_events(self) -> bool:
        """It would record its pooled event on the submitting thread's stream,
        not behind the forward."""
        return (
            getattr(self._adapter, "_producer_events", None) is not None
            and not self._adapter_takes_recorded_events()
        )

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

    def _dispatch(
        self,
        submit: Any,
        req: LMCacheReqMeta,
        event: Any,
        ready: Callable[[], bool] | None = None,
    ) -> None:
        if self._transfer is None:
            submit(req, event)
            return
        owner = str(req.req_id)
        with self._lock:
            self._queued_requests[owner] = self._queued_requests.get(owner, 0) + 1
        self._transfer.submit(self._submit_queued, submit, req, event, ready=ready)

    def _submit_queued(self, submit: Any, req: LMCacheReqMeta, event: Any) -> None:
        """One queued submission, on the transfer thread."""
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
        event must not be destroyed on that thread) and surfaces a
        submission that raised.
        """
        transfer = self._transfer
        if transfer is None:
            return
        transfer.take_spent()
        transfer.take_returned()
        transfer.raise_error()

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

    def _submit_load(self, req: LMCacheReqMeta, event: Any) -> None:
        from lmcache.integration.atom import AtomMPTransferSpec

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
            op = AtomMPTransferSpec(
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

    def _submit_save(self, req: LMCacheReqMeta, event: Any) -> None:
        from lmcache.integration.atom import AtomMPTransferSpec

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
            op = AtomMPTransferSpec(
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

    def _poll_round(self) -> None:
        """Advance every pending transfer once: on the forward thread at each
        report, or on a polling transfer thread (`run_transfers_off_step`)."""
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
                self._retire_future(entry.future)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_load_operations,
                    self._completed_load_operation_order,
                )
                if result is not True:
                    self._polled_failed_load.add(entry.completion)
                else:
                    self._polled_done_load.add(entry.completion)

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
                self._retire_future(entry.future)
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
            self._polled_done_save |= done_save
            self._polled_completions |= completions

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
        Waits on the transfer thread's rounds, up to the transfer deadline, and
        fail-stops past it. The completions stay for `get_finished`, which
        releases their leases.
        """
        wanted = {str(req_id) for req_id in req_ids}
        if not wanted:
            return
        transfer = self._transfer
        if transfer is None:
            raise NotImplementedError(
                "LMCache MP fence: only a polling transfer thread fences "
                "(`run_transfers_off_step`)"
            )

        def outstanding() -> list[str]:
            with self._lock:
                return self._outstanding_locked(wanted)

        transfer.fence(
            outstanding, self._transfer_deadline_s, f"requests {sorted(wanted)}"
        )

    def _retire_future(self, future: Any) -> None:
        """A terminal future; called with `_lock` held.

        On a polling transfer thread, handed back to be dropped on the forward
        thread. Elsewhere this is the forward thread, and the caller drops it.
        """
        transfer = self._transfer
        if transfer is not None and transfer.polls:
            transfer.hand_back(future)

    def get_finished(self) -> KVConnectorOutput:
        if self._adapter is None:
            return KVConnectorOutput()
        self._finished_housekeeping()
        transfer = self._transfer
        if transfer is not None and transfer.polls:
            with self._lock:
                in_flight = self._transfers_in_flight_locked()
                wait_s = _LAST_LOOK_WAIT_S if self._pending_loads else 0.0
            if in_flight:
                # A wake -- a step that queues nothing (an idle engine waiting
                # on a load) raises none of its own -- and a last look at a load.
                transfer.look(wait_s)
        else:
            self._poll_round()
        output = KVConnectorOutput()
        with self._lock:
            # Heartbeat health is a control-plane signal, not proof that GPU
            # work has quiesced, so a save is only reported once its device
            # event is terminal -- that check lives in `_poll_pending_saves`.
            output.finished_saving |= self._polled_done_save
            output.connector_completions |= self._polled_completions
            output.finished_loading |= self._polled_done_load
            output.failed_loading |= self._polled_failed_load
            self._polled_done_save = set()
            self._polled_completions = set()
            self._polled_done_load = set()
            self._polled_failed_load = set()
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
