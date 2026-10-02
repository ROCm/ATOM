# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker-side LMCache MP connector for backend-published PAGE views.

Attention backends publish block-major tensor views through
``KVTransferTensors``. LMCache MP can attach several physical kernel groups to
the same engine block-id space, so this connector registers those opaque views
without knowing which model or attention implementation produced them.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
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
    _transfer_mode,
    _validate_mp_config,
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

logger = logging.getLogger("atom")


# Where the MP connector's per-decode-step cost goes, as percentiles rather
# than means. Off unless ATOM_MP_STEP_PROBE=1; this is the instrument the
# throughput work in the recipe's Measured section is read from.
_STEP_PROBE_ON = __import__("os").environ.get("ATOM_MP_STEP_PROBE") == "1"

# Kill switch for the step-event pool: ATOM_MP_EVENT_POOL=0 exports a freshly
# created interprocess event every step instead of re-recording a pooled one.
# Pooling is safe on the lmcache_driven path (the server waits on its own
# event, imported from the handle) and is bypassed outright on the
# engine_driven path -- see `_share_step_event`.  The switch stays so a
# suspected event-lifetime problem can be ruled in or out without a rebuild.
_EVENT_POOL_ON = __import__("os").environ.get("ATOM_MP_EVENT_POOL", "1") != "0"

# tag -> [calls, total_s, max_s, [samples since the last report]]. A mean alone
# cannot tell a uniformly expensive call from a cheap one with a rare stall, and
# those want opposite fixes, so keep the samples and report percentiles.
_STEP_PROBE: dict[str, list] = {}
_STEP_PROBE_EVERY = 200


def _step_probe(tag: str, elapsed: float) -> None:
    bucket = _STEP_PROBE.setdefault(tag, [0, 0.0, 0.0, []])
    bucket[0] += 1
    bucket[1] += elapsed
    bucket[2] = max(bucket[2], elapsed)
    samples = bucket[3]
    samples.append(elapsed)
    if bucket[0] % _STEP_PROBE_EVERY:
        return
    samples.sort()
    last = len(samples) - 1
    logger.info(
        "MPSTEP %s calls=%s total_s=%.3f mean_us=%.1f "
        "p50_us=%.1f p90_us=%.1f p99_us=%.1f win_max_us=%.1f max_us=%.1f",
        tag,
        bucket[0],
        bucket[1],
        bucket[1] / bucket[0] * 1e6,
        samples[last // 2] * 1e6,
        samples[int(last * 0.90)] * 1e6,
        samples[int(last * 0.99)] * 1e6,
        samples[last] * 1e6,
        bucket[2] * 1e6,
    )
    samples.clear()


# tag -> samples since the last report, for plain numbers rather than durations.
_STEP_GAUGE: dict[str, list] = {}


def _step_gauge(tag: str, value: float) -> None:
    samples = _STEP_GAUGE.setdefault(tag, [])
    samples.append(value)
    if len(samples) % _STEP_PROBE_EVERY:
        return
    samples.sort()
    last = len(samples) - 1
    logger.info(
        "MPGAUGE %s n=%s p50=%.1f p90=%.1f p99=%.1f max=%.1f",
        tag,
        len(samples),
        samples[last // 2],
        samples[int(last * 0.90)],
        samples[int(last * 0.99)],
        samples[last],
    )
    samples.clear()


class _PooledIpcEvent:
    """One interprocess event, shared by every transfer submitted in a step.

    Recording a *fresh* interprocess event costs ~1.3 ms on this platform: the
    handle is allocated on the first record, not at construction. Re-recording
    one that already has a handle costs ~20 us and leaves the handle bytes
    unchanged, so the server keeps reusing the event it already imported.

    An event may only be re-recorded once no consumer still needs it to mean
    *that* step: `refs` counts the transfers that named it, and the worker
    takes it back only when the last of them reports terminal. Re-recording
    always moves the event forward, so a consumer that reads it late waits for
    a superset of the work it needed -- but waiting for unrelated later work is
    a stall, which is exactly what the refcount prevents.
    """

    __slots__ = ("event", "refs")

    def __init__(self, event: Any) -> None:
        self.event = event
        self.refs = 0


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
        self._num_recurrent_groups = 0
        self._adapter: Any = None
        self._is_kv_writer = True
        self._pending_saves: dict[str, _PendingSave] = {}
        # Submitting a transfer and polling it to completion are both HIP IPC
        # calls (`export_event` / `from_ipc_handle`). On an idle GPU they
        # cost microseconds, but under this rank's own kernel-launch
        # pressure they cost milliseconds each -- and under rank collapse
        # only the writer pays, so the whole TP group waits at the next
        # collective. Both release the GIL, so a drain thread turns a hard
        # per-step stall into a brief launch-latency tax instead.
        #
        # This holds for the load leg exactly as it does for the save leg, and
        # a conc16 step probe prices it: on the step thread `submit_retrieve_
        # request` costs 18.7 ms a call and a terminal load poll 13.8 ms, which
        # over 8400 steps is 890 + 656 us of fixed cost on *every* step --
        # against a measured 2.27 ms/step ITL gap to in-process. A pending
        # (non-terminal) poll costs 0.2 us, so what is expensive is the IPC
        # handle import, not waiting for the transfer.
        #
        # Both legs share one queue so that a request's load still reaches the
        # server before its save, as it did when the step thread issued them.
        self._transfer_queue: queue.Queue = queue.Queue()
        self._drain_thread: threading.Thread | None = None
        self._drain_stop = threading.Event()
        # Demand-driven, not timer-driven. A fixed poll interval on this thread
        # is a GIL tax on the step thread of every rank: a 2 ms cadence with
        # loads outstanding measured ~1400 polls/s/rank and cost more
        # end-to-end (ITL p50 23.1 -> 28.8 ms) than the step-thread work it
        # moved off, even though it did move that work off (`start_load_kv`
        # 1112 -> 16 us/step). The step signals instead, from `start_load_kv`
        # -- so the IPC work lands while the step thread is inside the model
        # forward with the GIL released, and `get_finished` later in the same
        # step finds the results already settled.
        self._drain_wake = threading.Event()
        self._drain_rounds = 0
        self._drained_done_save: set[SaveCompletionId] = set()
        self._drained_completions: set[ConnectorCompletion] = set()
        self._drained_done_load: set[LoadCompletionId] = set()
        self._drained_failed_load: set[LoadCompletionId] = set()
        # Deadline expiry is fail-stop, so it has to reach the step that
        # calls `get_finished` rather than just killing the drain thread.
        self._drain_error: BaseException | None = None
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
        self._drain_round_cv = threading.Condition(self._lock)
        self._event_pool: list[_PooledIpcEvent] = []
        # Engine-driven transfers stay inside this process, so their step event
        # needs neither an IPC handle nor the pool: LMCache waits on the very
        # object handed over, from its own commit thread, at an arbitrary later
        # time (phase 1 is a blocking RPC).  Re-recording a pooled event while
        # such a wait is in flight races hipEventRecord against
        # hipStreamWaitEvent on one hipEvent_t, which ROCm reports as
        # `HIP error: invalid argument` out of
        # async_engine_driven._prepare_gather_and_commit.  Measured on K3 TP8
        # conc16: pooled -> 5 failed stores, 5 write-locked orphans, tier 95.6%
        # full, nothing ever stored; unpooled -> 0 failures, 0 orphans, 4
        # objects stored.  lmcache_driven is unaffected because the server waits
        # on a separate event imported from the handle, not on this object.
        self._share_step_event = True

    def register_kv_caches(
        self,
        _kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
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
        # Only engine_driven needs the token axis made explicit; see
        # `_token_major_view`.  lmcache_driven's kernels address pages by this
        # shape and hang if it is split.
        views = _build_cache_views(
            transfer_tensors,
            num_blocks=normalized_num_blocks,
            tokens_per_block=self.block_size,
            expose_token_axis=_transfer_mode(self._config) == "engine_driven",
        )
        block_regions = getattr(transfer_tensors, "block_regions", None) or []
        expected = sum(int(region.unit_bytes) for region in block_regions)
        if expected != views.bytes_per_block:
            raise ValueError(
                "lmcache_mp block geometry mismatch: "
                f"views={views.bytes_per_block} transfer_regions={expected}"
            )

        adapter = _make_worker_adapter(self._config, rank)
        self._share_step_event = _transfer_mode(self._config) != "engine_driven"
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=layer_indices,
                tokens_per_block=self.block_size,
            )
            for layer_indices in views.layer_groups
        ]
        # Recurrent ordinal ``j`` gets engine group ``1 + j`` -- the same
        # numbering `native_state_layout` uses, and for the same reason: an
        # engine group is an address space, and a recurrent snapshot is neither
        # counted in attention blocks nor read in full. ``sw_size_tokens`` one
        # block wide is what makes the server restore the last snapshot only;
        # the earlier positions the transport fills with the null block id are
        # never committed. The server must therefore run with
        # ``--separate-object-groups``: without it every kernel group collapses
        # into one object group (`kv_layer_groups.build_object_groups`), the
        # null-chunk test then sees the attention group's real ids and skips
        # nothing, and the window that restores only the last snapshot is gone.
        groups.extend(
            EngineGroupInfo(
                engine_group_id=1 + ordinal,
                layer_indices=recurrent.tensor_indices,
                tokens_per_block=recurrent.tokens_per_block,
                sw_size_tokens=recurrent.tokens_per_block,
                recurrent_state=True,
            )
            for ordinal, recurrent in enumerate(views.recurrent)
        )
        self._num_recurrent_groups = len(views.recurrent)
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
        logger.info(
            "LMCache MP registered rank=%d tensors=%d groups=%d "
            "recurrent_groups=%d bytes_per_block=%d chunk=%d tp_replication=%d "
            "writer=%s save=%s load=%s",
            rank,
            len(views.tensors),
            len(views.layer_groups),
            len(views.recurrent),
            views.bytes_per_block,
            self.chunk_size,
            requested_replication,
            self._is_kv_writer,
            self._do_save,
            self._do_load,
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
        self._drain_stop.set()
        self._drain_wake.set()
        drain, self._drain_thread = self._drain_thread, None
        if drain is not None:
            # Bounded: the loop only ever waits on a 2 ms queue poll plus one
            # round of futures, and the adapter shutdown below is the backstop.
            drain.join(timeout=30.0)
        adapter, self._adapter = self._adapter, None
        if adapter is None:
            return
        try:
            adapter.shutdown()
        except Exception:
            # Teardown continues; the server's reaper is the backstop.
            logger.warning("LMCache MP worker shutdown failed", exc_info=True)

    def start_load_kv(self, metadata: Any) -> None:
        if _STEP_PROBE_ON:
            _probe_t0 = time.perf_counter()
            try:
                return self._start_load_kv_inner(metadata)
            finally:
                _step_probe("start_load_kv", time.perf_counter() - _probe_t0)
        return self._start_load_kv_inner(metadata)

    def _start_load_kv_inner(self, metadata: Any) -> None:
        # Raised before anything else in the step: the drain thread's useful
        # window is the model forward that follows, where the step thread is in
        # C with the GIL released. By `get_finished` the results are settled.
        self._drain_wake.set()
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        if self._adapter is None or self.chunk_size is None:
            raise RuntimeError("lmcache_mp KV caches are not registered")

        requests = [
            req
            for req in metadata.requests
            if (req.load_spec is not None and self._do_load)
            or (req.save_spec is not None and self._do_save)
        ]
        if not requests:
            return
        _t0 = time.perf_counter() if _STEP_PROBE_ON else 0.0
        pooled = self._acquire_step_event()
        if _STEP_PROBE_ON:
            _step_probe("acquire_event", time.perf_counter() - _t0)
        try:
            for req in requests:
                if req.load_spec is not None and self._do_load:
                    self._submit_load(req, pooled)
                if req.save_spec is not None and self._do_save:
                    self._submit_save(req, pooled)
        finally:
            with self._lock:
                # Drop the step's own hold. Once the submits that named it
                # have reported too, the event goes back to the pool; if none
                # did -- an invalid descriptor, a non-writer rank, an empty
                # save range -- this returns it here rather than leaking it.
                self._release_step_event(pooled)

    def _acquire_step_event(self) -> _PooledIpcEvent:
        """Return this step's recorded interprocess event; see `_PooledIpcEvent`.

        Comes back holding one reference for the step itself, so a submit that
        reports terminal while later submits are still being issued cannot put
        the event back in the pool mid-step.
        """

        share = self._share_step_event
        with self._lock:
            pooled = self._event_pool.pop() if (share and self._event_pool) else None
            if not _EVENT_POOL_ON:
                pooled = None
            if pooled is None:
                pooled = _PooledIpcEvent(torch.cuda.Event(interprocess=share))
            pooled.refs = 1
        pooled.event.record(torch.cuda.current_stream())
        return pooled

    def _release_step_event(self, pooled: _PooledIpcEvent | None) -> None:
        """Give one transfer's hold on a step event back. Caller holds the lock."""

        if pooled is None:
            return
        pooled.refs -= 1
        if pooled.refs <= 0 and _EVENT_POOL_ON and self._share_step_event:
            self._event_pool.append(pooled)

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

    def _recurrent_block_ids(
        self, req: LMCacheReqMeta, start: int, end: int
    ) -> list[list[int]]:
        """One block-id list per recurrent group, null everywhere but the end.

        A recurrent snapshot exists only at a chunk boundary, so an operation
        spanning ``n`` chunks carries ``n - 1`` null ids and the boundary block
        last. The null id is ``0``, vLLM's null block, which is what LMCache's
        `all_null_chunk_masks` tests for -- it asks whether any id in the chunk
        is truthy, so a sentinel of ``-1`` would read as a real block and the
        empty chunks would be committed as content-hashed garbage. It is not
        configurable: no released LMCache exposes a null-block-id knob.
        """
        if not self._num_recurrent_groups:
            if req.recurrent_state is not None:
                raise ValueError(
                    f"LMCache MP request {req.req_id} carries recurrent state, "
                    "but no recurrent group was registered"
                )
            return []
        state = req.recurrent_state
        if state is None:
            raise ValueError(
                f"LMCache MP request {req.req_id} has no recurrent state for "
                f"[{start}, {end}), but {self._num_recurrent_groups} recurrent "
                "group(s) are registered; a PAGE-only object would restore a "
                "prefix whose recurrent state is someone else's"
            )
        if len(state.block_ids) != self._num_recurrent_groups:
            raise ValueError(
                f"LMCache MP request {req.req_id} named "
                f"{len(state.block_ids)} recurrent blocks for "
                f"{self._num_recurrent_groups} group(s)"
            )
        if int(state.boundary_tokens) != end:
            raise ValueError(
                f"LMCache MP request {req.req_id} snapshotted recurrent state "
                f"at token {state.boundary_tokens}, but the transfer ends at "
                f"{end}; the state would not continue the KV it ships with"
            )
        count = (end - start) // int(self.chunk_size)
        return [[0] * (count - 1) + [int(block)] for block in state.block_ids]

    def _submit_load(self, req: LMCacheReqMeta, pooled: _PooledIpcEvent) -> None:
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
            block_ids = self._block_slice(req, start, end)
            _t0 = time.perf_counter() if _STEP_PROBE_ON else 0.0
            op = AtomMPTransferSpec(
                token_ids=list(req.token_ids),
                block_ids=[block_ids] + self._recurrent_block_ids(req, start, end),
                start=start,
                end=end,
            )
            if _STEP_PROBE_ON:
                _step_probe("load_descriptor", time.perf_counter() - _t0)
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
        with self._lock:
            # Held until this load reports terminal: LMCache keeps a reference
            # to the event for as long as its future is live. Taken here, on
            # the step thread, because the step drops its own hold as soon as
            # it returns -- the drain thread must not be the first to claim
            # the event, or the pool could hand it out again mid-transfer.
            pooled.refs += 1
        # `_submitting_loads` still holds this slot: the drain thread hands it
        # over to `_pending_loads` once the submit lands, so a duplicate
        # operation id is still refused while the submit is in flight.
        self._transfer_queue.put(
            (
                "load",
                operation_id,
                completion,
                request_id,
                op,
                pooled,
                0,
                0,
                time.monotonic(),
            )
        )
        self._ensure_drain_thread()
        self._drain_wake.set()

    def _submit_save(self, req: LMCacheReqMeta, pooled: _PooledIpcEvent) -> None:
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
            _t0 = time.perf_counter() if _STEP_PROBE_ON else 0.0
            block_ids = self._block_slice(req, start, end)
            recurrent_block_ids = self._recurrent_block_ids(req, start, end)
            op = AtomMPTransferSpec(
                token_ids=list(req.token_ids),
                block_ids=[block_ids] + recurrent_block_ids,
                start=start,
                end=end,
            )
            if _STEP_PROBE_ON:
                _step_probe("save_descriptor", time.perf_counter() - _t0)
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
            # Nothing was queued, so start the drain thread explicitly: it is
            # the only place this terminal entry is ever reported from.
            self._ensure_drain_thread()
            self._drain_wake.set()
            return
        with self._lock:
            # See the load twin: released when this save reports terminal. Taken
            # here, on the step thread, because the step releases its own hold as
            # soon as it returns -- the drain thread must not be the first to
            # claim the event.
            pooled.refs += 1
        # `_submitting_saves` still holds this slot: the drain thread hands it
        # over to `_pending_saves` once the submit lands, so a duplicate
        # operation id is still refused while the submit is in flight.
        self._transfer_queue.put(
            (
                "save",
                operation_id,
                completion,
                request_id,
                op,
                pooled,
                start,
                end,
                time.monotonic(),
            )
        )
        self._ensure_drain_thread()
        self._drain_wake.set()

    def _ensure_drain_thread(self) -> None:
        if self._drain_thread is not None:
            return
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            name="atom-lmcache-mp-save",
            daemon=True,
        )
        self._drain_thread.start()

    def _drain_loop(self) -> None:
        while not self._drain_stop.is_set():
            try:
                # The timeout is a backstop for teardown and for a rank that
                # has stopped stepping, not the working cadence.
                self._drain_wake.wait(timeout=0.05)
                # Cleared before the work, so a wake raised while this round is
                # running is kept for the next round rather than swallowed.
                self._drain_wake.clear()
                while True:
                    try:
                        item = self._transfer_queue.get_nowait()
                    except queue.Empty:
                        break
                    kind, rest = item[0], item[1:]
                    if kind == "load":
                        self._perform_load_submit(*rest[:5], rest[7])
                    else:
                        self._perform_save_submit(*rest)
                self._poll_pending_loads()
                self._poll_pending_saves()
                with self._drain_round_cv:
                    self._drain_rounds += 1
                    self._drain_round_cv.notify_all()
            except BaseException as exc:
                # Deadline expiry and any unexpected failure are fail-stop, and
                # this thread is not the one that can stop the engine.
                if self._drain_error is None:
                    self._drain_error = exc
                logger.exception("LMCache MP transfer drain failed")
                return

    def _settle_transfers(self, timeout: float = 5.0) -> None:
        """Block until every queued transfer has been submitted and polled once.

        Both legs settle on `_drain_loop`, so a caller that needs the result of
        a specific submit -- a test, or a teardown that wants the queue drained
        before the adapter goes away -- needs a barrier rather than a sleep.
        """
        drain = self._drain_thread
        if drain is None:
            return
        # `perf_counter`, not `monotonic`: the transfer deadline is expressed in
        # `monotonic` and tests drive it from a fake clock, which must not also
        # freeze this barrier's own timeout.
        deadline = time.perf_counter() + timeout
        while True:
            if self._drain_error is not None:
                raise self._drain_error
            self._drain_wake.set()
            with self._drain_round_cv:
                # A round drains the whole queue, but a submit can be in flight
                # when this one starts, so re-check the queue after waiting
                # rather than assuming a fixed number of rounds.
                target = self._drain_rounds + 2
                while self._drain_rounds < target:
                    if self._drain_error is not None or not drain.is_alive():
                        break
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError("LMCache MP save drain did not settle")
                    self._drain_round_cv.wait(min(remaining, 0.05))
            if self._drain_error is not None:
                raise self._drain_error
            if not drain.is_alive() or self._transfer_queue.empty():
                return

    def _perform_load_submit(
        self,
        operation_id: str,
        completion: LoadCompletionId,
        request_id: str,
        op: Any,
        pooled: _PooledIpcEvent,
        queued_at: float,
    ) -> None:
        try:
            _t0 = time.perf_counter() if _STEP_PROBE_ON else 0.0
            transfer = self._adapter.submit_retrieve_request(
                request_id,
                op,
                pooled.event,
            )
            if _STEP_PROBE_ON:
                _step_probe("load_submit", time.perf_counter() - _t0)
        except Exception:
            # The server may have taken the request before the connection
            # raised, and may still write the destination blocks, so the
            # destination stays leased until a terminal report -- which the
            # transfer deadline turns into a fail-stop if it never comes.
            logger.exception("LMCache MP load submission unprovable for %s", request_id)
            transfer = _UnprovableSubmission()
        with self._lock:
            self._submitting_loads.discard(operation_id)
            self._pending_loads[operation_id] = _PendingLoad(
                completion=completion,
                future=transfer,
                started_at=queued_at,
                event=pooled,
            )

    def _poll_pending_loads(self) -> None:
        """Advance every pending load, off the step thread.

        The save twin's reasoning applies unchanged: a terminal poll imports
        the peer's IPC event and costs milliseconds, so it runs without `_lock`
        held, and removal happens here alone -- the step thread only ever adds
        -- so an entry the snapshot missed is picked up on the next round.
        """
        with self._lock:
            pending = list(self._pending_loads.items())
        if not pending:
            return
        terminal_ops: list[tuple[str, _PendingLoad, Any]] = []
        for operation_id, entry in pending:
            _t0 = time.perf_counter() if _STEP_PROBE_ON else 0.0
            terminal, result = _terminal_future_result(entry.future)
            if _STEP_PROBE_ON:
                _step_probe(
                    "terminal_poll_load" if terminal else "pending_poll_load",
                    time.perf_counter() - _t0,
                )
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
                if _STEP_PROBE_ON:
                    # Enqueue to terminal: what the engine actually waits
                    # for, and the only leg never yet priced against
                    # in-process LMCache's logged per-op cost.
                    _step_probe("load_roundtrip", time.monotonic() - entry.started_at)
                self._pending_loads.pop(operation_id, None)
                self._release_step_event(entry.event)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_load_operations,
                    self._completed_load_operation_order,
                )
                if result is not True:
                    self._drained_failed_load.add(entry.completion)
                else:
                    self._drained_done_load.add(entry.completion)

    def _perform_save_submit(
        self,
        operation_id: str,
        completion: SaveCompletionId,
        request_id: str,
        op: Any,
        pooled: _PooledIpcEvent,
        start: int,
        end: int,
        queued_at: float,
    ) -> None:
        try:
            submit = getattr(
                self._adapter,
                "submit_store_request_with_chunk_events",
                self._adapter.submit_store_request,
            )
            _t0 = time.perf_counter() if _STEP_PROBE_ON else 0.0
            transfer = submit(
                request_id,
                op,
                pooled.event,
            )
            if _STEP_PROBE_ON:
                _step_probe("save_submit", time.perf_counter() - _t0)
        except Exception:
            # The server may have received the request before the connection
            # raised: keep the source leased until a terminal report, which
            # the transfer deadline turns into a fail-stop if it never comes.
            logger.exception("LMCache MP save submission unprovable for %s", request_id)
            transfer = _UnprovableSubmission()
        with self._lock:
            self._submitting_saves.discard(operation_id)
            self._pending_saves[operation_id] = _PendingSave(
                completion=completion,
                future=transfer,
                started_at=queued_at,
                start=start,
                end=end,
                event=pooled,
            )

    def _poll_pending_saves(self) -> None:
        """Advance every pending save, off the step thread.

        The polls themselves run without `_lock` held: a terminal poll imports
        the peer's IPC event and costs milliseconds, and holding the lock across
        it would just move the step's stall from the poll to the lock.  The step
        thread only ever *adds* to `_pending_saves` (the pre-submit failure
        path); removal happens here alone, so an entry the snapshot missed is
        simply picked up on the next round.
        """
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
            _t0 = time.perf_counter() if _STEP_PROBE_ON else 0.0
            terminal, result = _terminal_future_result(entry.future)
            if _STEP_PROBE_ON:
                _step_probe(
                    "terminal_poll_save" if terminal else "pending_poll_save",
                    time.perf_counter() - _t0,
                )
            if not terminal:
                _enforce_transfer_deadline(
                    operation_id, entry.started_at, self._transfer_deadline_s
                )
                continue
            terminal_ops.append((operation_id, entry, result))
        with self._lock:
            for operation_id, entry, result in terminal_ops:
                if _STEP_PROBE_ON:
                    # Enqueue to terminal: what the engine actually waits
                    # for, and the only leg never yet priced against
                    # in-process LMCache's logged per-op cost.
                    _step_probe("save_roundtrip", time.monotonic() - entry.started_at)
                self._pending_saves.pop(operation_id, None)
                self._release_step_event(entry.event)
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

    def get_finished(self) -> KVConnectorOutput:
        if self._adapter is None:
            return KVConnectorOutput()
        if _STEP_PROBE_ON:
            _probe_t0 = time.perf_counter()
            try:
                return self._get_finished_inner()
            finally:
                _step_probe("get_finished", time.perf_counter() - _probe_t0)
        return self._get_finished_inner()

    def _get_finished_inner(self) -> KVConnectorOutput:
        done_load: set[LoadCompletionId] = set()
        failed_load: set[LoadCompletionId] = set()
        done_save: set[SaveCompletionId] = set()
        connector_completions: set[ConnectorCompletion] = set()
        if self._drain_error is not None:
            # The save drain thread cannot stop the engine; the step can.
            raise self._drain_error
        with self._lock:
            # The save side is advanced by `_drain_loop`; take whatever it
            # has settled since the last step. Heartbeat health is a control-plane
            # signal, not proof that GPU work has quiesced, so a save is only
            # reported once its device event is terminal -- that check lives in
            # `_poll_pending_saves`.
            done_save |= self._drained_done_save
            connector_completions |= self._drained_completions
            self._drained_done_save = set()
            self._drained_completions = set()
            # Loads are advanced by the same drain thread, for the same reason
            # (see the queue's comment in `__init__`): the terminal poll's IPC
            # handle import is what costs milliseconds, and it has no business
            # on a thread the whole TP group waits for.
            done_load |= self._drained_done_load
            failed_load |= self._drained_failed_load
            self._drained_done_load = set()
            self._drained_failed_load = set()
            done_save.update(self._immediate_saves)
            for completion, token_range in self._immediate_saves.items():
                if isinstance(completion, SaveOperationId):
                    if token_range is not None:
                        connector_completions |= _source_safe_completions(
                            completion,
                            _chunk_ranges(*token_range, int(self.chunk_size)),
                        )
                    connector_completions.add(
                        ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, completion, True)
                    )
            failed_load.update(self._immediate_load_failures)
            self._immediate_saves.clear()
            self._immediate_load_failures.clear()
        return KVConnectorOutput(
            finished_loading=done_load,
            failed_loading=failed_load,
            finished_saving=done_save,
            connector_completions=connector_completions,
        )


__all__ = ["LMCacheMPConnector"]
