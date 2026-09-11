# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM standalone LMCache CPU/NVMe KV-offload connector.

Design:

* **Use LMCache engine orchestration** — worker-side save/load calls
  ``CacheEngine.store()`` / ``CacheEngine.retrieve()`` so LMCache owns chunking,
  key generation, lookup pins, and storage-manager put/get.
* **ATOM-owned raw-byte GPU connector** — LMCache's stock vLLM GPU connectors
  cannot represent ATOM's x-packed AITER KV layout
  (``K=(nb,H,D//x,bs,x)``). We pass an ATOM ``GPUConnectorInterface``
  implementation that moves opaque per-block bytes with
  :class:`DenseKVByteCodec`.
* **Daemon-after-forward copies** — ``start_load_kv`` only ``submit``s to a single
  serial copy daemon (ThreadPoolExecutor max_workers=1) and returns immediately, so
  the worker RPC thread is free for ``forward``; completions are polled in
  ``get_finished`` (called post-forward by ``async_proc_aggregation``). This is the
  fix for 005's "load blocks/starves prefill" (corr(TTFT, prefill-conc)=0.773).
* **Cross-process hit lookup** — scheduler (EngineCore process) queries worker hits
  via LMCache's ZMQ ``LookupClient``/``LookupServer`` (no homegrown mirror).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future
from contextlib import nullcontext

import torch

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadOperationId,
    SaveCompletionId,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload._offload_common import (
    OffloadSchedulerMixin,
    OffloadWorkerMixin,
    build_offload_engine,
    max_pending_saves,
    pp_aware_rank_and_world,
    validated_kv_role,
)
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.dense.save_admission import build_save_budget
from atom.kv_transfer.offload.dense.save_executor import (
    CancellableSaveExecutor,
    SaveQueueFull,
    SeenSaveGenerations,
    save_admission_enabled,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)

logger = logging.getLogger("atom")

DENSE_PAGE_SOURCE_SAFE_CHANNEL = "dense.page.source_safe"
DENSE_PAGE_STORE_CHANNEL = "dense.page.store"
DENSE_PAGE_RETIRED_CHANNEL = "dense.page.retired"


# =====================================================================
# Worker side
# =====================================================================
class DenseOffloadConnector(OffloadWorkerMixin, KVConnectorBase):
    # Offload is a *consumer* from the scheduler's POV (it loads KV back). Saves
    # are fire-and-forget on the worker and must NOT be reported as
    # finished_sending (the scheduler frees blocks on finished_sending — a P/D
    # producer semantic that would wrongly deallocate live offload blocks).
    # Executor plumbing + get_finished come from OffloadWorkerMixin.

    # Whether a per-request recurrent-state tensor in the registered kv_caches is
    # tolerated. The plain dense path has no rule keeping a restored KV prefix
    # aligned with linear-attention state, so it must reject such a model
    # (GDN: Qwen3-Next, Qwen3.5) and fail fast. A hybrid connector
    # that owns a state tier (kimi_k3) overrides this to True.
    _permit_per_request_state = False
    _supports_early_block_release = True

    def __init__(self, config) -> None:
        self._config = config
        self._init_worker_common(config)  # kv_role, executors, lock, tallies
        self.block_size = int(config.kv_cache_block_size)
        self.virtual_block_size = self.block_size * int(
            getattr(config, "decode_context_parallel_size", 1) or 1
        )
        self.chunk_size: int | None = None
        self._engine = None
        self._codec: DenseKVByteCodec | None = None
        self._lookup_server = None
        self._early_release = bool(self._supports_early_block_release)
        if self._early_release:
            n_save = int(os.environ.get("OFFLOAD_COPY_WORKERS", "1"))
            kvc = getattr(config, "kv_transfer_config", {}) or {}
            self._save_capacity = (
                max_pending_saves(kvc, n_save) if save_admission_enabled() else None
            )
            # The common TPE has not received work. Only PAGE early-release
            # workers use the physically cancellable queue; K3 keeps its path.
            self._save_executor.shutdown(wait=True)
            self._save_executor = CancellableSaveExecutor(
                max_workers=n_save,
                capacity=self._save_capacity,
                thread_name_prefix="offload-save",
            )
            self._save_dispatch_lock = threading.RLock()
            self._save_generations = SeenSaveGenerations()
            self._save_futures: dict[SaveOperationId, Future] = {}

    def close(self) -> None:
        super().close()
        gpu_connector = getattr(getattr(self, "_engine", None), "gpu_connector", None)
        close = getattr(gpu_connector, "close", None)
        if callable(close):
            close()

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        from aiter.dist.parallel_state import get_tp_group

        tp = get_tp_group()
        rank, world = pp_aware_rank_and_world(self._config, tp)
        self._rank = rank

        # Scheduler blocks, threaded from the model runner. MLA stores its KV
        # token-major, so the codec cannot infer the count from shape[0].
        self._codec = DenseKVByteCodec(
            kv_caches,
            num_blocks=num_blocks,
            permit_per_request_state=self._permit_per_request_state,
        )
        # Shared opaque-uint8 engine build; the chunked GPU connector needs
        # cfg.chunk_size, so it's built inside the factory once cfg exists.
        self._engine, cfg, meta = build_offload_engine(
            self._config,
            engine_id=f"{offcfg.lmcache_engine_id(self._config)}-{rank}",
            block_size=self.virtual_block_size,
            bytes_per_block=self._codec.bytes_per_block,
            gpu_connector_factory=lambda cfg, meta: BlockGPUConnector(
                self._codec,
                self.block_size,
                chunk_size=int(cfg.chunk_size),
                virtual_block_size=self.virtual_block_size,
                source_safe_callback=(
                    self._source_group_safe
                    if getattr(self, "_early_release", False)
                    else None
                ),
            ),
            world=world,
            rank=rank,
        )
        self.chunk_size = int(cfg.chunk_size)

        # ZMQ lookup server so the scheduler process can query our hit counts.
        try:
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            self._lookup_server = LookupClientFactory.create_lookup_server(
                self._engine, meta
            )
        except Exception as e:  # noqa: BLE001  # optional save-only dependency
            logger.warning("LMCache offload: lookup server not started: %s", e)

        gpu_connector = self._engine.gpu_connector
        logger.info(
            "LMCache offload worker rank=%d: bytes_per_block=%d chunk=%d "
            "gpu_staging_chunk_bytes=%d gpu_staging_buffer_chunks=%d "
            "gpu_staging_buffer_bytes=%d release_gpu_staging=%s "
            "save=%s load=%s",
            rank,
            self._codec.bytes_per_block,
            self.chunk_size,
            gpu_connector.gpu_staging_chunk_bytes,
            gpu_connector.gpu_staging_buffer_chunks,
            gpu_connector.gpu_staging_buffer_bytes,
            gpu_connector.release_gpu_staging_after_transfer,
            self._do_save,
            self._do_load,
        )

    # -- per-step (RPC thread): only enqueue, never copy ------------------
    def start_load_kv(self, metadata) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        if self._early_release:
            for operation in getattr(metadata, "cancel_save_operations", ()):
                self._cancel_save_operation(operation)
        load_requests = [
            req
            for req in metadata.requests
            if req.load_spec is not None and self._do_load
        ]
        loading_lookup_ids = {str(req.req_id) for req in load_requests}
        for lookup_id in metadata.lookup_requests_in_step:
            if str(lookup_id) not in loading_lookup_ids:
                self._lookup_unpin(lookup_id)
        for req in metadata.requests:
            if req.load_spec is not None and self._do_load:
                self._load_executor.submit(self._guard, "load", self._do_load_req, req)
            if req.save_spec is not None and self._do_save:
                if self._early_release and isinstance(
                    req.save_operation, SaveOperationId
                ):
                    self._submit_save_operation(req)
                else:
                    self._save_executor.submit(
                        self._guard, "save", self._do_save_req, req
                    )

    def _trace_save_event(self, event, operation, *, queue_wait_ms=0.0, reason=""):
        if not self._profile_enabled():
            return
        queued, running = self._save_executor.counts()
        logger.info(
            "[OFFLOAD-SAVE-QUEUE] rank=%s req=%s generation=%d event=%s "
            "queued=%d running=%d queue_wait_ms=%.2f reason=%s",
            getattr(self, "_rank", "?"),
            operation.req_id,
            operation.generation,
            event,
            queued,
            running,
            queue_wait_ms,
            reason or "-",
        )

    def _submit_save_operation(self, req: LMCacheReqMeta) -> None:
        operation = req.save_operation
        with self._save_dispatch_lock:
            if not self._save_generations.remember(operation.generation):
                return
            enqueued_at = time.perf_counter()
            try:
                future = self._save_executor.submit(
                    self._run_save_operation, req, enqueued_at
                )
            except Exception as exc:  # noqa: BLE001 - transactional submit boundary
                # Our executor rejects before enqueueing. Unlike a TPE whose
                # thread creation can fail after enqueue, no task can survive
                # this failed submission and subsequently read source blocks.
                self._publish_store_outcome(operation, False)
                self._record_save_retired(operation)
                reason = (
                    "worker_queue_full"
                    if isinstance(exc, SaveQueueFull)
                    else "submit_failed"
                )
                self._trace_save_event("rejected", operation, reason=reason)
                return
            self._save_futures[operation] = future
            future.add_done_callback(
                lambda done, op=operation: self._save_operation_done(op, done)
            )
            self._trace_save_event("enqueued", operation)

    def _cancel_save_operation(self, operation: SaveOperationId) -> None:
        with self._save_dispatch_lock:
            future = self._save_futures.get(operation)
            if future is not None:
                cancelled = future.cancel()
                self._trace_save_event(
                    "cancelled" if cancelled else "cancel_running", operation
                )
                return
            if self._save_generations.remember(operation.generation):
                # Cancellation may precede dispatch. Keep a generation fence
                # so a delayed metadata batch cannot start this operation.
                self._publish_store_outcome(operation, False)
                self._record_save_retired(operation)
                self._trace_save_event("cancelled_unseen", operation)
            # A known completed operation has already reported. A known but
            # unfenced failure must not become safe merely because it is retried.

    def _run_save_operation(self, req: LMCacheReqMeta, enqueued_at: float) -> bool:
        # Publish the enqueue record and install Future tracking before the
        # worker can begin, including for a store that returns immediately.
        with self._save_dispatch_lock:
            self._trace_save_event(
                "started",
                req.save_operation,
                queue_wait_ms=(time.perf_counter() - enqueued_at) * 1000,
            )
        try:
            self._do_save_req(req)
        except Exception:
            logger.exception("offload save failed for %s", req.save_operation)
            self._record_store_terminal(req, False)
            # This runs after the save call unwound, on its own executor
            # thread. It cannot enqueue more GPU reads after these fences.
            return self._fence_save_source()
        return True

    def _fence_save_source(self) -> bool:
        gpu_connector = getattr(getattr(self, "_engine", None), "gpu_connector", None)
        state_factory = getattr(gpu_connector, "_thread_state", None)
        if not callable(state_factory):
            logger.error("offload save cannot prove source safety: no staging state")
            return False
        try:
            state = state_factory()
            streams = (
                ("pack_stream", state.pack_stream),
                ("copy_stream", state.copy_stream),
            )
        except Exception:
            logger.exception("offload save could not inspect staging streams")
            return False
        fenced = True
        for name, stream in streams:
            if stream is None:
                continue
            try:
                stream.synchronize()
            except Exception:
                fenced = False
                logger.exception("offload save source fence failed: %s", name)
        return fenced

    def _save_operation_done(self, operation: SaveOperationId, future: Future) -> None:
        with self._save_dispatch_lock:
            self._save_futures.pop(operation, None)
        if future.cancelled():
            self._publish_store_outcome(operation, False)
            safe = True
        else:
            try:
                safe = future.result()
            except BaseException:
                # An unexpected executor-level failure has no source fence.
                self._publish_store_outcome(operation, False)
                logger.exception("offload save executor failed for %s", operation)
                safe = False
        if safe:
            self._record_save_retired(operation)
            self._trace_save_event("retired", operation)
        else:
            self._trace_save_event("unfenced", operation)

    # -- copy daemon thread ----------------------------------------------
    def _source_group_safe(self, identity: SaveSourceGroupId) -> None:
        """Publish one locally source-safe PAGE staging group for TP quorum."""

        with self._lock:
            self._connector_completions.add(
                ConnectorCompletion(
                    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
                    identity,
                    True,
                )
            )

    def _record_store_terminal(self, req: LMCacheReqMeta, succeeded: bool) -> None:
        operation = req.save_operation
        if getattr(self, "_early_release", False) and isinstance(
            operation, SaveOperationId
        ):
            self._publish_store_outcome(operation, succeeded)
            return
        with self._lock:
            self._done_save.add(self._save_completion_id(req))

    def _record_save_failure(self, req) -> None:
        self._record_store_terminal(req, False)

    def _publish_store_outcome(self, operation: SaveOperationId, succeeded: bool):
        with self._lock:
            self._connector_completions.add(
                ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, operation, succeeded)
            )

    def _record_save_retired(self, operation: SaveOperationId) -> None:
        with self._lock:
            self._connector_completions.add(
                ConnectorCompletion(DENSE_PAGE_RETIRED_CHANNEL, operation, True)
            )
            # Legacy deferred-free/MultiConnector pairing is a safety signal.
            # Never publish it from a failed, unfenced store result alone.
            self._done_save.add(operation)

    def _do_load_req(self, req: LMCacheReqMeta) -> None:
        ls = req.load_spec
        assert ls is not None
        hbm = int(ls.hbm_cached_tokens)
        lmc = int(ls.lmcache_cached_tokens)
        toks = req.token_ids[:lmc]
        t_total0 = time.perf_counter()
        if lmc <= hbm:
            self._lookup_unpin(req.req_id)
            with self._lock:
                self._done_load.add(self._load_completion_id(req))
            return
        chunk_size = int(self.chunk_size or 256)
        if hbm % chunk_size != 0:
            logger.warning(
                "LMCache offload: HBM prefix is not chunk-aligned req=%s "
                "hbm=%d chunk=%d; re-prefill",
                req.req_id,
                hbm,
                chunk_size,
            )
            self._lookup_unpin(req.req_id)
            with self._lock:
                self._failed_load.add(self._load_completion_id(req))
            return

        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:hbm] = False

        t_retrieve0 = time.perf_counter()
        self._reset_gpu_connector_transfer_stats()
        ret_mask = self._engine.retrieve(
            torch.tensor(toks),
            mask=mask,
            block_ids=req.block_ids,
            req_id=str(req.req_id),
        )
        retrieve_ms = (time.perf_counter() - t_retrieve0) * 1000
        transfer_stats = self._last_gpu_connector_transfer_stats()
        self._lookup_unpin(req.req_id)
        loaded = bool(ret_mask[hbm:lmc].all().item())
        with self._lock:
            if loaded:
                self._done_load.add(self._load_completion_id(req))
            else:
                self._failed_load.add(self._load_completion_id(req))
        total_ms = (time.perf_counter() - t_total0) * 1000
        if self._profile_enabled():
            logger.info(
                "[OFFLOAD-LOAD-PROF] rank=%s req=%s hbm=%d lmc=%d "
                "retrieved=%d status=%s chunks=%d groups=%d "
                "max_chunk_bytes=%d max_group_bytes=%d "
                "gpu_staging_chunk_bytes=%d gpu_staging_buffer_chunks=%d "
                "gpu_staging_buffer_bytes=%d total_bytes=%d "
                "pack_ms=%.2f copy_ms=%.2f sync_ms=%.2f "
                "transfer_ms=%.2f effective_gbps=%.2f "
                "retrieve_ms=%.2f total_ms=%.2f",
                getattr(self, "_rank", "?"),
                req.req_id,
                hbm,
                lmc,
                int(ret_mask.sum().item()),
                "ok" if loaded else "miss",
                int(transfer_stats.get("chunks", 0)),
                int(transfer_stats.get("groups", 0)),
                int(transfer_stats.get("max_chunk_bytes", 0)),
                int(transfer_stats.get("max_group_bytes", 0)),
                int(transfer_stats.get("gpu_staging_chunk_bytes", 0)),
                int(transfer_stats.get("gpu_staging_buffer_chunks", 0)),
                int(transfer_stats.get("gpu_staging_buffer_bytes", 0)),
                int(transfer_stats.get("total_bytes", 0)),
                float(transfer_stats.get("pack_ms", 0.0)),
                float(transfer_stats.get("copy_ms", 0.0)),
                float(transfer_stats.get("sync_ms", 0.0)),
                float(transfer_stats.get("transfer_ms", 0.0)),
                float(transfer_stats.get("effective_gbps", 0.0)),
                retrieve_ms,
                total_ms,
            )

    def _do_save_req(self, req: LMCacheReqMeta) -> None:
        ss = req.save_spec
        assert ss is not None
        toks = req.token_ids
        if not req.is_last_prefill:
            toks = toks[: (len(toks) // self.chunk_size) * self.chunk_size]
        skip = (ss.skip_leading_tokens // self.chunk_size) * self.chunk_size
        if skip >= len(toks):
            self._record_store_terminal(req, True)
            return

        t_total0 = time.perf_counter()
        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:skip] = False

        t_store0 = time.perf_counter()
        self._reset_gpu_connector_transfer_stats()
        gpu_connector = self._engine.gpu_connector
        track_source = getattr(gpu_connector, "track_save_source", None)
        source_context = (
            track_source(req.save_operation)
            if getattr(self, "_early_release", False) and callable(track_source)
            else nullcontext()
        )
        with source_context:
            self._engine.store(
                torch.tensor(toks),
                mask=mask,
                block_ids=req.block_ids,
                req_id=str(req.req_id),
            )
        store_ms = (time.perf_counter() - t_store0) * 1000
        transfer_stats = self._last_gpu_connector_transfer_stats()
        total_ms = (time.perf_counter() - t_total0) * 1000
        if self._profile_enabled():
            logger.info(
                "[OFFLOAD-SAVE-PROF] rank=%s req=%s toks=%d skip=%d "
                "chunks=%d groups=%d max_chunk_bytes=%d max_group_bytes=%d "
                "gpu_staging_chunk_bytes=%d "
                "gpu_staging_buffer_chunks=%d gpu_staging_buffer_bytes=%d "
                "total_bytes=%d pack_ms=%.2f copy_ms=%.2f sync_ms=%.2f "
                "transfer_ms=%.2f effective_gbps=%.2f "
                "store_ms=%.2f total_ms=%.2f",
                getattr(self, "_rank", "?"),
                req.req_id,
                len(toks),
                skip,
                int(transfer_stats.get("chunks", 0)),
                int(transfer_stats.get("groups", 0)),
                int(transfer_stats.get("max_chunk_bytes", 0)),
                int(transfer_stats.get("max_group_bytes", 0)),
                int(transfer_stats.get("gpu_staging_chunk_bytes", 0)),
                int(transfer_stats.get("gpu_staging_buffer_chunks", 0)),
                int(transfer_stats.get("gpu_staging_buffer_bytes", 0)),
                int(transfer_stats.get("total_bytes", 0)),
                float(transfer_stats.get("pack_ms", 0.0)),
                float(transfer_stats.get("copy_ms", 0.0)),
                float(transfer_stats.get("sync_ms", 0.0)),
                float(transfer_stats.get("transfer_ms", 0.0)),
                float(transfer_stats.get("effective_gbps", 0.0)),
                store_ms,
                total_ms,
            )
        self._record_store_terminal(req, True)

    # get_finished / get_finished_recv_blocks inherited from OffloadWorkerMixin
    # (finished_recving wakes loaded reqs, failed_recving -> recompute,
    # finished_saving releases deferred frees).


# =====================================================================
# Scheduler side
# =====================================================================
class DenseOffloadScheduler(OffloadSchedulerMixin, KVConnectorSchedulerBase):
    # Consumer semantics: finished_recving wakes parked seqs (the engine asserts
    # `not is_producer` on that path). Offload never uses finished_sending.
    is_producer = False
    # Opt the scheduler into offload-wake (suffix prefill) instead of the P/D
    # decode-jump in Scheduler.schedule(); see Scheduler._is_offload_connector.
    is_offload = True
    _supports_early_block_release = True

    def __init__(self, config) -> None:
        self._init_offload_statistics()
        self._config = config
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = validated_kv_role(kvc)
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        self.block_size = offcfg._strict_integer(
            "Dense block size",
            config.kv_cache_block_size,
            minimum=1,
        )
        self.virtual_block_size = self.block_size * int(
            getattr(config, "decode_context_parallel_size", 1) or 1
        )
        self.chunk_size: int | None = None
        self._lookup_client = None

        # req_id -> LoadSpec (pending load decided at match time)
        self._load_specs: dict[str, LoadSpec] = {}
        # req_id -> Sequence (queued to recv this step)
        self._reqs_need_recv: dict[str, object] = {}
        # req_id -> HBM chunk frontier for an emitted load. If the load fails,
        # lower the save frontier to this value so recomputed chunks can be
        # stored again.
        self._load_save_floors: dict[str, int] = {}
        # req_id -> LMCache chunk frontier observed by lookup. The scheduler
        # should not re-save this already-persisted prefix unless a later load
        # actually fails.
        self._hit_save_floors: dict[str, int] = {}
        # Persistent save tracker: sid -> [seq, saved_offset]. A seq's prompt
        # prefix is stored to LMCache once prefill computes it
        # (seq.prefix_hashes_published flips True), chunk by chunk.
        self._save_tracker: dict[str, list] = {}
        # Round-robin cursor over `_save_tracker`: the last sid that emitted a
        # save. `build_connector_meta` resumes the save scan just after it so a
        # bounded save queue (`_may_emit_save`) is shared fairly. Without it the
        # scan always restarts at the insertion-ordered head, and a long
        # multi-chunk request there re-wins the freed slot every step and
        # starves later requests (their blocks stay pinned by should_defer_free).
        self._save_rr_last: str | None = None
        # sid -> exact save generation.  Exact matching prevents a delayed TP
        # notification for an older request lifecycle from releasing the
        # current request's deferred blocks.
        self._save_inflight: dict[str, SaveCompletionId] = {}
        # Early block-release bookkeeping. Operation maps retain the exact
        # token-index -> block-id relationship frozen at save emission; the
        # per-request lease set exists only after request teardown transfers a
        # refcount share from the request to the save. Source-safe and
        # store-terminal are separate connector completions.
        self._early_release = bool(self._supports_early_block_release)
        self._save_operation_blocks: dict[SaveOperationId, dict[int, int]] = {}
        self._save_operation_safe: dict[SaveOperationId, set[int]] = {}
        self._save_operation_owner: dict[SaveOperationId, object] = {}
        self._save_lease_blocks: dict[int, set[int]] = {}
        self._save_lease_at: dict[int, float] = {}
        self._save_lease_owner: dict[int, object] = {}
        self._pending_source_safe_releases: list[frozenset] = []
        self._source_safe_waiting_for_store: dict[SaveOperationId, set[int]] = {}
        self._save_outcomes: dict[SaveOperationId, bool] = {}
        self._save_retired: set[SaveOperationId] = set()
        self._save_cancel_requested: set[SaveOperationId] = set()
        self._pending_save_cancels: list[SaveOperationId] = []
        self._prepared_saves: list[LMCacheReqMeta] = []
        self._save_budget = (
            build_save_budget(config, self.virtual_block_size)
            if self._early_release
            else None
        )
        if self._early_release:
            self._max_pending_saves = self._save_budget.max_operations
            logger.info(
                "LMCache PAGE save admission: max_ops=%s source_blocks=%s pending_bytes=%s bytes_per_block=%d",
                self._max_pending_saves,
                self._save_budget.max_source_blocks,
                self._save_budget.max_pending_bytes,
                self._save_budget.bytes_per_block,
            )
        self._save_queue_timeout_s = float(
            os.environ.get("OFFLOAD_SAVE_QUEUE_TIMEOUT_S", "2")
        )
        if self._save_queue_timeout_s < 0:
            raise ValueError("OFFLOAD_SAVE_QUEUE_TIMEOUT_S must be nonnegative")
        self._save_admission_stats = dict.fromkeys(
            (
                "save_ops_admitted",
                "save_tokens_admitted",
                "save_ops_dropped",
                "save_tokens_dropped",
                "save_cancel_requests",
            ),
            0,
        )
        self._save_nonce = 0
        self._load_nonce = 0
        self._load_lifecycles: dict[str, object] = {}
        self._active_load_operations: dict[str, tuple[object, LoadOperationId]] = {}
        self._lookup_in_step: list[str] = []
        self._lookup_results: dict[str, tuple[object, int]] = {}
        self._handoff_loads: set[str] = set()
        # Unaligned handoff is always on: when the HBM prefix-cache hit is not
        # chunk-aligned, recompute the misaligned head up to the next chunk
        # boundary, then load the aligned remainder from CPU. (Previously gated
        # by the OFFLOAD_UNALIGNED_HANDOFF env var; now unconditional.)
        try:
            self._min_load_tokens = max(
                0, int(os.environ.get("OFFLOAD_MIN_LOAD_TOKENS", "8192"))
            )
        except ValueError:
            logger.warning(
                "LMCache offload scheduler: invalid OFFLOAD_MIN_LOAD_TOKENS=%r; "
                "using 8192",
                os.environ.get("OFFLOAD_MIN_LOAD_TOKENS"),
            )
            self._min_load_tokens = 8192

        # Configuration is required even though the lookup service is optional.
        # Do not turn invalid storage or geometry into a cache miss at startup.
        cfg = offcfg.build_lmcache_config(kvc)
        self.chunk_size = offcfg._strict_integer(
            "LMCache chunk size",
            cfg.chunk_size,
            minimum=1,
        )
        world = offcfg.lmcache_replica_world_size(config)
        meta = offcfg.build_lmcache_metadata(config, cfg, world, 0)
        try:
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            self._lookup_client = LookupClientFactory.create_lookup_client(cfg, meta)
            logger.info(
                "LMCache offload scheduler: lookup client on %s (world=%d)",
                meta.engine_id,
                world,
            )
        except Exception as e:  # noqa: BLE001  # optional lookup service
            logger.warning(
                "LMCache offload scheduler: lookup client unavailable: %s", e
            )

    # -- match: how many extra tokens can come from CPU/NVMe -------------
    def _begin_load_lifecycle(self, seq) -> None:
        sid = str(seq.id)
        previous = self._load_lifecycles.get(sid)
        if previous is not None and previous is not seq:
            self._clear_pending_load(sid)
            self._active_load_operations.pop(sid, None)
        self._load_lifecycles[sid] = seq

    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        if not self._do_load or self._lookup_client is None:
            return 0, False
        self._begin_load_lifecycle(seq)
        num_prompt = seq.num_prompt_tokens
        token_ids = list(seq.token_ids[:num_prompt])
        sid = str(seq.id)
        pending = self._lookup_results.get(sid)
        if pending is not None and pending[0] is not seq:
            # An older lifecycle still owns this worker-side pin. Its cleanup
            # must be dispatched before the ID can acquire a new lease.
            return 0, False
        try:
            if pending is None:
                if sid not in self._lookup_in_step:
                    self._lookup_in_step.append(sid)
                self._lookup_results[sid] = (seq, 0)
                hit = self._lookup_client.lookup(token_ids, lookup_id=sid)
                if hit is None:
                    self._lookup_results.pop(sid, None)
                else:
                    self._lookup_results[sid] = (seq, int(hit))
            else:
                hit = pending[1]
        except Exception:
            logger.exception("LMCache offload lookup failed for seq %s", seq.id)
            return 0, False
        if logger.isEnabledFor(logging.DEBUG):
            _lh = None
            try:
                tdb = getattr(self._lookup_client, "token_database", None)
                if tdb is not None:
                    _lh = [
                        k
                        for (_s, _e, k) in list(
                            tdb.process_tokens(token_ids, make_key=False)
                        )[:3]
                    ]
            except Exception as e:  # noqa: BLE001  # debug-only introspection
                _lh = f"err:{e}"
            logger.debug(
                "[OFFLOAD-LOOKUP] seq=%s num_prompt=%d hbm_cached=%d hit=%s lookuphash3=%s",
                seq.id,
                num_prompt,
                int(seq.num_cached_tokens),
                hit,
                _lh,
            )
        if not hit:
            return 0, False
        sid = str(seq.id)
        hit = int(hit)
        if hit == num_prompt:  # full-prompt hit → recompute last token
            hit -= 1
        self._hit_save_floors[sid] = self._chunk_floor(hit)
        need = hit - int(seq.num_cached_tokens)
        if need <= 0:
            self._clear_pending_load(sid)
            self._hit_save_floors[sid] = self._chunk_floor(hit)
            return 0, False
        self._load_specs[sid] = LoadSpec(
            hbm_cached_tokens=int(seq.num_cached_tokens),
            lmcache_cached_tokens=hit,
            can_load=False,
        )
        return need, True  # True => park in WAITING_FOR_REMOTE_KVS

    def update_state_after_alloc(self, seq) -> None:
        self._begin_load_lifecycle(seq)
        sid = str(seq.id)
        ls = self._load_specs.get(sid) if self._do_load else None
        logger.debug(
            "[OFFLOAD-ALLOC] seq=%s ls_found=%s num_cached_now=%s",
            seq.id,
            ls is not None,
            int(getattr(seq, "num_cached_tokens", -1)),
        )
        if ls is not None:
            ls.can_load = True
            self._reqs_need_recv[sid] = seq
        # Track for save; build_connector_meta stores chunks once the scheduler's
        # computed frontier (seq.num_cached_tokens) has advanced past them.
        #
        # If LMCache lookup already found a prefix for this request, do not save
        # that prefix again. This covers both direct loads and the
        # hbm_satisfies_after_alloc case where HBM prefix cache already covers
        # the lookup hit. Only suffix chunks computed by this request should be
        # stored.
        initial_saved = max(
            self._lmcache_hit_save_floor(ls),
            int(self._hit_save_floors.get(sid, 0)),
        )
        if self._do_save:
            entry = self._save_tracker.get(sid)
            if self._early_release:
                initial_saved = max(
                    initial_saved, getattr(seq, "_offload_save_processed", 0)
                )
            if entry is None or entry[0] is not seq:
                self._save_tracker[sid] = [seq, initial_saved]
            else:
                entry[1] = max(int(entry[1]), initial_saved)

    @property
    def requires_save_retirement(self) -> bool:
        """Elapsed time alone never authorizes recycling PAGE save sources."""
        return self._early_release

    def _drop_save_range(self, seq, aligned: int, reason: str) -> None:
        entry = self._save_tracker.get(str(seq.id))
        if entry is None or entry[0] is not seq:
            return
        tokens = max(0, aligned - int(entry[1]))
        if tokens:
            self._save_admission_stats["save_ops_dropped"] += 1
            self._save_admission_stats["save_tokens_dropped"] += tokens
            logger.debug(
                "[OFFLOAD-SAVE-DROP] req=%s tokens=%d reason=%s", seq.id, tokens, reason
            )
        entry[1] = max(int(entry[1]), aligned)
        seq._offload_save_processed = entry[1]
        seq._offload_save_disabled = True

    def _prepare_save(self, seq, entry) -> LMCacheReqMeta | None:
        """Reserve credits and freeze source ownership before any block free."""
        aligned = self._save_frontier(seq)
        saved = int(entry[1])
        if aligned <= saved:
            return None
        if getattr(seq, "_offload_save_disabled", False):
            self._drop_save_range(seq, aligned, "lifecycle_disabled")
            return None
        operation = SaveOperationId(seq.id, self._save_nonce)
        block_ids = list(getattr(seq, "_offload_finished_block_ids", seq.block_table))
        first = saved // self.virtual_block_size
        end = -(-aligned // self.virtual_block_size)
        if end > len(block_ids):
            raise ValueError("save frontier exceeds the allocated PAGE block table")
        block_map = {index: block_ids[index] for index in range(first, end)}
        request = LMCacheReqMeta(
            req_id=seq.id,
            token_ids=list(seq.token_ids[:aligned]),
            block_ids=block_ids,
            save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
            is_last_prefill=aligned >= int(seq.num_prompt_tokens),
            save_operation=operation,
        )
        reason = self._save_budget.reserve(
            operation, block_map.values(), aligned - saved
        )
        if reason is not None:
            self._drop_save_range(seq, aligned, reason)
            return None
        self._save_nonce += 1
        self._save_admission_stats["save_ops_admitted"] += 1
        self._save_admission_stats["save_tokens_admitted"] += aligned - saved
        self._track_save_statistics(operation, aligned - saved)
        self._save_operation_blocks[operation] = block_map
        self._save_operation_safe[operation] = set()
        self._save_operation_owner[operation] = seq
        self._save_inflight[str(seq.id)] = operation
        entry[1] = aligned
        seq._offload_save_processed = aligned
        self._save_rr_last = str(seq.id)
        return request

    def _request_save_cancel(self, operation: SaveOperationId) -> None:
        if (
            operation not in self._save_operation_blocks
            or operation in self._save_cancel_requested
        ):
            return
        self._save_cancel_requested.add(operation)
        self._pending_save_cancels.append(operation)
        self._save_admission_stats["save_cancel_requests"] += 1

    def _cancel_aged_saves(self) -> None:
        if not self._early_release or self._save_queue_timeout_s <= 0:
            return
        now = time.monotonic()
        for operation, reservation in self._save_budget.operations.items():
            if now - reservation.created_at >= self._save_queue_timeout_s:
                self._request_save_cancel(operation)

    def _clear_pending_load(self, sid: str) -> None:
        self._load_specs.pop(sid, None)
        self._reqs_need_recv.pop(sid, None)
        self._handoff_loads.discard(sid)
        self._load_save_floors.pop(sid, None)
        self._hit_save_floors.pop(sid, None)
        # clear_lookup_status only clears the client's memo; it does not
        # release worker pins. Keep the ID for the next metadata dispatch.
        if self._lookup_client is not None:
            try:
                self._lookup_client.clear_lookup_status(sid)
            except Exception:
                logger.debug(
                    "LMCache offload: lookup status cleanup failed for req=%s",
                    sid,
                    exc_info=True,
                )

    def _decide_load_after_alloc(
        self, seq, ls: LoadSpec
    ) -> tuple[bool, str, int, int, int, int]:
        hbm = int(getattr(seq, "num_cached_tokens", ls.hbm_cached_tokens))
        lmc = int(ls.lmcache_cached_tokens)
        ls.hbm_cached_tokens = hbm
        chunk = int(self.chunk_size or 256)
        need = lmc - hbm
        if lmc <= hbm:
            return False, "hbm_satisfies_after_alloc", hbm, lmc, need, chunk
        if hbm % chunk != 0:
            return False, "unaligned_hbm_prefill", hbm, lmc, need, chunk
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        if need < min_load:
            return False, "too_small", hbm, lmc, need, chunk
        return True, "aligned_large_hit", hbm, lmc, need, chunk

    def adjust_prefill_chunk_after_alloc(self, seq, chunk: int) -> int:
        sid = str(seq.id)
        if sid not in self._handoff_loads:
            return chunk
        boundary = getattr(seq, "offload_handoff_boundary_tokens", None)
        if boundary is None:
            return chunk
        hbm = int(getattr(seq, "num_cached_tokens", 0))
        limit = int(boundary) - hbm
        if limit <= 0:
            return chunk
        adjusted = min(int(chunk), limit)
        return max(1, adjusted)

    def _may_emit_save(self) -> bool:
        """Legacy layout hook; Dense/M3 uses `_prepare_save` admission instead."""
        return True

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        meta = LMCacheOffloadMetadata()
        early_release = getattr(self, "_early_release", False)
        if early_release:
            self._cancel_aged_saves()
            meta.requests.extend(self._prepared_saves)
            self._prepared_saves = []
            meta.cancel_save_operations = self._pending_save_cancels
            self._pending_save_cancels = []

        # Loads
        logger.debug("[OFFLOAD-BUILD] reqs_need_recv=%d", len(self._reqs_need_recv))
        loading_sids: set[str] = set()
        load_items = list(self._reqs_need_recv.items()) if self._do_load else []
        for sid, seq in load_items:
            ls = self._load_specs.pop(sid, None)
            if ls is None or not ls.can_load:
                logger.debug(
                    "[OFFLOAD-LOAD-SKIP] seq=%s ls=%s can_load=%s",
                    sid,
                    ls is not None,
                    getattr(ls, "can_load", None),
                )
                continue
            # ★ Use the REAL HBM-cached count as the load floor.
            # get_num_new_matched_tokens runs BEFORE the prefix-cache match in
            # block_manager.allocate, so seq.num_cached_tokens was stale (often
            # 0) when the LoadSpec was recorded. By now (post-allocate) it is the
            # true HBM hit. Loading below this floor would overwrite HBM
            # prefix-cache blocks (possibly shared with other seqs) -> output
            # corruption. So load only [hbm_cached, offload_hit).
            should_load, reason, hbm, lmc, need, chunk = self._decide_load_after_alloc(
                seq, ls
            )
            if not should_load:
                self._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
                self._clear_pending_load(sid)
                continue
            # num_cached after load = max(HBM, offload); never drop below HBM.
            seq.offload_loaded_tokens = self._claim_after_load(seq, hbm, lmc)
            # req_id MUST be the raw seq.id (the type the scheduler compares
            # against in _update_waiting_for_remote_kv); str(seq.id) is only for
            # LMCache's lookup/pin API. A str here silently never wakes the seq.
            logger.debug(
                "[OFFLOAD-LOAD-EMIT] seq=%s hbm_cached=%d lmc_cached=%d "
                "offload_loaded=%d need=%d min_load=%d nblocks=%d reason=aligned_large_hit",
                seq.id,
                hbm,
                lmc,
                seq.offload_loaded_tokens,
                need,
                int(getattr(self, "_min_load_tokens", 8192)),
                len(list(seq.block_table)),
            )
            loading_sids.add(sid)
            self._load_save_floors[sid] = self._chunk_floor(hbm)
            load_operation = LoadOperationId(seq.id, self._load_nonce)
            self._load_nonce += 1
            seq._load_operation = load_operation
            self._active_load_operations[sid] = (seq, load_operation)
            self._track_load_statistics(load_operation, lmc - hbm)
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:lmc]),
                    block_ids=list(seq.block_table),
                    load_spec=ls,
                    load_operation=load_operation,
                )
            )
        meta.lookup_requests_in_step = [
            sid
            for sid in self._lookup_in_step
            if sid in loading_sids or sid not in self._load_specs
        ]
        # Saves: store fully computed prompt chunks. Under scheduler-side
        # chunked prefill, seq.num_cached_tokens advances after each prefill
        # chunk's forward has completed; use it as the D2H-safe frontier.
        chunk = self.chunk_size or 256
        # Round-robin start: resume just after the last sid we emitted a save
        # for, so a bounded `_may_emit_save` queue is shared fairly instead of
        # always favouring the insertion-ordered head (see `_save_rr_last`).
        tracker_sids = list(self._save_tracker.keys())
        if tracker_sids and self._save_rr_last in self._save_tracker:
            start = (tracker_sids.index(self._save_rr_last) + 1) % len(tracker_sids)
            tracker_sids = tracker_sids[start:] + tracker_sids[:start]
        for sid in tracker_sids:
            entry = self._save_tracker[sid]
            if not self._do_save:
                continue
            if not early_release and not self._may_emit_save():
                break
            seq, saved = entry
            if sid in self._reqs_need_recv or sid in loading_sids:
                continue  # loading this step; defer its save
            if early_release:
                if any(owner is seq for owner in self._save_operation_owner.values()):
                    continue
                request = self._prepare_save(seq, entry)
                if request is not None:
                    meta.add_request(request)
                continue
            if sid in self._save_inflight:
                continue  # keep at most one save per request in flight
            computed = min(
                int(
                    getattr(
                        seq,
                        "_offload_finished_cached_tokens",
                        getattr(seq, "num_cached_tokens", 0),
                    )
                ),
                int(seq.num_prompt_tokens),
            )
            is_last_prefill = computed >= int(seq.num_prompt_tokens)
            aligned = (computed // chunk) * chunk
            if aligned <= saved:
                continue
            logger.debug(
                "[OFFLOAD-SAVE-EMIT] seq=%s computed=%d num_prompt=%d aligned=%d saved=%d",
                seq.id,
                computed,
                int(seq.num_prompt_tokens),
                aligned,
                saved,
            )
            save_operation = SaveOperationId(seq.id, self._save_nonce)
            self._save_nonce += 1
            self._track_save_statistics(save_operation, aligned - saved)
            block_ids = list(
                getattr(seq, "_offload_finished_block_ids", seq.block_table)
            )
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:aligned]),
                    block_ids=block_ids,
                    save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
                    is_last_prefill=is_last_prefill,
                    save_operation=save_operation,
                )
            )
            entry[1] = aligned
            self._save_inflight[sid] = save_operation
            self._save_rr_last = sid
            if getattr(self, "_early_release", False):
                # Freeze the exact token-index -> block-id mapping before a
                # finished request clears its block table. The lease itself is
                # activated only by `activate_block_leases` at teardown.
                source_block_size = getattr(self, "virtual_block_size", self.block_size)
                start_block = saved // source_block_size
                end_block = -(-aligned // source_block_size)  # ceil div
                self._save_operation_blocks[save_operation] = {
                    index: block_ids[index]
                    for index in range(start_block, min(end_block, len(block_ids)))
                }
                self._save_operation_safe[save_operation] = set()
                self._save_operation_owner[save_operation] = seq
        dispatched = set(meta.lookup_requests_in_step)
        for sid in dispatched:
            self._lookup_results.pop(sid, None)
        self._lookup_in_step = [
            sid for sid in self._lookup_in_step if sid not in dispatched
        ]
        self._reqs_need_recv.clear()
        return meta

    def should_defer_free(self, seq) -> bool:
        if self._has_active_load(seq):
            return True
        if not self._do_save:
            return False
        if getattr(self, "_early_release", False):
            return bool(self.protected_block_ids(seq))
        sid = str(seq.id)
        operation_blocks = getattr(self, "_save_operation_blocks", {})
        operation_safe = getattr(self, "_save_operation_safe", {})
        operation_owner = getattr(self, "_save_operation_owner", {})
        unsafe_retired_operation = any(
            owner is seq
            and not set(operation_blocks.get(operation, {}).values()).issubset(
                operation_safe.get(operation, set())
            )
            for operation, owner in operation_owner.items()
        )
        return (
            sid in self._save_inflight
            or self._has_pending_save(seq)
            or unsafe_retired_operation
        )

    def protected_block_ids(self, seq) -> frozenset | None:
        """Only admitted source blocks not yet known source-safe.

        None means "this connector cannot narrow the protection" (the layout
        does not support exact leases, or a load is in flight, since a load also
        touches HBM blocks this connector does not track per-range) -- the
        scheduler falls back to deferring the whole request. Final saves must
        acquire admission in request_finished, before block deallocation.
        """
        if not self._early_release or self._has_active_load(seq):
            return None
        protected: set[int] = set()
        for operation, blocks in self._save_operation_blocks.items():
            if self._save_operation_owner.get(operation) is not seq:
                continue
            safe = self._save_operation_safe.get(operation, set())
            protected.update(
                block_id for block_id in blocks.values() if block_id not in safe
            )

        return frozenset(protected)

    def activate_block_leases(self, seq, block_ids: frozenset[int]) -> None:
        """Record the refcount shares transferred at request deallocation."""

        if not self._early_release or not block_ids:
            return
        lease_key = id(seq)
        leased = self._save_lease_blocks.setdefault(lease_key, set())
        self._save_lease_owner[lease_key] = seq
        added = set(block_ids) - leased
        leased.update(added)
        if added:
            self._save_lease_at.setdefault(lease_key, time.monotonic())
            self.total_leased_source_blocks += len(added)

    def take_source_safe_releases(self) -> list[frozenset]:
        """Drain block-ID sets whose lease became source-safe since the last poll."""
        out = self._pending_source_safe_releases
        self._pending_source_safe_releases = []
        return out

    def reclaim_stale_leases(self, timeout_s: float) -> list[frozenset]:
        """Request cancellation of old leases; time is not a source-read fence."""
        if timeout_s <= 0 or not self._save_lease_at:
            return []
        now = time.monotonic()
        stale_keys = [
            lease_key
            for lease_key, at in self._save_lease_at.items()
            if now - at >= timeout_s
        ]
        for lease_key in stale_keys:
            for operation, owner in self._save_operation_owner.items():
                if id(owner) == lease_key:
                    self._request_save_cancel(operation)
        return []

    def release_stalled_save(self, seq) -> None:
        """Drop bookkeeping for a stall-escaped save the scheduler is freeing.

        No-op on dense: its `should_defer_free` has no stall escape, so a request
        with a pending save always defers and is never preemptable. K3 overrides
        this to pop its `_save_tracker`. Defined here so every offload impl
        answers the scheduler's `release_stalled_save` forward uniformly.
        """

    def has_pending_work(self) -> bool:
        """Keep polling admitted saves, cancellation acknowledgements and loads."""
        return (
            bool(self._reqs_need_recv)
            or bool(self._save_inflight)
            or bool(getattr(self, "_save_lease_blocks", {}))
            or bool(self._prepared_saves)
            or bool(self._pending_save_cancels)
            or bool(self._save_operation_blocks)
            or any(sid not in self._load_specs for sid in self._lookup_in_step)
        )

    def save_finished(self, req_id) -> None:
        if self._early_release and isinstance(req_id, SaveOperationId):
            # Legacy notification wakes the engine, but does not replace the
            # independent outcome and no-future-source-read quorums.
            return
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        active = self._save_inflight.get(sid)
        if isinstance(req_id, SaveOperationId):
            if active != req_id:
                return
        elif isinstance(active, SaveOperationId):
            # Once this lifecycle has an exact identity, a raw request ID
            # cannot complete it.  Raw IDs still clear explicitly legacy
            # entries should one be restored from older scheduler state.
            return
        self._save_inflight.pop(sid, None)
        self._finish_save_statistics(req_id)
        self._release_operation_lease(req_id)
        self._finish_retired_request(sid)

    def connector_completion(self, completion: ConnectorCompletion) -> bool | None:
        """Apply TP/PP-quorumed source-safe and store-terminal reports."""

        if completion.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL:
            identity = completion.operation_id
            if not isinstance(identity, SaveSourceGroupId):
                return False
            if completion.succeeded:
                self._source_group_finished(identity)
            return None
        if completion.channel == DENSE_PAGE_RETIRED_CHANNEL:
            operation = completion.operation_id
            if not isinstance(operation, SaveOperationId):
                return False
            if completion.succeeded and operation in self._save_operation_blocks:
                self._save_retired.add(operation)
                self._mark_sources_safe(
                    operation, self._save_operation_blocks[operation].values()
                )
                self._try_retire_save(operation)
            return True
        if completion.channel != DENSE_PAGE_STORE_CHANNEL:
            return False
        operation = completion.operation_id
        if not isinstance(operation, SaveOperationId):
            return False
        self._store_finished(operation, succeeded=completion.succeeded)
        return True

    def _source_group_finished(self, identity: SaveSourceGroupId) -> None:
        operation = identity.save_operation
        block_map = self._save_operation_blocks.get(operation)
        if block_map is None:
            return
        source_blocks: set[int] = set()
        for start, end in identity.ranges:
            start_block = start // self.virtual_block_size
            end_block = -(-end // self.virtual_block_size)
            source_blocks.update(
                block_map[index]
                for index in range(start_block, end_block)
                if index in block_map
            )
        self._mark_sources_safe(operation, source_blocks)

    def _mark_sources_safe(self, operation, source_blocks) -> None:
        safe = self._save_operation_safe.setdefault(operation, set())
        source_blocks = set(source_blocks)
        newly_safe = source_blocks - safe
        safe.update(newly_safe)
        self._save_budget.source_safe(operation, newly_safe)
        if not newly_safe:
            return
        owner = self._save_operation_owner.get(operation)
        self._release_safe_owner_leases(owner)
        if operation not in self._save_outcomes:
            self._source_safe_waiting_for_store.setdefault(operation, set()).update(
                newly_safe
            )

    def _release_safe_owner_leases(self, owner) -> None:
        lease_key = id(owner) if owner is not None else None
        leased = self._save_lease_blocks.get(lease_key)
        if not leased:
            return
        unsafe = set()
        for operation, candidate_owner in self._save_operation_owner.items():
            if candidate_owner is owner:
                unsafe.update(
                    set(self._save_operation_blocks[operation].values())
                    - self._save_operation_safe.get(operation, set())
                )
        releasable = leased - unsafe
        if releasable:
            leased.difference_update(releasable)
            self._pending_source_safe_releases.append(frozenset(releasable))
            self.total_source_safe_released_blocks += len(releasable)
            if not leased:
                self._save_lease_blocks.pop(lease_key, None)
                self._save_lease_at.pop(lease_key, None)
                self._save_lease_owner.pop(lease_key, None)

    def _store_finished(self, operation: SaveOperationId, *, succeeded: bool) -> None:
        if operation not in self._save_operation_blocks:
            return
        self._save_outcomes[operation] = (
            self._save_outcomes.get(operation, True) and succeeded
        )
        self._source_safe_waiting_for_store.pop(operation, None)
        self._try_retire_save(operation)

    def _try_retire_save(self, operation) -> None:
        if operation not in self._save_retired or operation not in self._save_outcomes:
            return
        succeeded = self._save_outcomes.pop(operation)
        self._save_retired.discard(operation)
        self._save_cancel_requested.discard(operation)
        self._pending_save_cancels = [
            op for op in self._pending_save_cancels if op != operation
        ]
        owner = self._save_operation_owner.get(operation)
        if not succeeded and owner is not None:
            self._drop_save_range(owner, self._save_frontier(owner), "store_failed")
        if succeeded:
            self._finish_save_statistics(operation)
        else:
            self._cancel_save_statistics(operation)
        self._save_budget.retire(operation)
        self._release_operation_lease(operation)
        self._source_safe_waiting_for_store.pop(operation, None)
        sid = str(operation.req_id)
        if self._save_inflight.get(sid) == operation:
            self._save_inflight.pop(sid, None)
        self._finish_retired_request(sid)

    def _release_operation_lease(self, operation) -> None:
        if not isinstance(operation, SaveOperationId):
            return
        self._save_operation_blocks.pop(operation, None)
        self._save_operation_safe.pop(operation, None)
        owner = self._save_operation_owner.pop(operation, None)
        self._release_safe_owner_leases(owner)

    def _finish_retired_request(self, sid: str) -> None:
        entry = self._save_tracker.get(sid)
        if entry is None:
            return
        seq = entry[0]
        if hasattr(seq, "_offload_finished_block_ids") and not any(
            owner is seq for owner in self._save_operation_owner.values()
        ):
            self._save_tracker.pop(sid, None)

    def blocks_waiting_for_store(self) -> int:
        return sum(
            len(blocks) for blocks in self._source_safe_waiting_for_store.values()
        )

    def get_statistics(self) -> dict[str, int]:
        statistics = super().get_statistics()
        if self._early_release:
            statistics.update(self._save_admission_stats)
            statistics.update(
                save_ops_unretired=len(self._save_budget.operations),
                save_pending_bytes=self._save_budget.pending_bytes,
                source_blocks_reserved=self._save_budget.source_blocks,
                source_blocks_leased=sum(map(len, self._save_lease_blocks.values())),
                save_oldest_age_ms=self._save_budget.oldest_age_ms(),
            )
        return statistics

    def abandon_save(self, req_id) -> None:
        """Request PAGE cancellation, or abandon an unsupported legacy layout."""
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        if self._early_release:
            for operation in self._save_operation_blocks:
                if operation == req_id or (
                    not isinstance(req_id, SaveOperationId)
                    and str(operation.req_id) == sid
                ):
                    self._request_save_cancel(operation)
            return
        operation = self._save_inflight.pop(sid, None)
        if operation is not None:
            self._cancel_save_statistics(operation)
        self._save_tracker.pop(sid, None)
        owner = self._save_operation_owner.pop(operation, None)
        lease_key = id(owner) if owner is not None else None
        self._save_lease_at.pop(lease_key, None)
        blocks = self._save_lease_blocks.pop(lease_key, None)
        self._save_lease_owner.pop(lease_key, None)
        if isinstance(operation, SaveOperationId):
            self._save_operation_blocks.pop(operation, None)
            self._save_operation_safe.pop(operation, None)
            self._source_safe_waiting_for_store.pop(operation, None)
        if blocks:
            self._pending_source_safe_releases.append(frozenset(blocks))
            self.total_abnormal_lease_reclaims += len(blocks)

    def load_failed(self, req_id) -> bool:
        sid = str(req_id.req_id if isinstance(req_id, LoadOperationId) else req_id)
        active = self._active_load_operations.get(sid)
        if isinstance(req_id, LoadOperationId):
            if active is None or active[1] != req_id:
                return False
            self._active_load_operations.pop(sid, None)
        elif active is not None:
            # Once this lifecycle has an exact generation, a legacy raw request
            # ID cannot complete it (including after request-ID reuse).
            return False
        self._finish_load_statistics(req_id, succeeded=False)
        floor = self._load_save_floors.get(sid)
        entry = self._save_tracker.get(sid)
        if (
            floor is not None
            and entry is not None
            and not getattr(entry[0], "_offload_save_disabled", False)
        ):
            # The LMCache hit was not actually loaded. Let the recomputed
            # [HBM, LMC) chunks be saved again instead of permanently treating
            # them as already persisted.
            entry[1] = self._chunk_floor(floor)
            if self._early_release:
                entry[1] = max(
                    entry[1], getattr(entry[0], "_offload_save_processed", 0)
                )
        self._clear_pending_load(sid)
        return True

    def load_finished(self, req_id) -> bool:
        sid = str(req_id.req_id if isinstance(req_id, LoadOperationId) else req_id)
        active = self._active_load_operations.get(sid)
        if isinstance(req_id, LoadOperationId):
            if active is None or active[1] != req_id:
                return False
            self._active_load_operations.pop(sid, None)
        elif active is not None:
            return False
        self._finish_load_statistics(req_id, succeeded=True)
        self._load_save_floors.pop(sid, None)
        return True

    def cancel_pending_load(self, seq) -> None:
        sid = str(seq.id)
        if self._load_lifecycles.get(sid) is not seq:
            return
        self._clear_pending_load(sid)
        active = self._active_load_operations.get(sid)
        if active is not None and active[0] is seq:
            self._active_load_operations.pop(sid, None)
            operation = active[1]
            self._cancel_load_statistics(operation)
            if getattr(seq, "_load_operation", None) == operation:
                delattr(seq, "_load_operation")

    def request_finished(self, seq) -> None:
        sid = str(seq.id)
        if self._load_lifecycles.get(sid) is seq:
            self._clear_pending_load(sid)
            active = self._active_load_operations.get(sid)
            if active is not None and active[0] is seq:
                self._active_load_operations.pop(sid, None)
                self._cancel_load_statistics(active[1])
            self._load_lifecycles.pop(sid, None)
        entry = self._save_tracker.get(sid)
        if entry is not None and entry[0] is seq:
            if self._early_release:
                if not hasattr(seq, "_offload_finished_block_ids"):
                    seq._offload_finished_block_ids = list(seq.block_table)
                    seq._offload_finished_cached_tokens = min(
                        int(getattr(seq, "num_cached_tokens", 0)),
                        int(seq.num_prompt_tokens),
                    )
                if self._has_pending_save(seq):
                    if any(
                        owner is seq for owner in self._save_operation_owner.values()
                    ):
                        self._drop_save_range(
                            seq, self._save_frontier(seq), "final_save_busy"
                        )
                    else:
                        request = self._prepare_save(seq, entry)
                        if request is not None:
                            self._prepared_saves.append(request)
                self._finish_retired_request(sid)
            elif not self.should_defer_free(seq):
                self._save_tracker.pop(sid, None)
        if hasattr(seq, "_load_operation"):
            delattr(seq, "_load_operation")
