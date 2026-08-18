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
  :class:`ATOMKVByteCodec`.
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
from concurrent.futures import ThreadPoolExecutor

import torch

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import KVConnectorOutput, ReqId
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.atom_kv_byte_codec import ATOMKVByteCodec
from atom.kv_transfer.offload.atom_lmcache_gpu_connector import (
    ATOMLMCacheGPUConnector,
)
from atom.kv_transfer.offload.metadata import (
    ATOMRawBytesLMCacheMetadata,
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)

logger = logging.getLogger("atom")


# =====================================================================
# Worker side
# =====================================================================
class LMCacheOffloadConnector(KVConnectorBase):
    # Offload is a *consumer* from the scheduler's POV (it loads KV back). Saves
    # are fire-and-forget on the worker and must NOT be reported as
    # finished_sending (the scheduler frees blocks on finished_sending — a P/D
    # producer semantic that would wrongly deallocate live offload blocks).
    is_producer = False

    def __init__(self, config) -> None:
        self._config = config
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = kvc.get("kv_role", "offload")
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        self.block_size = int(config.kv_cache_block_size)
        self.chunk_size: int | None = None

        # Copy daemons: keep GPU<->host copies off the RPC thread. SEPARATE
        # executors for LOAD vs SAVE so a load (on the TTFT critical path — a
        # parked seq is waiting for it) never queues behind a backlog of fire-
        # and-forget saves (Phase 4 root cause: with one shared serial daemon, a
        # reload sat behind ~N filler saves -> request hung well past timeout).
        # The ATOM LMCache GPU connector owns per-thread staging streams.
        # OFFLOAD_COPY_WORKERS tunes the SAVE pool only.
        n_save_workers = int(os.environ.get("OFFLOAD_COPY_WORKERS", "1"))
        self._load_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lmc-offload-load"
        )
        self._save_executor = ThreadPoolExecutor(
            max_workers=n_save_workers, thread_name_prefix="lmc-offload-save"
        )
        self._lock = threading.Lock()
        self._done_load: set[ReqId] = set()
        self._done_save: set[ReqId] = set()
        self._failed_load: set[ReqId] = set()

        self._engine = None
        self._codec: ATOMKVByteCodec | None = None
        self._lookup_server = None
        # Built in `register_kv_caches` when the state tier is enabled and the
        # backend owns per-request state. Always defined, because
        # `AttentionBackend._submit_state_spills` probes for it on every batch
        # and an AttributeError there would be a forward-path crash.
        self._state_tier = None
        from atom.kv_transfer.offload.state_tier import _JointPark

        # Requests whose KV leg and state leg must both land before the engine
        # is told anything. Inert until something arms it, so it costs a
        # dictionary lookup per report while `OFFLOAD_STATE_JOINT_KV` is off --
        # the engine only ever issues both legs for one request with it on.
        self._joint_park = _JointPark()

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        from aiter.dist.parallel_state import get_tp_group
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.v1.memory_management import MemoryFormat

        tp = get_tp_group()
        rank, world = tp.rank_in_group, tp.world_size
        self._rank = rank

        cfg = offcfg.build_lmcache_config(
            getattr(self._config, "kv_transfer_config", None)
        )
        self.chunk_size = int(cfg.chunk_size)
        # num_blocks is the scheduler-visible block count, threaded from the
        # model runner. MLA stores its KV token-major, so the codec cannot infer
        # this count from tensor.shape[0] (the page-size-1 physical row count).
        self._codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks)
        self._validate_block_geometry(transfer_tensors)
        base_meta = offcfg.build_lmcache_metadata(self._config, cfg, world, rank)
        meta = ATOMRawBytesLMCacheMetadata(
            base_meta,
            atom_block_size=self.block_size,
            bytes_per_block=self._codec.bytes_per_block,
        )
        gpu_connector = ATOMLMCacheGPUConnector(
            self._codec,
            self.block_size,
            chunk_size=self.chunk_size,
        )

        self._engine = LMCacheEngineBuilder.get_or_create(
            f"atom-offload-{rank}",
            cfg,
            meta,
            gpu_connector,
            lambda t, s: None,
            lambda o, s: o,
        )
        # LMCache's LocalCPU allocator does not accept BINARY for normal
        # MemoryObj allocation. The metadata shape/dtype already make this an
        # opaque uint8 object, so keep a supported tensor MemoryFormat.
        self._engine.fmt = MemoryFormat.KV_2LTD
        self._engine.post_init()
        self._validate_and_log_storage_backends(cfg)

        self._maybe_build_state_tier(gpu_connector, transfer_tensors, meta, rank, world)

        # ZMQ lookup server so the scheduler process can query our hit counts.
        try:
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            self._lookup_server = LookupClientFactory.create_lookup_server(
                self._engine, meta
            )
        except Exception as e:  # noqa: BLE001  # optional third-party service
            logger.warning("LMCache offload: lookup server not started: %s", e)

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

    def _maybe_build_state_tier(
        self, gpu_connector, transfer_tensors, meta, rank: int, world: int
    ) -> None:
        """Stand up the worker half of the state offload tier, if it is on.

        Gated on `state_offload_staging_groups()`, the same function the arena
        and the pool size themselves from, so the runner and the engine cannot
        disagree about whether the tier exists. A second `os.environ` read here
        would let the engine queue spills into a ring the runner never drains.

        Three things this needs and where each comes from:
        * the backend, for `state_entry_views` -- carried on
          `transfer_tensors.state_backend`, set by the model runner, the only
          scope holding both the builder and this connector;
        * a `StagedTransfer` -- the KV GPU connector already owns one sized to
          the bounded staging buffer, and reusing it is what keeps the state
          path on the same HBM bound as KV. Note it does not share the *buffer*:
          `StagedTransfer` keeps those in `threading.local`, and the tier packs
          on its own `lmc-state` worker, so the tier adds one more resident
          staging buffer of `gpu_staging_buffer_bytes` per rank. What reuse
          buys is a single place that bound is configured, which is what makes
          the `entry_bytes > staging_bytes` refusal below meaningful;
        * the per-entry byte count -- measured off the backend's own views
          rather than read from the sizing plan, because the pack allocates
          exactly what the views sum to and a plan figure that rounded
          differently would fail inside `memory_tensor` on the first spill.
        """
        from atom.model_engine.state_offload import state_offload_staging_groups

        if state_offload_staging_groups() <= 0:
            return
        # Pipeline parallelism breaks the tier in two independent ways, so it is
        # refused outright rather than half-supported.
        #
        # First the key. `StateByteCodec.key` builds a `CacheEngineKey` from
        # (model, world_size, worker_id, hash), and `worker_id` here is
        # `tp.rank_in_group` -- there is no PP component anywhere in it. Every PP
        # stage holds a *different* slice of the layers, so stage 0 and stage 1
        # at the same TP rank would write different bytes under an identical key
        # and silently overwrite each other. A later load would then restore one
        # stage's state into another's layers: wrong output, no error.
        #
        # Adding a PP component to the key would fix that half and still leave
        # the second. `pp_engine_core.py` has every stage call `forward` on the
        # head-pickled batch, so `_submit_state_spills` fires on all of them,
        # but only the head runs `_poll_kv_transfer_progress`. The non-head
        # stages' `state_staging_released` reports are never drained, so the
        # ring never gets those slots back and the tier quietly stops spilling
        # after `staging_depth` evictions. Wiring that up is a scheduler change,
        # not a key change, so PP + OFFLOAD_STATE is out of scope for now.
        #
        # Paged-KV offload is unaffected: this returns before the tier is built
        # and touches nothing else.
        pp_size = int(getattr(self._config, "pipeline_parallel_size", 1) or 1)
        if pp_size > 1:
            logger.warning(
                "state offload: OFFLOAD_STATE is not supported with pipeline "
                "parallelism (pipeline_parallel_size=%d); the state tier stays "
                "off and nothing spills. Paged-KV offload is unaffected.",
                pp_size,
            )
            return
        backend = getattr(transfer_tensors, "state_backend", None)
        if backend is None:
            logger.warning(
                "state offload: OFFLOAD_STATE is set but no attention backend "
                "was published; the tier stays off and nothing spills."
            )
            return
        try:
            views = backend.state_entry_views(0)
            entry_bytes = sum(int(v.numel()) * v.element_size() for v in views)
        except (NotImplementedError, AttributeError):
            # `NotImplementedError` is the base class's own refusal;
            # `AttributeError` is a builder that predates the method. Both mean
            # the same thing to us -- this backend has no per-request state we
            # can name bytes for -- and neither is worth killing model load
            # over, so both warn and disable rather than propagate.
            # `IndexError` is deliberately NOT caught: it means group 0 does not
            # exist, i.e. a zero-entry state pool with the tier switched on,
            # which is a sizing bug that must be loud rather than degrade into a
            # server that silently never spills.
            logger.warning(
                "state offload: %s owns no per-request state views; the tier "
                "stays off and nothing spills.",
                type(backend).__name__,
            )
            return
        # One entry is MB-scale (53.6 MiB measured) while the shared staging
        # buffer is sized for KV chunks, so the two can easily fail to fit.
        # `StagedTransfer.pack` would then raise inside `ensure_buffer`, and
        # `StateOffloadTier._do_spill`'s broad `except` would turn every single
        # spill into a warning plus a slot release -- a tier that looks healthy,
        # burns a D2D copy per eviction, and stores nothing, forever. Refuse to
        # build it instead, and name both numbers so the fix is obvious.
        from atom.kv_transfer.offload.staged_transfer import StagedTransfer

        staging_bytes = int(getattr(gpu_connector, "gpu_staging_buffer_bytes", 0))
        staged = gpu_connector._staged
        if entry_bytes > staging_bytes:
            # A state entry is one *entry*; the KV buffer is sized in LMCache
            # chunks, and at the shipped default of 2 chunks it is ~8 MiB
            # against a 55 MiB entry. This used to refuse the tier, which is
            # the worst of the three options: the engine-side index still
            # exists, so it keeps handing out staging slots and counting spills
            # that never happen, and the only symptom is this line in a
            # 100k-line log. Growing the shared buffer is the other, and it
            # charges every KV worker thread for a size only the state thread
            # needs. So the tier gets its own, of exactly one entry, on the
            # `lmc-state` thread that packs into it.
            logger.info(
                "state offload: one state entry is %.2f MiB but the KV staging "
                "buffer holds %.2f MiB, so the tier takes its own buffer of one "
                "entry (one per rank). Set OFFLOAD_GPU_STAGING_CHUNKS >= %d to "
                "have both share the KV buffer instead.",
                entry_bytes / (1 << 20),
                staging_bytes / (1 << 20),
                -(-entry_bytes // max(1, gpu_connector.gpu_staging_chunk_bytes)),
            )
            staged = StagedTransfer(
                gpu_connector.device,
                staging_buffer_bytes=entry_bytes,
                release_after_transfer=gpu_connector.release_gpu_staging_after_transfer,
            )
        from atom.kv_transfer.offload.state_object import StateByteCodec
        from atom.kv_transfer.offload.state_tier import StateOffloadTier

        codec = StateByteCodec(
            backend,
            staged,
            entry_bytes,
            model_name=meta.model_name,
            world_size=world,
            worker_id=rank,
        )
        codec.bind_storage_manager(self._engine.storage_manager)
        # No index: `StateOffloadIndex` lives in the engine process. Both
        # directions report and the engine applies.
        self._state_tier = StateOffloadTier(codec)

    def _validate_block_geometry(self, transfer_tensors) -> None:
        """Cross-check codec blocks against existing transfer-region metadata."""
        block_regions = getattr(transfer_tensors, "block_regions", None) or []
        if not block_regions:
            return
        expected = sum(int(region.unit_bytes) for region in block_regions)
        if self._codec.bytes_per_block != expected:
            raise ValueError(
                "LMCache offload KV block geometry mismatch: "
                f"codec={self._codec.bytes_per_block} bytes, "
                f"transfer_regions={expected} bytes, "
                f"num_blocks={self._codec.num_blocks}"
            )

    def _validate_and_log_storage_backends(self, cfg) -> None:
        """Report the realized LMCache tier topology and validate NVMe startup."""
        storage_manager = getattr(self._engine, "storage_manager", None)
        backend_names: list[str] = []
        if storage_manager is not None:
            list_backends = getattr(storage_manager, "list_backends", None)
            if callable(list_backends):
                backend_names = sorted(str(name) for name in list_backends())
            else:
                storage_backends = getattr(storage_manager, "storage_backends", {})
                backend_names = sorted(str(name) for name in storage_backends)

        local_disk = getattr(cfg, "local_disk", None)
        disk_size_gib = float(getattr(cfg, "max_local_disk_size", 0.0) or 0.0)
        disk_configured = bool(local_disk) and disk_size_gib > 0
        if disk_configured and "LocalDiskBackend" not in backend_names:
            raise RuntimeError(
                "LMCache local-disk offload was configured but LocalDiskBackend "
                f"was not created on rank {self._rank}; backends={backend_names}"
            )

        logger.info(
            "LMCache offload worker rank=%d storage: backends=%s "
            "local_cpu=%s max_local_cpu_gib=%s local_disk=%s "
            "max_local_disk_gib=%s store_location=%s retrieve_locations=%s",
            self._rank,
            backend_names,
            getattr(cfg, "local_cpu", None),
            getattr(cfg, "max_local_cpu_size", None),
            local_disk,
            getattr(cfg, "max_local_disk_size", None),
            getattr(cfg, "store_location", None),
            getattr(cfg, "retrieve_locations", None),
        )

    # -- per-step (RPC thread): only enqueue, never copy ------------------
    def start_load_kv(self, metadata) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        self._arm_joint_loads(metadata)
        self._start_state_loads(metadata)
        loading_lookup_ids = {
            str(req.req_id)
            for req in metadata.requests
            if req.load_spec is not None and self._do_load
        }
        for lookup_id in metadata.lookup_requests_in_step:
            if str(lookup_id) not in loading_lookup_ids:
                self._lookup_unpin(lookup_id)
        save_ready_event = None
        if self._do_save and any(
            req.save_spec is not None for req in metadata.requests
        ):
            # Forward kernels publish KV on the RPC thread's current stream.
            # Save workers pack it on independent streams, so carry an event
            # across the thread boundary instead of racing the producer.
            save_ready_event = torch.cuda.Event()
            save_ready_event.record(torch.cuda.current_stream())
        for req in metadata.requests:
            if req.load_spec is not None and self._do_load:
                self._load_executor.submit(self._guard, "load", self._do_load_req, req)
            if req.save_spec is not None and self._do_save:
                self._save_executor.submit(
                    self._guard,
                    "save",
                    self._do_save_req,
                    req,
                    save_ready_event,
                )

    def _arm_joint_loads(self, metadata) -> None:
        """Hold a request that has both legs until both of them report.

        The two legs come back through one channel -- `get_finished` unions the
        tier's report into the KV one -- so an id in both collapses to a single
        wake, and the engine would resume the suffix prefill while the other
        transfer is still writing. Whichever leg is slower decides which half of
        the prefix is garbage, and neither raises.

        Armed per step from the metadata that carries both lists, which is all
        the pairing there is to do: a request is parked while either leg is in
        flight, so a second pair cannot be issued before this one resolves.
        Same `req_id` values on both sides (`seq.id`, not stringified) -- that
        is exactly why they collide in the union, and it has to hold here too.
        """
        loads = getattr(metadata, "state_loads", None) or ()
        state_ids = {req_id for req_id, _h, _slot in loads}
        if not state_ids or not self._do_load:
            return
        for req in metadata.requests:
            if req.load_spec is not None and req.req_id in state_ids:
                self._joint_park.arm(req.req_id, needs_kv=True, needs_state=True)

    def _start_state_loads(self, metadata) -> None:
        """Hand this step's state-tier loads to the tier's own executor.

        No producer fence, unlike the save path: a load *writes* the state
        entry and nothing on the compute stream is reading it yet. The group
        belongs to a request that is parked precisely so no forward touches it,
        and `StagedTransfer.unpack` synchronizes the stream that produced the
        bytes before it returns, so by the time the report reaches the engine
        the entry is readable from any stream.

        With no tier the loads are reported failed rather than dropped. The
        tier can legitimately refuse to build (wrong connector, pipeline
        parallelism, a backend with no state views, an entry larger than the
        staging buffer) while the engine's index exists and keeps offering
        loads; each of those requests is already parked, and only a report
        unparks it.
        """
        loads = getattr(metadata, "state_loads", None)
        if not loads:
            return
        if self._state_tier is None:
            logger.warning(
                "state offload: %d load(s) arrived but this worker built no "
                "state tier; failing them so their requests recompute instead "
                "of waiting forever.",
                len(loads),
            )
            self._fail_state_loads(loads)
            return
        for req_id, h, group in loads:
            self._state_tier.submit_load(req_id, int(h), int(group))

    def _fail_state_loads(self, loads) -> None:
        with self._lock:
            self._failed_load.update(req_id for req_id, _h, _group in loads)

    def _guard(self, kind: str, fn, req, *args) -> None:
        try:
            fn(req, *args)
        except Exception:
            logger.exception(
                "LMCache offload: %s failed for %s", fn.__name__, req.req_id
            )
            if kind == "load":
                self._lookup_unpin(req.req_id)
            with self._lock:
                if kind == "load":
                    self._failed_load.add(req.req_id)
                else:
                    # A failed save should not keep blocks pinned forever. The
                    # request simply loses this offload opportunity.
                    self._done_save.add(req.req_id)

    def _lookup_unpin(self, req_id) -> None:
        if getattr(self, "_engine", None) is None:
            return
        try:
            self._engine.lookup_unpin(str(req_id))
        except Exception:  # best-effort third-party cleanup
            logger.debug(
                "LMCache offload: lookup unpin failed for %s",
                req_id,
                exc_info=True,
            )

    def _profile_enabled(self) -> bool:
        return os.environ.get("OFFLOAD_PROFILE", "0").lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def _last_gpu_connector_transfer_stats(self) -> dict[str, int | float]:
        gpu_connector = getattr(getattr(self, "_engine", None), "gpu_connector", None)
        if gpu_connector is None or not hasattr(gpu_connector, "last_transfer_stats"):
            return {}
        try:
            return dict(gpu_connector.last_transfer_stats())
        except Exception:  # optional instrumentation hook
            logger.debug("Failed to read GPU transfer stats", exc_info=True)
            return {}

    def _reset_gpu_connector_transfer_stats(self) -> None:
        gpu_connector = getattr(getattr(self, "_engine", None), "gpu_connector", None)
        if gpu_connector is None or not hasattr(gpu_connector, "reset_transfer_stats"):
            return
        try:
            gpu_connector.reset_transfer_stats()
        except Exception:  # optional instrumentation hook
            logger.debug("Failed to reset GPU transfer stats", exc_info=True)

    # -- copy daemon thread ----------------------------------------------
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
                self._done_load.add(req.req_id)
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
                self._failed_load.add(req.req_id)
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
                self._done_load.add(req.req_id)
            else:
                self._failed_load.add(req.req_id)
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

    def _do_save_req(self, req: LMCacheReqMeta, producer_event=None) -> None:
        ss = req.save_spec
        assert ss is not None
        toks = req.token_ids
        if not req.is_last_prefill:
            toks = toks[: (len(toks) // self.chunk_size) * self.chunk_size]
        skip = (ss.skip_leading_tokens // self.chunk_size) * self.chunk_size
        if skip >= len(toks):
            with self._lock:
                self._done_save.add(req.req_id)
            return

        # Wait only for the forward work that produces this save's KV. This
        # blocks the background save worker, not the model RPC thread, and is
        # intentionally narrower than a device-wide synchronize().
        if producer_event is not None:
            producer_event.synchronize()

        t_total0 = time.perf_counter()
        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:skip] = False

        t_store0 = time.perf_counter()
        self._reset_gpu_connector_transfer_stats()
        self._engine.store(
            torch.tensor(toks),
            mask=mask,
            block_ids=req.block_ids,
            req_id=str(req.req_id),
        )
        store_ms = (time.perf_counter() - t_store0) * 1000
        transfer_stats = self._last_gpu_connector_transfer_stats()
        with self._lock:
            self._done_save.add(req.req_id)
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

    # -- per-step (RPC thread, post-forward): poll completions ------------
    def get_finished(self) -> KVConnectorOutput:
        # Offload uses extended completion states:
        # - finished_loading wakes successfully loaded requests.
        # - failed_loading wakes them for recompute using already allocated blocks.
        # - finished_saving releases blocks whose free was deferred during save.
        with self._lock:
            dl = set(self._done_load)
            fl = set(self._failed_load)
            ds = set(self._done_save)
            self._done_save.clear()
            self._done_load.clear()
            self._failed_load.clear()
        # The state tier's spill reports: not request-keyed, because the
        # request that owned a spilled checkpoint is gone by the time its bytes
        # land. Its *load* reports are, so they merge into the two channels a
        # KV load already uses -- which is what gives the state leg the
        # aggregator's per-request quorum for free. Downstream does not
        # distinguish the two legs and does not need to: the engine's
        # `settle_state_load` is a no-op for an id its state index never issued,
        # and a request that has both legs is held by `_joint_park` until both
        # have reported, so what leaves here is one event either way.
        if self._state_tier is not None:
            indexed, released, index_failed = self._state_tier.take_spill_reports()
            state_done, state_failed = self._state_tier.get_finished()
            dl, fl = self._settle_joint(dl, fl, state_done, state_failed)
        else:
            indexed, released, index_failed = set(), set(), set()
        return KVConnectorOutput(
            finished_sending=set(),
            finished_loading=dl,
            failed_loading=fl,
            finished_saving=ds,
            state_indexed=indexed,
            state_staging_released=released,
            state_index_failed=index_failed,
        )

    def _settle_joint(
        self,
        kv_done: set,
        kv_failed: set,
        state_done: set,
        state_failed: set,
    ) -> tuple[set, set]:
        """Merge the two report channels, holding back armed pairs.

        Everything not armed passes through exactly as it did before the joint
        load existed, which is the case that has to stay free: `waits_for` is
        asked first because `_JointPark._settle` ignores ids it never armed, and
        an ignored settle is indistinguishable from a leg that landed.

        Either leg failing fails the pair. Half a load leaves the state claiming
        a prefix whose KV never arrived, and `failed_loading` already means
        "wake for recompute over the blocks you hold", which is what that wants.
        """
        park = self._joint_park
        legs = (
            (park.settle_kv, kv_done, True),
            (park.settle_kv, kv_failed, False),
            (park.settle_state, state_done, True),
            (park.settle_state, state_failed, False),
        )
        passthrough_done: set = set()
        passthrough_failed: set = set()
        for settle, reports, ok in legs:
            for req_id in reports:
                if park.waits_for(req_id):
                    settle(req_id, ok)
                elif ok:
                    passthrough_done.add(req_id)
                else:
                    passthrough_failed.add(req_id)
        ready, ready_failed = park.take_ready()
        return passthrough_done | ready, passthrough_failed | ready_failed

    def get_finished_recv_blocks(self) -> list[int]:
        # Local CUDA copies are ordered by the copy stream + synchronize() before
        # we mark done; no RDMA-style GPU fence needed.
        return []


# =====================================================================
# Scheduler side
# =====================================================================
class LMCacheOffloadConnectorScheduler(KVConnectorSchedulerBase):
    # Consumer semantics: finished_recving wakes parked seqs (the engine asserts
    # `not is_producer` on that path). Offload never uses finished_sending.
    is_producer = False
    # Opt the scheduler into offload-wake (suffix prefill) instead of the P/D
    # decode-jump in Scheduler.schedule(); see Scheduler._is_offload_connector.
    is_offload = True

    def __init__(self, config) -> None:
        self._config = config
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = kvc.get("kv_role", "offload")
        self.block_size = int(config.kv_cache_block_size)
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
        self._save_inflight: set[str] = set()
        # sid -> when its save was handed to the worker. A save that never
        # reports is what wedged two full benchmark runs: `should_defer_free`
        # holds a finished request's blocks until its save completes, so when
        # LMCache stopped completing stores the KV pool drained and the
        # scheduler stopped producing batches altogether -- requests kept
        # arriving, nothing was ever scheduled again, and no log said why.
        self._save_inflight_since: dict[str, float] = {}
        # Seconds before the save path is called stalled. Not a transfer
        # deadline: a 4096-token store takes ~65ms, so anything past this is a
        # backend that has stopped, not one that is slow.
        self._save_stall_s = float(os.environ.get("OFFLOAD_SAVE_STALL_S", "120"))
        # Ceiling on requests whose blocks a save may pin at once. Each one is
        # a request that has finished and cannot be freed.
        self._save_max_inflight = int(
            os.environ.get("OFFLOAD_SAVE_MAX_INFLIGHT", "64")
        )
        self._save_stalled = False
        self._warned_save_stalled = False
        self._load_inflight_tokens: dict[str, int] = {}
        self._save_inflight_tokens: dict[str, int] = {}
        self._lookup_in_step: list[str] = []
        self._handoff_loads: set[str] = set()
        self.total_load_requests = 0
        self.total_loaded_tokens = 0
        self.total_load_failures = 0
        self.total_save_requests = 0
        self.total_saved_tokens = 0
        # Whether to store paged KV for a sequence that owns per-request state.
        # Default off: `_decide_load_after_alloc` refuses the load leg for those
        # sequences unconditionally, so the bytes have no reader inside this
        # engine. Set `OFFLOAD_SAVE_PER_REQ_CACHE=1` to restore the old
        # behaviour, which is only useful when a separate stateless consumer
        # shares this LMCache instance.
        self._save_per_req_cache = os.environ.get(
            "OFFLOAD_SAVE_PER_REQ_CACHE", "0"
        ).strip().lower() not in ("", "0", "false", "no", "off")
        # State offload tier loads admitted this pass, drained by
        # `build_connector_meta`. Not keyed by req_id: two requests may resume
        # off the same hash in one pass, each into its own slot.
        self._pending_state_loads: list[tuple] = []
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

        try:
            cfg = offcfg.build_lmcache_config(kvc)
            self.chunk_size = int(cfg.chunk_size)
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            world = int(getattr(config, "tensor_parallel_size", 1) or 1)
            meta = offcfg.build_lmcache_metadata(config, cfg, world, 0)
            self._lookup_client = LookupClientFactory.create_lookup_client(cfg, meta)
        except Exception as e:  # noqa: BLE001  # optional third-party client
            logger.warning(
                "LMCache offload scheduler: lookup client unavailable: %s", e
            )

        self._warn_if_per_req_cache_model(config)

    def _warn_if_per_req_cache_model(self, config) -> None:
        """Say once, at startup, that this model will decline every KV load.

        `_decide_load_after_alloc` refuses loads for a per-request-cache
        sequence, which for a hybrid model is every sequence. Without this the
        operator sees a permanent 0% load rate and reads it as a broken cache
        rather than a deliberate restriction. Once per server -- the refusal
        itself is logged per request at DEBUG, and a warning at that position
        would print on every prefill.

        The model-type set is imported rather than restated so the two cannot
        drift; the per-sequence check stays on `seq.has_per_req_cache`.
        """
        model_type = getattr(getattr(config, "hf_config", None), "model_type", None)
        if model_type is None:
            return
        from atom.model_engine.llm_engine import InputOutputProcessor

        if model_type not in InputOutputProcessor._per_req_cache_model_types():
            return
        logger.warning(
            "LMCache offload: model_type=%s keeps a per-request recurrent "
            "state (linear/compressor layers) alongside its paged KV. The "
            "state cannot be restored from a paged-KV load, so loading KV "
            "past the state boundary would run the linear layers over history "
            "their state never saw. Every KV LOAD is therefore declined for "
            "this model and a 0%% load rate is expected; KV SAVES still run "
            "normally and populate the tier. Set OFFLOAD_STATE to enable the "
            "state offload tier, which restores both together and lifts the "
            "restriction.",
            model_type,
        )

    # -- match: how many extra tokens can come from CPU/NVMe -------------
    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        if self._lookup_client is None:
            return 0, False
        num_prompt = seq.num_prompt_tokens
        token_ids = list(seq.token_ids[:num_prompt])
        sid = str(seq.id)
        if sid not in self._lookup_in_step:
            self._lookup_in_step.append(sid)
        try:
            hit = self._lookup_client.lookup(token_ids, lookup_id=sid)
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
        hit = int(hit)
        if hit == num_prompt:  # full-prompt hit → recompute last token
            hit -= 1
        self._hit_save_floors[sid] = self._chunk_floor(hit)
        need = hit - int(seq.num_cached_tokens)
        if need <= 0:
            if self._lookup_client is not None:
                try:
                    self._lookup_client.clear_lookup_status(sid)
                except Exception:  # best-effort cleanup
                    logger.debug(
                        "LMCache offload: clear lookup status failed for %s",
                        sid,
                        exc_info=True,
                    )
            return 0, False
        self._load_specs[sid] = LoadSpec(
            hbm_cached_tokens=int(seq.num_cached_tokens),
            lmcache_cached_tokens=hit,
            can_load=False,
        )
        return need, True  # True => park in WAITING_FOR_REMOTE_KVS

    def update_state_after_alloc(self, seq) -> None:
        sid = str(seq.id)
        ls = self._load_specs.get(sid)
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
        if not self._save_per_req_cache and getattr(seq, "has_per_req_cache", False):
            # Do not track this sequence for saving at all. Its load leg is
            # refused unconditionally by `_decide_load_after_alloc`, so the
            # bytes would have no reader in this engine -- and, more sharply, a
            # tracked entry that is never emitted keeps `_has_pending_save`
            # true forever, which makes `should_defer_free` hold this
            # request's blocks for good.
            return
        initial_saved = max(
            self._lmcache_hit_save_floor(ls),
            int(self._hit_save_floors.get(sid, 0)),
        )
        if sid not in self._save_tracker:
            self._save_tracker[sid] = [seq, initial_saved]
        else:
            self._save_tracker[sid][0] = seq
            self._save_tracker[sid][1] = max(
                int(self._save_tracker[sid][1]), initial_saved
            )

    def _chunk_floor(self, tokens: int) -> int:
        chunk = int(self.chunk_size or 256)
        return (max(0, int(tokens)) // chunk) * chunk

    def _lmcache_hit_save_floor(self, ls: LoadSpec | None) -> int:
        if ls is None:
            return 0
        return self._chunk_floor(ls.lmcache_cached_tokens)

    def _set_save_frontier(self, sid: str, seq, saved: int) -> None:
        saved = self._chunk_floor(saved)
        if sid not in self._save_tracker:
            self._save_tracker[sid] = [seq, saved]
        else:
            self._save_tracker[sid][0] = seq
            self._save_tracker[sid][1] = saved

    def _clear_pending_load(self, sid: str) -> None:
        self._load_specs.pop(sid, None)
        self._reqs_need_recv.pop(sid, None)
        self._handoff_loads.discard(sid)
        self._load_save_floors.pop(sid, None)
        self._hit_save_floors.pop(sid, None)
        if self._lookup_client is not None:
            try:
                self._lookup_client.clear_lookup_status(sid)
            except Exception:  # best-effort cleanup
                logger.debug(
                    "LMCache offload: clear lookup status failed for %s",
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
        # A hybrid model (GDN/KDA recurrent state, V4 compressor ring) carries
        # a per-request state that is the compressed history of exactly
        # `[0, hbm)` -- `BlockManager.allocate` shrank the HBM hit to a
        # boundary a state checkpoint covers, so right now the state boundary
        # and the KV-loaded length agree. Raising the KV-loaded length to `lmc`
        # breaks that: the scheduler would forward only `[lmc, num_prompt)`,
        # the linear layers would never see `[hbm, lmc)`, and at hbm == 0 the
        # freshly recycled state group makes `has_initial_state` True over
        # another request's leftovers. Silent wrong output, no exception.
        #
        # Refusing costs the hybrid nothing it could have had: ATOM runs one
        # forward over the whole batch, so the linear layers must walk
        # `[hbm, lmc)` token by token regardless of whether the full-attention
        # layers' KV is present. The load buys no work saving, only risk.
        # Saves are untouched -- this request's KV still populates the tier for
        # a stateless reader.
        #
        # Lifting the guard takes a boundary both legs are held to, and one
        # matcher has to pick it: the KV leg alone comes from LMCache's
        # `lookup()` floored to `chunk_size`, the state leg alone from
        # `BlockManager._gated_hit` -- a fixpoint over the state caches snapped
        # to a checkpoint rung and gated by `min_fork_tokens` -- and they agree
        # only by coincidence. So `can_allocate` picks B for both
        # (`_joint_kv_boundary`) and this clamps L down to it. Without a B the
        # refusal stands, which is also the whole behaviour with
        # `OFFLOAD_STATE_JOINT_KV` off.
        if getattr(seq, "has_per_req_cache", False):
            joint = int(getattr(seq, "state_joint_boundary_tokens", 0) or 0)
            if joint <= hbm:
                return False, "per_req_cache_state_boundary", hbm, lmc, need, chunk
            if joint > lmc:
                # The engine aimed past what this connector's lookup found.
                # Nothing else may run: the state leg is already aimed at B and
                # a shorter KV load would leave it claiming history the forward
                # never sees.
                return False, "joint_boundary_above_lookup", hbm, lmc, need, chunk
            if hbm % chunk != 0:
                # The KV leg moves whole chunks, and the blocks below `hbm` are
                # shared HBM cache blocks another request may be reading, so
                # rounding the start down is not available. `can_allocate` is
                # supposed to have declined the joint boundary for this; the
                # scheduler disowns it if we still get here.
                return False, "joint_unaligned_hbm_prefill", hbm, lmc, need, chunk
            # Transfer to the chunk that covers the boundary, claim only the
            # boundary (`_claim_after_load`). The engine picked both numbers.
            kv_target = int(getattr(seq, "state_joint_kv_tokens", 0) or 0) or joint
            if kv_target > lmc:
                return False, "joint_boundary_above_lookup", hbm, lmc, need, chunk
            lmc = kv_target
            ls.lmcache_cached_tokens = lmc
            need = lmc - hbm
            # Deliberately past the `min_load` floor below: the boundary was
            # chosen for both legs together, and refusing this one on size would
            # leave the state leg claiming a prefix whose KV never came.
            return True, "joint_state_and_kv", hbm, lmc, need, chunk
        if lmc <= hbm:
            return False, "hbm_satisfies_after_alloc", hbm, lmc, need, chunk
        if hbm % chunk != 0:
            return False, "unaligned_hbm_prefill", hbm, lmc, need, chunk
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        if need < min_load:
            return False, "too_small", hbm, lmc, need, chunk
        return True, "aligned_large_hit", hbm, lmc, need, chunk

    def _claim_after_load(self, seq, hbm: int, lmc: int) -> int:
        """How far the request may call itself cached once the load lands.

        Normally the whole loaded prefix. For a joint load it is the *state*
        boundary, which sits at or below the transfer's end: the KV leg is aimed
        at the LMCache chunk covering that boundary, and claiming the rounded-up
        figure would have the forward skip tokens the recurrent state does not
        cover -- wrong output, and silent, since nothing downstream re-derives
        the state's reach.
        """
        joint = int(getattr(seq, "state_joint_boundary_tokens", 0) or 0)
        if joint:
            return max(hbm, min(joint, lmc))
        return max(hbm, lmc)

    def _maybe_start_unaligned_handoff(
        self,
        seq,
        ls: LoadSpec,
        hbm: int,
        lmc: int,
        chunk: int,
    ) -> bool:
        boundary = ((hbm + chunk - 1) // chunk) * chunk
        remaining_after_boundary = lmc - boundary
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        if boundary <= hbm or remaining_after_boundary < min_load:
            return False

        sid = str(seq.id)
        ls.hbm_cached_tokens = boundary
        ls.can_load = True
        self._reqs_need_recv.pop(sid, None)
        self._handoff_loads.add(sid)
        seq.offload_loaded_tokens = hbm
        seq.offload_handoff_boundary_tokens = boundary
        logger.debug(
            "[OFFLOAD-LOAD-HANDOFF] seq=%s hbm_cached=%d boundary=%d "
            "lmc_cached=%d need_after_boundary=%d min_load=%d chunk=%d",
            seq.id,
            hbm,
            boundary,
            lmc,
            remaining_after_boundary,
            min_load,
            chunk,
        )
        return True

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

    def should_park_partial_prefill_for_load(self, seq) -> bool:
        sid = str(seq.id)
        if sid not in self._handoff_loads:
            return False
        ls = self._load_specs.get(sid)
        if ls is None:
            self._handoff_loads.discard(sid)
            return False
        boundary = int(getattr(seq, "offload_handoff_boundary_tokens", 0) or 0)
        hbm = int(getattr(seq, "num_cached_tokens", 0))
        if boundary > 0 and hbm < boundary:
            return False

        should_load, reason, hbm, lmc, need, chunk = self._decide_load_after_alloc(
            seq, ls
        )
        if not should_load:
            self._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
            self._clear_pending_load(sid)
            return False

        ls.can_load = True
        self._reqs_need_recv[sid] = seq
        self._handoff_loads.discard(sid)
        seq.offload_loaded_tokens = self._claim_after_load(seq, hbm, lmc)
        logger.debug(
            "[OFFLOAD-LOAD-HANDOFF-READY] seq=%s hbm_cached=%d "
            "lmc_cached=%d offload_loaded=%d need=%d",
            seq.id,
            hbm,
            lmc,
            seq.offload_loaded_tokens,
            need,
        )
        return True

    def _mark_load_skip(
        self,
        seq,
        reason: str,
        hbm: int,
        lmc: int,
        need: int,
        chunk: int,
    ) -> None:
        seq.offload_loaded_tokens = hbm
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        logger.debug(
            "[OFFLOAD-LOAD-SKIP] seq=%s hbm_cached=%d lmc_cached=%d "
            "need=%d min_load=%d chunk=%d reason=%s",
            seq.id,
            hbm,
            lmc,
            need,
            min_load,
            chunk,
            reason,
        )

    def should_park_for_load_after_alloc(self, seq) -> bool:
        sid = str(seq.id)
        ls = self._load_specs.get(sid)
        if ls is None:
            return False
        should_load, reason, hbm, lmc, need, chunk = self._decide_load_after_alloc(
            seq, ls
        )
        if not should_load:
            if (
                reason == "unaligned_hbm_prefill"
                and self._maybe_start_unaligned_handoff(seq, ls, hbm, lmc, chunk)
            ):
                return False
            self._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
            self._clear_pending_load(sid)
            return False
        seq.offload_loaded_tokens = self._claim_after_load(seq, hbm, lmc)
        return True

    def enqueue_state_loads(self, loads) -> bool:
        """Take this pass's state-tier loads, for the next `build_connector_meta`.

        Returns whether they were taken. Always True here; the answer exists
        for `MultiConnectorScheduler`, which can be asked when no sub-connector
        carries state loads at all. The caller must fail a refusal rather than
        drop it -- every one of these requests is parked on a report.

        `(req_id, state_hash, target_group)`, already decided by the engine:
        `BlockManager._attach_state_group` chose the group and
        `StateOffloadIndex.request_load` vouched for the hash. Nothing is
        re-decided here -- unlike the KV leg, whose `LoadSpec` is re-checked at
        build time because `num_cached_tokens` moves under it between match and
        allocate. A state load is decided *after* allocate, against a group
        this request already holds, so there is nothing left to move.
        """
        self._pending_state_loads.extend(loads)
        return True

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        meta = LMCacheOffloadMetadata()
        # Drained, not copied: every load handed over is submitted exactly once,
        # and a second submission would write the same entry into a group the
        # first transfer is already filling.
        meta.state_loads = self._pending_state_loads
        self._pending_state_loads = []

        # Loads
        logger.debug("[OFFLOAD-BUILD] reqs_need_recv=%d", len(self._reqs_need_recv))
        loading_sids: set[str] = set()
        for sid, seq in list(self._reqs_need_recv.items()):
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
            # Persist the physical load start on the sequence. The scheduler
            # combines it with offload_loaded_tokens after all TP workers
            # succeed to publish the restored GPU prefix.
            seq.offload_load_start_tokens = hbm
            seq.offload_loaded_tokens = self._claim_after_load(seq, hbm, lmc)
            # req_id MUST be the raw seq.id (the type the scheduler compares
            # against in _update_waiting_for_remote_kv); str(seq.id) is only for
            # LMCache's lookup/pin API. A str here silently never wakes the seq.
            logger.debug(
                "[OFFLOAD-LOAD-EMIT] seq=%s hbm_cached=%d lmc_cached=%d "
                "offload_loaded=%d need=%d min_load=%d nblocks=%d reason=%s",
                seq.id,
                hbm,
                lmc,
                seq.offload_loaded_tokens,
                need,
                int(getattr(self, "_min_load_tokens", 8192)),
                len(list(seq.block_table)),
                reason,
            )
            loading_sids.add(sid)
            self._load_save_floors[sid] = self._chunk_floor(hbm)
            self._load_inflight_tokens[sid] = max(0, lmc - hbm)
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:lmc]),
                    block_ids=list(seq.block_table),
                    load_spec=ls,
                )
            )
        meta.lookup_requests_in_step = [
            sid for sid in self._lookup_in_step if sid not in self._handoff_loads
        ]
        self._lookup_in_step = [
            sid for sid in self._lookup_in_step if sid in self._handoff_loads
        ]
        # Saves: store fully computed prompt chunks. Under scheduler-side
        # chunked prefill, seq.num_cached_tokens advances after each prefill
        # chunk's forward has completed; use it as the D2H-safe frontier.
        chunk = self.chunk_size or 256
        self._refresh_save_stall()
        for sid, entry in self._save_tracker.items():
            seq, saved = entry
            if self._save_stalled:
                break  # nothing new goes out while the backend is not draining
            if len(self._save_inflight) >= self._save_max_inflight:
                break  # do not pin another finished request's blocks
            if sid in self._reqs_need_recv or sid in loading_sids:
                continue  # loading this step; defer its save
            if sid in self._save_inflight:
                continue  # keep at most one save per request in flight
            computed = min(
                int(getattr(seq, "num_cached_tokens", 0)),
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
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:aligned]),
                    block_ids=list(seq.block_table),
                    save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
                    is_last_prefill=is_last_prefill,
                )
            )
            entry[1] = aligned
            self._save_inflight.add(sid)
            self._save_inflight_since[sid] = time.monotonic()
            self._save_inflight_tokens[sid] = max(0, aligned - int(saved))
        self._reqs_need_recv.clear()
        return meta

    def _refresh_save_stall(self) -> None:
        """Decide whether the save path has stopped draining.

        One number decides it: how long the oldest in-flight save has been out.
        A 4096-token store costs ~65ms, so a save outstanding for minutes is a
        backend that stopped, not one that is busy -- and while it is out,
        every finished request behind it keeps its blocks.
        """
        if not self._save_inflight_since:
            if self._save_stalled:
                logger.info("LMCache offload: save path draining again")
            self._save_stalled = False
            self._warned_save_stalled = False
            return
        oldest = min(self._save_inflight_since.values())
        self._save_stalled = (time.monotonic() - oldest) > self._save_stall_s
        if self._save_stalled and not self._warned_save_stalled:
            self._warned_save_stalled = True
            logger.warning(
                "LMCache offload: no save has completed in %.0fs (%d in flight). "
                "Emitting no further saves and releasing the blocks of requests "
                "whose save was never sent; the ones already handed to the "
                "backend still have to wait, because it is reading those blocks.",
                time.monotonic() - oldest,
                len(self._save_inflight),
            )

    def _save_frontier(self, seq) -> int:
        computed = min(
            int(getattr(seq, "num_cached_tokens", 0)),
            int(getattr(seq, "num_prompt_tokens", 0)),
        )
        return self._chunk_floor(computed)

    def _has_pending_save(self, seq) -> bool:
        sid = str(seq.id)
        entry = self._save_tracker.get(sid)
        if entry is None:
            return False
        return self._save_frontier(seq) > int(entry[1])

    def should_defer_free(self, seq) -> bool:
        sid = str(seq.id)
        if sid in self._save_inflight:
            # The backend is reading these blocks right now. Freeing them would
            # let the next request write into them mid-transfer and index the
            # result under this prefix's hash -- corrupt bytes that look valid
            # to every later load. Waiting is the only safe answer, which is why
            # the emission side caps how many requests can be in this state.
            return True
        if not self._has_pending_save(seq):
            return False
        if self._save_stalled:
            # This save was never handed out, so nothing is reading these
            # blocks and dropping it costs only the store. Holding them is what
            # turned a stopped backend into a stopped engine.
            self._save_tracker.pop(sid, None)
            return False
        return True

    def save_finished(self, req_id) -> None:
        sid = str(req_id)
        self._save_inflight.discard(sid)
        self._save_inflight_since.pop(sid, None)
        saved_tokens = self._save_inflight_tokens.pop(sid, 0)
        self.total_save_requests += 1
        self.total_saved_tokens += saved_tokens

    def load_finished(self, req_id) -> None:
        sid = str(req_id)
        loaded_tokens = self._load_inflight_tokens.pop(sid, 0)
        self.total_load_requests += 1
        self.total_loaded_tokens += loaded_tokens

    def load_failed(self, req_id) -> None:
        sid = str(req_id)
        self._load_inflight_tokens.pop(sid, None)
        self.total_load_failures += 1
        floor = self._load_save_floors.get(sid)
        entry = self._save_tracker.get(sid)
        if floor is not None and entry is not None:
            # The LMCache hit was not actually loaded. Let the recomputed
            # [HBM, LMC) chunks be saved again instead of permanently treating
            # them as already persisted.
            entry[1] = self._chunk_floor(floor)
        self._clear_pending_load(sid)

    def request_finished(self, seq) -> None:
        sid = str(seq.id)
        self._clear_pending_load(sid)
        self._load_inflight_tokens.pop(sid, None)
        if not self.should_defer_free(seq):
            self._save_inflight_tokens.pop(sid, None)
            self._save_tracker.pop(sid, None)

    def get_statistics(self) -> dict[str, int]:
        """Return cumulative and queue-depth stats without worker RPCs."""
        return {
            "load_requests": self.total_load_requests,
            "loaded_tokens": self.total_loaded_tokens,
            "load_failures": self.total_load_failures,
            "save_requests": self.total_save_requests,
            "saved_tokens": self.total_saved_tokens,
            "loads_pending": len(self._load_inflight_tokens),
            "saves_pending": len(self._save_inflight_tokens),
        }
