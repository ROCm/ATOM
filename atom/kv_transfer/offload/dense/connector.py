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
import time

import torch

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import (
    LoadOperationId,
    SaveCompletionId,
    SaveOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload._offload_common import (
    OffloadSchedulerMixin,
    OffloadWorkerMixin,
    build_offload_engine,
    pp_aware_rank_and_world,
    validated_kv_role,
)
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)

logger = logging.getLogger("atom")


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

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        from aiter.dist.parallel_state import get_tp_group

        tp = get_tp_group()
        rank, world = pp_aware_rank_and_world(self._config, tp)
        self._rank = rank

        # num_blocks is the physical block count (num_physical_kvcache_blocks),
        # threaded from the model runner. MLA stores its KV token-major, so the
        # codec can't infer the block count from tensor.shape[0]; pass it.
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
                # Producer fence for the async KV save. The save gathers KV
                # pages on a private `pack_stream` (kv_byte_codec.gpu_to_chunk_
                # major_device_buffer) with no ordering against the forward's
                # compute stream. The shared staging class documents that the KV
                # caller -- this connector -- must record this fence
                # (hybrid/kimi_k3/staging.py:43-59; hybrid/dsv4/connector.py is
                # the only other caller that does). Record it here on the RPC
                # thread, BEFORE this step's forward, so the event captures the
                # prior forwards that wrote this prefix's [0, boundary) KV; the
                # save worker waits it before the gather. Without it the gather
                # reads torn/partially-written latent that is then faithfully
                # reloaded -- silent, reload-count-scaling accuracy corruption.
                if torch.cuda.is_available():
                    ev = torch.cuda.Event()
                    ev.record(torch.cuda.current_stream())
                    req.producer_event = ev
                self._save_executor.submit(self._guard, "save", self._do_save_req, req)

    # -- copy daemon thread ----------------------------------------------
    def _do_load_req(self, req: LMCacheReqMeta) -> None:
        """Fetch this request's KV, then report it. Composition only.

        Split so a hybrid layout can put a second leg between the two halves
        and still emit exactly ONE completion for the pair. Behaviour for dense
        is unchanged: same completion ids, same sets, same profiling record.
        """
        self._finish_load(req, self._load_kv_bytes(req))

    def _load_kv_bytes(self, req: LMCacheReqMeta) -> bool:
        """Move the KV bytes. Reports nothing -- that is `_finish_load`.

        Returns True when the request's KV is where it should be, including
        the no-op case where HBM already covers the lookup.
        """
        ls = req.load_spec
        assert ls is not None
        hbm = int(ls.hbm_cached_tokens)
        lmc = int(ls.lmcache_cached_tokens)
        toks = req.token_ids[:lmc]
        t_total0 = time.perf_counter()
        if lmc <= hbm:
            # Nothing to fetch: HBM already covers the lookup. Still a
            # success -- the request has the KV it asked for.
            return True
        chunk_size = int(self.chunk_size or 256)
        if hbm % chunk_size != 0:
            logger.warning(
                "LMCache offload: HBM prefix is not chunk-aligned req=%s "
                "hbm=%d chunk=%d; re-prefill",
                req.req_id,
                hbm,
                chunk_size,
            )
            return False

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
        loaded = bool(ret_mask[hbm:lmc].all().item())
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
        return loaded

    def _finish_load(self, req: LMCacheReqMeta, ok: bool) -> None:
        """Release the lookup pin and emit this request's ONE load completion.

        Every path through `_do_load_req` ends here, including a raise in a
        subclass's second leg, so one dispatch produces exactly one report.
        """
        self._lookup_unpin(req.req_id)
        with self._lock:
            if ok:
                self._done_load.add(self._load_completion_id(req))
            else:
                self._failed_load.add(self._load_completion_id(req))

    def _do_save_req(self, req: LMCacheReqMeta) -> None:
        ss = req.save_spec
        assert ss is not None
        toks = req.token_ids
        if not req.is_last_prefill:
            toks = toks[: (len(toks) // self.chunk_size) * self.chunk_size]
        skip = (ss.skip_leading_tokens // self.chunk_size) * self.chunk_size
        if skip >= len(toks):
            with self._lock:
                self._done_save.add(self._save_completion_id(req))
            return

        t_total0 = time.perf_counter()
        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:skip] = False

        # Wait for the forward that produced these KV pages before the async
        # gather reads them (fence recorded in `start_load_kv`). This runs on the
        # off-TTFT save worker, so the host sync costs no serving latency; it is
        # what makes the gather's source quiescent, the guarantee the staging
        # class relies on but does not itself provide.
        producer_event = getattr(req, "producer_event", None)
        if producer_event is not None:
            producer_event.synchronize()

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
            self._done_save.add(self._save_completion_id(req))
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
        self._save_nonce = 0
        self._load_nonce = 0
        self._load_lifecycles: dict[str, object] = {}
        self._active_load_operations: dict[str, tuple[object, LoadOperationId]] = {}
        self._lookup_in_step: list[str] = []
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
        try:
            hit = self._lookup_client.lookup(token_ids, lookup_id=str(seq.id))
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
            if self._lookup_client is not None:
                try:
                    self._lookup_client.clear_lookup_status(sid)
                except Exception:
                    logger.debug(
                        "LMCache offload: lookup status cleanup failed for req=%s",
                        sid,
                        exc_info=True,
                    )
            return 0, False
        self._lookup_in_step.append(sid)
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
            if entry is None or entry[0] is not seq:
                self._save_tracker[sid] = [seq, initial_saved]
            else:
                entry[1] = max(int(entry[1]), initial_saved)

    def _clear_pending_load(self, sid: str) -> None:
        self._load_specs.pop(sid, None)
        self._reqs_need_recv.pop(sid, None)
        self._handoff_loads.discard(sid)
        self._load_save_floors.pop(sid, None)
        self._hit_save_floors.pop(sid, None)
        self._lookup_in_step = [
            req_id for req_id in self._lookup_in_step if req_id != sid
        ]
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
        """Seam for a subclass to bound how many saves may be outstanding.

        Unbounded here. It matters for a layout whose `should_defer_free` pins
        a finished request's blocks until its save drains -- an unbounded queue
        then lets a slow backend hold an unbounded slice of the pool.
        """
        return True

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        meta = LMCacheOffloadMetadata()

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
        meta.lookup_requests_in_step = list(self._lookup_in_step)
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
            if not self._may_emit_save():
                break
            seq, saved = entry
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
            save_operation = SaveOperationId(seq.id, self._save_nonce)
            self._save_nonce += 1
            self._track_save_statistics(save_operation, aligned - saved)
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:aligned]),
                    block_ids=list(seq.block_table),
                    save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
                    is_last_prefill=is_last_prefill,
                    save_operation=save_operation,
                )
            )
            entry[1] = aligned
            self._save_inflight[sid] = save_operation
            self._save_rr_last = sid
        dispatched = set(meta.lookup_requests_in_step)
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
        sid = str(seq.id)
        return sid in self._save_inflight or self._has_pending_save(seq)

    def release_stalled_save(self, seq) -> None:
        """Drop bookkeeping for a stall-escaped save the scheduler is freeing.

        No-op on dense: its `should_defer_free` has no stall escape, so a request
        with a pending save always defers and is never preemptable. K3 overrides
        this to pop its `_save_tracker`. Defined here so every offload impl
        answers the scheduler's `release_stalled_save` forward uniformly.
        """

    def has_pending_work(self) -> bool:
        """True while a load still needs dispatch or a save is unreported.

        Feeds ``EngineCore.has_pending_kv_work()``, so it reads only state
        that clears itself: ``_reqs_need_recv`` is emptied by every
        ``build_connector_meta`` and ``_save_inflight`` by ``save_finished``
        (or ``abandon_save`` when the scheduler reclaims a stalled save).
        Saves that are queued but not yet dispatched are covered there by the
        scheduler's ``deferred_free_blocks``, which ``should_defer_free``
        keeps populated for exactly those requests.
        """
        return bool(self._reqs_need_recv) or bool(self._save_inflight)

    def save_finished(self, req_id) -> None:
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

    def abandon_save(self, req_id) -> None:
        """Force-drop a save the scheduler reclaimed after it stalled.

        `save_finished` is the completion path: it matches the exact
        `SaveOperationId` and, handed a raw request id while an exact generation
        is parked, deliberately refuses -- a delayed TP notification must not
        complete a newer lifecycle. Reclamation is the opposite need. The
        scheduler's `_reconcile_stalled_deferred_saves` has already freed the
        blocks of a save the backend never reported (LMCache force-unpins a
        stalled save without a completion) and holds only the raw request id, so
        drop the entry unconditionally. Without this the entry lingers,
        `should_defer_free` stays True and `has_pending_work` never clears, and
        the engine busy-loops with every GPU idle. Not a completion: the bytes
        were never persisted, so the statistics are *cancelled*, not finished,
        and the tracker entry is dropped so the save loop cannot re-emit it
        against freed blocks.
        """
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        operation = self._save_inflight.pop(sid, None)
        if operation is not None:
            self._cancel_save_statistics(operation)
        self._save_tracker.pop(sid, None)

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
        if floor is not None and entry is not None:
            # The LMCache hit was not actually loaded. Let the recomputed
            # [HBM, LMC) chunks be saved again instead of permanently treating
            # them as already persisted.
            entry[1] = self._chunk_floor(floor)
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
        if entry is not None and entry[0] is seq and not self.should_defer_free(seq):
            self._save_tracker.pop(sid, None)
        if hasattr(seq, "_load_operation"):
            delattr(seq, "_load_operation")
