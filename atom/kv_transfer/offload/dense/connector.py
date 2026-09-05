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
import time

import torch

from atom.kv_transfer.disaggregation.base import KVConnectorBase
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload._offload_common import (
    OffloadWorkerMixin,
    build_offload_engine,
    pp_aware_rank_and_world,
    validated_kv_role,
)
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
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
                self._save_executor.submit(self._guard, "save", self._do_save_req, req)

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
            with self._lock:
                self._done_save.add(self._save_completion_id(req))
            return

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
class DenseOffloadScheduler(ChunkedOffloadSchedulerBase):
    """Scheduler for the legacy in-process dense LMCache transport."""

    def __init__(self, config) -> None:
        kvc = getattr(config, "kv_transfer_config", {}) or {}

        # Preserve the legacy constructor's fail-fast ordering. Configuration
        # metadata needs model fields that lightweight validation callers may
        # intentionally omit, so reject the role and block geometry first.
        validated_kv_role(kvc)
        offcfg._strict_integer(
            "Dense block size",
            config.kv_cache_block_size,
            minimum=1,
        )

        # Configuration is required even though the lookup service is optional.
        # Do not turn invalid storage or geometry into a cache miss at startup.
        cfg = offcfg.build_lmcache_config(kvc)
        chunk_size = offcfg._strict_integer(
            "LMCache chunk size",
            cfg.chunk_size,
            minimum=1,
        )
        world = offcfg.lmcache_replica_world_size(config)
        meta = offcfg.build_lmcache_metadata(config, cfg, world, 0)
        lookup_client = None
        try:
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            lookup_client = LookupClientFactory.create_lookup_client(cfg, meta)
            logger.info(
                "LMCache offload scheduler: lookup client on %s (world=%d)",
                meta.engine_id,
                world,
            )
        except Exception as e:  # noqa: BLE001  # optional lookup service
            logger.warning(
                "LMCache offload scheduler: lookup client unavailable: %s", e
            )

        super().__init__(
            config,
            chunk_size=chunk_size,
            lookup_client=lookup_client,
        )
