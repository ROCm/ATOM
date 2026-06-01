# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM standalone LMCache CPU/NVMe KV-offload connector.

Design (see ../../../../PLAN_impl_lmcache_offload_v5.md + 005 LEARN notes):

* **Reuse real LMCache as a storage tier only** — per-rank ``LMCacheEngine`` for its
  ``StorageManager`` (CPU LRU + NVMe L3) + ``ChunkedTokenDatabase`` (chunk-256 keys).
  We bypass ``engine.store/retrieve`` (its token-major GPU path can't represent ATOM's
  x-packed KV storage layout — ``K=(nb,H,D//x,bs,x)``, see ``ATOMKVByteCodec`` docstring;
  loosely "swizzle", but a persistent storage layout, not LDS bank-swizzle) and instead
  move **opaque per-block bytes** via :class:`ATOMKVByteCodec` into pinned
  ``KV_2LTD``-as-uint8 ``MemoryObj``s.
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
from atom.kv_transfer.offload.gpu_connector import ATOMKVByteCodec
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.kv_transfer.offload.trace import offload_trace

logger = logging.getLogger("atom")


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class _UnusedGPUConnector:
    """Satisfies LMCacheEngineBuilder.get_or_create; never invoked (we do our own
    byte-copy and never call engine.store/retrieve)."""

    def to_gpu(self, *a, **k):
        raise NotImplementedError

    def from_gpu(self, *a, **k):
        raise NotImplementedError

    def batched_from_gpu(self, *a, **k):
        raise NotImplementedError

    def batched_to_gpu(self, *a, **k):
        raise NotImplementedError

    def get_shape(self, num_tokens):
        return torch.Size((num_tokens,))


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
        # Each worker thread gets its OWN CUDA stream (disjoint block_ids -> no
        # write conflict). OFFLOAD_COPY_WORKERS tunes the SAVE pool only.
        n_save_workers = int(os.environ.get("OFFLOAD_COPY_WORKERS", "1"))
        self._load_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lmc-offload-load"
        )
        self._save_executor = ThreadPoolExecutor(
            max_workers=n_save_workers, thread_name_prefix="lmc-offload-save"
        )
        self._tls = threading.local()  # per-thread copy stream
        self._lock = threading.Lock()
        self._done_load: set[ReqId] = set()
        self._done_save: set[ReqId] = set()
        self._failed_load: set[ReqId] = set()
        self._load_active = threading.Event()
        self._request_fastpath = os.environ.get(
            "OFFLOAD_REQUEST_FASTPATH", "1"
        ).lower() not in ("0", "false", "no", "off")

        self._engine = None
        self._sm = None
        self._tdb = None
        self._codec: ATOMKVByteCodec | None = None
        self._lookup_server = None

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict, transfer_tensors=None) -> None:
        from aiter.dist.parallel_state import get_tp_group
        from lmcache.v1.cache_engine import LMCacheEngineBuilder

        tp = get_tp_group()
        rank, world = tp.rank_in_group, tp.world_size
        self._rank = rank

        cfg = offcfg.build_lmcache_config()
        offcfg.apply_extra_overrides(cfg, getattr(self._config, "kv_transfer_config", None))
        meta = offcfg.build_lmcache_metadata(self._config, cfg, world, rank)
        self.chunk_size = int(cfg.chunk_size)

        self._engine = LMCacheEngineBuilder.get_or_create(
            f"atom-offload-{rank}", cfg, meta, _UnusedGPUConnector(),
            lambda t, s: None, lambda o, s: o,
        )
        self._engine.post_init()
        self._sm = self._engine.storage_manager
        self._tdb = self._engine.token_database
        self._codec = ATOMKVByteCodec(kv_caches)

        # DEBUG: wrap engine.lookup to capture EVERY call (incl. the ones the ZMQ
        # lookup_server makes on behalf of the scheduler) — args + result.
        _orig_lookup = self._engine.lookup
        _rk = rank
        def _logged_lookup(*a, **k):
            r = _orig_lookup(*a, **k)
            h = k.get("hashes")
            logger.debug("[ENGINE.LOOKUP] rank=%s lookup_id=%s nhashes=%s first3=%s -> %s",
                         _rk, k.get("lookup_id"), (len(h) if h is not None else None),
                         (list(h[:3]) if h else None), r)
            return r
        self._engine.lookup = _logged_lookup

        # ZMQ lookup server so the scheduler process can query our hit counts.
        try:
            from lmcache.v1.lookup_client.factory import LookupClientFactory
            self._lookup_server = LookupClientFactory.create_lookup_server(
                self._engine, meta
            )
        except Exception as e:  # lookup server optional for save-only smoke
            logger.warning("LMCache offload: lookup server not started: %s", e)

        logger.info(
            "LMCache offload worker rank=%d: bytes_per_block=%d chunk=%d "
            "codec_layout=%s save=%s load=%s",
            rank, self._codec.bytes_per_block, self.chunk_size, self._codec.layout,
            self._do_save, self._do_load,
        )

    # -- per-step (RPC thread): only enqueue, never copy ------------------
    def start_load_kv(self, metadata) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        for req in metadata.requests:
            if req.load_spec is not None and self._do_load:
                offload_trace(
                    "worker_load_enqueue",
                    rank=getattr(self, "_rank", "?"),
                    req=req.req_id,
                    hbm=req.load_spec.hbm_cached_tokens,
                    lmc=req.load_spec.lmcache_cached_tokens,
                    blocks=len(req.block_ids),
                )
                self._load_executor.submit(self._guard, "load", self._do_load_req, req)
            if req.save_spec is not None and self._do_save:
                offload_trace(
                    "worker_save_enqueue",
                    rank=getattr(self, "_rank", "?"),
                    req=req.req_id,
                    skip=req.save_spec.skip_leading_tokens,
                    toks=len(req.token_ids),
                    blocks=len(req.block_ids),
                )
                self._save_executor.submit(self._guard, "save", self._do_save_req, req)

    def _guard(self, kind: str, fn, req) -> None:
        load_active = getattr(self, "_load_active", None)
        if kind == "load" and load_active is None:
            load_active = threading.Event()
            self._load_active = load_active
        if kind == "load":
            load_active.set()
        try:
            fn(req)
        except Exception:
            logger.exception("LMCache offload: %s failed for %s", fn.__name__, req.req_id)
            if kind == "load":
                self._lookup_unpin(req.req_id)
            with self._lock:
                if kind == "load":
                    self._failed_load.add(req.req_id)
                else:
                    # A failed save should not keep blocks pinned forever. The
                    # request simply loses this offload opportunity.
                    self._done_save.add(req.req_id)
        finally:
            if kind == "load":
                load_active.clear()

    def _lookup_unpin(self, req_id) -> None:
        if getattr(self, "_engine", None) is None:
            return
        try:
            self._engine.lookup_unpin([str(req_id)])  # LMCache pin keyed by str id
        except Exception:
            pass

    def _copy_device(self) -> torch.device | None:
        codec = getattr(self, "_codec", None)
        device = getattr(codec, "device", None)
        if device is None:
            return None
        device = torch.device(device)
        if device.type != "cuda":
            return None
        return device

    def _stream(self) -> torch.cuda.Stream:
        """A CUDA stream owned by the calling copy-daemon thread and device."""
        device = self._copy_device()
        key = str(device) if device is not None else "default"
        streams = getattr(self._tls, "streams", None)
        if streams is None:
            streams = {}
            self._tls.streams = streams
        s = streams.get(key)
        if s is None:
            if device is None:
                s = torch.cuda.Stream()
            else:
                with torch.cuda.device(device):
                    s = torch.cuda.Stream()
            streams[key] = s
        return s

    def _host_tmp(self, nbytes: int) -> torch.Tensor:
        """Pinned CPU scratch buffer owned by the calling copy-daemon thread."""
        buf = getattr(self._tls, "host_tmp", None)
        if buf is None or int(buf.numel()) < int(nbytes):
            try:
                buf = torch.empty((int(nbytes),), dtype=torch.uint8, pin_memory=True)
            except RuntimeError:
                logger.warning(
                    "LMCache offload: pinned host scratch allocation failed; "
                    "falling back to pageable CPU memory",
                    exc_info=True,
                )
                buf = torch.empty((int(nbytes),), dtype=torch.uint8)
            self._tls.host_tmp = buf
        return buf[: int(nbytes)]

    def _pause_save_for_load(self, stream: torch.cuda.Stream) -> None:
        """Let critical-path loads drain before fire-and-forget save copies."""
        load_active = getattr(self, "_load_active", None)
        if load_active is None or not load_active.is_set():
            return
        stream.synchronize()
        while load_active.is_set():
            time.sleep(0.001)

    def _block_ids(self, req: LMCacheReqMeta, start: int, end: int) -> list[int]:
        return req.block_ids[start // self.block_size : _cdiv(end, self.block_size)]

    def _profile_enabled(self) -> bool:
        return os.environ.get("OFFLOAD_PROFILE", "1").lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def _request_fastpath_enabled(self) -> bool:
        return (
            bool(getattr(self, "_request_fastpath", False))
            and self._codec is not None
            and self._codec.layout == "segment_indexed"
        )

    def _request_level_key(self, chunks, token_count: int):
        """Synthetic key for a whole-prefix segment-major object.

        The normal LMCache chunk keys remain authoritative for lookup. This key
        is an optional per-rank fast path for exact full-prefix reloads: it uses
        the last chunk's prefix hash plus tags, so it cannot collide with normal
        chunk entries and stays stable across scheduler/worker processes.
        """
        if not chunks:
            return None
        key = chunks[-1][2]
        request_configs = dict(getattr(key, "request_configs", None) or {})
        request_configs["lmcache.tag.atom_offload"] = "request"
        request_configs["lmcache.tag.atom_offload_tokens"] = str(int(token_count))
        request_configs["lmcache.tag.atom_offload_layout"] = str(
            getattr(self._codec, "layout", "unknown")
        )
        return key.__class__(
            model_name=key.model_name,
            world_size=key.world_size,
            worker_id=key.worker_id,
            chunk_hash=key.chunk_hash,
            dtype=key.dtype,
            request_configs=request_configs,
        )

    # -- copy daemon thread ----------------------------------------------
    def _do_load_req(self, req: LMCacheReqMeta) -> None:
        ls = req.load_spec
        assert ls is not None
        hbm = int(ls.hbm_cached_tokens)
        toks = req.token_ids[: ls.lmcache_cached_tokens]
        t_total0 = time.perf_counter()
        offload_trace(
            "worker_load_start",
            rank=getattr(self, "_rank", "?"),
            req=req.req_id,
            hbm=hbm,
            lmc=ls.lmcache_cached_tokens,
            toks=len(toks),
            blocks=len(req.block_ids),
        )
        if int(ls.lmcache_cached_tokens) <= hbm:
            self._lookup_unpin(req.req_id)
            with self._lock:
                self._done_load.add(req.req_id)
            offload_trace(
                "worker_load_done",
                rank=getattr(self, "_rank", "?"),
                req=req.req_id,
                status="hbm_only",
                total_ms=f"{(time.perf_counter() - t_total0) * 1000:.2f}",
            )
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
            offload_trace(
                "worker_load_done",
                rank=getattr(self, "_rank", "?"),
                req=req.req_id,
                status="unaligned_hbm",
                hbm=hbm,
                chunk=chunk_size,
                total_ms=f"{(time.perf_counter() - t_total0) * 1000:.2f}",
            )
            return
        stream = self._stream()
        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:hbm] = False
        t0 = time.perf_counter()
        chunks = list(self._tdb.process_tokens(torch.tensor(toks), mask=mask))
        process_ms = (time.perf_counter() - t0) * 1000
        logger.debug("offload _do_load req=%s hbm=%d lmc=%d chunks=%d",
                     req.req_id, hbm, ls.lmcache_cached_tokens, len(chunks))

        # All-or-nothing above the HBM prefix: a partial load would let attention
        # read uninitialized blocks, and a chunk that overlaps an HBM-cache hit
        # could overwrite shared prefix-cache blocks. In either case the seq
        # wakes and re-prefills from its HBM floor.
        if not chunks:
            logger.warning("LMCache offload: no loadable chunks req=%s; re-prefill",
                           req.req_id)
            self._lookup_unpin(req.req_id)
            with self._lock:
                self._failed_load.add(req.req_id)
            offload_trace(
                "worker_load_done",
                rank=getattr(self, "_rank", "?"),
                req=req.req_id,
                status="no_chunks",
                total_ms=f"{(time.perf_counter() - t_total0) * 1000:.2f}",
            )
            return
        for (s, _e, _key) in chunks:
            if s < hbm:
                logger.warning(
                    "LMCache offload: chunk overlaps HBM prefix req=%s hbm=%d "
                    "chunk_start=%d; re-prefill",
                    req.req_id, hbm, s,
                )
                self._lookup_unpin(req.req_id)
                with self._lock:
                    self._failed_load.add(req.req_id)
                offload_trace(
                    "worker_load_done",
                    rank=getattr(self, "_rank", "?"),
                    req=req.req_id,
                    status="overlap_hbm",
                    hbm=hbm,
                    chunk_start=s,
                    total_ms=f"{(time.perf_counter() - t_total0) * 1000:.2f}",
                )
                return
        contains_ms = 0.0
        loaded_objs = []
        get_ms = 0.0
        host_alloc_ms = 0.0
        stitch_ms = 0.0
        h2d_submit_ms = 0.0
        sync_ms = 0.0
        nblocks = 0
        nbytes = 0
        copy_calls = 0
        chunk_bids: list[list[int]] = [
            self._block_ids(req, s, e) for (s, e, _key) in chunks
        ]
        all_bids = [bid for bids in chunk_bids for bid in bids]
        nblocks = len(all_bids)
        nbytes = nblocks * self._codec.bytes_per_block

        request_key = None
        if hbm == 0 and self._request_fastpath_enabled():
            request_key = self._request_level_key(chunks, len(toks))
        if request_key is not None:
            req_mo = None
            t0 = time.perf_counter()
            request_location = self._sm.contains(request_key)
            contains_ms += (time.perf_counter() - t0) * 1000
            if request_location:
                try:
                    t0 = time.perf_counter()
                    req_mo = self._sm.get(request_key)
                    get_ms += (time.perf_counter() - t0) * 1000
                    if req_mo is not None:
                        copy_calls = self._codec.copy_calls_for_block_ids(all_bids)
                        t0 = time.perf_counter()
                        self._codec.host_to_gpu(req_mo.tensor, all_bids, stream)
                        h2d_submit_ms += (time.perf_counter() - t0) * 1000
                        t0 = time.perf_counter()
                        stream.synchronize()
                        sync_ms += (time.perf_counter() - t0) * 1000
                        self._lookup_unpin(req.req_id)
                        with self._lock:
                            self._done_load.add(req.req_id)
                        total_ms = (time.perf_counter() - t_total0) * 1000
                        offload_trace(
                            "worker_load_done",
                            rank=getattr(self, "_rank", "?"),
                            req=req.req_id,
                            status="ok_request",
                            chunks=len(chunks),
                            blocks=nblocks,
                            bytes_gib=f"{nbytes / 1024**3:.3f}",
                            stitch_ms=f"{stitch_ms:.2f}",
                            h2d_submit_ms=f"{h2d_submit_ms:.2f}",
                            sync_ms=f"{sync_ms:.2f}",
                            total_ms=f"{total_ms:.2f}",
                        )
                        if self._profile_enabled():
                            logger.info(
                                "[OFFLOAD-LOAD-PROF] rank=%s req=%s hbm=%d lmc=%d "
                                "chunks=%d blocks=%d bytes=%.3fGiB copy_calls=%d "
                                "layout=%s fastpath=request process_ms=%.2f "
                                "contains_ms=%.2f get_ms=%.2f host_alloc_ms=%.2f "
                                "stitch_ms=%.2f h2d_submit_ms=%.2f sync_ms=%.2f "
                                "total_ms=%.2f",
                                getattr(self, "_rank", "?"),
                                req.req_id,
                                hbm,
                                ls.lmcache_cached_tokens,
                                len(chunks),
                                nblocks,
                                nbytes / 1024**3,
                                copy_calls,
                                self._codec.layout,
                                process_ms,
                                contains_ms,
                                get_ms,
                                host_alloc_ms,
                                stitch_ms,
                                h2d_submit_ms,
                                sync_ms,
                                total_ms,
                            )
                        logger.info("offload _do_load DONE req=%s", req.req_id)
                        return
                finally:
                    if req_mo is not None:
                        req_mo.ref_count_down()

        for (_s, _e, key) in chunks:
            t0 = time.perf_counter()
            if not self._sm.contains(key):
                contains_ms += (time.perf_counter() - t0) * 1000
                logger.warning("LMCache offload: load miss req=%s; re-prefill", req.req_id)
                self._lookup_unpin(req.req_id)
                with self._lock:
                    self._failed_load.add(req.req_id)
                offload_trace(
                    "worker_load_done",
                    rank=getattr(self, "_rank", "?"),
                    req=req.req_id,
                    status="miss",
                    chunks=len(chunks),
                    total_ms=f"{(time.perf_counter() - t_total0) * 1000:.2f}",
                )
                return
            contains_ms += (time.perf_counter() - t0) * 1000

        try:
            for (s, e, key) in chunks:
                t0 = time.perf_counter()
                mo = self._sm.get(key)
                get_ms += (time.perf_counter() - t0) * 1000
                if mo is None:
                    t0 = time.perf_counter()
                    stream.synchronize()
                    sync_ms += (time.perf_counter() - t0) * 1000
                    for loaded_mo in loaded_objs:
                        loaded_mo.ref_count_down()
                    self._lookup_unpin(req.req_id)
                    with self._lock:
                        self._failed_load.add(req.req_id)
                    offload_trace(
                        "worker_load_done",
                        rank=getattr(self, "_rank", "?"),
                        req=req.req_id,
                        status="get_none",
                        chunks=len(chunks),
                        total_ms=f"{(time.perf_counter() - t_total0) * 1000:.2f}",
                    )
                    return
                loaded_objs.append(mo)
                bids = chunk_bids[len(loaded_objs) - 1]
                if self._codec.layout != "segment_indexed":
                    copy_calls += self._codec.copy_calls_for_block_ids(bids)
                    t0 = time.perf_counter()
                    self._codec.host_to_gpu(mo.tensor, bids, stream)
                    h2d_submit_ms += (time.perf_counter() - t0) * 1000
            if self._codec.layout == "segment_indexed":
                copy_calls = self._codec.copy_calls_for_block_ids(all_bids)
                t0 = time.perf_counter()
                req_buf = self._host_tmp(nbytes)
                host_alloc_ms += (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                self._codec.stitch_chunk_buffers(
                    req_buf,
                    [mo.tensor for mo in loaded_objs],
                    [len(bids) for bids in chunk_bids],
                )
                stitch_ms += (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                self._codec.host_to_gpu(req_buf, all_bids, stream)
                h2d_submit_ms += (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            stream.synchronize()
            sync_ms += (time.perf_counter() - t0) * 1000
        except Exception:
            try:
                t0 = time.perf_counter()
                stream.synchronize()
                sync_ms += (time.perf_counter() - t0) * 1000
            finally:
                for loaded_mo in loaded_objs:
                    loaded_mo.ref_count_down()
                self._lookup_unpin(req.req_id)
            raise
        for mo in loaded_objs:
            mo.ref_count_down()
        # Release the lookup pin (taken by the scheduler's LookupClient.lookup)
        # now that the chunks are safely in GPU; lets the pool evict them later.
        self._lookup_unpin(req.req_id)
        with self._lock:
            self._done_load.add(req.req_id)
        total_ms = (time.perf_counter() - t_total0) * 1000
        offload_trace(
            "worker_load_done",
            rank=getattr(self, "_rank", "?"),
            req=req.req_id,
            status="ok",
            chunks=len(chunks),
            blocks=nblocks,
            bytes_gib=f"{nbytes / 1024**3:.3f}",
            stitch_ms=f"{stitch_ms:.2f}",
            h2d_submit_ms=f"{h2d_submit_ms:.2f}",
            sync_ms=f"{sync_ms:.2f}",
            total_ms=f"{total_ms:.2f}",
        )
        if self._profile_enabled():
            logger.info(
                "[OFFLOAD-LOAD-PROF] rank=%s req=%s hbm=%d lmc=%d "
                "chunks=%d blocks=%d bytes=%.3fGiB copy_calls=%d "
                "layout=%s fastpath=chunk process_ms=%.2f contains_ms=%.2f "
                "get_ms=%.2f host_alloc_ms=%.2f stitch_ms=%.2f "
                "h2d_submit_ms=%.2f sync_ms=%.2f total_ms=%.2f",
                getattr(self, "_rank", "?"),
                req.req_id,
                hbm,
                ls.lmcache_cached_tokens,
                len(chunks),
                nblocks,
                nbytes / 1024**3,
                copy_calls,
                self._codec.layout,
                process_ms,
                contains_ms,
                get_ms,
                host_alloc_ms,
                stitch_ms,
                h2d_submit_ms,
                sync_ms,
                total_ms,
            )
        logger.info("offload _do_load DONE req=%s", req.req_id)

    def _do_save_req(self, req: LMCacheReqMeta) -> None:
        from lmcache.v1.memory_management import MemoryFormat

        ss = req.save_spec
        assert ss is not None
        stream = self._stream()
        toks = req.token_ids
        if not req.is_last_prefill:
            toks = toks[: (len(toks) // self.chunk_size) * self.chunk_size]
        skip = (ss.skip_leading_tokens // self.chunk_size) * self.chunk_size
        if skip >= len(toks):
            with self._lock:
                self._done_save.add(req.req_id)
            offload_trace(
                "worker_save_done",
                rank=getattr(self, "_rank", "?"),
                req=req.req_id,
                status="skip",
                toks=len(toks),
            )
            return

        t_total0 = time.perf_counter()
        offload_trace(
            "worker_save_start",
            rank=getattr(self, "_rank", "?"),
            req=req.req_id,
            skip=skip,
            toks=len(toks),
            blocks=len(req.block_ids),
        )
        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:skip] = False
        t0 = time.perf_counter()
        chunks = list(self._tdb.process_tokens(torch.tensor(toks), mask=mask))
        process_ms = (time.perf_counter() - t0) * 1000

        keys, objs, already = [], [], 0
        request_key = None
        request_obj = None
        request_fastpath = "off"
        if skip == 0 and self._request_fastpath_enabled() and chunks:
            request_key = self._request_level_key(chunks, len(toks))
            request_fastpath = "miss"
            t0 = time.perf_counter()
            if self._sm.contains(request_key):
                request_key = None
                request_fastpath = "hit"
            contains_ms = (time.perf_counter() - t0) * 1000
        else:
            contains_ms = 0.0
        put_started = False
        alloc_ms = 0.0
        host_alloc_ms = 0.0
        d2h_submit_ms = 0.0
        sync_ms = 0.0
        split_ms = 0.0
        put_ms = 0.0
        nblocks = 0
        total_nbytes = 0
        copy_calls = 0
        chunk_bids: list[list[int]] = []
        try:
            for (s, e, key) in chunks:
                self._pause_save_for_load(stream)
                t0 = time.perf_counter()
                if self._sm.contains(key):  # already offloaded → skip wasted D2H
                    contains_ms += (time.perf_counter() - t0) * 1000
                    already += 1
                    continue
                contains_ms += (time.perf_counter() - t0) * 1000
                bids = self._block_ids(req, s, e)
                chunk_nbytes = len(bids) * self._codec.bytes_per_block
                t0 = time.perf_counter()
                mo = self._sm.allocate(torch.Size((chunk_nbytes,)), torch.uint8,
                                       fmt=MemoryFormat.KV_2LTD)
                alloc_ms += (time.perf_counter() - t0) * 1000
                if mo is None:  # pool under pressure; stop here
                    break
                keys.append(key)
                objs.append(mo)
                chunk_bids.append(bids)
                nblocks += len(bids)
                total_nbytes += chunk_nbytes
                if self._codec.layout != "segment_indexed":
                    copy_calls += self._codec.copy_calls_for_block_ids(bids)
                    # D2H on this thread's dedicated copy stream (off compute stream).
                    t0 = time.perf_counter()
                    self._codec.gpu_to_host(mo.tensor, bids, stream)
                    d2h_submit_ms += (time.perf_counter() - t0) * 1000

            if keys:
                if self._codec.layout == "segment_indexed":
                    all_bids = [bid for bids in chunk_bids for bid in bids]
                    copy_calls = self._codec.copy_calls_for_block_ids(all_bids)
                    if request_key is not None and len(keys) == len(chunks):
                        t0 = time.perf_counter()
                        request_obj = self._sm.allocate(
                            torch.Size((total_nbytes,)),
                            torch.uint8,
                            fmt=MemoryFormat.KV_2LTD,
                        )
                        alloc_ms += (time.perf_counter() - t0) * 1000
                        if request_obj is not None:
                            req_buf = request_obj.tensor
                            request_fastpath = "stored"
                        else:
                            request_key = None
                            request_fastpath = "alloc_failed"
                    else:
                        request_key = None
                        if request_fastpath == "miss":
                            request_fastpath = "partial_skip"
                    if request_obj is None:
                        t0 = time.perf_counter()
                        req_buf = self._host_tmp(total_nbytes)
                        host_alloc_ms += (time.perf_counter() - t0) * 1000
                    t0 = time.perf_counter()
                    self._codec.gpu_to_host(req_buf, all_bids, stream)
                    d2h_submit_ms += (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                stream.synchronize()  # stream-specific
                sync_ms += (time.perf_counter() - t0) * 1000
                if self._codec.layout == "segment_indexed":
                    t0 = time.perf_counter()
                    self._codec.split_request_buffer(
                        req_buf,
                        [mo.tensor for mo in objs],
                        [len(bids) for bids in chunk_bids],
                    )
                    split_ms += (time.perf_counter() - t0) * 1000
                put_started = True
                t0 = time.perf_counter()
                put_keys = list(keys)
                put_objs = list(objs)
                if request_key is not None and request_obj is not None:
                    put_keys.append(request_key)
                    put_objs.append(request_obj)
                self._sm.batched_put(put_keys, put_objs)
                put_ms += (time.perf_counter() - t0) * 1000
        except Exception:
            if not put_started:
                try:
                    t0 = time.perf_counter()
                    stream.synchronize()
                    sync_ms += (time.perf_counter() - t0) * 1000
                finally:
                    cleanup_objs = list(objs)
                    if request_obj is not None:
                        cleanup_objs.append(request_obj)
                    for mo in cleanup_objs:
                        mo.ref_count_down()
            raise
        with self._lock:
            self._done_save.add(req.req_id)
        total_ms = (time.perf_counter() - t_total0) * 1000
        offload_trace(
            "worker_save_done",
            rank=getattr(self, "_rank", "?"),
            req=req.req_id,
            status="ok",
            toks=len(toks),
            chunks=len(chunks),
            stored=len(keys),
            blocks=nblocks,
            bytes_gib=f"{total_nbytes / 1024**3:.3f}",
            d2h_submit_ms=f"{d2h_submit_ms:.2f}",
            sync_ms=f"{sync_ms:.2f}",
            split_ms=f"{split_ms:.2f}",
            request_fastpath=request_fastpath,
            total_ms=f"{total_ms:.2f}",
        )
        if self._profile_enabled():
            logger.info(
                "[OFFLOAD-SAVE-PROF] rank=%s req=%s toks=%d chunks=%d "
                "stored=%d already=%d blocks=%d bytes=%.3fGiB copy_calls=%d "
                "layout=%s request_fastpath=%s process_ms=%.2f "
                "contains_ms=%.2f alloc_ms=%.2f host_alloc_ms=%.2f "
                "d2h_submit_ms=%.2f sync_ms=%.2f split_ms=%.2f "
                "put_ms=%.2f total_ms=%.2f",
                getattr(self, "_rank", "?"),
                req.req_id,
                len(toks),
                len(chunks),
                len(keys),
                already,
                nblocks,
                total_nbytes / 1024**3,
                copy_calls,
                self._codec.layout,
                request_fastpath,
                process_ms,
                contains_ms,
                alloc_ms,
                host_alloc_ms,
                d2h_submit_ms,
                sync_ms,
                split_ms,
                put_ms,
                total_ms,
            )
        if logger.isEnabledFor(logging.DEBUG):
            _kh = [getattr(k, "chunk_hash", None) for k in keys[:2]]
            _contains = [bool(self._sm.contains(k)) for k in keys[:2]]
            logger.debug("[OFFLOAD-SAVE] rank=%s req=%s toks=%d chunks=%d stored=%d already=%d "
                         "chunkhash2=%s contains=%s",
                         self._rank, req.req_id, len(toks), len(chunks), len(keys),
                         already, _kh, _contains)

    # -- per-step (RPC thread, post-forward): poll completions ------------
    def get_finished(self) -> KVConnectorOutput:
        # Offload uses extended completion states:
        # - finished_recving wakes successfully loaded requests.
        # - failed_recving wakes them for recompute using already allocated blocks.
        # - finished_saving releases blocks whose free was deferred during save.
        with self._lock:
            dl = set(self._done_load)
            fl = set(self._failed_load)
            ds = set(self._done_save)
            self._done_save.clear()
            self._done_load.clear()
            self._failed_load.clear()
        if dl or fl or ds:
            offload_trace(
                "worker_get_finished",
                rank=getattr(self, "_rank", "?"),
                done_load=sorted(dl),
                failed_load=sorted(fl),
                done_save=sorted(ds),
            )
        return KVConnectorOutput(
            finished_sending=set(),
            finished_recving=dl,
            failed_recving=fl,
            finished_saving=ds,
        )

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
        # Persistent save tracker: sid -> [seq, saved_offset]. A seq's prompt
        # prefix is stored to LMCache once prefill computes it
        # (seq.prefix_hashes_published flips True), chunk by chunk.
        self._save_tracker: dict[str, list] = {}
        self._save_inflight: set[str] = set()
        self._lookup_in_step: list[str] = []

        try:
            cfg = offcfg.build_lmcache_config()
            offcfg.apply_extra_overrides(cfg, kvc)
            self.chunk_size = int(cfg.chunk_size)
            from lmcache.v1.lookup_client.factory import LookupClientFactory
            world = int(getattr(config, "tensor_parallel_size", 1) or 1)
            meta = offcfg.build_lmcache_metadata(config, cfg, world, 0)
            self._lookup_client = LookupClientFactory.create_lookup_client(cfg, meta)
        except Exception as e:
            logger.warning("LMCache offload scheduler: lookup client unavailable: %s", e)

    # -- match: how many extra tokens can come from CPU/NVMe -------------
    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        if self._lookup_client is None:
            return 0, False
        num_prompt = seq.num_prompt_tokens
        token_ids = list(seq.token_ids[:num_prompt])
        offload_trace(
            "scheduler_lookup_start",
            req=seq.id,
            prompt=num_prompt,
            hbm=seq.num_cached_tokens,
        )
        t_lookup0 = time.perf_counter()
        try:
            hit = self._lookup_client.lookup(token_ids, lookup_id=str(seq.id))
        except Exception:
            logger.exception("LMCache offload lookup failed for seq %s", seq.id)
            offload_trace(
                "scheduler_lookup_done",
                req=seq.id,
                status="exception",
                lookup_ms=f"{(time.perf_counter() - t_lookup0) * 1000:.2f}",
            )
            return 0, False
        lookup_ms = (time.perf_counter() - t_lookup0) * 1000
        if logger.isEnabledFor(logging.DEBUG):
            _lh = None
            try:
                tdb = getattr(self._lookup_client, "token_database", None)
                if tdb is not None:
                    _lh = [k for (_s, _e, k) in list(
                        tdb.process_tokens(token_ids, make_key=False))[:3]]
            except Exception as e:
                _lh = f"err:{e}"
            logger.debug("[OFFLOAD-LOOKUP] seq=%s num_prompt=%d hbm_cached=%d hit=%s lookuphash3=%s",
                         seq.id, num_prompt, int(seq.num_cached_tokens), hit, _lh)
        if not hit:
            offload_trace(
                "scheduler_lookup_done",
                req=seq.id,
                status="miss",
                hit=hit,
                lookup_ms=f"{lookup_ms:.2f}",
            )
            return 0, False
        sid = str(seq.id)
        hit = int(hit)
        if hit == num_prompt:  # full-prompt hit → recompute last token
            hit -= 1
        need = hit - int(seq.num_cached_tokens)
        if need <= 0:
            if self._lookup_client is not None:
                try:
                    self._lookup_client.clear_lookup_status(sid)
                except Exception:
                    pass
            offload_trace(
                "scheduler_lookup_done",
                req=seq.id,
                status="hbm_satisfies",
                hit=hit,
                hbm=seq.num_cached_tokens,
                lookup_ms=f"{lookup_ms:.2f}",
            )
            return 0, False
        self._lookup_in_step.append(sid)
        self._load_specs[sid] = LoadSpec(
            hbm_cached_tokens=int(seq.num_cached_tokens),
            lmcache_cached_tokens=hit,
            can_load=False,
        )
        offload_trace(
            "scheduler_lookup_done",
            req=seq.id,
            status="need_load",
            hit=hit,
            hbm=seq.num_cached_tokens,
            need=need,
            lookup_ms=f"{lookup_ms:.2f}",
        )
        return need, True  # True => park in WAITING_FOR_REMOTE_KVS

    def update_state_after_alloc(self, seq) -> None:
        sid = str(seq.id)
        ls = self._load_specs.get(sid)
        logger.debug("[OFFLOAD-ALLOC] seq=%s ls_found=%s num_cached_now=%s",
                     seq.id, ls is not None, int(getattr(seq, "num_cached_tokens", -1)))
        if ls is not None:
            ls.can_load = True
            self._reqs_need_recv[sid] = seq
            offload_trace(
                "scheduler_load_alloc_ready",
                req=seq.id,
                hbm=seq.num_cached_tokens,
                lmc=ls.lmcache_cached_tokens,
                blocks=len(seq.block_table),
            )
        # Track for save; build_connector_meta stores chunks once the scheduler's
        # computed frontier (seq.num_cached_tokens) has advanced past them.
        if sid not in self._save_tracker:
            self._save_tracker[sid] = [seq, 0]

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        meta = LMCacheOffloadMetadata()
        meta.lookup_requests_in_step = self._lookup_in_step
        self._lookup_in_step = []

        # Loads
        logger.debug("[OFFLOAD-BUILD] reqs_need_recv=%d", len(self._reqs_need_recv))
        for sid, seq in self._reqs_need_recv.items():
            ls = self._load_specs.pop(sid, None)
            if ls is None or not ls.can_load:
                logger.debug("[OFFLOAD-LOAD-SKIP] seq=%s ls=%s can_load=%s",
                             sid, ls is not None, getattr(ls, "can_load", None))
                continue
            # ★ Use the REAL HBM-cached count as the load floor.
            # get_num_new_matched_tokens runs BEFORE the prefix-cache match in
            # block_manager.allocate, so seq.num_cached_tokens was stale (often
            # 0) when the LoadSpec was recorded. By now (post-allocate) it is the
            # true HBM hit. Loading below this floor would overwrite HBM
            # prefix-cache blocks (possibly shared with other seqs) -> output
            # corruption. So load only [hbm_cached, offload_hit).
            ls.hbm_cached_tokens = int(seq.num_cached_tokens)
            if ls.hbm_cached_tokens >= int(ls.lmcache_cached_tokens):
                seq.offload_loaded_tokens = int(seq.num_cached_tokens)
                logger.info(
                    "[OFFLOAD-LOAD-SKIP] seq=%s hbm_cached=%d lmc_cached=%d "
                    "reason=hbm_satisfies_after_alloc",
                    seq.id,
                    ls.hbm_cached_tokens,
                    ls.lmcache_cached_tokens,
                )
                offload_trace(
                    "scheduler_load_hbm_satisfies_after_alloc",
                    req=seq.id,
                    hbm=ls.hbm_cached_tokens,
                    lmc=ls.lmcache_cached_tokens,
                    blocks=len(list(seq.block_table)),
                )
                # The request may already be parked in WAITING_FOR_REMOTE_KVS.
                # Emit a no-op load so every worker reports finished_recving via
                # the normal aggregation path instead of trying to complete it
                # locally in the scheduler process.
                meta.add_request(LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[: ls.lmcache_cached_tokens]),
                    block_ids=list(seq.block_table),
                    load_spec=ls,
                ))
                continue
            chunk = self.chunk_size or 256
            if ls.hbm_cached_tokens % chunk != 0:
                seq.offload_loaded_tokens = int(seq.num_cached_tokens)
                logger.info(
                    "[OFFLOAD-LOAD-SKIP] seq=%s hbm_cached=%d lmc_cached=%d "
                    "reason=unaligned_hbm chunk=%d",
                    seq.id,
                    ls.hbm_cached_tokens,
                    ls.lmcache_cached_tokens,
                    chunk,
                )
                offload_trace(
                    "scheduler_load_unaligned_hbm",
                    req=seq.id,
                    hbm=ls.hbm_cached_tokens,
                    lmc=ls.lmcache_cached_tokens,
                    chunk=chunk,
                    blocks=len(list(seq.block_table)),
                )
                # LMCache chunks can only be loaded from a chunk boundary. Do
                # not round down and overwrite HBM prefix-cache blocks that may
                # be shared with other requests; wake the parked request and let
                # it continue prefill from the HBM floor.
                meta.add_request(LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[: ls.hbm_cached_tokens]),
                    block_ids=list(seq.block_table),
                    load_spec=LoadSpec(
                        hbm_cached_tokens=ls.hbm_cached_tokens,
                        lmcache_cached_tokens=ls.hbm_cached_tokens,
                        can_load=True,
                    ),
                ))
                continue
            # num_cached after load = max(HBM, offload); never drop below HBM.
            seq.offload_loaded_tokens = max(
                int(seq.num_cached_tokens), int(ls.lmcache_cached_tokens)
            )
            # req_id MUST be the raw seq.id (the type the scheduler compares
            # against in _update_waiting_for_remote_kv); str(seq.id) is only for
            # LMCache's lookup/pin API. A str here silently never wakes the seq.
            logger.info("[OFFLOAD-LOAD-EMIT] seq=%s hbm_cached=%d lmc_cached=%d offload_loaded=%d nblocks=%d",
                        seq.id, ls.hbm_cached_tokens, ls.lmcache_cached_tokens,
                        seq.offload_loaded_tokens, len(list(seq.block_table)))
            offload_trace(
                "scheduler_load_emit",
                req=seq.id,
                hbm=ls.hbm_cached_tokens,
                lmc=ls.lmcache_cached_tokens,
                offload_loaded=seq.offload_loaded_tokens,
                blocks=len(list(seq.block_table)),
            )
            meta.add_request(LMCacheReqMeta(
                req_id=seq.id,
                token_ids=list(seq.token_ids[: ls.lmcache_cached_tokens]),
                block_ids=list(seq.block_table),
                load_spec=ls,
            ))
        # Saves: store fully computed prompt chunks. Under scheduler-side
        # chunked prefill, seq.num_cached_tokens advances after each prefill
        # chunk's forward has completed; use it as the D2H-safe frontier.
        chunk = self.chunk_size or 256
        for sid, entry in self._save_tracker.items():
            seq, saved = entry
            if sid in self._reqs_need_recv:
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
            offload_trace(
                "scheduler_save_emit",
                req=seq.id,
                prompt=seq.num_prompt_tokens,
                computed=computed,
                aligned=aligned,
                saved=saved,
                blocks=len(seq.block_table),
            )
            meta.add_request(LMCacheReqMeta(
                req_id=seq.id,
                token_ids=list(seq.token_ids[:aligned]),
                block_ids=list(seq.block_table),
                save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
                is_last_prefill=is_last_prefill,
            ))
            entry[1] = aligned
            self._save_inflight.add(sid)
        self._reqs_need_recv.clear()
        return meta

    def _save_frontier(self, seq) -> int:
        chunk = self.chunk_size or 256
        computed = min(
            int(getattr(seq, "num_cached_tokens", 0)),
            int(getattr(seq, "num_prompt_tokens", 0)),
        )
        return (computed // chunk) * chunk

    def _has_pending_save(self, seq) -> bool:
        sid = str(seq.id)
        entry = self._save_tracker.get(sid)
        if entry is None:
            return False
        return self._save_frontier(seq) > int(entry[1])

    def should_defer_free(self, seq) -> bool:
        sid = str(seq.id)
        return sid in self._save_inflight or self._has_pending_save(seq)

    def save_finished(self, req_id) -> None:
        self._save_inflight.discard(str(req_id))

    def load_failed(self, req_id) -> None:
        self._load_specs.pop(str(req_id), None)
        self._reqs_need_recv.pop(str(req_id), None)

    def request_finished(self, seq) -> None:
        sid = str(seq.id)
        self._load_specs.pop(sid, None)
        self._reqs_need_recv.pop(sid, None)
        if not self.should_defer_free(seq):
            self._save_tracker.pop(sid, None)
        if self._lookup_client is not None:
            try:
                self._lookup_client.clear_lookup_status(sid)
            except Exception:
                pass
