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
from atom.kv_transfer.offload.gpu_connector import ATOMKVByteCodec
from atom.kv_transfer.offload.lmcache_compat import (
    ATOMLMCacheGPUConnector,
    ATOMRawBytesLMCacheMetadata,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.kv_transfer.offload.trace import offload_trace

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
        # The LMCache-compatible GPU connector owns per-thread staging streams.
        # OFFLOAD_COPY_WORKERS tunes the SAVE pool only.
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

        self._engine = None
        self._sm = None
        self._tdb = None
        self._codec: ATOMKVByteCodec | None = None
        self._lookup_server = None

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict, transfer_tensors=None) -> None:
        from aiter.dist.parallel_state import get_tp_group
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.v1.memory_management import MemoryFormat

        tp = get_tp_group()
        rank, world = tp.rank_in_group, tp.world_size
        self._rank = rank

        cfg = offcfg.build_lmcache_config()
        offcfg.apply_extra_overrides(
            cfg, getattr(self._config, "kv_transfer_config", None)
        )
        self.chunk_size = int(cfg.chunk_size)
        self._codec = ATOMKVByteCodec(kv_caches)
        base_meta = offcfg.build_lmcache_metadata(self._config, cfg, world, rank)
        meta = ATOMRawBytesLMCacheMetadata(
            base_meta,
            atom_block_size=self.block_size,
            bytes_per_block=self._codec.bytes_per_block,
        )
        gpu_connector = ATOMLMCacheGPUConnector(self._codec, self.block_size)

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
        self._sm = self._engine.storage_manager
        self._tdb = self._engine.token_database

        # DEBUG: wrap engine.lookup to capture EVERY call (incl. the ones the ZMQ
        # lookup_server makes on behalf of the scheduler) — args + result.
        _orig_lookup = self._engine.lookup
        _rk = rank

        def _logged_lookup(*a, **k):
            r = _orig_lookup(*a, **k)
            h = k.get("hashes")
            logger.debug(
                "[ENGINE.LOOKUP] rank=%s lookup_id=%s nhashes=%s first3=%s -> %s",
                _rk,
                k.get("lookup_id"),
                (len(h) if h is not None else None),
                (list(h[:3]) if h else None),
                r,
            )
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
            rank,
            self._codec.bytes_per_block,
            self.chunk_size,
            self._codec.layout,
            self._do_save,
            self._do_load,
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
        try:
            fn(req)
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

    def _profile_enabled(self) -> bool:
        return os.environ.get("OFFLOAD_PROFILE", "1").lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    # -- copy daemon thread ----------------------------------------------
    def _do_load_req(self, req: LMCacheReqMeta) -> None:
        ls = req.load_spec
        assert ls is not None
        hbm = int(ls.hbm_cached_tokens)
        lmc = int(ls.lmcache_cached_tokens)
        toks = req.token_ids[:lmc]
        t_total0 = time.perf_counter()
        offload_trace(
            "worker_load_start",
            rank=getattr(self, "_rank", "?"),
            req=req.req_id,
            hbm=hbm,
            lmc=lmc,
            toks=len(toks),
            blocks=len(req.block_ids),
        )
        if lmc <= hbm:
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

        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:hbm] = False

        t_retrieve0 = time.perf_counter()
        ret_mask = self._engine.retrieve(
            torch.tensor(toks),
            mask=mask,
            block_ids=req.block_ids,
            req_id=str(req.req_id),
        )
        retrieve_ms = (time.perf_counter() - t_retrieve0) * 1000
        self._lookup_unpin(req.req_id)
        loaded = bool(ret_mask[hbm:lmc].all().item()) if lmc > hbm else True
        with self._lock:
            if loaded:
                self._done_load.add(req.req_id)
            else:
                self._failed_load.add(req.req_id)
        total_ms = (time.perf_counter() - t_total0) * 1000
        offload_trace(
            "worker_load_done",
            rank=getattr(self, "_rank", "?"),
            req=req.req_id,
            status="ok" if loaded else "miss",
            hbm=hbm,
            lmc=lmc,
            retrieved=int(ret_mask.sum().item()),
            retrieve_ms=f"{retrieve_ms:.2f}",
            total_ms=f"{total_ms:.2f}",
        )
        if self._profile_enabled():
            logger.info(
                "[OFFLOAD-LOAD-PROF] rank=%s req=%s hbm=%d lmc=%d "
                "retrieved=%d status=%s retrieve_ms=%.2f total_ms=%.2f",
                getattr(self, "_rank", "?"),
                req.req_id,
                hbm,
                lmc,
                int(ret_mask.sum().item()),
                "ok" if loaded else "miss",
                retrieve_ms,
                total_ms,
            )
        logger.info("offload _do_load DONE req=%s", req.req_id)

    def _do_save_req(self, req: LMCacheReqMeta) -> None:
        ss = req.save_spec
        assert ss is not None
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

        t_store0 = time.perf_counter()
        self._engine.store(
            torch.tensor(toks),
            mask=mask,
            block_ids=req.block_ids,
            req_id=str(req.req_id),
        )
        store_ms = (time.perf_counter() - t_store0) * 1000
        with self._lock:
            self._done_save.add(req.req_id)
        total_ms = (time.perf_counter() - t_total0) * 1000
        offload_trace(
            "worker_save_done",
            rank=getattr(self, "_rank", "?"),
            req=req.req_id,
            status="ok",
            toks=len(toks),
            skip=skip,
            store_ms=f"{store_ms:.2f}",
            total_ms=f"{total_ms:.2f}",
        )
        if self._profile_enabled():
            logger.info(
                "[OFFLOAD-SAVE-PROF] rank=%s req=%s toks=%d skip=%d "
                "store_ms=%.2f total_ms=%.2f",
                getattr(self, "_rank", "?"),
                req.req_id,
                len(toks),
                skip,
                store_ms,
                total_ms,
            )

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
        self._lookup_in_step: list[str] = []
        self._handoff_loads: set[str] = set()
        self._allow_unaligned_handoff = os.environ.get(
            "OFFLOAD_UNALIGNED_HANDOFF", "0"
        ).lower() in ("1", "true", "yes", "on")
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
            cfg = offcfg.build_lmcache_config()
            offcfg.apply_extra_overrides(cfg, kvc)
            self.chunk_size = int(cfg.chunk_size)
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            world = int(getattr(config, "tensor_parallel_size", 1) or 1)
            meta = offcfg.build_lmcache_metadata(config, cfg, world, 0)
            self._lookup_client = LookupClientFactory.create_lookup_client(cfg, meta)
        except Exception as e:
            logger.warning(
                "LMCache offload scheduler: lookup client unavailable: %s", e
            )

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
                    _lh = [
                        k
                        for (_s, _e, k) in list(
                            tdb.process_tokens(token_ids, make_key=False)
                        )[:3]
                    ]
            except Exception as e:
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
        self._hit_save_floors[sid] = self._chunk_floor(hit)
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
        logger.debug(
            "[OFFLOAD-ALLOC] seq=%s ls_found=%s num_cached_now=%s",
            seq.id,
            ls is not None,
            int(getattr(seq, "num_cached_tokens", -1)),
        )
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
        self._lookup_in_step = [
            req_id for req_id in self._lookup_in_step if req_id != sid
        ]
        if self._lookup_client is not None:
            try:
                self._lookup_client.clear_lookup_status(sid)
            except Exception:
                pass

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

    def _maybe_start_unaligned_handoff(
        self,
        seq,
        ls: LoadSpec,
        hbm: int,
        lmc: int,
        chunk: int,
    ) -> bool:
        if not getattr(self, "_allow_unaligned_handoff", False):
            return False
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
        logger.info(
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
        offload_trace(
            "scheduler_load_handoff_start",
            req=seq.id,
            hbm=hbm,
            boundary=boundary,
            lmc=lmc,
            need_after_boundary=remaining_after_boundary,
            min_load=min_load,
            chunk=chunk,
            blocks=len(list(seq.block_table)),
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
        offload_trace(
            "scheduler_load_handoff_prefill_boundary",
            req=seq.id,
            hbm=hbm,
            boundary=int(boundary),
            original_chunk=int(chunk),
            adjusted_chunk=adjusted,
        )
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
        seq.offload_loaded_tokens = max(hbm, lmc)
        logger.info(
            "[OFFLOAD-LOAD-HANDOFF-READY] seq=%s hbm_cached=%d "
            "lmc_cached=%d offload_loaded=%d need=%d",
            seq.id,
            hbm,
            lmc,
            seq.offload_loaded_tokens,
            need,
        )
        offload_trace(
            "scheduler_load_handoff_ready",
            req=seq.id,
            hbm=hbm,
            lmc=lmc,
            need=need,
            blocks=len(list(seq.block_table)),
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
        logger.info(
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
        offload_trace(
            "scheduler_load_skip",
            req=seq.id,
            reason=reason,
            hbm=hbm,
            lmc=lmc,
            need=need,
            min_load=min_load,
            chunk=chunk,
            blocks=len(list(seq.block_table)),
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
        seq.offload_loaded_tokens = max(hbm, lmc)
        return True

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        meta = LMCacheOffloadMetadata()

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
            seq.offload_loaded_tokens = max(hbm, lmc)
            # req_id MUST be the raw seq.id (the type the scheduler compares
            # against in _update_waiting_for_remote_kv); str(seq.id) is only for
            # LMCache's lookup/pin API. A str here silently never wakes the seq.
            logger.info(
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
            offload_trace(
                "scheduler_load_emit",
                req=seq.id,
                hbm=hbm,
                lmc=lmc,
                need=need,
                min_load=int(getattr(self, "_min_load_tokens", 8192)),
                offload_loaded=seq.offload_loaded_tokens,
                blocks=len(list(seq.block_table)),
            )
            loading_sids.add(sid)
            self._load_save_floors[sid] = self._chunk_floor(hbm)
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:lmc]),
                    block_ids=list(seq.block_table),
                    load_spec=ls,
                )
            )
        meta.lookup_requests_in_step = self._lookup_in_step
        self._lookup_in_step = []
        # Saves: store fully computed prompt chunks. Under scheduler-side
        # chunked prefill, seq.num_cached_tokens advances after each prefill
        # chunk's forward has completed; use it as the D2H-safe frontier.
        chunk = self.chunk_size or 256
        for sid, entry in self._save_tracker.items():
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
            offload_trace(
                "scheduler_save_emit",
                req=seq.id,
                prompt=seq.num_prompt_tokens,
                computed=computed,
                aligned=aligned,
                saved=saved,
                blocks=len(seq.block_table),
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
        self._reqs_need_recv.clear()
        return meta

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
        return sid in self._save_inflight or self._has_pending_save(seq)

    def save_finished(self, req_id) -> None:
        self._save_inflight.discard(str(req_id))

    def load_failed(self, req_id) -> None:
        sid = str(req_id)
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
        if not self.should_defer_free(seq):
            self._save_tracker.pop(sid, None)
