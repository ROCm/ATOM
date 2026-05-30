# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM standalone LMCache CPU/NVMe KV-offload connector.

Design (see ../../../../PLAN_impl_lmcache_offload_v5.md + 005 LEARN notes):

* **Reuse real LMCache as a storage tier only** — per-rank ``LMCacheEngine`` for its
  ``StorageManager`` (CPU LRU + NVMe L3) + ``ChunkedTokenDatabase`` (chunk-256 keys).
  We bypass ``engine.store/retrieve`` (token-major GPU path can't represent AITER's
  swizzle) and instead move **opaque per-block bytes** via :class:`ATOMKVByteCodec`
  into pinned ``KV_2LTD``-as-uint8 ``MemoryObj``s.
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
from concurrent.futures import ThreadPoolExecutor

import torch

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.gpu_connector import ATOMKVByteCodec
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)

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
        self._done_load: set[str] = set()
        self._done_save: set[str] = set()

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
            logger.info("[ENGINE.LOOKUP] rank=%s lookup_id=%s nhashes=%s first3=%s -> %s",
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
            "LMCache offload worker rank=%d: bytes_per_block=%d chunk=%d save=%s load=%s",
            rank, self._codec.bytes_per_block, self.chunk_size,
            self._do_save, self._do_load,
        )

    # -- per-step (RPC thread): only enqueue, never copy ------------------
    def start_load_kv(self, metadata) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        for req in metadata.requests:
            if req.load_spec is not None and self._do_load:
                self._load_executor.submit(self._guard, self._do_load_req, req)
            if req.save_spec is not None and self._do_save:
                self._save_executor.submit(self._guard, self._do_save_req, req)

    def _guard(self, fn, req) -> None:
        try:
            fn(req)
        except Exception:
            logger.exception("LMCache offload: %s failed for %s", fn.__name__, req.req_id)
            # Wake the seq anyway so it is not stuck parked; scheduler re-derives
            # how much is actually cached (load) / proceeds (save).
            with self._lock:
                (self._done_load if fn is self._do_load_req else self._done_save).add(
                    req.req_id
                )

    def _stream(self) -> torch.cuda.Stream:
        """A CUDA stream owned by the calling copy-daemon thread (lazily made)."""
        s = getattr(self._tls, "stream", None)
        if s is None:
            s = torch.cuda.Stream()
            self._tls.stream = s
        return s

    def _block_ids(self, req: LMCacheReqMeta, start: int, end: int) -> list[int]:
        return req.block_ids[start // self.block_size : _cdiv(end, self.block_size)]

    # -- copy daemon thread ----------------------------------------------
    def _do_load_req(self, req: LMCacheReqMeta) -> None:
        ls = req.load_spec
        assert ls is not None
        stream = self._stream()
        hbm = (ls.hbm_cached_tokens // self.chunk_size) * self.chunk_size
        toks = req.token_ids[: ls.lmcache_cached_tokens]
        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:hbm] = False
        chunks = list(self._tdb.process_tokens(torch.tensor(toks), mask=mask))
        logger.debug("offload _do_load req=%s hbm=%d lmc=%d chunks=%d",
                     req.req_id, hbm, ls.lmcache_cached_tokens, len(chunks))

        # All-or-nothing: a partial load would let attention read uninitialized
        # blocks. If any chunk is gone (evicted between lookup and load), skip the
        # whole load — the seq wakes and re-prefills the suffix (loaded 0).
        for (_s, _e, key) in chunks:
            if not self._sm.contains(key):
                logger.warning("LMCache offload: load miss req=%s; re-prefill", req.req_id)
                with self._lock:
                    self._done_load.add(req.req_id)
                return

        for (s, e, key) in chunks:
            mo = self._sm.get(key)
            if mo is None:
                with self._lock:
                    self._done_load.add(req.req_id)
                return
            self._codec.host_to_gpu(mo.tensor, self._block_ids(req, s, e), stream)
            mo.ref_count_down()
        stream.synchronize()
        # Release the lookup pin (taken by the scheduler's LookupClient.lookup)
        # now that the chunks are safely in GPU; lets the pool evict them later.
        try:
            self._engine.lookup_unpin([str(req.req_id)])  # LMCache pin keyed by str id
        except Exception:
            pass
        with self._lock:
            self._done_load.add(req.req_id)
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
            return

        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:skip] = False
        chunks = list(self._tdb.process_tokens(torch.tensor(toks), mask=mask))

        keys, objs, already = [], [], 0
        for (s, e, key) in chunks:
            if self._sm.contains(key):  # already offloaded → skip wasted D2H
                already += 1
                continue
            bids = self._block_ids(req, s, e)
            nbytes = len(bids) * self._codec.bytes_per_block
            mo = self._sm.allocate(torch.Size((nbytes,)), torch.uint8,
                                   fmt=MemoryFormat.KV_2LTD)
            if mo is None:  # pool under pressure; stop here
                break
            # D2H on this thread's dedicated copy stream (off the compute stream).
            self._codec.gpu_to_host(mo.tensor, bids, stream)
            keys.append(key)
            objs.append(mo)

        if keys:
            stream.synchronize()  # stream-specific
            self._sm.batched_put(keys, objs)
        with self._lock:
            self._done_save.add(req.req_id)
        _kh = [getattr(k, "chunk_hash", None) for k in keys[:2]]
        _contains = [bool(self._sm.contains(k)) for k in keys[:2]]
        logger.info("[OFFLOAD-SAVE] rank=%s req=%s toks=%d chunks=%d stored=%d already=%d "
                    "chunkhash2=%s contains=%s",
                    self._rank, req.req_id, len(toks), len(chunks), len(keys),
                    already, _kh, _contains)

    # -- per-step (RPC thread, post-forward): poll completions ------------
    def get_finished(self) -> tuple[set, set]:
        # (finished_sending, finished_recving). Offload SAVES are fire-and-forget
        # (they don't free blocks), so finished_sending is ALWAYS empty; only
        # completed LOADS are reported, to wake parked seqs for suffix prefill.
        with self._lock:
            dl = set(self._done_load)
            self._done_save.clear()
            self._done_load.clear()
        return set(), dl

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
        try:
            hit = self._lookup_client.lookup(token_ids, lookup_id=str(seq.id))
        except Exception:
            logger.exception("LMCache offload lookup failed for seq %s", seq.id)
            return 0, False
        _lh = None
        try:
            tdb = getattr(self._lookup_client, "token_database", None)
            if tdb is not None:
                _lh = [k for (_s, _e, k) in list(
                    tdb.process_tokens(token_ids, make_key=False))[:3]]
        except Exception as e:
            _lh = f"err:{e}"
        logger.info("[OFFLOAD-LOOKUP] seq=%s num_prompt=%d hbm_cached=%d hit=%s lookuphash3=%s",
                    seq.id, num_prompt, int(seq.num_cached_tokens), hit, _lh)
        if not hit:
            return 0, False
        self._lookup_in_step.append(str(seq.id))
        need = int(hit) - int(seq.num_cached_tokens)
        if int(hit) == num_prompt:  # full-prompt hit → recompute last token
            need -= 1
        if need <= 0:
            return 0, False
        self._load_specs[str(seq.id)] = LoadSpec(
            hbm_cached_tokens=int(seq.num_cached_tokens),
            lmcache_cached_tokens=int(hit),
            can_load=False,
        )
        return need, True  # True => park in WAITING_FOR_REMOTE_KVS

    def update_state_after_alloc(self, seq) -> None:
        sid = str(seq.id)
        ls = self._load_specs.get(sid)
        logger.info("[OFFLOAD-ALLOC] seq=%s ls_found=%s num_cached_now=%s",
                    seq.id, ls is not None, int(getattr(seq, "num_cached_tokens", -1)))
        if ls is not None:
            ls.can_load = True
            self._reqs_need_recv[sid] = seq
        # Track for save; the prompt prefix is offloaded later, once prefill has
        # actually computed it (checked via prefix_hashes_published in build).
        if sid not in self._save_tracker:
            self._save_tracker[sid] = [seq, 0]

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        meta = LMCacheOffloadMetadata()
        meta.lookup_requests_in_step = self._lookup_in_step
        self._lookup_in_step = []

        # Loads
        logger.info("[OFFLOAD-BUILD] reqs_need_recv=%d", len(self._reqs_need_recv))
        for sid, seq in self._reqs_need_recv.items():
            ls = self._load_specs.pop(sid, None)
            if ls is None or not ls.can_load:
                logger.info("[OFFLOAD-LOAD-SKIP] seq=%s ls=%s can_load=%s",
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
            meta.add_request(LMCacheReqMeta(
                req_id=seq.id,
                token_ids=list(seq.token_ids[: ls.lmcache_cached_tokens]),
                block_ids=list(seq.block_table),
                load_spec=ls,
            ))
        # Saves: store the prompt prefix once prefill has computed it. We detect
        # "computed" via seq.prefix_hashes_published (set in postprocess after the
        # prefill step), so the blocks we D2H are already written -- no race with
        # forward. Persistent tracker: each chunk is stored once.
        chunk = self.chunk_size or 256
        for sid, entry in self._save_tracker.items():
            seq, saved = entry
            if sid in self._reqs_need_recv:
                continue  # loading this step; defer its save
            if not getattr(seq, "prefix_hashes_published", False):
                continue  # prefill not finished computing the prompt yet
            aligned = (int(seq.num_prompt_tokens) // chunk) * chunk
            if aligned <= saved:
                continue
            logger.info("[OFFLOAD-SAVE-EMIT] seq=%s num_prompt=%d aligned=%d saved=%d",
                        seq.id, int(seq.num_prompt_tokens), aligned, saved)
            meta.add_request(LMCacheReqMeta(
                req_id=seq.id,
                token_ids=list(seq.token_ids[:aligned]),
                block_ids=list(seq.block_table),
                save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
            ))
            entry[1] = aligned
        self._reqs_need_recv.clear()
        return meta

    def request_finished(self, seq) -> None:
        sid = str(seq.id)
        self._load_specs.pop(sid, None)
        self._reqs_need_recv.pop(sid, None)
        self._save_tracker.pop(sid, None)
        if self._lookup_client is not None:
            try:
                self._lookup_client.clear_lookup_status(sid)
            except Exception:
                pass
