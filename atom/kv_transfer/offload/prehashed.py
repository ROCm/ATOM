# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prehashed LMCache saves for the in-process offload path.

``ATOM_OFFLOAD_PREHASHED_SAVE``. Without it every save of the in-process
``lmcache_offload`` connector costs O(context) host work, most of it on the TP
workers and all of it repeated per save of the same request:

* the scheduler copies the request's whole computed token prefix into the
  save's ``LMCacheReqMeta.token_ids`` (``list(seq.token_ids[:aligned])``),
  pickles it into the step's connector metadata, and every TP worker unpickles
  it on its forward thread before the step's forward (~2 MiB / 10 ms at 512K
  tokens);
* each worker's save thread then turns it into a tensor (``tokens_to_tensor``)
  and ``LMCacheEngine.store`` re-hashes the WHOLE prefix chunk by chunk
  (``ChunkedTokenDatabase.process_tokens`` slices the tensor and calls
  ``.cpu().tolist()`` per 256-token chunk, ~2000 times at 512K) only to skip
  the masked, already-saved chunks.

That save-thread work holds the GIL in thousands of short slices, so for tens
of milliseconds after every prefill each rank's forward thread waits up to a
switch interval (5 ms) at every GIL hand-off, by a different amount on each
rank: the ranks fall out of step and the leading rank spins in the next
all-reduce. Measured on MiniMax-M3.1 (TP4, 510K context, 2K new tokens): a
28 ms engine gap before the first decode, then ~8 decode steps of 17-30 ms
instead of 6 ms.

With the flag, the scheduler hashes each prompt's LMCache chunk chain once (with
the lookup client's own ``ChunkedTokenDatabase``, so the hashes are exactly the
ones LMCache itself would compute), sends the tier lookup as those hashes
(LMCache's own lookup message), and emits each save as
:class:`~atom.kv_transfer.offload.metadata.PrehashedSave` -- the hashes of the
chunks the save adds. The worker builds the keys with LMCache's
``_make_key_by_hash`` and runs the same allocate / ``batched_from_gpu`` /
``batched_put`` sequence ``LMCacheEngine.store`` runs, for those chunks only.
Keys, chunk boundaries, the producer fence, source-safe reporting and
completion reporting are unchanged, so a later lookup or retrieve (which still
hashes tokens) finds exactly what was stored.

``ATOM_OFFLOAD_ASYNC_TIER_LOOKUP`` (needs the above) also takes the tier lookup
off the scheduler thread. The in-process lookup is a synchronous ZMQ round trip
to every TP worker, whose lookup-server thread builds one key per prompt chunk,
pins every hit chunk and touches them in the cache policy -- tens of
milliseconds at 512K tokens, during which the scheduler dispatches nothing. All
lookups now run on one scheduler-side thread (the ZMQ client is not
thread-safe), and a request whose HBM prefix already leaves fewer than
``OFFLOAD_MIN_LOAD_TOKENS`` uncached -- so no tier answer could be loaded -- is
admitted without waiting, through the scheduler's existing detach path
(``OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT``), which until now only the LMCache MP
client could take. Its lookup is still sent, for the CPU tier's LRU recency;
its answer lowers the request's save floor if the tier holds less than the HBM
prefix (those chunks are stored again), and its pins are released through the
next step's ``lookup_requests_in_step``, like any lookup that did not load.
"""

from __future__ import annotations

import logging
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch

from atom.kv_transfer.offload.metadata import (
    LMCacheReqMeta,
    PrehashedLoad,
    PrehashedSave,
    SaveSpec,
)
from atom.utils import envs

logger = logging.getLogger("atom")
# A prehashed load logs LMCache's own retrieve line through LMCache's logger.
_lmcache_logger = logging.getLogger("lmcache.v1.cache_engine")

# Rebuild the per-request chain table's liveness view every this many inserts.
_PRUNE_EVERY = 64


def _chunk_list(tokens, start: int, end: int) -> list[int]:
    """`tokens[start:end]` as a list of ints (`array.array` or list input)."""
    chunk = tokens[start:end]
    tolist = getattr(chunk, "tolist", None)
    return tolist() if callable(tolist) else list(chunk)


class PrehashedSaveSchedulerMixin:
    """Scheduler half: one chunk-hash chain per request, reused by lookup and saves.

    Mixed in ahead of a ``ChunkedOffloadSchedulerBase`` subclass whose lookup
    client is LMCache's in-process ``LMCacheLookupClient``. Call
    :meth:`_init_prehashed_save` at the end of ``__init__``.
    """

    def _init_prehashed_save(self) -> None:
        self._prehashed = False
        self._prehash_tdb = None
        self._prehash_verified = False
        # sid -> (weak ref to the sequence, its prompt's chunk-hash chain so far)
        self._prehash_chains: dict[str, tuple[object, list[int]]] = {}
        self._prehash_inserts = 0
        # ATOM_OFFLOAD_ASYNC_TIER_LOOKUP: the one thread every lookup RPC runs
        # on, and the answers of detached lookups for the scheduler thread.
        self._lookup_pool: ThreadPoolExecutor | None = None
        self._detached_answers: queue.SimpleQueue = queue.SimpleQueue()
        if not envs.ATOM_OFFLOAD_PREHASHED_SAVE:
            if envs.ATOM_OFFLOAD_ASYNC_TIER_LOOKUP:
                logger.warning(
                    "ATOM_OFFLOAD_ASYNC_TIER_LOOKUP ignored: it needs "
                    "ATOM_OFFLOAD_PREHASHED_SAVE"
                )
            return
        client = getattr(self, "_lookup_client", None)
        tdb = getattr(client, "token_database", None)
        transport = getattr(client, "transport", None)
        reason = None
        if client is None or tdb is None or transport is None:
            reason = "no in-process LMCache lookup client"
        elif getattr(client, "enable_blending", False):
            reason = "LMCache blending sends tokens, not hashes"
        elif not callable(getattr(tdb, "_hash_tokens", None)) or not callable(
            getattr(tdb, "_get_init_hash", None)
        ):
            reason = "LMCache token database has no chunk hash hooks"
        elif getattr(tdb, "hash_func", None) is hash and os.environ.get(
            "PYTHONHASHSEED", "random"
        ) in ("", "random"):
            # Builtin hash differs per process without a fixed seed, so a key
            # hashed here would not be the one a worker hashes on retrieve.
            reason = "builtin hash without PYTHONHASHSEED"
        if reason is not None:
            logger.warning(
                "ATOM_OFFLOAD_PREHASHED_SAVE ignored: %s; saves ship tokens", reason
            )
            return
        self._prehash_tdb = tdb
        self._prehashed = True
        if envs.ATOM_OFFLOAD_ASYNC_TIER_LOOKUP:
            self._lookup_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="lmc-tier-lookup"
            )
        logger.info(
            "LMCache offload scheduler: prehashed saves on (chunk=%d), "
            "async tier lookup %s",
            int(self.chunk_size),
            "on" if self._lookup_pool is not None else "off",
        )

    # -- the chain ---------------------------------------------------------
    def _prune_prehash_chains(self) -> None:
        live = []
        for sid, (ref, _chain) in self._prehash_chains.items():
            seq = ref()
            if seq is None:
                continue
            if sid in self._save_tracker or self._load_lifecycles.get(sid) is seq:
                live.append(sid)
        self._prehash_chains = {sid: self._prehash_chains[sid] for sid in live}

    def _prefix_chunk_hashes(self, seq, n_chunks: int) -> list[int] | None:
        """LMCache's hashes of `seq`'s first `n_chunks` chunks, or None.

        Extends the request's cached chain only by the chunks it lacks, so a
        request hashes each prompt chunk once over its lifetime (its lookup
        usually hashes the whole prompt; its saves then only slice). None turns
        the feature off: the first chain is checked against LMCache's own
        ``process_tokens`` and a mismatch must not reach a store.
        """
        if not self._prehashed:
            return None
        chunk = int(self.chunk_size)
        n_chunks = max(0, int(n_chunks))
        tokens = seq.token_ids
        if n_chunks * chunk > len(tokens):
            raise ValueError(
                f"prehashed save: {n_chunks} chunks exceed {len(tokens)} tokens "
                f"of seq {seq.id}"
            )
        sid = str(seq.id)
        entry = self._prehash_chains.get(sid)
        if entry is None or entry[0]() is not seq:
            chain: list[int] = []
            self._prehash_chains[sid] = (self._weak_seq(seq), chain)
            self._prehash_inserts += 1
            if self._prehash_inserts % _PRUNE_EVERY == 0:
                self._prune_prehash_chains()
                self._prehash_chains.setdefault(sid, (self._weak_seq(seq), chain))
        else:
            chain = entry[1]
        if len(chain) < n_chunks:
            t0 = time.perf_counter()
            self._extend_prehash_chain(chain, tokens, n_chunks)
            self._perf_bump("prehash_us", int((time.perf_counter() - t0) * 1e6))
            self._perf_bump("prehash_chunks", n_chunks)
        if not self._prehash_verified and n_chunks > 0:
            if not self._verify_prehash_chain(tokens, chain, n_chunks):
                return None
        return chain

    def _extend_prehash_chain(self, chain: list[int], tokens, n_chunks: int) -> None:
        """Append the chained hashes of `tokens`' chunks `len(chain)` to `n_chunks`."""
        tdb = self._prehash_tdb
        chunk = int(self.chunk_size)
        prev = chain[-1] if chain else tdb._get_init_hash()
        for i in range(len(chain), n_chunks):
            prev = tdb._hash_tokens(
                _chunk_list(tokens, i * chunk, (i + 1) * chunk), prev
            )
            chain.append(prev)

    def _verify_prehash_chain(self, tokens, chain: list[int], n_chunks: int) -> bool:
        """Compare the first chain with LMCache's own hashing, once."""
        chunk = int(self.chunk_size)
        expected = [
            h
            for (_s, _e, h) in self._prehash_tdb.process_tokens(
                _chunk_list(tokens, 0, n_chunks * chunk), make_key=False
            )
        ][:n_chunks]
        if expected != chain[:n_chunks]:
            logger.error(
                "ATOM_OFFLOAD_PREHASHED_SAVE disabled: the scheduler's chunk "
                "hashes do not match LMCache's process_tokens (first %d chunks); "
                "saves ship tokens",
                n_chunks,
            )
            self._prehashed = False
            self._prehash_chains.clear()
            return False
        self._prehash_verified = True
        return True

    # -- lookup: LMCache's own message, built from the cached chain ---------
    def _lookup_rpc(self, hashes: list[int], sid: str) -> int | None:
        """`LMCacheLookupClient.lookup` minus its hashing; None if no rank answered.

        Runs on the scheduler thread, or on the lookup thread under
        ATOM_OFFLOAD_ASYNC_TIER_LOOKUP -- never both, the transport's sockets
        are not thread-safe.
        """
        chunk = int(self.chunk_size)
        client = self._lookup_client
        responses = client.transport.send_and_recv_all(
            [hashes, [chunk] * len(hashes), sid, ""]
        )
        if not responses:
            return None
        results = [int.from_bytes(resp, "big") for resp in responses]
        if len(set(results)) > 1:
            logger.debug(
                "LMCache prehashed lookup %s: hits differ across ranks %s", sid, results
            )
        hit = min(results)
        status = getattr(client, "reqs_status", None)
        if isinstance(status, dict):
            status[sid] = hit
        return hit

    def _tier_lookup(self, seq, sid: str) -> int | None:
        chunk = int(self.chunk_size)
        n_chunks = int(seq.num_prompt_tokens) // chunk
        chain = self._prefix_chunk_hashes(seq, n_chunks)
        if chain is None:
            return super()._tier_lookup(seq, sid)
        start = min(self._tier_lookup_start(seq) // chunk, n_chunks)
        if start == n_chunks:
            # LMCacheLookupClient.lookup: no hashes -> a hit of 0, no RPC.
            return 0
        hashes = chain[start:n_chunks]
        if self._lookup_pool is not None:
            hit = self._lookup_pool.submit(self._lookup_rpc, hashes, sid).result()
        else:
            hit = self._lookup_rpc(hashes, sid)
        # LMCacheLookupClient answers 0, not None, when the transport failed.
        # The tier's contiguous run from `start` counts from the prompt's head;
        # none at all is no hit.
        return start * chunk + hit if hit else 0

    # -- detached lookups (ATOM_OFFLOAD_ASYNC_TIER_LOOKUP) -------------------
    def can_detach_lookup(self, seq, num_hbm_tokens: int) -> bool:
        if self._lookup_pool is None or not self._prehashed:
            return super().can_detach_lookup(seq, num_hbm_tokens)
        if not self._load_ruled_out(seq, num_hbm_tokens):
            return False
        sid = str(seq.id)
        if (
            sid in self._load_specs
            or sid in self._lookup_results
            or sid in self._active_load_operations
        ):
            return False
        return self._lookup_len(seq) > 0

    def detach_lookup(self, seq, num_hbm_tokens: int) -> bool:
        """Admit `seq` now; its lookup runs on the lookup thread.

        The save floor starts at the HBM prefix, as the base detach does, and
        `_drain_detached_answers` lowers it if the tier holds less.
        """
        if self._lookup_pool is None or not self._prehashed:
            return super().detach_lookup(seq, num_hbm_tokens)
        if not self.can_detach_lookup(seq, num_hbm_tokens):
            return False
        chunk = int(self.chunk_size)
        n_chunks = int(seq.num_prompt_tokens) // chunk
        chain = self._prefix_chunk_hashes(seq, n_chunks)
        if chain is None:
            return False
        sid = str(seq.id)
        self._lookup_defer_since.pop(sid, None)
        self._forget_tier_hit(sid)
        floor = self._chunk_floor(max(0, int(num_hbm_tokens)))
        if floor > 0:
            self._hit_save_floors[sid] = floor
        if n_chunks > 0:
            self._lookup_pool.submit(
                self._detached_lookup_job, chain[:n_chunks], sid, self._weak_seq(seq)
            )
        self._perf_bump("lookup_detached")
        return True

    def _detached_lookup_job(self, hashes: list[int], sid: str, ref) -> None:
        try:
            hit = self._lookup_rpc(hashes, sid)
        except Exception:  # noqa: BLE001  # a lost answer only costs recency
            logger.warning("LMCache detached lookup %s failed", sid, exc_info=True)
            hit = None
        self._detached_answers.put((sid, ref, hit))

    def _drain_detached_answers(self) -> None:
        """Apply detached lookups' answers on the scheduler thread.

        Every answered lookup pinned its hit on the workers, so its id joins
        this step's `lookup_requests_in_step`, whose dispatch unpins it. A
        tier holding less than the request's save floor lowers the floor (the
        missing chunks are stored again); no answer lowers it to 0.
        """
        while True:
            try:
                sid, ref, hit = self._detached_answers.get_nowait()
            except queue.Empty:
                return
            if sid not in self._lookup_in_step:
                self._lookup_in_step.append(sid)
            seq = ref()
            if seq is None:
                continue
            floor = 0 if hit is None else self._chunk_floor(max(0, int(hit)))
            if self._lower_save_floor(sid, seq, floor):
                self._perf_bump("detached_floor_lowered")

    def build_connector_meta(self):
        if self._lookup_pool is not None:
            self._drain_detached_answers()
        return super().build_connector_meta()

    # -- save: ship the new chunks' hashes ---------------------------------
    def _build_save_request(
        self,
        seq,
        saved: int,
        aligned: int,
        operation,
        block_ids: list[int],
        is_last_prefill: bool,
    ) -> LMCacheReqMeta | None:
        chunk = int(self.chunk_size)
        chain = None
        if saved % chunk == 0 and aligned % chunk == 0:
            try:
                chain = self._prefix_chunk_hashes(seq, aligned // chunk)
            except Exception:  # noqa: BLE001  # the token path still works
                logger.warning(
                    "LMCache prehashed save of %s fell back to tokens",
                    seq.id,
                    exc_info=True,
                )
        if chain is None:
            return super()._build_save_request(
                seq, saved, aligned, operation, block_ids, is_last_prefill
            )
        self._perf_bump("prehashed_saves")
        return LMCacheReqMeta(
            req_id=seq.id,
            token_ids=[],
            block_ids=block_ids,
            save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
            is_last_prefill=is_last_prefill,
            save_operation=operation,
            prehashed_save=PrehashedSave(
                start=int(saved),
                hashes=tuple(chain[saved // chunk : aligned // chunk]),
            ),
        )

    # -- load: ship the loaded chunks' hashes (ATOM_OFFLOAD_PREHASHED_LOAD) --
    def _build_load_request(self, seq, load_spec, load_operation, transfer_end):
        """The load as the hashes of chunks ``[hbm, lmc)``, not the prompt prefix.

        The token path pickles ``prompt[:transfer_end]`` into the step's
        metadata, every worker unpickles it on its forward thread, and its
        load thread re-hashes the whole prefix chunk by chunk, handing the GIL
        to the forward thread about twice per chunk.
        """
        chunk = int(self.chunk_size)
        hbm = int(load_spec.hbm_cached_tokens)
        lmc = int(load_spec.lmcache_cached_tokens)
        chain = None
        if (
            envs.ATOM_OFFLOAD_PREHASHED_LOAD
            and hbm % chunk == 0
            and lmc % chunk == 0
            and transfer_end == lmc
        ):
            try:
                chain = self._prefix_chunk_hashes(seq, lmc // chunk)
            except Exception:  # noqa: BLE001  # the token path still works
                logger.warning(
                    "LMCache prehashed load of %s fell back to tokens",
                    seq.id,
                    exc_info=True,
                )
        if chain is None:
            return super()._build_load_request(
                seq, load_spec, load_operation, transfer_end
            )
        self._perf_bump("prehashed_loads")
        return LMCacheReqMeta(
            req_id=seq.id,
            token_ids=[],
            block_ids=list(seq.block_table),
            load_spec=load_spec,
            load_operation=load_operation,
            prehashed_load=PrehashedLoad(
                start=hbm, hashes=tuple(chain[hbm // chunk : lmc // chunk])
            ),
        )


class PrehashedSaveWorkerMixin:
    """Worker half: store a :class:`PrehashedSave` without its tokens.

    Mixed in ahead of ``DenseOffloadConnector`` (or a subclass). A request
    without ``prehashed_save`` takes the parent's token path unchanged.
    """

    def _do_save_req(self, req: LMCacheReqMeta, *, producer_event=None) -> None:
        prehashed = getattr(req, "prehashed_save", None)
        if prehashed is None:
            return super()._do_save_req(req, producer_event=producer_event)
        return self._do_prehashed_save(req, prehashed, producer_event)

    @torch.inference_mode()
    def _do_prehashed_save(
        self, req: LMCacheReqMeta, prehashed: PrehashedSave, producer_event
    ) -> None:
        """``LMCacheEngine.store`` for the chunks ``prehashed`` names.

        The same steps in the same order as ``store``'s non-layerwise path --
        its early exits, a prefix-contiguous allocation that stops at the first
        refusal, ``batched_from_gpu`` with the store kwargs this connector
        passes on the token path, then ``batched_put`` -- with the keys built
        from the shipped hashes instead of from re-hashing the prefix, and
        under ``torch.inference_mode`` as ``store`` is (the KV pool and
        LMCache's buffers may be inference tensors). KV events and LMCache's
        store statistics are not produced.

        ``must_store_all`` (a lazy demotion): chunks the tier already holds are
        skipped, and a refused allocation or an engine that stores nothing
        reports a failure instead of ``store``'s shorter success.
        """
        chunk = int(self.chunk_size)
        start = int(prehashed.start)
        hashes = prehashed.hashes
        if start % chunk != 0:
            raise ValueError(
                f"prehashed save of {req.req_id} starts off a chunk boundary: "
                f"{start} % {chunk}"
            )
        if not hashes:
            self._record_store_terminal(req, True)
            return
        t0 = time.perf_counter()
        synchronize = getattr(producer_event, "synchronize", None)
        if synchronize is not None:
            synchronize()
        engine = self._engine
        must_store_all = prehashed.must_store_all
        is_passive = getattr(engine, "_is_passive", None)
        if (
            not engine.is_healthy()
            or (callable(is_passive) and is_passive())
            or engine.is_frozen()
        ):
            # `store` returns without storing in all three cases, and the token
            # path then reports the save as done.
            self._record_store_terminal(req, not must_store_all)
            return
        token_database = engine.token_database
        storage_manager = engine.storage_manager
        shapes = engine.metadata.get_shapes(chunk)
        dtypes = engine.metadata.get_dtypes()
        busy_loop = engine.config.get_extra_config_value("force_store_wait", False)
        keys, memory_objs, starts, ends = [], [], [], []
        refused = False
        for index, chunk_hash in enumerate(hashes):
            key = token_database._make_key_by_hash(int(chunk_hash))
            if must_store_all and storage_manager.contains(key) is not None:
                continue
            memory_obj = storage_manager.allocate(
                shapes, dtypes, busy_loop=busy_loop, fmt=engine.fmt
            )
            if memory_obj is None:
                refused = True
                logger.warning(
                    "LMCache offload: CPU tier under pressure, storing %d of %d "
                    "chunks of req=%s%s",
                    len(memory_objs),
                    len(hashes),
                    req.req_id,
                    " (reported failed)" if must_store_all else "",
                )
                break
            chunk_start = start + index * chunk
            keys.append(key)
            memory_objs.append(memory_obj)
            starts.append(chunk_start)
            ends.append(chunk_start + chunk)
        if memory_objs:
            self._reset_gpu_connector_transfer_stats()
            gpu_connector = engine.gpu_connector
            track_source = getattr(gpu_connector, "track_save_source", None)
            source_context = (
                track_source(req.save_operation)
                if getattr(self, "_early_release", False) and callable(track_source)
                else nullcontext()
            )
            kwargs = {"block_ids": req.block_ids, "req_id": str(req.req_id)}
            if producer_event is not None:
                kwargs["producer_event"] = producer_event
            try:
                with source_context:
                    gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
            except BaseException:
                # Not put, so nothing else drops the allocation's reference.
                for memory_obj in memory_objs:
                    try:
                        memory_obj.ref_count_down()
                    except Exception:  # noqa: BLE001  # best-effort cleanup
                        pass
                raise
            storage_manager.batched_put(
                keys, memory_objs, transfer_spec=None, location=engine.store_location
            )
        if self._profile_enabled():
            stats = self._last_gpu_connector_transfer_stats()
            logger.info(
                "[OFFLOAD-SAVE-PROF] rank=%s req=%s prehashed start=%d chunks=%d "
                "stored=%d total_bytes=%d producer_fenced=%d pack_ms=%.2f "
                "copy_ms=%.2f sync_ms=%.2f total_ms=%.2f",
                getattr(self, "_rank", "?"),
                req.req_id,
                start,
                len(hashes),
                len(memory_objs),
                int(stats.get("total_bytes", 0)),
                int(stats.get("producer_fenced", 0)),
                float(stats.get("pack_ms", 0.0)),
                float(stats.get("copy_ms", 0.0)),
                float(stats.get("sync_ms", 0.0)),
                (time.perf_counter() - t0) * 1000,
            )
        self._record_store_terminal(req, not (refused and must_store_all))

    # -- load ----------------------------------------------------------------
    def _check_prehashed_load_engine(self) -> None:
        """Refuse at startup an engine whose retrieve `_retrieve` does not mirror."""
        if not envs.ATOM_OFFLOAD_PREHASHED_LOAD:
            return
        engine = self._engine
        unsupported = [
            name
            for name in (
                "async_loading",
                "save_only_first_rank",
                "remove_after_retrieve",
            )
            if getattr(engine, name, False)
        ]
        if unsupported:
            raise ValueError(
                "ATOM_OFFLOAD_PREHASHED_LOAD does not support an LMCache engine "
                f"with {', '.join(unsupported)}"
            )

    @torch.inference_mode()
    def _retrieve(self, req: LMCacheReqMeta, hbm: int, lmc: int) -> tuple[bool, int]:
        """``LMCacheEngine.retrieve`` for the chunks ``req.prehashed_load`` names.

        Its synchronous path in the same order -- the lookup pin's location
        (else the storage manager's prefix block mapping), ``batched_get``,
        the run of chunks up to the first one that cannot be read,
        ``batched_to_gpu``, one reference dropped per chunk -- with the keys
        built from the shipped hashes. LMCache's retrieve statistics and its
        ``Retrieved ...`` line are kept, so logs and metrics read the same.
        """
        prehashed = getattr(req, "prehashed_load", None)
        if prehashed is None:
            return super()._retrieve(req, hbm, lmc)
        chunk = int(self.chunk_size)
        start = int(prehashed.start)
        if start != hbm or hbm + len(prehashed.hashes) * chunk != lmc:
            raise ValueError(
                f"prehashed load of {req.req_id} covers [{start}, "
                f"{start + len(prehashed.hashes) * chunk}), not [{hbm}, {lmc})"
            )
        engine = self._engine
        is_passive = getattr(engine, "_is_passive", None)
        if not engine.is_healthy() or (callable(is_passive) and is_passive()):
            return False, 0
        sid = str(req.req_id)
        stats = engine.stats_monitor.on_retrieve_request(lmc - hbm)
        storage_manager = engine.storage_manager
        make_key = engine.token_database._make_key_by_hash
        infos = [
            (make_key(int(h)), start + i * chunk, start + (i + 1) * chunk)
            for i, h in enumerate(prehashed.hashes)
        ]
        pins = engine.lookup_pins.get(sid)
        if pins is not None and len(pins) == 1:
            block_mapping = {next(iter(pins)): infos}
        else:
            block_mapping = storage_manager.get_block_mapping(infos)
        got: list[tuple[object, int, int]] = []
        first_missing = None
        for location, blocks in block_mapping.items():
            objs = storage_manager.batched_get(
                keys=[key for key, _s, _e in blocks], location=location
            )
            used = 0
            for (_key, s, e), obj in zip(blocks, objs):
                if obj is None:
                    if first_missing is None or s < first_missing:
                        first_missing = s
                    break
                got.append((obj, s, e))
                used += 1
            for obj in objs[used:]:
                if obj is not None:
                    obj.ref_count_down()
        if first_missing is not None:
            kept = []
            for item in got:
                if item[2] <= first_missing:
                    kept.append(item)
                else:
                    item[0].ref_count_down()
            got = kept
        retrieved = 0
        nbytes = 0
        try:
            if got:
                objs, starts, ends = zip(*got)
                nbytes = sum(obj.get_size() for obj in objs)
                with stats.profile_to_gpu():
                    engine.gpu_connector.batched_to_gpu(
                        list(objs),
                        list(starts),
                        list(ends),
                        block_ids=req.block_ids,
                        req_id=sid,
                    )
                retrieved = sum(e - s for _o, s, e in got)
        finally:
            for obj, _s, _e in got:
                obj.ref_count_down()
        engine.stats_monitor.on_retrieve_finished(stats, retrieved)
        cost = stats.time_to_retrieve()
        _lmcache_logger.info(
            "[req_id=%s] Retrieved %d out of %d required tokens "
            "(from %d total tokens). size: %.4f gb, "
            "cost %.4f ms, throughput: %.4f GB/s; prehashed",
            sid,
            retrieved,
            lmc - hbm,
            lmc,
            nbytes / 1024**3,
            cost * 1000,
            nbytes / cost / 1024**3 if cost > 0 else 0,
        )
        # The chunks run from `hbm` with no gap: a missing one ends the run.
        return retrieved == lmc - hbm, retrieved


__all__ = ["PrehashedSaveSchedulerMixin", "PrehashedSaveWorkerMixin"]
