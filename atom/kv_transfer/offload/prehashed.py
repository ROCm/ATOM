# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prehashed LMCache saves for the in-process offload path.

On the token path every save ships the request's whole token prefix to each TP
worker, whose save thread re-hashes it chunk by chunk to skip the chunks already
saved: O(context) Python per save that holds the GIL against the forward thread.
Here the scheduler hashes each prompt's chunk chain once, with the lookup
client's own ``ChunkedTokenDatabase``, answers the tier lookup from it, and ships
a save as the hashes of the chunks it adds. The worker builds the keys with
``_make_key_by_hash`` and runs ``LMCacheEngine.store``'s non-layerwise steps for
those chunks only, so the stored keys are the ones the token path stores.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext

import torch

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import (
    salt_kwargs,
    tier_cache_salt,
    wait_save_fence,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheReqMeta,
    PrehashedSave,
    SaveSpec,
)

logger = logging.getLogger("atom")

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
        self._prehash_ship_tokens = False
        self._prehash_tdb = None
        self._prehash_verified = False
        # sid -> (weak ref to the sequence, its prompt's chunk-hash chain so far)
        self._prehash_chains: dict[str, tuple[object, list[int]]] = {}
        self._prehash_inserts = 0
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
            logger.info("LMCache prehashed saves off: %s; saves ship tokens", reason)
            return
        self._prehash_tdb = tdb
        self._prehashed = True
        # The worker's store emits KV events with the chunks' token ids.
        kvc = getattr(self._config, "kv_transfer_config", None)
        self._prehash_ship_tokens = bool(
            getattr(offcfg.build_lmcache_config(kvc), "enable_kv_events", False)
        )
        logger.info(
            "LMCache offload scheduler: prehashed saves on (chunk=%d)",
            int(self.chunk_size),
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
        if (
            not self._prehash_verified
            and n_chunks > 0
            and not self._verify_prehash_chain(tokens, chain, n_chunks)
        ):
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
                "LMCache prehashed saves off: the scheduler's chunk "
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
    def _lookup_configs(self, seq) -> str:
        """The lookup message's `request_configs` frame for `seq`: its media
        salt as JSON, empty for an unsalted prompt (as `LMCacheLookupClient`)."""
        configs = self._tier_lookup_kwargs(seq).get("request_configs")
        return json.dumps(configs) if configs else ""

    def _lookup_rpc(self, hashes: list[int], sid: str, configs: str) -> int | None:
        """`LMCacheLookupClient.lookup` minus its hashing; None if no rank answered."""
        chunk = int(self.chunk_size)
        client = self._lookup_client
        responses = client.transport.send_and_recv_all(
            [hashes, [chunk] * len(hashes), sid, configs]
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
        hit = self._lookup_rpc(chain[start:n_chunks], sid, self._lookup_configs(seq))
        # LMCacheLookupClient answers 0, not None, when the transport failed.
        # The tier's contiguous run from `start` counts from the prompt's head;
        # none at all is no hit.
        return start * chunk + hit if hit else 0

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
            except Exception:  # the token path still works
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
        return self._request_meta_type(
            req_id=seq.id,
            token_ids=[],
            block_ids=block_ids,
            save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
            is_last_prefill=is_last_prefill,
            save_operation=operation,
            cache_salt=tier_cache_salt(seq),
            prehashed_save=PrehashedSave(
                start=int(saved),
                hashes=tuple(chain[saved // chunk : aligned // chunk]),
                token_ids=(
                    tuple(_chunk_list(seq.token_ids, saved, aligned))
                    if self._prehash_ship_tokens
                    else None
                ),
            ),
        )


class PrehashedSaveWorkerMixin:
    """Worker half: store a :class:`PrehashedSave` without its tokens.

    Mixed in ahead of ``DenseOffloadConnector`` (or a subclass). A request
    without ``prehashed_save`` takes the parent's token path unchanged.
    """

    def _do_save_req(self, req: LMCacheReqMeta, *, producer_event) -> None:
        prehashed = getattr(req, "prehashed_save", None)
        if prehashed is None:
            return super()._do_save_req(req, producer_event=producer_event)
        return self._do_prehashed_save(req, prehashed, producer_event)

    @torch.inference_mode()
    def _do_prehashed_save(
        self, req: LMCacheReqMeta, prehashed: PrehashedSave, producer_event
    ) -> None:
        """``LMCacheEngine.store``'s non-layerwise steps for the chunks ``prehashed``
        names, keyed from the shipped hashes: its early exits, a prefix-contiguous
        allocation that stops at the first refusal, ``batched_from_gpu`` with the
        token path's kwargs, then ``batched_put``, recording the same store
        statistics and KV events.
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
        # As the token path: wait on the producing stream before this thread
        # reads KV, and fail closed without a fence.
        wait_save_fence(producer_event)
        engine = self._engine
        is_passive = getattr(engine, "_is_passive", None)
        if (
            not engine.is_healthy()
            or (callable(is_passive) and is_passive())
            or engine.is_frozen()
        ):
            # `store` returns without storing in all three cases, and the token
            # path then reports the save as done.
            self._record_store_terminal(req, True)
            return
        token_database = engine.token_database
        storage_manager = engine.storage_manager
        shapes = engine.metadata.get_shapes(chunk)
        dtypes = engine.metadata.get_dtypes()
        busy_loop = engine.config.get_extra_config_value("force_store_wait", False)
        # The media salt travels as the key's tags, as on the token path.
        salt = salt_kwargs(req.cache_salt)
        request_configs = salt.get("request_configs")
        # LMCache's store statistics and KV events, as `store` records them.
        store_stats = engine.stats_monitor.on_store_request(len(hashes) * chunk)
        events = (
            engine.kv_events if getattr(engine, "kv_events_enabled", False) else None
        )
        tokens = prehashed.token_ids or ()
        prev_hash = 0
        keys, memory_objs, starts, ends = [], [], [], []
        with store_stats.profile_process_tokens():
            for index, chunk_hash in enumerate(hashes):
                memory_obj = storage_manager.allocate(
                    shapes, dtypes, busy_loop=busy_loop, fmt=engine.fmt
                )
                if memory_obj is None:
                    logger.warning(
                        "LMCache offload: CPU tier under pressure, storing %d of %d "
                        "chunks of req=%s",
                        len(memory_objs),
                        len(hashes),
                        req.req_id,
                    )
                    break
                chunk_start = start + index * chunk
                keys.append(
                    token_database._make_key_by_hash(int(chunk_hash), request_configs)
                )
                memory_objs.append(memory_obj)
                starts.append(chunk_start)
                ends.append(chunk_start + chunk)
                if events is not None:
                    from lmcache.utils import CacheStoreEvent

                    # `store`'s fields, its token slice [start, end] included.
                    offset = index * chunk
                    events.append(
                        CacheStoreEvent(
                            block_hashes=[keys[-1].chunk_hash],
                            parent_block_hash=None if chunk_start == 0 else prev_hash,
                            token_ids=list(tokens[offset : offset + chunk + 1]),
                            block_size=chunk,
                            lora_id=None,
                            medium=torch.device("cpu"),
                            lora_name=None,
                        )
                    )
                    prev_hash = keys[-1].chunk_hash
        if memory_objs:
            self._reset_gpu_connector_transfer_stats()
            gpu_connector = engine.gpu_connector
            track_source = getattr(gpu_connector, "track_save_source", None)
            source_context = (
                track_source(req.save_operation)
                if getattr(self, "_early_release", False) and callable(track_source)
                else nullcontext()
            )
            kwargs = {
                "block_ids": req.block_ids,
                "req_id": str(req.req_id),
                "producer_event": producer_event,
                **salt,
            }
            try:
                with source_context, store_stats.profile_from_gpu():
                    gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
            except BaseException:
                # Not put, so nothing else drops the allocation's reference.
                for memory_obj in memory_objs:
                    try:
                        memory_obj.ref_count_down()
                    except Exception:  # best-effort cleanup
                        logger.debug(
                            "LMCache prehashed save: releasing a chunk failed",
                            exc_info=True,
                        )
                raise
            with store_stats.profile_put():
                storage_manager.batched_put(
                    keys,
                    memory_objs,
                    transfer_spec=None,
                    location=engine.store_location,
                )
            engine.stats_monitor.on_store_finished(store_stats, len(keys) * chunk)
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
        self._record_store_terminal(req, True)


__all__ = ["PrehashedSaveSchedulerMixin", "PrehashedSaveWorkerMixin"]
