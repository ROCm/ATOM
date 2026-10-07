# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler side of the native Mooncake Store offload.

The policy -- what to save and when, whether a hit is worth loading, early
block release, exact operation completions -- is all
:class:`ChunkedOffloadSchedulerBase`'s. This module answers the one question it
asks a transport, how much of a prompt the Store holds, and shapes the requests
the workers receive: each carries the prompt's chunk digests, which name the
Store objects, instead of the prompt's token ids.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from atom.kv_transfer.disaggregation.types import LoadOperationId, SaveOperationId
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import (
    max_pending_saves,
    validated_kv_role,
)
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec, SaveSpec
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store.config import (
    MooncakeStoreOffloadConfig,
    check_engine_compatibility,
    parse_mooncake_store_config,
)
from atom.kv_transfer.offload.mooncake_store.keys import (
    DIGEST_BYTES,
    chain_seed,
    chunk_hash_chain,
    rank_key_prefix,
    store_namespace,
)
from atom.utils import envs

logger = logging.getLogger("atom")

# Seconds a lookup answers "no answer" after the Store could not be reached,
# before it tries again, or ten times as long as the failed connect blocked
# (`_LOOKUP_BACKOFF_FACTOR`), whichever is longer. Connecting blocks the
# scheduler thread (Mooncake's setup holds the GIL, and retries a master that
# does not answer for minutes), so a missing master must not cost that on
# every step.
_RECONNECT_INTERVAL_S = 30.0
# After a lookup the Store failed to answer, lookups answer "no answer" without
# asking it for this long, or for this many times as long as the failed one
# blocked the scheduler thread, whichever is longer: a master that is down
# holds each call for seconds, a hung one for coro_rpc's 30 s request timeout.
_LOOKUP_BACKOFF_S = 10.0
_LOOKUP_BACKOFF_FACTOR = 10.0
# Seconds between two warnings about failing lookups.
_LOOKUP_WARNING_INTERVAL_S = 60.0
# Seconds between two `[OFFLOAD-LOOKUP-STATS]` lines.
_LOOKUP_STATS_INTERVAL_S = 60.0
# A hit older than this is looked up again right before its load is
# dispatched. Only the lookup's read lease (10 s by default) keeps the chunks
# from eviction, and a request can wait far longer than that for KV blocks.
_LOOKUP_REUSE_S = 1.0


class _PromptView:
    """A sequence's prompt, standing in for its token ids in a lookup.

    The chunked scheduler hands the lookup client whatever
    ``_lookup_token_ids`` returns and tests it for emptiness. Copying a long
    prompt into a list on every lookup would cost more than the lookup; this
    carries the sequence, whose chunk digests are hashed once.
    """

    __slots__ = ("seq",)

    def __init__(self, seq: Any) -> None:
        self.seq = seq

    def __len__(self) -> int:
        return int(self.seq.num_prompt_tokens)


def prompt_chunk_hashes(seq: Any, chunk_tokens: int) -> bytes:
    """Digests of every full chunk of ``seq``'s prompt, hashed once per sequence.

    Kept on the sequence (``_mooncake_store_chunk_hashes``): its prompt never
    changes, and lookups, saves and loads of it all need the same chain.
    """
    num_prompt = int(seq.num_prompt_tokens)
    expected = (num_prompt // int(chunk_tokens)) * DIGEST_BYTES
    cached = getattr(seq, "_mooncake_store_chunk_hashes", None)
    if isinstance(cached, bytes) and len(cached) == expected:
        return cached
    hashes = chunk_hash_chain(
        seq.token_ids,
        chunk_tokens,
        num_tokens=num_prompt,
        seed=chain_seed(getattr(seq, "cache_seed", None)),
    )
    seq._mooncake_store_chunk_hashes = hashes
    return hashes


class _StoreLookup:
    """The chunked scheduler's lookup client, answered by the Store's masters.

    A hit is the prefix every rank holds: per rank, the run of chunks present
    from chunk 0, and the minimum over ranks, since a load is all-or-nothing
    across the PP stages and TP ranks. ``batch_is_exist`` is one master RPC per
    call, and it grants each present key a read lease (10 s by default) that
    keeps it from being evicted for that long and no longer, so the scheduler
    asks again before it dispatches a load whose hit is older than
    ``_LOOKUP_REUSE_S``.

    The Store client is opened on the first lookup, not at construction: every
    PP stage builds a scheduler, and only the head ever schedules. A lookup
    blocks the scheduler thread, so one the Store fails to answer pauses the
    next ones (``_LOOKUP_BACKOFF_S``).
    """

    def __init__(
        self,
        cfg: MooncakeStoreOffloadConfig,
        *,
        namespace: str,
        world: int,
        chunk_tokens: int,
    ) -> None:
        self._cfg = cfg
        self._namespace = namespace
        self._world = int(world)
        self._chunk_tokens = int(chunk_tokens)
        # Per-NIC pools: a rank's chunks live in the pool of its NIC, which the
        # scheduler cannot see, so every pool is asked for every key.
        self._masters = cfg.store_masters()
        self._clients: list[store_client.MooncakeStoreClient] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._retry_connect_at = 0.0
        self._retry_lookup_at = 0.0
        self._next_warning_at = 0.0
        # Answered lookups; those where the ranks held prefixes of different
        # lengths; and the rank objects past the shared prefix they found,
        # present but unusable while another rank lacks the chunk -- evicted,
        # failed, or still being written by a slower stage.
        self.lookups = 0
        self.uneven_lookups = 0
        self.stranded_objects = 0
        self._next_stats_at = time.monotonic() + _LOOKUP_STATS_INTERVAL_S

    def lookup(self, token_ids: Any, lookup_id: str | None = None) -> int | None:
        """Tokens of the prompt's prefix the Store holds on every rank.

        None, not 0, when the Store did not answer -- unreachable, a failed
        call, or any per-key error: the scheduler retries a non-answer later,
        where a 0 would be remembered as a miss.
        """
        started = time.perf_counter()
        if isinstance(token_ids, _PromptView):
            hashes = prompt_chunk_hashes(token_ids.seq, self._chunk_tokens)
        else:
            hashes = chunk_hash_chain(token_ids, self._chunk_tokens)
        chunks = len(hashes) // DIGEST_BYTES
        if not chunks:
            return 0
        if time.monotonic() < self._retry_lookup_at:
            return None
        clients = self._connected_clients()
        if clients is None:
            return None
        digests = [
            hashes[index * DIGEST_BYTES : (index + 1) * DIGEST_BYTES].hex()
            for index in range(chunks)
        ]
        keys = [
            prefix + digest
            for rank in range(self._world)
            for prefix in (rank_key_prefix(self._namespace, rank, self._world),)
            for digest in digests
        ]
        asked_at = time.monotonic()
        try:
            codes = self._exists(clients, keys)
        except Exception:  # noqa: BLE001  # any failure is a non-answer
            self._warn(
                "a Store lookup of %d keys failed; lookups pause for %.0fs",
                len(keys),
                self._back_off(asked_at),
                exc_info=True,
            )
            return None
        errors = [code for code in codes if code < 0]
        if errors:
            self._warn(
                "a Store lookup got %d error(s) for %d keys, first %s; lookups "
                "pause for %.0fs",
                len(errors),
                len(keys),
                store_client.describe(errors[0]),
                self._back_off(asked_at),
            )
            return None
        presents = []
        for rank in range(self._world):
            shard = codes[rank * chunks : (rank + 1) * chunks]
            presents.append(
                next((index for index, code in enumerate(shard) if code != 1), chunks)
            )
        hit_chunks = min(presents)
        self._count_lookup(sum(present - hit_chunks for present in presents))
        hit = hit_chunks * self._chunk_tokens
        logger.debug(
            "[OFFLOAD-LOOKUP] lookup_id=%s chunks=%d keys=%d hit=%d elapsed_ms=%.2f",
            lookup_id,
            chunks,
            len(keys),
            hit,
            (time.perf_counter() - started) * 1000,
        )
        return hit

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Nothing to release: a Store read lease expires by itself."""

    def close(self) -> None:
        clients, self._clients = self._clients or [], None
        for client in clients:
            _close_quietly(client)
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _exists(
        self, clients: list[store_client.MooncakeStoreClient], keys: list[str]
    ) -> list[int]:
        """One answer per key: 1 if any pool holds it, else an error, else 0."""
        if len(clients) == 1:
            return clients[0].exists(keys)
        assert self._executor is not None
        answers = list(self._executor.map(lambda client: client.exists(keys), clients))
        merged = []
        for codes in zip(*answers):
            if 1 in codes:
                merged.append(1)
            else:
                merged.append(min(min(codes), 0))
        return merged

    def _back_off(self, asked_at: float) -> float:
        """Pause lookups after one the Store failed to answer; the seconds paused."""
        now = time.monotonic()
        pause = max(_LOOKUP_BACKOFF_S, _LOOKUP_BACKOFF_FACTOR * (now - asked_at))
        self._retry_lookup_at = now + pause
        return pause

    def _connected_clients(self) -> list[store_client.MooncakeStoreClient] | None:
        if self._clients is not None:
            return self._clients
        now = time.monotonic()
        if now < self._retry_connect_at:
            return None
        clients: list[store_client.MooncakeStoreClient] = []
        try:
            for pool in self._masters:
                # Lookups are master RPCs only; tcp needs no NIC of its own.
                clients.append(
                    store_client.MooncakeStoreClient(
                        local_hostname=self._cfg.local_hostname,
                        metadata_server=pool.metadata,
                        master_server_addr=pool.master,
                        protocol="tcp",
                        rdma_devices="",
                        lookup_batch_keys=self._cfg.lookup_batch_keys,
                    )
                )
        except Exception:
            for client in clients:
                _close_quietly(client)
            # From when the failed setup returned, which can be minutes after
            # it started: a pause counted from before it would already be over.
            failed_at = time.monotonic()
            pause = max(
                _RECONNECT_INTERVAL_S, _LOOKUP_BACKOFF_FACTOR * (failed_at - now)
            )
            self._retry_connect_at = failed_at + pause
            logger.warning(
                "Mooncake Store offload: the scheduler cannot reach the Store "
                "(%s); lookups answer nothing for the next %.0fs",
                ", ".join(pool.master for pool in self._masters),
                pause,
                exc_info=True,
            )
            return None
        if len(clients) > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=len(clients), thread_name_prefix="mooncake-store-lookup"
            )
        self._clients = clients
        logger.info(
            "Mooncake Store offload: scheduler lookups use %s",
            ", ".join(pool.master for pool in self._masters),
        )
        return clients

    def _count_lookup(self, stranded: int) -> None:
        """Count an answered lookup, logging the totals once a minute."""
        self.lookups += 1
        if stranded:
            self.uneven_lookups += 1
            self.stranded_objects += stranded
        now = time.monotonic()
        if now < self._next_stats_at:
            return
        self._next_stats_at = now + _LOOKUP_STATS_INTERVAL_S
        logger.info(
            "[OFFLOAD-LOOKUP-STATS] lookups=%d uneven_lookups=%d stranded_objects=%d",
            self.lookups,
            self.uneven_lookups,
            self.stranded_objects,
        )

    def _warn(self, message: str, *args: Any, exc_info: bool = False) -> None:
        now = time.monotonic()
        if now < self._next_warning_at:
            logger.debug("Mooncake Store offload: " + message, *args)
            return
        self._next_warning_at = now + _LOOKUP_WARNING_INTERVAL_S
        logger.warning(
            "Mooncake Store offload: " + message + "; the request prefills instead",
            *args,
            exc_info=exc_info,
        )


def _close_quietly(client: store_client.MooncakeStoreClient) -> None:
    try:
        client.close()
    except Exception:
        logger.debug("Mooncake Store offload: client close failed", exc_info=True)


class MooncakeStoreOffloadScheduler(ChunkedOffloadSchedulerBase):
    """Chunked offload scheduling with the Mooncake Store as the tier."""

    # Workers publish one source-safe group per chunk once its GPU pack is
    # done (the block GPU connector), separately from the store terminal.
    _supports_early_block_release = True

    def __init__(self, config: Any) -> None:
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        validated_kv_role(kvc)
        cfg = parse_mooncake_store_config(kvc)
        check_engine_compatibility(cfg, config)
        world = offcfg.lmcache_replica_world_size(config)
        namespace = store_namespace(config, cfg.chunk_tokens)
        self._store_cfg = cfg
        self._namespace = namespace
        # time.monotonic() of each request's last lookup; see `_ensure_lookup_pin`.
        self._looked_up_at: dict[str, float] = {}
        super().__init__(
            config,
            chunk_size=cfg.chunk_tokens,
            lookup_client=_StoreLookup(
                cfg, namespace=namespace, world=world, chunk_tokens=cfg.chunk_tokens
            ),
        )
        extra = kvc.get("kv_connector_extra_config", kvc) or {}
        # Unbounded unless asked for, like the dense offload: one save per
        # request is in flight at most either way.
        self._max_pending_saves = (
            max_pending_saves(kvc, envs.OFFLOAD_COPY_WORKERS)
            if "max_pending_saves" in extra
            else None
        )
        logger.info(
            "Mooncake Store offload scheduler: namespace=%s world=%d chunk=%d "
            "masters=%s max_pending_saves=%s abandon_timeout=%.0fs",
            namespace,
            world,
            cfg.chunk_tokens,
            ",".join(pool.master for pool in cfg.store_masters()),
            self._max_pending_saves,
            cfg.save_abandon_timeout_s,
        )

    def _lookup_token_ids(self, seq: Any) -> _PromptView:
        return _PromptView(seq)

    def _fresh_tier_lookup(self, seq: Any, sid: str) -> int | None:
        # Stamped before asking: the lease runs from the master's answer.
        self._looked_up_at[sid] = time.monotonic()
        return super()._fresh_tier_lookup(seq, sid)

    def _ensure_lookup_pin(self, seq: Any, sid: str, spec: LoadSpec) -> bool:
        """Renew a hit's read leases before its load, which always goes ahead.

        The base trusts a hit it still holds from this request's lookup, a
        live pin for LMCache. A Store lookup holds its chunks only for its
        read lease, which a request waiting for KV blocks outlives, so a hit
        older than ``_LOOKUP_REUSE_S``, or one answered from the memo, is
        looked up again: that renews the lease right before the get.

        The answer does not decide the load. The engine parked the request
        when it admitted it, before this dispatch, and only the load's report
        wakes it: a load dropped here would leave it parked for good, holding
        its KV blocks and a ``max_num_seqs`` slot. A prefix evicted meanwhile
        fails the worker's all-or-nothing get instead, and the request prefills
        once that failure reports (``load_failed``).
        """
        looked_up_at = self._looked_up_at.get(sid)
        pending = self._lookup_results.get(sid)
        if (
            pending is None
            or pending[0] is not seq
            or looked_up_at is None
            or time.monotonic() - looked_up_at > _LOOKUP_REUSE_S
        ):
            hit = self._fresh_tier_lookup(seq, sid)
            if hit is None or hit < int(spec.lmcache_cached_tokens):
                logger.debug(
                    "[OFFLOAD-LOOKUP] seq=%s the Store now answers %s for a load "
                    "of %d tokens; dispatched anyway, a missing chunk fails it",
                    seq.id,
                    hit,
                    int(spec.lmcache_cached_tokens),
                )
        return True

    def _clear_pending_load(self, sid: str) -> None:
        super()._clear_pending_load(sid)
        self._looked_up_at.pop(sid, None)

    def request_finished(self, seq: Any) -> None:
        super().request_finished(seq)
        self._looked_up_at.pop(str(seq.id), None)

    def _may_emit_save(self) -> bool:
        cap = self._max_pending_saves
        return cap is None or len(self._save_inflight) < cap

    def _build_save_request(
        self,
        seq: Any,
        saved: int,
        aligned: int,
        operation: SaveOperationId,
        block_ids: list[int],
        is_last_prefill: bool,
    ) -> LMCacheReqMeta:
        """Save chunks ``[saved, aligned)``: the digests of ``[0, aligned)``.

        No token ids: every PP stage receives each request, and the digests
        are all the worker needs to name its objects. The dispatch time starts
        the workers' source read deadline where the reclaim clock starts.
        """
        hashes = prompt_chunk_hashes(seq, self.chunk_size)
        return LMCacheReqMeta(
            req_id=seq.id,
            token_ids=[],
            block_ids=block_ids,
            save_spec=SaveSpec(skip_leading_tokens=saved, can_save=True),
            is_last_prefill=is_last_prefill,
            save_operation=operation,
            chunk_hashes=hashes[: (aligned // self.chunk_size) * DIGEST_BYTES],
            dispatched_at=time.time(),
        )

    def _build_load_request(
        self,
        seq: Any,
        load_spec: LoadSpec,
        load_operation: LoadOperationId,
        transfer_end: int,
    ) -> LMCacheReqMeta:
        """Load ``[hbm, transfer_end)``: the digests of ``[0, transfer_end)``."""
        hashes = prompt_chunk_hashes(seq, self.chunk_size)
        return LMCacheReqMeta(
            req_id=seq.id,
            token_ids=[],
            block_ids=list(seq.block_table),
            load_spec=load_spec,
            load_operation=load_operation,
            chunk_hashes=hashes[: (transfer_end // self.chunk_size) * DIGEST_BYTES],
        )

    def _new_load_operation(self, seq: Any) -> LoadOperationId:
        operation = super()._new_load_operation(seq)
        # The worker restores KV straight into allocated blocks, which never
        # pass through `hash_blocks()`. Naming where the load starts lets
        # `Scheduler._mark_offload_load_ready` publish the loaded prefix into
        # the HBM prefix index once the load succeeds, so a later turn hits it
        # in HBM instead of loading it again; without it the suffix prefill
        # finds its parent block unhashed and cannot register its own blocks
        # either. Loads are dispatched only from a chunk-aligned post-allocate
        # HBM frontier, which is always a hash-block boundary.
        seq.offload_load_start_tokens = int(seq.num_cached_tokens)
        return operation

    def save_abandon_timeout_s(self) -> float:
        """Seconds a deferred save may sit before the engine reclaims it.

        A setting of this connector (``mooncake_store.save_abandon_timeout_s``)
        rather than a pin timeout: the workers stop reading a save's source
        blocks well before it runs out (see the worker's source read deadline).
        """
        return self._store_cfg.save_abandon_timeout_s


__all__ = ["MooncakeStoreOffloadScheduler", "prompt_chunk_hashes"]
