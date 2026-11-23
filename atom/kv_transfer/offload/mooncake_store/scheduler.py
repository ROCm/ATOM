# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler side of the native Mooncake Store offload.

The policy, the digest-carrying requests and the paced save retries are the
store-agnostic :class:`KVStoreOffloadScheduler`'s. This module answers the one
question it asks a transport, how much of a prompt the Store holds, from the
masters of its pools, minding the Store's taint marker, and renews the read
leases a load relies on.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.kv_store import scheduler as _store_scheduler
from atom.kv_transfer.offload.kv_store.scheduler import (  # noqa: F401  # re-exported
    _LOOKUP_STATS_INTERVAL_S,
    LOOKUP_BACKOFF_S,
    LOOKUP_RECONNECT_INTERVAL_S,
    KVStoreOffloadScheduler,
    LookupWarnings,
    StoreLookupBase,
    monotonic_clock_id,
    pause_lookups,
    prompt_chunk_hashes,
)
from atom.kv_transfer.offload.lookup_client import LookupPinStatus
from atom.kv_transfer.offload.metadata import LoadSpec
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store.async_lookup import (
    AsyncStoreLookup,
    DaemonExecutor,
)
from atom.kv_transfer.offload.mooncake_store.config import (
    MooncakeStoreOffloadConfig,
    StorePool,
    check_engine_compatibility,
    parse_mooncake_store_config,
)
from atom.kv_transfer.offload.mooncake_store.keys import (
    DIGEST_BYTES,
    STORE_TAINT_KEY,
    store_namespace,
)
from atom.model_engine.sequence import SequenceStatus
from atom.utils import envs

logger = logging.getLogger("atom")

# Lookups pause a pool after a failed call (`pause_lookups`): Mooncake's setup
# holds the GIL and retries a master that does not answer for minutes, a
# master that is down holds each call for seconds, a hung one for coro_rpc's
# 30 s request timeout.
# A hit older than this is looked up again right before its load is
# dispatched. Only the lookup's read lease (10 s by default) keeps the chunks
# from eviction, and a request can wait far longer than that for KV blocks.
_LOOKUP_REUSE_S = 1.0
# Store lookups an arrival batch may queue at once (`offer_lookups`).
_MAX_OFFERS_PER_BATCH = 64


def require_monotonic_clock_id() -> str:
    """The boot ID the saves are timed on, or a refusal to start without one.

    See :func:`atom.kv_transfer.offload.kv_store.scheduler.require_monotonic_clock_id`.
    """
    return _store_scheduler.require_monotonic_clock_id("the Mooncake Store offload")


class _PoolLookups:
    """One Store pool as the scheduler's lookups reach it.

    Its client is opened on the first lookup that asks it. A pool that cannot
    be reached, or fails a lookup, is not asked until ``paused_until``: every
    call to it holds the lookup worker, and the other pools answer meanwhile. A pool found to carry the taint marker is ``tainted`` for good
    and holds nothing for the lookups from then on.
    """

    __slots__ = ("client", "executor", "paused_until", "pool", "tainted")

    def __init__(self, pool: StorePool) -> None:
        self.pool = pool
        self.client: store_client.MooncakeStoreClient | None = None
        # Its own daemon worker when several pools are asked at once
        # (`_StoreLookup._exists_on`); None for a single pool.
        self.executor: DaemonExecutor | None = None
        self.paused_until = 0.0
        self.tainted = False


class _StoreLookup(StoreLookupBase):
    """The chunked scheduler's lookup client, answered by the Store's masters.

    A hit is the prefix every rank holds: per rank, the run of chunks present
    from chunk 0, and the minimum over ranks, since a load is all-or-nothing
    across the PP stages and TP ranks. ``batch_is_exist`` is one master RPC per
    call, and it grants each present key a read lease (10 s by default) that
    keeps it from being evicted for that long and no longer, so the scheduler
    asks again before it dispatches a load whose hit is older than
    ``_LOOKUP_REUSE_S``.

    Every pool is asked for every key (the scheduler cannot see which NIC a
    rank writes through). A key is present if any pool holds it, absent if a
    pool answered and none holds it, and unanswered if no pool answered: a
    pool that fails is absent for every key another pool answers. A lookup
    answers only what the unanswered keys cannot change, so a pool that fails
    never makes a hit longer.

    Each call also asks for the Store's taint marker (:mod:`.taint`). A pool
    that carries it holds nothing from then on, for this scheduler's life, and
    is not asked again; one that cannot say whether it does answers nothing.

    A pool's Store client is opened on the first lookup that asks it, not at
    construction: every PP stage builds a scheduler, and only the head ever
    schedules. A pool the scheduler cannot reach, or that fails a lookup, is
    not asked for a while (``LOOKUP_RECONNECT_INTERVAL_S``, ``LOOKUP_BACKOFF_S``).
    """

    def __init__(
        self,
        cfg: MooncakeStoreOffloadConfig,
        *,
        namespace: str,
        world: int,
        chunk_tokens: int,
    ) -> None:
        super().__init__(namespace=namespace, world=world, chunk_tokens=chunk_tokens)
        self._cfg = cfg
        # Exact while every server on these pools puts each rank on the same
        # NIC; see the offload README for what a different placement costs.
        self._pools = [_PoolLookups(pool) for pool in cfg.store_masters()]
        # Several pools are asked at once, each on a daemon worker of its own:
        # a master RPC that hangs must not keep the engine process alive at
        # exit, as a `ThreadPoolExecutor` worker would (it is joined then).
        if len(self._pools) > 1:
            for index, pool in enumerate(self._pools):
                pool.executor = DaemonExecutor(f"mooncake-store-lookup-{index}")
        # Per pool master: when its next warning may be logged.
        self._warnings = LookupWarnings("Mooncake Store offload")

    def lookup(self, seq: Any, lookup_id: str | None = None) -> int | None:
        """Tokens of ``seq``'s prompt prefix the Store holds on every rank.

        None, not 0, when the Store did not answer -- no pool reached, or the
        keys no pool answered decide the hit: the scheduler retries a
        non-answer later, where a 0 would be remembered as a miss.

        Store keys chain from `Sequence.cache_seed`, the source of the media
        salt, so the lookup takes no salt argument.
        """
        started = time.perf_counter()
        hashes = prompt_chunk_hashes(seq, self._chunk_tokens)
        chunks = len(hashes) // DIGEST_BYTES
        if not chunks:
            return 0
        keys = self._rank_keys(hashes, 0, chunks)
        merged = self._ask(keys)
        if merged is None:
            return None
        # Per rank, the run of chunks from chunk 0 some pool holds: at least
        # `held` (an unanswered key counts as absent), at most `bound` (as
        # present).
        held, bound = self._rank_prefixes(merged, chunks)
        hit_chunks = min(held)
        if min(bound) != hit_chunks:
            logger.debug(
                "[OFFLOAD-LOOKUP] lookup_id=%s chunks=%d: keys no pool answered "
                "decide the hit (%d to %d chunks); no answer",
                lookup_id,
                chunks,
                hit_chunks,
                min(bound),
            )
            return None
        self._count_lookup(sum(present - hit_chunks for present in held))
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

    def renew(self, seq: Any, start: int, end: int) -> int:
        """Renew the read leases of every rank's chunks in tokens ``[start, end)``.

        Right before a load reads them: the leases of the lookup behind it may
        have run out while the request waited for KV blocks. Returns how many
        of those objects some pool holds; the load goes ahead whatever the
        answer, and its all-or-nothing get fails on a chunk the Store lost.
        A pool that fails is paused as after a lookup.
        """
        hashes = prompt_chunk_hashes(seq, self._chunk_tokens)
        first = int(start) // self._chunk_tokens
        last = min(int(end) // self._chunk_tokens, len(hashes) // DIGEST_BYTES)
        pools = self._pools_to_ask() if first < last else []
        if not pools:
            return 0
        keys = self._rank_keys(hashes, first, last)
        answers = [a for a in self._exists_on(pools, keys) if a is not None]
        return sum(1 for codes in zip(*answers) if 1 in codes)

    # The two below serve only the asynchronous renewal (ATOM_STORE_ASYNC_RENEW).
    def renew_key_snapshot(self, seq: Any, start: int, end: int) -> tuple[str, ...]:
        """Snapshot keys on the scheduler thread before background renewal."""
        # `renew`'s range math, repeated so that `renew` stays main's code.
        hashes = prompt_chunk_hashes(seq, self._chunk_tokens)
        first = int(start) // self._chunk_tokens
        last = min(int(end) // self._chunk_tokens, len(hashes) // DIGEST_BYTES)
        return tuple(self._rank_keys(hashes, first, last))

    def renew_keys(self, keys: tuple[str, ...]) -> int:
        answers = self._ask(list(keys)) if keys else None
        return sum(1 for code in answers or () if code == 1)

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Nothing to release: a Store read lease expires by itself."""

    def close(self) -> None:
        """Stop the pool workers; one inside a hung RPC is a daemon thread."""
        for pool in self._pools:
            if pool.executor is not None:
                pool.executor.shutdown()

    def _ask(self, keys: list[str]) -> list[int | None] | None:
        """Per key: 1 if a pool holds it, 0 if a pool answered and none holds
        it, None if no pool answered. None when no pool answered any key.
        """
        answers: list[list[int]] = []
        if any(pool.tainted for pool in self._pools):
            answers.append([0] * len(keys))
        pools = self._pools_to_ask()
        if pools:
            answers += [
                answer for answer in self._exists_on(pools, keys) if answer is not None
            ]
        if not answers:
            return None
        merged: list[int | None] = []
        for codes in zip(*answers):
            if 1 in codes:
                merged.append(1)
            elif 0 in codes:
                merged.append(0)
            else:
                merged.append(None)
        if all(code is None for code in merged):
            return None
        return merged

    def _pools_to_ask(self) -> list[_PoolLookups]:
        """The pools neither tainted nor paused, each connected, connecting
        those not yet."""
        now = time.monotonic()
        return [
            pool
            for pool in self._pools
            if not pool.tainted
            and now >= pool.paused_until
            and (pool.client is not None or self._connect(pool))
        ]

    def _exists_on(
        self, pools: list[_PoolLookups], keys: list[str]
    ) -> list[list[int] | None]:
        """Each pool's codes for ``keys``, concurrently.

        None for a pool that raised or could not say whether it carries the
        taint marker, and all 0 for one that does. A pool that raised or
        answered an error is paused (`pause_lookups`).
        """
        asked = [STORE_TAINT_KEY, *keys]
        asked_at = time.monotonic()
        futures = (
            [pool.executor.submit(pool.client.exists, asked) for pool in pools]
            if all(pool.executor is not None for pool in pools)
            else None
        )
        answers: list[list[int] | None] = []
        for index, pool in enumerate(pools):
            try:
                taint, *codes = (
                    futures[index].result()
                    if futures is not None
                    else pool.client.exists(asked)
                )
            except Exception as exc:  # noqa: BLE001  # the pool is not asked a while
                self._warn(
                    pool,
                    "the Store pool at %s failed a lookup of %d keys; it is not "
                    "asked for %.0fs",
                    pool.pool.master,
                    len(keys),
                    pause_lookups(pool, LOOKUP_BACKOFF_S, asked_at),
                    exc_info=exc,
                )
                answers.append(None)
                continue
            if taint == 1:
                pool.tainted = True
                logger.error(
                    "Mooncake Store offload: the Store pool at %s carries the "
                    "taint marker %s: a put a worker gave up on may still write "
                    "memory the Store has since given to other objects. Lookups "
                    "count nothing it holds from now on, so no load reads it; "
                    "restart the Store, then this engine, to load from it again",
                    pool.pool.master,
                    STORE_TAINT_KEY,
                )
                answers.append([0] * len(keys))
                continue
            errors = [code for code in (taint, *codes) if code < 0]
            if errors:
                self._warn(
                    pool,
                    "the Store pool at %s got %d error(s) for %d keys, first %s; "
                    "it is not asked for %.0fs",
                    pool.pool.master,
                    len(errors),
                    len(asked),
                    store_client.describe(errors[0]),
                    pause_lookups(pool, LOOKUP_BACKOFF_S, asked_at),
                )
            # A pool that cannot say whether it carries the marker answers
            # nothing: what it holds may be what a late write reached.
            answers.append(None if taint < 0 else codes)
        return answers

    def _connect(self, pool: _PoolLookups) -> bool:
        """Open ``pool``'s client; on failure pause it and return False."""
        started = time.monotonic()
        try:
            # Lookups are master RPCs only; tcp needs no NIC of its own.
            pool.client = store_client.MooncakeStoreClient(
                local_hostname=self._cfg.local_hostname,
                metadata_server=pool.pool.metadata,
                master_server_addr=pool.pool.master,
                protocol="tcp",
                rdma_devices="",
            )
        except Exception:
            pause = pause_lookups(pool, LOOKUP_RECONNECT_INTERVAL_S, started)
            logger.warning(
                "Mooncake Store offload: the scheduler cannot reach the Store "
                "pool at %s; lookups do not ask it for the next %.0fs",
                pool.pool.master,
                pause,
                exc_info=True,
            )
            return False
        logger.info(
            "Mooncake Store offload: scheduler lookups use %s", pool.pool.master
        )
        return True

    def _warn(
        self, pool: _PoolLookups, message: str, *args: Any, exc_info: Any = None
    ) -> None:
        """At most one warning per pool and `LOOKUP_WARNING_INTERVAL_S`."""
        self._warnings.warn(pool.pool.master, message, *args, exc_info=exc_info)


class MooncakeStoreOffloadScheduler(KVStoreOffloadScheduler):
    """Chunked offload scheduling with the Mooncake Store as the tier."""

    LABEL = "Mooncake Store offload"
    # It answers `lookup_ready`, so the scheduler may start its loads early.
    supports_early_load = True

    def __init__(self, config: Any) -> None:
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        validated_kv_role(kvc)
        cfg = parse_mooncake_store_config(kvc)
        check_engine_compatibility(cfg, config)
        require_monotonic_clock_id()
        world = offcfg.lmcache_replica_world_size(config)
        namespace = store_namespace(config, cfg.chunk_tokens)
        # time.monotonic() of each request's last lookup; see `_ensure_lookup_pin`.
        self._looked_up_at: dict[str, float] = {}
        self._async_renew = envs.ATOM_STORE_ASYNC_RENEW
        self.cancels_undispatched_loads = self._async_renew
        super().__init__(
            config,
            store_cfg=cfg,
            namespace=namespace,
            world=world,
            lookup_client=AsyncStoreLookup(
                _StoreLookup(
                    cfg, namespace=namespace, world=world, chunk_tokens=cfg.chunk_tokens
                )
            ),
            endpoint="masters=" + ",".join(pool.master for pool in cfg.store_masters()),
        )
        if self._async_renew:
            # Bound only with the flag: without it these are the class's.
            self._pending_loads: dict[str, Any] = {}
            self._renew_before_load = self._renew_before_load_async
            self._clear_pending_load = self._clear_pending_load_async
            self.build_connector_meta = self._build_connector_meta_async
            self.has_pending_work = self._has_pending_work_async
        # The Store reads whole chunks into fresh blocks, so an HBM hit that
        # ends inside a chunk is lowered to the chunk floor
        # (`BlockManager.lower_hit_to_chunk`). The registry holds the flag, one
        # source for this and the DP router's tier profile.
        self.lowers_hbm_hit_to_chunk = KVConnectorFactory.lowers_hbm_hit_to_chunk(
            "mooncake_store"
        )

    def offer_lookups(self, seqs):
        # Called on arrival after tokenization, before admission/prefill.
        if not self._do_load:
            return
        offered = 0
        for seq in seqs:
            # Not a request that left WAITING meanwhile, and a bounded number per
            # arrival batch: the rest reach their lookup through the scheduler's
            # own prefetch (`prefetch_lookups`), as the queue drains.
            if offered >= _MAX_OFFERS_PER_BATCH:
                break
            if self._lookup_wanted(seq) and self._lookup_client.submit_if(
                seq, str(seq.id), lambda seq=seq: seq.status == SequenceStatus.WAITING
            ):
                offered += 1

    def _lookup_starting(self, sid: str) -> None:
        # Stamped before asking: the lease runs from the master's answer.
        self._looked_up_at[sid] = time.monotonic()

    def _fresh_tier_lookup(self, seq: Any, sid: str) -> int | None:
        hit = super()._fresh_tier_lookup(seq, sid)
        # An answer the asynchronous lookup took earlier is as old as its query.
        started = self._lookup_client.query_started(sid)
        if started is not None and started < self._looked_up_at.get(sid, started):
            self._looked_up_at[sid] = started
        return hit

    def _ensure_lookup_pin(
        self, seq: Any, sid: str, spec: LoadSpec
    ) -> bool | LookupPinStatus:
        """Renew the read leases of what a load reads; the load always goes ahead.

        The base trusts a hit it still holds from this request's lookup, a
        live pin for LMCache. A Store lookup holds its chunks only for its
        read lease, which a request waiting for KV blocks outlives, so for a
        hit older than ``_LOOKUP_REUSE_S``, or one answered from the memo, the
        chunks the load reads, ``[hbm, transfer_end)`` of every rank, are asked
        for again: that renews their leases right before the get.

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
            pending is not None
            and pending[0] is seq
            and looked_up_at is not None
            and time.monotonic() - looked_up_at <= _LOOKUP_REUSE_S
        ):
            return True
        start = int(spec.hbm_cached_tokens)
        end = int(
            spec.lmcache_cached_tokens
            if spec.transfer_end_tokens is None
            else spec.transfer_end_tokens
        )
        return self._renew_before_load(seq, sid, start, end)

    def _renew_before_load(self, seq: Any, sid: str, start: int, end: int) -> bool:
        """Renew the leases of tokens ``[start, end)`` now, on this thread.

        With ``ATOM_STORE_ASYNC_RENEW`` ``__init__`` binds
        ``_renew_before_load_async`` instead.
        """
        held = self._lookup_client.renew(seq, start, end)
        logger.debug(
            "[OFFLOAD-LOOKUP] seq=%s renewed the leases of tokens [%d, %d): the "
            "Store holds %d of their objects; dispatched whatever it holds",
            seq.id,
            start,
            end,
            held,
        )
        return True

    def _clear_pending_load(self, sid: str) -> None:
        super()._clear_pending_load(sid)
        self._looked_up_at.pop(sid, None)

    def _renew_before_load_async(
        self, seq: Any, sid: str, start: int, end: int
    ) -> bool | LookupPinStatus:
        """`_renew_before_load` with ``ATOM_STORE_ASYNC_RENEW`` (bound in ``__init__``).

        The renewal runs on the renewal executor
        (``AsyncStoreLookup.renew_async``). While it has not finished and was
        submitted less than its timeout ago (queued or running) the load stays
        parked and is retried next step (``PENDING``), so its GET goes out one
        or more steps later than on the synchronous path. A fresh hit (the
        caller returns before this), a rejected submission, or a new submission while an earlier renewal is
        still stuck (``renewal_busy``) dispatches in the same step; a renewal
        already queued behind a hung RPC stays parked until its own timeout.
        ``renew_parked`` counts the polls while parked (the idle drain polls
        every 1 ms, so it is not a step count). Any other outcome (finished,
        timed out, error, cancelled, rejected, busy) dispatches the get exactly
        as the synchronous path does (zero held keys included: the get
        validates residency); each fallback reason is counted as
        ``renew_fallback_<reason>``. Cancellation before dispatch retires the
        renewal and the reservation without worker work. Once dispatch has
        handed the blocks to workers, cleanup waits for their terminal report
        so it cannot release blocks still used by DMA.
        """
        # The keys of `[start, end)` never change for one request: they are
        # built only when the renewal is submitted, not on each retry.
        state = self._lookup_client.renew_async(
            sid,
            (start, end),
            functools.partial(
                self._lookup_client.backend.renew_key_snapshot, seq, start, end
            ),
        )
        if state == "pending":
            self._perf_bump("renew_parked")
            return LookupPinStatus.PENDING
        if state != "ready":
            self._perf_bump(f"renew_fallback_{state}")
            n = self._perf_counters[f"renew_fallback_{state}"]
            # The lookup logs an "error" itself, at most once a minute; the
            # counter counts every one.
            if state != "error" and (n == 1 or n % 100 == 0):
                logger.warning(
                    "Mooncake Store renewal for %s: %s; dispatching the load "
                    "anyway (%d so far)",
                    sid,
                    state,
                    n,
                )
        return True

    def _build_connector_meta_async(self):
        """The class's build, then the loads left ``PENDING`` wait for the next.

        ``_reqs_need_recv`` is emptied by every build; a load whose renewal has
        not finished is put back, so it keeps reporting work
        (``has_pending_work``) and is retried on the next build.
        """
        self._pending_loads.clear()
        meta = type(self).build_connector_meta(self)
        if self._pending_loads:
            self._reqs_need_recv.update(self._pending_loads)
            self._pending_loads.clear()
        return meta

    def _has_pending_work_async(self) -> bool:
        """The class's, plus loads handed to workers and not yet reported.

        With the flag a parked load's GET can go out from an idle or shutdown
        drain build; it leaves `_reqs_need_recv` then, so without this the
        drain could stop polling before the load's terminal report.
        """
        return type(self).has_pending_work(self) or bool(self._active_load_operations)

    def _clear_pending_load_async(self, sid: str) -> None:
        self._lookup_client.cancel_renewal(sid)
        type(self)._clear_pending_load(self, sid)

    def cancel_undispatched_load(self, seq: Any) -> bool:
        sid = str(seq.id)
        if (
            self._load_lifecycles.get(sid) is not seq
            or self._reqs_need_recv.get(sid) is not seq
        ):
            return False
        # Workers never got this load: `build_connector_meta` drops a load
        # from `_reqs_need_recv` in the same call that hands it to workers, so
        # one still there is undispatched. A running renewal retains immutable
        # keys, not these blocks.
        self.cancel_pending_load(seq)
        return True


__all__ = [
    "MooncakeStoreOffloadScheduler",
    "monotonic_clock_id",
    "prompt_chunk_hashes",
    "require_monotonic_clock_id",
]
