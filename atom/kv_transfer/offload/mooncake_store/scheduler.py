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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.kv_store import scheduler as _store_scheduler
from atom.kv_transfer.offload.kv_store.scheduler import (  # noqa: F401  # re-exported
    _LOOKUP_STATS_INTERVAL_S,
    KVStoreOffloadScheduler,
    StoreLookupBase,
    monotonic_clock_id,
    prompt_chunk_hashes,
)
from atom.kv_transfer.offload.metadata import LoadSpec
from atom.kv_transfer.offload.mooncake_store import client as store_client
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

logger = logging.getLogger("atom")

# Seconds lookups do not ask a pool the scheduler could not reach. Connecting
# blocks the scheduler thread (Mooncake's setup holds the GIL, and retries a
# master that does not answer for minutes), so a missing master must not cost
# that on every step.
_RECONNECT_INTERVAL_S = 30.0
# The same for a pool that failed a lookup: a master that is down holds each
# call for seconds, a hung one for coro_rpc's 30 s request timeout.
_LOOKUP_BACKOFF_S = 10.0
# Either pause lasts at least this many times as long as the failed call
# blocked the scheduler thread (`_StoreLookup._pause`).
_LOOKUP_BACKOFF_FACTOR = 10.0
# Seconds between two warnings about one pool's failing lookups.
_LOOKUP_WARNING_INTERVAL_S = 60.0
# A hit older than this is looked up again right before its load is
# dispatched. Only the lookup's read lease (10 s by default) keeps the chunks
# from eviction, and a request can wait far longer than that for KV blocks.
_LOOKUP_REUSE_S = 1.0


def require_monotonic_clock_id() -> str:
    """The boot ID the saves are timed on, or a refusal to start without one.

    See :func:`atom.kv_transfer.offload.kv_store.scheduler.require_monotonic_clock_id`.
    """
    return _store_scheduler.require_monotonic_clock_id("the Mooncake Store offload")


class _PoolLookups:
    """One Store pool as the scheduler's lookups reach it.

    Its client is opened on the first lookup that asks it. A pool that cannot
    be reached, or fails a lookup, is not asked until ``paused_until``: every
    call to it blocks the scheduler thread, and the other pools answer
    meanwhile. A pool found to carry the taint marker is ``tainted`` for good
    and holds nothing for the lookups from then on.
    """

    __slots__ = ("client", "paused_until", "pool", "tainted")

    def __init__(self, pool: StorePool) -> None:
        self.pool = pool
        self.client: store_client.MooncakeStoreClient | None = None
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
    schedules. A lookup blocks the scheduler thread, so a pool the scheduler
    cannot reach, or that fails a lookup, is not asked for a while
    (``_RECONNECT_INTERVAL_S``, ``_LOOKUP_BACKOFF_S``).
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
        # Several pools are asked at once; the executor starts no thread until
        # the first lookup.
        self._executor = (
            ThreadPoolExecutor(
                max_workers=len(self._pools),
                thread_name_prefix="mooncake-store-lookup",
            )
            if len(self._pools) > 1
            else None
        )
        # Per pool master: when its next warning may be logged.
        self._next_warning_at: dict[str, float] = {}

    def lookup(self, seq: Any, lookup_id: str | None = None) -> int | None:
        """Tokens of ``seq``'s prompt prefix the Store holds on every rank.

        None, not 0, when the Store did not answer -- no pool reached, or the
        keys no pool answered decide the hit: the scheduler retries a
        non-answer later, where a 0 would be remembered as a miss.
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

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Nothing to release: a Store read lease expires by itself."""

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
        answered an error is paused (`_pause`).
        """
        asked = [STORE_TAINT_KEY, *keys]
        asked_at = time.monotonic()
        futures = (
            [self._executor.submit(pool.client.exists, asked) for pool in pools]
            if self._executor is not None
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
                    self._pause(pool, _LOOKUP_BACKOFF_S, asked_at),
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
                    self._pause(pool, _LOOKUP_BACKOFF_S, asked_at),
                )
            # A pool that cannot say whether it carries the marker answers
            # nothing: what it holds may be what a late write reached.
            answers.append(None if taint < 0 else codes)
        return answers

    @staticmethod
    def _pause(pool: _PoolLookups, floor_s: float, since: float) -> float:
        """Pause ``pool`` after a failed call that started at ``since``.

        For ``floor_s`` or `_LOOKUP_BACKOFF_FACTOR` times as long as the call
        blocked, whichever is longer, counted from now: a failed setup can
        return minutes after it started, and a pause counted from before it
        would already be over. Returns the seconds paused.
        """
        now = time.monotonic()
        pause = max(floor_s, _LOOKUP_BACKOFF_FACTOR * (now - since))
        pool.paused_until = now + pause
        return pause

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
            pause = self._pause(pool, _RECONNECT_INTERVAL_S, started)
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
        """At most one warning per pool and `_LOOKUP_WARNING_INTERVAL_S`."""
        now = time.monotonic()
        if now < self._next_warning_at.get(pool.pool.master, 0.0):
            logger.debug("Mooncake Store offload: " + message, *args)
            return
        self._next_warning_at[pool.pool.master] = now + _LOOKUP_WARNING_INTERVAL_S
        logger.warning("Mooncake Store offload: " + message, *args, exc_info=exc_info)


class MooncakeStoreOffloadScheduler(KVStoreOffloadScheduler):
    """Chunked offload scheduling with the Mooncake Store as the tier."""

    LABEL = "Mooncake Store offload"

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
        super().__init__(
            config,
            store_cfg=cfg,
            namespace=namespace,
            world=world,
            lookup_client=_StoreLookup(
                cfg, namespace=namespace, world=world, chunk_tokens=cfg.chunk_tokens
            ),
            endpoint="masters=" + ",".join(pool.master for pool in cfg.store_masters()),
        )

    def _lookup_starting(self, sid: str) -> None:
        # Stamped before asking: the lease runs from the master's answer.
        self._looked_up_at[sid] = time.monotonic()

    def _ensure_lookup_pin(self, seq: Any, sid: str, spec: LoadSpec) -> bool:
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


__all__ = [
    "MooncakeStoreOffloadScheduler",
    "monotonic_clock_id",
    "prompt_chunk_hashes",
    "require_monotonic_clock_id",
]
