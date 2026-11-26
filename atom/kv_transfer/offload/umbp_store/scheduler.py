# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler side of the UMBP offload.

The policy, the digest-carrying requests and the paced save retries are the
store-agnostic :class:`KVStoreOffloadScheduler`'s. This module answers the one
question it asks a transport, how much of a prompt the server holds. Unlike a
Mooncake Store: one server instead of pools, no read leases (a UMBP lookup
holds nothing, so nothing is renewed before a load; the base dispatches it as
it is), and no cross-client taint marker (see :mod:`.taint`).
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from atom.kv_transfer.disaggregation.types import LoadOperationId, SaveOperationId
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.kv_store import scheduler as _store_scheduler
from atom.kv_transfer.offload.kv_store.keys import DIGEST_BYTES
from atom.kv_transfer.offload.kv_store.scheduler import (
    LOOKUP_BACKOFF_S,
    LOOKUP_RECONNECT_INTERVAL_S,
    KVStoreOffloadScheduler,
    LookupWarnings,
    StoreLookupBase,
    monotonic_clock_id,
    pause_lookups,
    prompt_chunk_hashes,
)
from atom.kv_transfer.offload.lazy_save import LazySaveSchedulerMixin
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, SaveSpec
from atom.kv_transfer.offload.miss_accounting import MissAccountant
from atom.kv_transfer.offload.tier_index import TierIndex
from atom.kv_transfer.offload.umbp_store import taint
from atom.kv_transfer.offload.umbp_store.client import UMBPStoreClient
from atom.kv_transfer.offload.umbp_store.config import (
    UMBPStoreOffloadConfig,
    check_engine_compatibility,
    parse_umbp_store_config,
    store_namespace,
)
from atom.utils import envs

logger = logging.getLogger("atom")

# A prefetched answer older than this is asked again at admission: the server
# evicts on its own, and a stale hit only costs a failed all-or-nothing load,
# but a fresh one is cheap once the request is at the head.
_ANSWER_TTL_S = 2.0
# Answers nobody consumed (the request finished, or was admitted on a
# remembered hit) are dropped after this many TTLs.
_ANSWER_KEEP_TTLS = 8


def require_monotonic_clock_id() -> str:
    """The boot ID the saves are timed on, or a refusal to start without one."""
    return _store_scheduler.require_monotonic_clock_id("the UMBP offload")


class _ServerLookup(StoreLookupBase):
    """The chunked scheduler's lookup client, answered by the UMBP server.

    A lookup asks ``batch_exists`` for every rank's key of each chunk (split
    into batches by the client); it grants nothing, so a chunk found may be
    evicted before its load, whose all-or-nothing get then fails.

    UMBP answers a failed RPC as every key absent, so a lookup cannot tell a
    server that did not answer from one that holds nothing: that costs a miss,
    never a wrong hit. Each call waits at most ``umbp_store.lookup_timeout_ms``
    and then answers every key absent too. A lookup answers None, which the
    scheduler retries later, when it got no answer it can use: the call
    raised, it took the whole deadline, or the server is not asked for a
    while (``_client_to_ask``) because it could not be reached or its last
    lookup raised or was slow.

    **Off the scheduler thread** (``umbp_store.lookup_threads`` > 0, the
    default): the chunked base prefetches the head of the waiting queue
    (`prefetch_lookups` -> :meth:`submit`); each prefetch asks for every chunk
    of the prompt and keeps the per-chunk answer, so admission can read the
    run from wherever its lookup starts -- chunk 0, or past the HBM hit when
    the tier holds what HBM evicted (``save_policy: lazy``), which is not known
    when the prefetch is sent. While a prefetch is in flight the base passes
    the request over (`lookup_pending`, bounded by ``OFFLOAD_LOOKUP_DEFER_S``).
    A lookup with no fresh answer asks for its own chunks on the lookup thread
    and waits for it. Every call to the server runs on the lookup threads,
    each with its own client, never on the scheduler thread.

    A server this process has marked (:mod:`.taint`) holds nothing for the
    lookups. The mark is this process's only; the workers, which place it,
    refuse their own loads either way.

    The clients are opened on the first lookup, not at construction: every PP
    stage builds a scheduler, and only the head ever schedules. A server the
    scheduler cannot reach, or whose lookup raised or was slow, is not asked
    for a while (``LOOKUP_RECONNECT_INTERVAL_S``, ``LOOKUP_BACKOFF_S``).
    """

    def __init__(
        self,
        cfg: UMBPStoreOffloadConfig,
        *,
        namespace: str,
        world: int,
        chunk_tokens: int,
    ) -> None:
        super().__init__(namespace=namespace, world=world, chunk_tokens=chunk_tokens)
        self._cfg = cfg
        self.client: UMBPStoreClient | None = None
        self.paused_until = 0.0
        self._warnings = LookupWarnings("UMBP offload")
        self._threads = int(cfg.lookup_threads)
        self._pool: ThreadPoolExecutor | None = None
        # One client per lookup thread: the scheduler thread never calls one.
        self._local = threading.local()
        self._lock = threading.Lock()
        # sid -> the prefetch in flight; sid -> (weak seq, chunks, per-chunk
        # answer or None for no answer, answered at).
        self._inflight: dict[str, Future] = {}
        self._answers: dict[str, tuple[Any, int, list[bool] | None, float]] = {}
        self.prefetched = 0
        self.prefetch_answers_used = 0
        self.prefetch_answers_stale = 0

    # -- the scheduler's lookup ------------------------------------------
    def lookup(
        self, seq: Any, lookup_id: str | None = None, start_chunk: int = 0
    ) -> int | None:
        """Tokens of ``seq``'s prompt prefix the server holds on every rank,
        counted from the prompt's head: the run of held chunks from
        ``start_chunk`` (0, or the HBM hit under ``save_policy: lazy``), 0 if
        it holds none of them.

        None, not 0, when the server could not be asked: the scheduler retries
        a non-answer later, where a 0 would be remembered as a miss.
        """
        started = time.perf_counter()
        hashes = prompt_chunk_hashes(seq, self._chunk_tokens)
        chunks = len(hashes) // DIGEST_BYTES
        if not chunks:
            return 0
        start = min(max(0, int(start_chunk)), chunks)
        if start == chunks:
            return 0
        answer, first, prefetched = None, start, False
        if lookup_id is not None:
            answer = self._take_answer(str(lookup_id), seq, chunks)
            if answer is not None:
                first, prefetched = 0, True
        if answer is None:
            answer = self._run(self._presence, hashes, start, chunks)
            if answer is None:
                return None
        run = 0
        for held in answer[start - first :]:
            if not held:
                break
            run += 1
        hit = (start + run) * self._chunk_tokens if run else 0
        logger.debug(
            "[OFFLOAD-LOOKUP] lookup_id=%s chunks=%d start=%d hit=%d "
            "prefetched=%s elapsed_ms=%.2f",
            lookup_id,
            chunks,
            start,
            hit,
            prefetched,
            (time.perf_counter() - started) * 1000,
        )
        return hit

    def peek(self, seq: Any, lookup_id: str, start_chunk: int) -> int | None:
        """The run a fresh prefetched answer gives from ``start_chunk``, in
        tokens from the prompt's head (0 if none held); None without one.
        Consumes nothing and sends nothing (`tier_probe`)."""
        with self._lock:
            entry = self._answers.get(str(lookup_id))
        if entry is None:
            return None
        ref, chunks, answer, at = entry
        if ref() is not seq or answer is None:
            return None
        if time.monotonic() - at > _ANSWER_TTL_S:
            return None
        start = min(max(0, int(start_chunk)), chunks)
        run = 0
        for held in answer[start:]:
            if not held:
                break
            run += 1
        return (start + run) * self._chunk_tokens if run else 0

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Forget ``lookup_id``'s prefetch: a UMBP lookup holds nothing."""
        with self._lock:
            self._answers.pop(str(lookup_id), None)
            self._inflight.pop(str(lookup_id), None)

    # -- the asynchronous protocol of the chunked base ----------------------
    def submit(self, seq: Any, lookup_id: str) -> bool:
        """Prefetch ``seq``'s per-chunk answer on a lookup thread; False when
        nothing was sent (no threads, a paused server, an empty prompt, or a
        prefetch of it already in flight or answered)."""
        if self._threads <= 0 or time.monotonic() < self.paused_until:
            return False
        sid = str(lookup_id)
        hashes = prompt_chunk_hashes(seq, self._chunk_tokens)
        chunks = len(hashes) // DIGEST_BYTES
        if not chunks:
            return False
        # Marked in flight before the job can run, so the job's own pop on
        # completion can never be undone by this thread: a done placeholder
        # until the real future replaces it (a lookup that meets the
        # placeholder finds no answer yet and asks itself).
        placeholder: Future = Future()
        placeholder.set_result(None)
        with self._lock:
            if sid in self._inflight or sid in self._answers:
                return False
            self._inflight[sid] = placeholder
        try:
            future = self._executor().submit(
                self._prefetch_job, hashes, chunks, sid, _weak(seq)
            )
        except Exception:
            with self._lock:
                if self._inflight.get(sid) is placeholder:
                    del self._inflight[sid]
            raise
        with self._lock:
            if self._inflight.get(sid) is placeholder:
                self._inflight[sid] = future
        self.prefetched += 1
        return True

    def pending_ids(self):
        with self._lock:
            return set(self._inflight) | set(self._answers)

    def is_pending(self, lookup_id) -> bool:
        with self._lock:
            return str(lookup_id) in self._inflight

    def poll(self, lookup_id) -> bool:
        """Whether ``lookup_id``'s prefetch has answered (or none is in flight)."""
        return not self.is_pending(lookup_id)

    def answered(self, lookup_id) -> bool:
        with self._lock:
            return str(lookup_id) in self._answers

    def pump(self) -> None:
        """Drop answers nobody consumed."""
        cutoff = time.monotonic() - _ANSWER_KEEP_TTLS * _ANSWER_TTL_S
        with self._lock:
            stale = [
                sid
                for sid, (ref, _, _, at) in self._answers.items()
                if at < cutoff or ref() is None
            ]
            for sid in stale:
                del self._answers[sid]

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    # -- internals ------------------------------------------------------------
    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=max(1, self._threads), thread_name_prefix="umbp-lookup"
            )
        return self._pool

    def _run(self, fn, *args):
        """``fn(*args)`` on a lookup thread when there are any, else here."""
        if self._threads <= 0:
            return fn(*args)
        return self._executor().submit(fn, *args).result()

    def _prefetch_job(self, hashes: bytes, chunks: int, sid: str, ref) -> None:
        try:
            answer = self._presence(hashes, 0, chunks)
        except Exception:  # a lost prefetch costs a sync lookup
            logger.warning(
                "UMBP offload: prefetched lookup %s failed", sid, exc_info=True
            )
            answer = None
        with self._lock:
            self._inflight.pop(sid, None)
            self._answers[sid] = (ref, chunks, answer, time.monotonic())

    def _take_answer(self, sid: str, seq: Any, chunks: int) -> list[bool] | None:
        """Consume ``sid``'s prefetched per-chunk answer, waiting (bounded) for
        one in flight; None when there is none fresh for this prompt."""
        with self._lock:
            future = self._inflight.get(sid)
        if future is not None:
            try:
                future.result(timeout=self._cfg.lookup_timeout_ms / 1000.0 + 1.0)
            except FutureTimeout:
                return None
            except Exception:  # noqa: BLE001  # the job logs its own failures
                return None
        with self._lock:
            entry = self._answers.pop(sid, None)
        if entry is None:
            return None
        ref, answered_chunks, answer, at = entry
        if ref() is not seq or answered_chunks != chunks or answer is None:
            return None
        if time.monotonic() - at > _ANSWER_TTL_S:
            self.prefetch_answers_stale += 1
            return None
        self.prefetch_answers_used += 1
        return answer

    def _presence(self, hashes: bytes, first: int, last: int) -> list[bool] | None:
        """Per chunk of ``[first, last)``, whether every rank's object is held;
        None when the server gave no usable answer (which pauses lookups)."""
        count = last - first
        if taint.reason(self._cfg.address) is not None:
            return [False] * count
        client = self._client_to_ask()
        if client is None:
            return None
        keys = self._rank_keys(hashes, first, last)
        asked_at = time.monotonic()
        try:
            present = client.exists(keys)
        except Exception as exc:  # noqa: BLE001  # the server is not asked a while
            self._warn(
                "the UMBP server at %s failed a lookup of %d keys; it is not "
                "asked for %.0fs",
                self._cfg.address,
                len(keys),
                pause_lookups(self, LOOKUP_BACKOFF_S, asked_at),
                exc_info=exc,
            )
            return None
        waited_ms = (time.monotonic() - asked_at) * 1000
        if waited_ms >= self._cfg.lookup_timeout_ms:
            # Past the deadline Mori answers every key absent: not an answer.
            self._warn(
                "the UMBP server at %s took %.0f ms to answer a lookup of %d "
                "keys (umbp_store.lookup_timeout_ms=%d); it is not asked for %.0fs",
                self._cfg.address,
                waited_ms,
                len(keys),
                self._cfg.lookup_timeout_ms,
                pause_lookups(self, LOOKUP_BACKOFF_S, asked_at),
            )
            return None
        # Per rank, the run from `first`; the objects past the shortest run are
        # stranded (another rank lacks the chunk).
        held, _ = self._rank_prefixes(present, count)
        shortest = min(held)
        self._count_lookup(sum(n - shortest for n in held))
        world = self._world
        return [
            all(present[rank * count + index] for rank in range(world))
            for index in range(count)
        ]

    def _client_to_ask(self) -> UMBPStoreClient | None:
        """This thread's client, connecting it if needed; None while paused or
        when it cannot connect (which pauses it)."""
        if time.monotonic() < self.paused_until:
            return None
        threaded = self._threads > 0
        client = getattr(self._local, "client", None) if threaded else self.client
        if client is not None:
            return client
        started = time.monotonic()
        try:
            # Lookups only: no memory is registered.
            client = UMBPStoreClient(
                address=self._cfg.address,
                startup_timeout_ms=self._cfg.startup_timeout_ms,
                lookup_timeout_ms=self._cfg.lookup_timeout_ms,
            )
        except Exception:
            pause = pause_lookups(self, LOOKUP_RECONNECT_INTERVAL_S, started)
            logger.warning(
                "UMBP offload: the scheduler cannot reach the server at %s; "
                "lookups do not ask it for the next %.0fs",
                self._cfg.address,
                pause,
                exc_info=True,
            )
            return None
        if threaded:
            self._local.client = client
        else:
            self.client = client
        logger.info(
            "UMBP offload: scheduler lookups use %s (%s)",
            self._cfg.address,
            f"{self._threads} lookup thread(s)" if threaded else "synchronous",
        )
        return client

    def _warn(self, message: str, *args: Any, exc_info: Any = None) -> None:
        """At most one warning per `LOOKUP_WARNING_INTERVAL_S`."""
        self._warnings.warn(self._cfg.address, message, *args, exc_info=exc_info)


def _sid_of(req_id: Any) -> str:
    return str(req_id.req_id if isinstance(req_id, LoadOperationId) else req_id)


def _weak(seq: Any):
    try:
        return weakref.ref(seq)
    except TypeError:
        return lambda seq=seq: seq


class UMBPStoreOffloadScheduler(KVStoreOffloadScheduler):
    """Chunked offload scheduling with a UMBP standalone server as the tier
    (``save_policy: eager``: every computed prompt chunk is put after its
    prefill)."""

    LABEL = "UMBP offload"

    def __init__(self, config: Any) -> None:
        kvc = config.kv_transfer_config or {}
        validated_kv_role(kvc)
        cfg = parse_umbp_store_config(kvc)
        check_engine_compatibility(cfg, config)
        require_monotonic_clock_id()
        world = offcfg.lmcache_replica_world_size(config)
        namespace = store_namespace(config, cfg.chunk_tokens)
        super().__init__(
            config,
            store_cfg=cfg,
            namespace=namespace,
            world=world,
            lookup_client=_ServerLookup(
                cfg, namespace=namespace, world=world, chunk_tokens=cfg.chunk_tokens
            ),
            endpoint=f"server={cfg.address}",
        )
        # What the server is known to hold, for `tier_probe` ranking only.
        self._tier_index: TierIndex | None = (
            TierIndex(cfg.tier_index_chunks) if cfg.tier_index_chunks else None
        )
        # ATOM_OFFLOAD_MISS_ACCOUNTING: where each prompt's reusable prefix went.
        self._miss_accounting: MissAccountant | None = None
        if (
            envs.ATOM_OFFLOAD_MISS_ACCOUNTING
            and envs.ATOM_OFFLOAD_MISS_ACCOUNTING_CHUNKS
        ):
            self._miss_accounting = MissAccountant(
                envs.ATOM_OFFLOAD_MISS_ACCOUNTING_CHUNKS,
                cfg.chunk_tokens,
                f"umbp_store save_policy={cfg.save_policy}",
            )

    def _prompt_digests(self, seq: Any, num_tokens: int) -> list[bytes]:
        """The store digests of `seq`'s first `num_tokens // chunk` chunks."""
        hashes = prompt_chunk_hashes(seq, int(self.chunk_size))
        n = min(int(num_tokens) // int(self.chunk_size), len(hashes) // DIGEST_BYTES)
        return [hashes[i * DIGEST_BYTES : (i + 1) * DIGEST_BYTES] for i in range(n)]

    # -- miss accounting (ATOM_OFFLOAD_MISS_ACCOUNTING) ----------------------
    def update_state_after_alloc(self, seq: Any) -> None:
        accounting = self._miss_accounting
        if accounting is None or not self._do_load:
            super().update_state_after_alloc(seq)
            return
        sid = str(seq.id)
        # Read before the base spends the remembered hit.
        tier_hit = self._last_tier_hit(seq, sid) or 0
        super().update_state_after_alloc(seq)
        spec = self._load_specs.get(sid)
        load = 0
        if spec is not None and spec.can_load:
            # What the dispatch will load. The spec still starts at the HBM
            # frontier of the lookup, before the allocation found the HBM hit;
            # the dispatch decides on a spec moved to that hit, so decide on a
            # copy here.
            loads, _reason, _hbm, _end, need, _chunk = self._decide_load_after_alloc(
                seq, copy.copy(spec)
            )
            load = need if loads else 0
        prompt = int(seq.num_prompt_tokens)
        accounting.admitted(
            sid,
            self._prompt_digests(seq, prompt),
            prompt_tokens=prompt,
            servable_tokens=self._loadable_hit(prompt, prompt),
            hbm_tokens=int(seq.num_cached_tokens),
            tier_hit_tokens=int(tier_hit),
            load_tokens=load,
        )

    def load_failed(self, req_id: Any) -> bool:
        if not super().load_failed(req_id):
            return False
        if self._miss_accounting is not None:
            self._miss_accounting.load_failed(_sid_of(req_id))
        return True

    def load_finished(self, req_id: Any) -> bool:
        if not super().load_finished(req_id):
            return False
        if self._miss_accounting is not None:
            self._miss_accounting.load_finished(_sid_of(req_id))
        return True

    def request_finished(self, seq: Any) -> None:
        accounting = self._miss_accounting
        if accounting is not None:
            # The prompt's KV now exists: the next prompt sharing it could reuse it.
            done = min(int(seq.num_hashed_tokens), int(seq.num_prompt_tokens))
            accounting.computed(self._prompt_digests(seq, done))
            accounting.forget(str(seq.id))
        super().request_finished(seq)

    def get_statistics(self) -> dict[str, int]:
        statistics = super().get_statistics()
        lookup = self._lookup_client
        statistics.update(
            {
                "umbp_lookups": lookup.lookups,
                "umbp_uneven_lookups": lookup.uneven_lookups,
                "umbp_lookups_prefetched": lookup.prefetched,
                "umbp_prefetch_answers_used": lookup.prefetch_answers_used,
                "umbp_prefetch_answers_stale": lookup.prefetch_answers_stale,
            }
        )
        if self._miss_accounting is not None:
            statistics.update(self._miss_accounting.statistics())
        return statistics

    def _tier_probe_start(self, num_hbm_tokens: int) -> int:
        """Where a lookup past an HBM hit of `num_hbm_tokens` starts: the whole
        prompt, since an eager tier holds what HBM holds too."""
        return 0

    def _tier_lookup(self, seq: Any, sid: str) -> int | None:
        chunk = int(self.chunk_size)
        start = int(self._tier_lookup_start(seq)) // chunk
        hit = self._lookup_client.lookup(seq, lookup_id=sid, start_chunk=start)
        if hit is not None and self._tier_index is not None:
            hashes = prompt_chunk_hashes(seq, chunk)
            chunks = len(hashes) // DIGEST_BYTES
            digests = [
                hashes[i * DIGEST_BYTES : (i + 1) * DIGEST_BYTES]
                for i in range(min(start, chunks), chunks)
            ]
            self._tier_index.observe_lookup(digests, max(0, int(hit) // chunk - start))
        return hit

    def tier_probe(self, seq: Any, num_hbm_tokens: int) -> int:
        """The prompt tokens the server likely holds past where `seq`'s lookup
        would start: from a fresh prefetched answer, else from the tier index;
        0 if neither knows. For `Scheduler._order_waiting` only: sends nothing
        and pins nothing."""
        if not self._do_load:
            return 0
        chunk = int(self.chunk_size)
        start = int(self._tier_probe_start(num_hbm_tokens)) // chunk
        hit = self._lookup_client.peek(seq, str(seq.id), start)
        if hit is not None:
            return hit
        index = self._tier_index
        if index is None:
            return 0
        hashes = prompt_chunk_hashes(seq, chunk)
        chunks = len(hashes) // DIGEST_BYTES
        pos = min(start, chunks)
        while (
            pos < chunks
            and hashes[pos * DIGEST_BYTES : (pos + 1) * DIGEST_BYTES] in index
        ):
            pos += 1
        return pos * chunk if pos > start else 0


class UMBPLazyStoreScheduler(LazySaveSchedulerMixin, UMBPStoreOffloadScheduler):
    """``umbp_store.save_policy: lazy``: a prompt block is put to the server
    only when HBM is about to evict it (`LazySaveSchedulerMixin`), so HBM and
    the server together hold more prefixes than an eager tier the size of the
    server's DRAM -- what a node whose DRAM is about its HBM's size needs.

    The tier holds what HBM evicted, so a lookup starts past the HBM hit
    (`needs_hbm_probe_before_lookup`); a demotion is an ordinary store save
    whose chunk i is block ``block_ids[i]`` under the digest of the prompt
    chunk that block holds, one chunk per KV block. Under PP the head builds
    the demotions and every stage stores its own layers of those blocks under
    its own keys; the demotion reports once every stage has (`PPKVAggregator`).
    """

    _lazy_label = "UMBP lazy"
    _lazy_demotion_req_id = "umbp-lazy-demotion"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # The mixin's chain is the store's digest chain (`prompt_chunk_hashes`).
        self._prehashed = True
        self._init_lazy_save()
        logger.info(
            "UMBP offload: save_policy lazy (a block is put when HBM is about to "
            "evict it); at most %d demotions of %d blocks in flight",
            self._max_demotions,
            self._demotion_blocks,
        )

    def _check_lazy_backend(self, chunk: int) -> None:
        if envs.ATOM_OFFLOAD_LAZY_EXCLUSIVE:
            raise ValueError(
                "ATOM_OFFLOAD_LAZY_EXCLUSIVE is not supported with umbp_store: "
                "UMBP has no call that makes an object the next it evicts"
            )

    def _prefix_chunk_hashes(self, seq: Any, n_chunks: int) -> list[bytes]:
        hashes = prompt_chunk_hashes(seq, int(self.chunk_size))
        n = min(int(n_chunks), len(hashes) // DIGEST_BYTES)
        return [hashes[i * DIGEST_BYTES : (i + 1) * DIGEST_BYTES] for i in range(n)]

    def _demotion_request(
        self, operation: SaveOperationId, block_ids: list[int], hashes: list
    ) -> LMCacheReqMeta:
        return LMCacheReqMeta(
            req_id=operation.req_id,
            token_ids=[],
            block_ids=list(block_ids),
            save_spec=SaveSpec(skip_leading_tokens=0, can_save=True),
            save_operation=operation,
            chunk_hashes=b"".join(hashes),
            dispatched_at=time.monotonic(),
            dispatch_clock=monotonic_clock_id(),
        )


def make_umbp_store_scheduler(config: Any) -> UMBPStoreOffloadScheduler:
    """The scheduler half of ``umbp_store`` for its ``save_policy``."""
    cfg = parse_umbp_store_config(config.kv_transfer_config or {})
    if cfg.save_policy == "lazy":
        return UMBPLazyStoreScheduler(config)
    return UMBPStoreOffloadScheduler(config)


__all__ = [
    "UMBPLazyStoreScheduler",
    "UMBPStoreOffloadScheduler",
    "make_umbp_store_scheduler",
    "require_monotonic_clock_id",
]
