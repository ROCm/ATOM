# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler-side LMCache MP lookups and their read-lock bookkeeping."""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.offload.mp.deployment import _mp_session_id
from atom.kv_transfer.offload.mp.lookup_driver import _DRIVER_POLL_S, _LookupDriver

logger = logging.getLogger("atom")

_ASYNC_ADAPTER_ATTRS = (
    "_client",
    "_parallel",
    "_create_key",
    "_pending_lookups",
    "_lookup_results",
)

# Status re-queries for a discarded lookup before its locks are left to the TTL.
_ORPHAN_STATUS_RETRIES = 8
# How long a discarded lookup may stay unanswered before it is given up and
# its locks are left to the TTL. Bounds how long it keeps the engine polling
# and its MP session open.
_ORPHAN_TIMEOUT_S = 120.0

_ASYNC_LOOKUP_FAILED = (
    "LMCache MP async lookup failed for request %s in phase %s; "
    "read locks it may hold are left to the read TTL"
)

# How long the lookup driver lets an answered lookup wait for its request's
# admission before releasing its read locks and asking again. Well below
# LMCache's default L1 read TTL (300 s): a lookup that outlives the TTL hands
# the retrieve chunks the server may already have evicted.
_MAX_UNCONSUMED_HOLD_S = 120.0


def _locked(method):
    """Run a `_MPLookupClient` method under the client's lock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class _LookupState:
    token_ids: list[int]
    hit: int | None = None
    retrieve_start: int | None = None
    retrieve_end: int | None = None


class _MPLookupClient:
    """Synchronous lookup facade used by ATOM's existing scheduler policy."""

    token_database = None

    def __init__(
        self,
        adapter: Any,
        *,
        config: Any,
        timeout: float,
        poll_interval: float,
        lock: Any = None,
        driver_window: int | None = None,
    ) -> None:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("LMCache MP lookup timeout and poll interval must be > 0")
        # Every public method holds it. Reentrant: methods call one another,
        # and a `_LockedAdapter` around `adapter` takes the same lock.
        self._lock = lock if lock is not None else threading.RLock()
        self._adapter = adapter
        self._config = config
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._lookups: dict[str, _LookupState] = {}
        # lookup_id -> [request_id, prompt length, phase, future]
        self._async: dict[str, list] = {}
        # Prompts of submitted lookups not yet consumed, to free their locks.
        self._async_tokens: dict[str, list[int]] = {}
        # Answered async hits. Kept here, not only in the adapter's result
        # cache, which `cleanup_lookup_result` clears by request ID.
        self._async_hits: dict[str, int] = {}
        self._orphans: set[str] = set()
        # Failed status queries retried per discarded lookup.
        self._status_retries: dict[str, int] = {}
        # When each discarded lookup became an orphan (monotonic seconds).
        self._orphan_since: dict[str, float] = {}
        # Hits of discarded lookups, drained by `pop_discarded_hits`.
        self._discarded_hits: dict[str, int] = {}
        # Driver only: arrived requests whose lookup is not sent yet, in
        # arrival order. Each maps to a callable returning the prompt to look
        # up, or None once the request no longer wants a lookup.
        self._offered: OrderedDict[str, Callable[[], list[int] | None]] = OrderedDict()
        # When each unconsumed async hit was answered (monotonic seconds).
        self._answered_at: dict[str, float] = {}
        self.driver_stats = {"sent": 0, "stale_released": 0}
        # The non-blocking path drives LMCache's ATOM adapter through its
        # internals (message-queue client and result caches). An adapter
        # without them falls back to the synchronous lookup.
        missing = [name for name in _ASYNC_ADAPTER_ATTRS if not hasattr(adapter, name)]
        self._async_supported = not missing
        if missing:
            logger.warning(
                "LMCache MP adapter %s lacks %s; tier lookups stay synchronous",
                type(adapter).__name__,
                ", ".join(missing),
            )
        self._driver: _LookupDriver | None = None
        if driver_window is not None and self._async_supported:
            self._driver = _LookupDriver(self, driver_window)
            self._driver.start()
            logger.info(
                "LMCache MP lookup driver started (window %d)", self._driver.window
            )

    @property
    def driving(self) -> bool:
        """Whether a driver thread sends and advances async lookups."""
        return self._driver is not None

    def close(self) -> None:
        """Stop the driver thread; lookups it left in flight stay to the TTL."""
        driver, self._driver = self._driver, None
        if driver is not None:
            driver.stop()

    # -- driver -------------------------------------------------------------
    #
    # With a driver, requests are offered when they arrive (input IO thread)
    # and the driver sends their lookups in arrival order while fewer than
    # `window` are in flight or answered and unconsumed: each answered one
    # holds tier read locks until its request consumes it. Everything else --
    # admission consuming the answer, `discard`, the save floor -- is the
    # same path a prefetched lookup takes.

    def offer(self, lookup_id: str, tokens: Callable[[], list[int] | None]) -> bool:
        """Queue an arrived request's lookup for the driver. False if not queued."""

        driver = self._driver
        if driver is None:
            return False
        with self._lock:
            if (
                lookup_id in self._offered
                or lookup_id in self._async
                or lookup_id in self._async_tokens
                or lookup_id in self._lookups
            ):
                return False
            self._offered[lookup_id] = tokens
        driver.notify()
        return True

    def _drive(self, window: int) -> bool:
        """One driver pass. True while a lookup is still in flight."""

        with self._lock:
            self.pump()
            self._release_stale()
        while True:
            with self._lock:
                if (
                    not self._offered
                    or len(self._async) + len(self._async_hits) >= window
                ):
                    break
                lookup_id, tokens = self._offered.popitem(last=False)
            # Outside the lock: copying a long prompt takes milliseconds, and
            # `submit` re-checks whether admission took the request meanwhile.
            token_ids = tokens()
            if token_ids is None:
                continue
            with self._lock:
                if self.submit(token_ids, lookup_id):
                    self.driver_stats["sent"] += 1
        with self._lock:
            return bool(self._async)

    def _release_stale(self) -> None:
        """Release answers left unconsumed past `_MAX_UNCONSUMED_HOLD_S`; re-ask.

        The request is still waiting (an admitted one consumes its answer and a
        finished one discards it), so it gets a fresh lookup whose locks start
        a new TTL instead of a retrieve that may find its chunks evicted.
        """
        if not self._answered_at:
            return
        now = time.monotonic()
        for lookup_id, answered in list(self._answered_at.items()):
            if now - answered < _MAX_UNCONSUMED_HOLD_S:
                continue
            del self._answered_at[lookup_id]
            token_ids = self._async_tokens.get(lookup_id)
            if lookup_id not in self._async_hits or token_ids is None:
                continue
            self._try_release_unconsumed(lookup_id)
            # A waiting request's answer, not a detached one's: no save floor.
            self._discarded_hits.pop(lookup_id, None)
            self.driver_stats["stale_released"] += 1
            self.submit(token_ids, lookup_id)

    # -- asynchronous submission ------------------------------------------
    #
    # A lookup costs a blocking round trip on the scheduler thread (the server
    # hashes the whole prompt before it answers). `submit` sends it while the
    # request still waits; `pump` advances it without blocking. Either way the
    # answered hit is kept in `_async_hits`, so the synchronous `lookup` that
    # later consumes it returns at once, and `discard` can still free its
    # locks after the adapter's result cache was cleaned up.

    @_locked
    def submit(self, token_ids: list[int], lookup_id: str) -> bool:
        """Send this request's lookup without waiting. False if not sent."""

        self._offered.pop(lookup_id, None)
        if (
            not self._async_supported
            or lookup_id in self._async
            or lookup_id in self._async_tokens
            or lookup_id in self._lookups
        ):
            return False
        adapter = self._adapter
        request_id = _mp_session_id(self._config, lookup_id)
        if (
            request_id in adapter._pending_lookups
            or request_id in adapter._lookup_results
        ):
            return False
        chunk = int(adapter.lmcache_tokens_per_chunk)
        aligned_end = (len(token_ids) // chunk) * chunk
        if aligned_end == 0:
            return False
        key = adapter._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            worker_id=None,
        )
        future = adapter._client.lookup(key, adapter._parallel.tp_size)
        self._async[lookup_id] = [request_id, len(token_ids), "lookup", future]
        self._async_tokens[lookup_id] = token_ids
        return True

    def supports_async(self) -> bool:
        """Whether `submit` can send a lookup without waiting for it."""
        return self._async_supported

    @_locked
    def is_pending(self, lookup_id: str) -> bool:
        """Submitted and not answered yet."""
        return lookup_id in self._async

    @_locked
    def poll(self, lookup_id: str) -> bool:
        """Advance one async lookup without blocking. True once it answered."""
        if lookup_id not in self._async:
            return True
        try:
            return self._advance(lookup_id)
        except Exception:
            # Let the synchronous path ask again and surface the error there.
            phase = self._drop_failed(lookup_id)
            logger.warning(_ASYNC_LOOKUP_FAILED, lookup_id, phase, exc_info=True)
            return True

    @_locked
    def pending_ids(self):
        """Lookups already submitted and not yet consumed (do not mutate)."""
        if self._driver is not None:
            # The driver adds and removes entries concurrently.
            return frozenset(self._async_tokens)
        return self._async_tokens.keys()

    def _advance(self, lookup_id: str) -> bool:
        """One non-blocking step of an async lookup. True once answered."""

        entry = self._async[lookup_id]
        request_id, _, phase, future = entry
        adapter = self._adapter
        if phase == "lookup":
            if not future.query():
                return False
            future.result(timeout=0)
            adapter._pending_lookups.add(request_id)
            entry[2] = phase = "status"
            entry[3] = future = adapter._client.query_prefetch_status(request_id)
        if not future.query():
            return False
        result = future.result(timeout=0)
        if result is None:
            entry[3] = adapter._client.query_prefetch_status(request_id)
            return False
        hit = int(result) * int(adapter.lmcache_tokens_per_chunk)
        adapter._lookup_results[request_id] = hit
        self._async_hits[lookup_id] = hit
        self._answered_at[lookup_id] = time.monotonic()
        del self._async[lookup_id]
        return True

    @_locked
    def pump(self) -> None:
        """Advance every async lookup; release the ones nobody will consume."""

        for lookup_id in list(self._async):
            try:
                answered = self._advance(lookup_id)
            except Exception:
                if self._requery_orphan_status(lookup_id):
                    continue
                phase = self._drop_failed(lookup_id)
                logger.warning(_ASYNC_LOOKUP_FAILED, lookup_id, phase, exc_info=True)
                if lookup_id in self._orphans:
                    # Nobody will consume or retry it.
                    self._forget_orphan(lookup_id)
                continue
            if lookup_id not in self._orphans:
                continue
            if answered:
                self._orphans.discard(lookup_id)
                self._status_retries.pop(lookup_id, None)
                self._try_release_unconsumed(lookup_id)
            elif self._orphan_expired(lookup_id):
                logger.warning(
                    "LMCache MP: discarded lookup %s unanswered after %.0fs; "
                    "leaving its read locks to the read TTL",
                    lookup_id,
                    _ORPHAN_TIMEOUT_S,
                )
                self._drop_failed(lookup_id)
                self._forget_orphan(lookup_id)

    def _orphan_expired(self, lookup_id: str) -> bool:
        """Whether a discarded lookup has waited past `_ORPHAN_TIMEOUT_S`.

        The clock starts on first sight when `discard` did not start it: a
        `lookup()` timeout orphans a lookup without a timestamp.
        """
        now = time.monotonic()
        return now - self._orphan_since.setdefault(lookup_id, now) > _ORPHAN_TIMEOUT_S

    def _requery_orphan_status(self, lookup_id: str) -> bool:
        """Re-query a discarded lookup whose status query failed.

        A discarded lookup has no consumer to ask again. Once the server
        acked it ("status" phase) it may hold read locks, and only an answered
        status lets them be released exactly -- releasing by range with the
        hit unknown could drop locks other requests hold on shared objects.
        So keep it in flight (which also keeps its MP session alive) and ask
        again, a bounded number of times; past that, leave the locks to the
        read TTL. A failure in the "lookup" phase never reached the server and
        holds nothing, so it is not retried.
        """
        entry = self._async.get(lookup_id)
        if entry is None or lookup_id not in self._orphans or entry[2] != "status":
            return False
        tries = self._status_retries.get(lookup_id, 0)
        if tries >= _ORPHAN_STATUS_RETRIES:
            self._status_retries.pop(lookup_id, None)
            return False
        # A failed re-query still counts as a try and keeps the lookup in
        # flight: the next pump re-raises on the stale future and lands here
        # again, so the bound above is what ends it, not the first failure.
        self._status_retries[lookup_id] = tries + 1
        try:
            entry[3] = self._adapter._client.query_prefetch_status(entry[0])
        except Exception:
            logger.warning(
                "LMCache MP: could not re-query discarded lookup %s (try %d/%d)",
                lookup_id,
                tries + 1,
                _ORPHAN_STATUS_RETRIES,
                exc_info=True,
            )
        logger.debug("LMCache MP: retrying status of discarded lookup %s", lookup_id)
        return True

    def _drop_failed(self, lookup_id: str) -> str | None:
        """Forget an async lookup that raised; return the phase it failed in.

        Its hit is unknown, so read locks the server may already hold for it
        cannot be freed here and expire only with the L1 read TTL.
        """
        entry = self._async.pop(lookup_id, None)
        return entry[2] if entry is not None else None

    @_locked
    def has_orphans(self) -> bool:
        """Whether a discarded lookup still has to answer and free its locks."""
        return bool(self._orphans)

    def _forget_orphan(self, lookup_id: str) -> None:
        """Drop every record of a discarded lookup that failed or timed out.

        Includes the adapter's pending/result entry, which `submit` checks:
        left behind, it would leak and block a reused request ID. The read
        locks themselves are left to the read TTL, as in
        `clear_lookup_status`.
        """
        self._orphans.discard(lookup_id)
        self._orphan_since.pop(lookup_id, None)
        self._status_retries.pop(lookup_id, None)
        self._async_tokens.pop(lookup_id, None)
        self._async_hits.pop(lookup_id, None)
        self._answered_at.pop(lookup_id, None)
        self._try_cleanup_lookup_result(lookup_id)

    def _try_cleanup_lookup_result(self, lookup_id: str) -> None:
        """Clear the adapter's entry for `lookup_id`; never raises."""
        try:
            self._adapter.cleanup_lookup_result(_mp_session_id(self._config, lookup_id))
        except Exception:
            logger.warning(
                "LMCache MP: cleaning up discarded lookup %s failed",
                lookup_id,
                exc_info=True,
            )

    def _try_release_unconsumed(self, lookup_id: str) -> None:
        """`_release_unconsumed` that never raises into the scheduler loop.

        A failed release leaves the locks to the session end / read TTL, the
        same fallback `request_finished` takes for its own cleanup.
        """
        self._orphan_since.pop(lookup_id, None)
        try:
            self._release_unconsumed(lookup_id)
        except Exception:
            logger.warning(
                "LMCache MP: releasing discarded lookup %s failed; its locks "
                "are left to the session end / read TTL",
                lookup_id,
                exc_info=True,
            )
            self._async_tokens.pop(lookup_id, None)
            self._async_hits.pop(lookup_id, None)
            # `free_lookup_locks` may have raised before the cleanup ran.
            self._try_cleanup_lookup_result(lookup_id)

    def _release_unconsumed(self, lookup_id: str) -> None:
        request_id = _mp_session_id(self._config, lookup_id)
        hit = self._async_hits.pop(lookup_id, None)
        token_ids = self._async_tokens.pop(lookup_id, None)
        self._answered_at.pop(lookup_id, None)
        if hit is not None:
            # Its answer is not consumed as a hit, but the scheduler still
            # needs it as a save floor (`pop_discarded_hits`).
            self._discarded_hits[lookup_id] = int(hit)
        if hit and token_ids is not None:
            self._adapter.free_lookup_locks(
                token_ids=token_ids, start=0, end=hit, request_id=request_id
            )
        self._adapter.cleanup_lookup_result(request_id)

    @_locked
    def pop_discarded_hits(self) -> dict[str, int]:
        """Tier hits of discarded lookups answered since the last call."""
        hits, self._discarded_hits = self._discarded_hits, {}
        return hits

    @_locked
    def discard(self, lookup_id: str) -> None:
        """Forget an async lookup its request will never consume."""

        self._offered.pop(lookup_id, None)
        if lookup_id in self._async:
            # Still in flight: its locks exist only once it answers.
            self._orphans.add(lookup_id)
            self._orphan_since.setdefault(lookup_id, time.monotonic())
            return
        if lookup_id in self._async_tokens:
            self._try_release_unconsumed(lookup_id)

    def lookup(self, token_ids: list[int], lookup_id: str) -> int | None:
        """Hit length for this prompt, or None if the tier never answered.

        None is a non-answer, not an empty answer: the caller must ask again
        rather than record "this tier has nothing" for a prompt the tier may
        well hold.

        Holds the lock only between waits, so a driver keeps advancing other
        lookups while this one is waited for.
        """

        with self._lock:
            self._orphans.discard(lookup_id)
            self._offered.pop(lookup_id, None)
            in_flight = lookup_id in self._async
        if in_flight:
            deadline = time.monotonic() + self._timeout
            while True:
                with self._lock:
                    if lookup_id not in self._async:
                        break
                    try:
                        if self._advance(lookup_id):
                            break
                    except Exception:
                        phase = self._drop_failed(lookup_id)
                        logger.warning(
                            _ASYNC_LOOKUP_FAILED, lookup_id, phase, exc_info=True
                        )
                        raise
                    if time.monotonic() >= deadline:
                        logger.warning(
                            "LMCache MP async lookup timed out after %.1fs for "
                            "request %s",
                            self._timeout,
                            lookup_id,
                        )
                        self._orphans.add(lookup_id)
                        return None
                time.sleep(self._wait_interval())
        with self._lock:
            self._async_tokens.pop(lookup_id, None)
            self._answered_at.pop(lookup_id, None)
            async_hit = self._async_hits.pop(lookup_id, None)
            state = _LookupState(token_ids=list(token_ids))
            self._lookups[lookup_id] = state
            if async_hit is not None:
                state.hit = async_hit
                return async_hit
            request_id = _mp_session_id(self._config, lookup_id)
            self._adapter.maybe_submit_lookup_request(request_id, token_ids)
        deadline = time.monotonic() + self._timeout
        while True:
            with self._lock:
                result = self._adapter.check_lookup_result(request_id)
                if result is not None:
                    hit = int(result)
                    state.hit = hit
                    return hit
            if time.monotonic() >= deadline:
                logger.warning(
                    "LMCache MP lookup timed out after %.1fs for request %s",
                    self._timeout,
                    lookup_id,
                )
                # The MP API has no cancel-prefetch call. Keep the adapter job
                # intact so request_finished() can release locks if the result
                # becomes available; eagerly cleaning it here would orphan the
                # server-side lookup and its locks. `state.hit` stays None,
                # which is what tells clear_lookup_status() the job is still
                # outstanding.
                return None
            time.sleep(self._poll_interval)

    def _wait_interval(self) -> float:
        """Sleep between checks of an in-flight async lookup."""
        if self._driver is not None:
            return min(self._poll_interval, _DRIVER_POLL_S)
        return self._poll_interval

    @_locked
    def prepare_retrieve(
        self,
        lookup_id: str,
        start: int,
        end: int | None = None,
    ) -> None:
        """Hand one subrange of a lookup hit to the worker retrieve.

        The worker owns and releases locks in ``[start, end)``. Any hit prefix
        already resident in HBM and any hit suffix beyond ``end`` will not be
        consumed by that retrieve, so release those locks here.
        """
        if start < 0 or (end is not None and end < start):
            raise ValueError(
                f"invalid retrieve range for {lookup_id}: start={start}, end={end}"
            )
        state = self._lookups.get(lookup_id)
        request_id = _mp_session_id(self._config, lookup_id)
        if (
            state is not None
            and state.hit is not None
            and end is not None
            and end > state.hit
        ):
            raise ValueError(
                f"retrieve end {end} exceeds lookup hit {state.hit} for {lookup_id}"
            )
        if state is not None and state.hit is not None and start > 0:
            self._adapter.free_lookup_locks(
                token_ids=state.token_ids,
                start=0,
                end=min(start, state.hit),
                request_id=request_id,
            )
        if (
            state is not None
            and state.hit is not None
            and end is not None
            and end < state.hit
        ):
            self._adapter.free_lookup_locks(
                token_ids=state.token_ids,
                start=end,
                end=state.hit,
                request_id=request_id,
            )
        if state is not None:
            state.retrieve_start = start
            state.retrieve_end = end
        self._adapter.cleanup_lookup_result(request_id)

    @_locked
    def complete_retrieve(self, lookup_id: str, *, succeeded: bool) -> None:
        # Once a retrieve is submitted, LMCache owns the remaining lookup
        # locks. Its lmcache-driven transfer releases them on both success and
        # failure (including partial failures). Releasing the range again from
        # the scheduler would decrement every TP rank's read locks twice. The
        # scheduler cannot distinguish a pre-submit failure; that rarer case is
        # left to request end_session/server TTL cleanup instead.
        self._lookups.pop(lookup_id, None)
        self._adapter.cleanup_lookup_result(_mp_session_id(self._config, lookup_id))

    @_locked
    def hit_tokens(self, lookup_id: str) -> int | None:
        state = self._lookups.get(lookup_id)
        return None if state is None else state.hit

    @_locked
    def clear_lookup_status(self, lookup_id: str) -> None:
        state = self._lookups.pop(lookup_id, None)
        request_id = _mp_session_id(self._config, lookup_id)
        if state is not None and state.hit is None:
            result = self._adapter.check_lookup_result(request_id)
            if result is None:
                logger.warning(
                    "LMCache MP lookup for request %s is still pending during "
                    "cleanup; dropping local state while server TTL/session "
                    "cleanup releases any eventual locks",
                    lookup_id,
                )
                self._adapter.cleanup_lookup_result(request_id)
                return
            state.hit = int(result)
        # Once retrieve has started, the transfer owns the remaining read
        # locks. Failed terminal loads call complete_retrieve(False) first.
        if (
            state is not None
            and state.retrieve_start is None
            and state.hit
            and state.hit > 0
        ):
            self._adapter.free_lookup_locks(
                token_ids=state.token_ids,
                start=0,
                end=state.hit,
                request_id=request_id,
            )
        self._adapter.cleanup_lookup_result(request_id)
