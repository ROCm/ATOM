# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler-side LMCache MP lookups and their read-lock bookkeeping."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.offload.mp.deployment import _mp_session_id

logger = logging.getLogger("atom")

_ASYNC_ADAPTER_ATTRS = (
    "_client",
    "_parallel",
    "_create_key",
    "_pending_lookups",
    "_lookup_results",
)

_ASYNC_LOOKUP_FAILED = (
    "LMCache MP async lookup failed for request %s in phase %s; "
    "read locks it may hold are left to the read TTL"
)


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
    ) -> None:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("LMCache MP lookup timeout and poll interval must be > 0")
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

    # -- asynchronous submission ------------------------------------------
    #
    # A lookup costs a blocking round trip on the scheduler thread (the server
    # hashes the whole prompt before it answers). `submit` sends it while the
    # request still waits; `pump` advances it without blocking. Either way the
    # answered hit is kept in `_async_hits`, so the synchronous `lookup` that
    # later consumes it returns at once, and `discard` can still free its
    # locks after the adapter's result cache was cleaned up.

    def submit(self, token_ids: list[int], lookup_id: str) -> bool:
        """Send this request's lookup without waiting. False if not sent."""

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

    def is_pending(self, lookup_id: str) -> bool:
        """Submitted and not answered yet."""
        return lookup_id in self._async

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

    def pending_ids(self):
        """Lookups already submitted and not yet consumed (do not mutate)."""
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
        del self._async[lookup_id]
        return True

    def pump(self) -> None:
        """Advance every async lookup; release the ones nobody will consume."""

        for lookup_id in list(self._async):
            try:
                answered = self._advance(lookup_id)
            except Exception:
                phase = self._drop_failed(lookup_id)
                logger.warning(_ASYNC_LOOKUP_FAILED, lookup_id, phase, exc_info=True)
                continue
            if answered and lookup_id in self._orphans:
                self._orphans.discard(lookup_id)
                self._release_unconsumed(lookup_id)

    def _drop_failed(self, lookup_id: str) -> str | None:
        """Forget an async lookup that raised; return the phase it failed in.

        Its hit is unknown, so read locks the server may already hold for it
        cannot be freed here and expire only with the L1 read TTL.
        """
        entry = self._async.pop(lookup_id, None)
        return entry[2] if entry is not None else None

    def _release_unconsumed(self, lookup_id: str) -> None:
        request_id = _mp_session_id(self._config, lookup_id)
        hit = self._async_hits.pop(lookup_id, None)
        token_ids = self._async_tokens.pop(lookup_id, None)
        if hit and token_ids is not None:
            self._adapter.free_lookup_locks(
                token_ids=token_ids, start=0, end=hit, request_id=request_id
            )
        self._adapter.cleanup_lookup_result(request_id)

    def discard(self, lookup_id: str) -> None:
        """Forget an async lookup its request will never consume."""

        if lookup_id in self._async:
            # Still in flight: its locks exist only once it answers.
            self._orphans.add(lookup_id)
            return
        if lookup_id in self._async_tokens:
            self._release_unconsumed(lookup_id)

    def lookup(self, token_ids: list[int], lookup_id: str) -> int | None:
        """Hit length for this prompt, or None if the tier never answered.

        None is a non-answer, not an empty answer: the caller must ask again
        rather than record "this tier has nothing" for a prompt the tier may
        well hold.
        """

        self._orphans.discard(lookup_id)
        if lookup_id in self._async:
            deadline = time.monotonic() + self._timeout
            while lookup_id in self._async:
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
                time.sleep(self._poll_interval)
        self._async_tokens.pop(lookup_id, None)
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

    def complete_retrieve(self, lookup_id: str, *, succeeded: bool) -> None:
        # Once a retrieve is submitted, LMCache owns the remaining lookup
        # locks. Its lmcache-driven transfer releases them on both success and
        # failure (including partial failures). Releasing the range again from
        # the scheduler would decrement every TP rank's read locks twice. The
        # scheduler cannot distinguish a pre-submit failure; that rarer case is
        # left to request end_session/server TTL cleanup instead.
        self._lookups.pop(lookup_id, None)
        self._adapter.cleanup_lookup_result(_mp_session_id(self._config, lookup_id))

    def hit_tokens(self, lookup_id: str) -> int | None:
        state = self._lookups.get(lookup_id)
        return None if state is None else state.hit

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
