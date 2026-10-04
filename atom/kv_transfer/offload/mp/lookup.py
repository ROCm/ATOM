# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler-side LMCache MP lookups and their read-lock bookkeeping."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.offload.chunked_scheduler import TIER_LOOKUP_PENDING
from atom.kv_transfer.offload.mp.deployment import _mp_session_id

logger = logging.getLogger("atom")


@dataclass
class _LookupState:
    token_ids: list[int]
    # First prompt token the question covers; the server locks only the hit
    # from here on.
    lookup_start: int = 0
    hit: int | None = None
    retrieve_start: int | None = None
    retrieve_end: int | None = None
    # time.monotonic() when the question went to the server.
    submitted_at: float = 0.0


class _MPLookupClient:
    """Lookup facade between ATOM's scheduler policy and the MP adapter.

    ``lookup`` waits for the server's answer. A server with an L2 answers only
    once it has prefetched the hit from L2 into its L1, so with a small L1
    nearly every hit makes ``lookup`` hold the scheduler thread for an L2 read.
    With ``nonblocking`` set, the scheduler asks through ``lookup_nowait``
    instead, which never waits out a prefetch: the request stays in line while
    the server prefetches, and the scheduler admits other work meanwhile.

    With ``starts_past_hbm_prefix`` set, the scheduler passes a ``start``: the
    prompt's HBM prefix, which it will never load. A server that prefetches
    from the key's start (``mooncake_l2_server``) then reads only the chunks
    past it, answers the hit as a prefix length all the same, and locks only
    its part from ``start`` on; it clamps every release to that start, so the
    releases below stay ranges from 0. A server that ignores ``start`` answers
    and locks as if it were 0.
    """

    token_database = None

    def __init__(
        self,
        adapter: Any,
        *,
        config: Any,
        timeout: float,
        poll_interval: float,
        nonblocking: bool = False,
        max_pending: int = 8,
        nowait_grace: float = 0.02,
        starts_past_hbm_prefix: bool = False,
    ) -> None:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("LMCache MP lookup timeout and poll interval must be > 0")
        if max_pending < 1 or nowait_grace < 0:
            raise ValueError(
                "LMCache MP max pending lookups must be >= 1 and the no-wait "
                "grace >= 0"
            )
        self._adapter = adapter
        self._config = config
        self._timeout = timeout
        self._poll_interval = poll_interval
        self.nonblocking = bool(nonblocking)
        self.starts_past_hbm_prefix = bool(starts_past_hbm_prefix)
        self._max_pending = int(max_pending)
        self._nowait_grace = float(nowait_grace)
        self._lookups: dict[str, _LookupState] = {}
        if self.nonblocking:
            logger.info(
                "LMCache MP lookups do not wait out L2 prefetches: at most %d "
                "outstanding, %.0f ms for an answer before a request waits",
                self._max_pending,
                self._nowait_grace * 1000,
            )
        if self.starts_past_hbm_prefix:
            logger.info("LMCache MP lookups start past each prompt's HBM prefix")

    def _submit(
        self, token_ids: list[int], lookup_id: str, start: int = 0
    ) -> _LookupState:
        state = _LookupState(
            token_ids=list(token_ids),
            lookup_start=int(start),
            submitted_at=time.monotonic(),
        )
        self._lookups[lookup_id] = state
        request_id = _mp_session_id(self._config, lookup_id)
        if start > 0:
            self._adapter.maybe_submit_lookup_request(
                request_id, token_ids, start=int(start)
            )
        else:
            self._adapter.maybe_submit_lookup_request(request_id, token_ids)
        return state

    def _num_outstanding(self, now: float) -> int:
        return sum(
            1
            for state in self._lookups.values()
            if state.hit is None and now - state.submitted_at < self._timeout
        )

    def lookup_nowait(
        self, token_ids: list[int], lookup_id: str, start: int = 0
    ) -> Any:
        """``lookup`` without waiting out an L2 prefetch.

        Answers the hit once the server has, ``TIER_LOOKUP_PENDING`` while it
        has not, and None once the question has been out for the lookup
        timeout -- the same non-answer ``lookup`` gives then. A new question
        waits up to the no-wait grace, which covers a hit that is all in L1;
        an outstanding one is checked once.

        Past ``max_pending`` outstanding questions a new one is not submitted:
        the answer is ``TIER_LOOKUP_PENDING`` and the request asks again later.
        That keeps the queue's head first in line for the server, and bounds
        how much of the server's L1 the prefetches of requests that are not
        admitted yet hold under read locks.
        """

        now = time.monotonic()
        state = self._lookups.get(lookup_id)
        if state is None or state.hit is not None:
            if self._num_outstanding(now) >= self._max_pending:
                return TIER_LOOKUP_PENDING
            state = self._submit(token_ids, lookup_id, start)
            deadline = state.submitted_at + self._nowait_grace
        else:
            deadline = now
        request_id = _mp_session_id(self._config, lookup_id)
        while True:
            result = self._adapter.check_lookup_result(request_id)
            if result is not None:
                state.hit = int(result)
                return state.hit
            now = time.monotonic()
            if now - state.submitted_at >= self._timeout:
                logger.warning(
                    "LMCache MP lookup timed out after %.1fs for request %s",
                    self._timeout,
                    lookup_id,
                )
                # As in `lookup`: the job stays with the adapter and
                # `state.hit` stays None, so cleanup can still release locks.
                return None
            if now >= deadline:
                return TIER_LOOKUP_PENDING
            time.sleep(self._poll_interval)

    def lookup(
        self, token_ids: list[int], lookup_id: str, start: int = 0
    ) -> int | None:
        """Hit length for this prompt, or None if the tier never answered.

        None is a non-answer, not an empty answer: the caller must ask again
        rather than record "this tier has nothing" for a prompt the tier may
        well hold.
        """

        state = self._submit(token_ids, lookup_id, start)
        request_id = _mp_session_id(self._config, lookup_id)
        deadline = state.submitted_at + self._timeout
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

    def lookup_start(self, lookup_id: str) -> int:
        """Where the live lookup for this request started, or 0 if none is live."""
        state = self._lookups.get(lookup_id)
        return 0 if state is None else state.lookup_start

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
