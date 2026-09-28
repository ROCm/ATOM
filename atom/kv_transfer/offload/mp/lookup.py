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

    def lookup(self, token_ids: list[int], lookup_id: str) -> int | None:
        """Hit length for this prompt, or None if the tier never answered.

        None is a non-answer, not an empty answer: the caller must ask again
        rather than record "this tier has nothing" for a prompt the tier may
        well hold.
        """

        state = _LookupState(token_ids=list(token_ids))
        self._lookups[lookup_id] = state
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
