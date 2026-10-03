# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler lookups across LMCache servers that each hold some PP stages.

With ``lmcache.mp.stage_servers`` every server stores only its stage group's
layers of each chunk, so a prefix is loadable only as far as every server
holds it. The fan-out adapter answers the minimum of the servers' hits and,
like vLLM's multi-server connector, releases each longer server's read locks
beyond that minimum. From then on every live server holds locks on exactly
``[0, hit)``, so the single-server lock bookkeeping in ``lookup.py`` applies
unchanged to each of them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom")

# How long a stage server whose submit or poll raised is skipped. Lookups in
# that window answer 0 at once: a prefix one server cannot serve is a miss.
# Each failure in a row doubles the window up to the cap, because the first
# lookup after a window blocks the scheduler for up to the adapter's timeout
# on a hung server; the cap bounds that to about a tenth of the time.
_STAGE_SERVER_RETRY_S = 30.0
_STAGE_SERVER_MAX_RETRY_S = 300.0


@dataclass
class _FanOutLookup:
    token_ids: list[int]
    submitted: set[int] = field(default_factory=set)
    hits: dict[int, int] = field(default_factory=dict)
    failed: set[int] = field(default_factory=set)
    answer: int | None = None

    def live_servers(self) -> list[int]:
        return sorted(self.submitted - self.failed)


class _StageServersSchedulerAdapter:
    """Scheduler adapter that looks a prefix up on every stage server."""

    def __init__(self, adapters: list[Any], urls: list[str]) -> None:
        if len(adapters) != len(urls) or not adapters:
            raise ValueError("stage server adapters and urls must pair up")
        chunk_sizes = {int(adapter.lmcache_tokens_per_chunk) for adapter in adapters}
        if len(chunk_sizes) != 1:
            raise ValueError(
                "LMCache stage servers must share one chunk size, got "
                + ", ".join(
                    f"{url}={adapter.lmcache_tokens_per_chunk}"
                    for url, adapter in zip(urls, adapters)
                )
            )
        self.lmcache_tokens_per_chunk = chunk_sizes.pop()
        self._adapters = list(adapters)
        self._urls = list(urls)
        self._retry_at = [0.0] * len(adapters)
        self._failures_in_a_row = [0] * len(adapters)
        self._lookups: dict[str, _FanOutLookup] = {}

    def _mark_down(self, index: int) -> float:
        """Skip server ``index`` from now, after its failed call returned."""
        self._failures_in_a_row[index] += 1
        window = min(
            _STAGE_SERVER_RETRY_S * 2 ** (self._failures_in_a_row[index] - 1),
            _STAGE_SERVER_MAX_RETRY_S,
        )
        self._retry_at[index] = time.monotonic() + window
        return window

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int]
    ) -> None:
        if request_id in self._lookups:
            return
        state = _FanOutLookup(token_ids=list(token_ids))
        self._lookups[request_id] = state
        now = time.monotonic()
        if any(now < retry_at for retry_at in self._retry_at):
            # A known-down server makes the minimum 0; locking the others
            # only to release them again would be wasted RPCs.
            state.answer = 0
            return
        for index, adapter in enumerate(self._adapters):
            state.submitted.add(index)
            try:
                adapter.maybe_submit_lookup_request(request_id, token_ids)
            except Exception:
                state.failed.add(index)
                window = self._mark_down(index)
                logger.warning(
                    "LMCache MP stage server %s failed to submit a lookup for "
                    "%s; every lookup is a miss for %.0fs",
                    self._urls[index],
                    request_id,
                    window,
                    exc_info=True,
                )

    def check_lookup_result(self, request_id: str) -> int | None:
        state = self._lookups.get(request_id)
        if state is None:
            return 0
        if state.answer is not None:
            return state.answer
        pending = False
        for index in state.live_servers():
            if index in state.hits:
                continue
            try:
                result = self._adapters[index].check_lookup_result(request_id)
            except Exception:
                state.failed.add(index)
                window = self._mark_down(index)
                logger.warning(
                    "LMCache MP stage server %s failed to answer a lookup for "
                    "%s; every lookup is a miss for %.0fs",
                    self._urls[index],
                    request_id,
                    window,
                    exc_info=True,
                )
                continue
            if result is None:
                pending = True
            else:
                state.hits[index] = int(result)
                self._failures_in_a_row[index] = 0
        if pending:
            return None
        live_hits = {i: h for i, h in state.hits.items() if i not in state.failed}
        hit = 0 if state.failed else min(live_hits.values(), default=0)
        for index, server_hit in live_hits.items():
            if server_hit > hit:
                self._free_on(index, state.token_ids, hit, server_hit, request_id)
        state.answer = hit
        return hit

    def _free_on(
        self, index: int, token_ids: list[int], start: int, end: int, request_id: str
    ) -> None:
        try:
            self._adapters[index].free_lookup_locks(
                token_ids=token_ids, start=start, end=end, request_id=request_id
            )
        except Exception:
            logger.warning(
                "LMCache MP stage server %s failed to free lookup locks "
                "[%d, %d) for %s",
                self._urls[index],
                start,
                end,
                request_id,
                exc_info=True,
            )

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> None:
        """Release ``[start, end)`` on every server still holding the lookup.

        Every range ``lookup.py`` releases lies inside the reconciled hit, and
        every live server holds exactly that range, so each is freed once.
        """
        state = self._lookups.get(request_id)
        servers = range(len(self._adapters)) if state is None else state.live_servers()
        for index in servers:
            self._free_on(index, token_ids, start, end, request_id)

    def cleanup_lookup_result(self, request_id: str) -> None:
        state = self._lookups.pop(request_id, None)
        if state is not None and state.answer is None:
            # No hit was ever returned for this lookup, so no caller owns the
            # locks the answered servers took. A server answering later is
            # left to session/TTL cleanup, as with a single server.
            for index, server_hit in state.hits.items():
                if index not in state.failed and server_hit > 0:
                    self._free_on(index, state.token_ids, 0, server_hit, request_id)
        for adapter in self._adapters:
            adapter.cleanup_lookup_result(request_id)

    def end_session(self, request_id: str) -> None:
        # Every server refreshes the prefix's LRU position, which keeps the
        # stage groups' eviction order aligned.
        for url, adapter in zip(self._urls, self._adapters):
            try:
                adapter.end_session(request_id)
            except Exception:
                logger.warning(
                    "LMCache MP stage server %s failed to end session %s",
                    url,
                    request_id,
                    exc_info=True,
                )

    def shutdown(self) -> None:
        for url, adapter in zip(self._urls, self._adapters):
            try:
                adapter.shutdown()
            except Exception:
                logger.warning(
                    "Failed to shut down LMCache MP stage server adapter %s",
                    url,
                    exc_info=True,
                )


__all__ = ["_StageServersSchedulerAdapter"]
