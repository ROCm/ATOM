# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler-side lookups for a TP group whose ranks span several hosts.

An LMCache MP server moves KV through GPU IPC, so it can serve only its own
host's ranks: each host runs one server, and its ranks store and retrieve their
shards there as a host-local world. A prefix is loadable only as far as every
rank can retrieve its shard, so the engine's scheduler, which runs on one host,
asks every host's server and loads only what all of them hold.

``TPHostsSchedulerAdapter`` presents those per-host scheduler adapters as one,
with the attributes and methods `lookup._MPLookupClient` drives, on the
scheduler thread or on its lookup driver thread.

Hosts store in lockstep (every rank saves the same chunks at the same
boundaries), so their answers normally agree. They can still differ: each
server evicts on its own, and a store can land on one host before another.
What every host holds is then the shortest hit, but a longer host's read locks
do not cover it: its STATE image lock sits at its own endpoint, not at the
shorter one, and a retrieve there would release a lock it never took. So each
host whose hit is longer releases its locks and is asked again for the prompt
cut at the shortest hit, and the answers are combined again -- and narrowed
again if a re-asked host now answers shorter still. At a shortest hit of 0, or
after `_MAX_NARROWING_ROUNDS` rounds, every host's locks are released and the
request recomputes.

A server serves these requests on a thread pool, so a release and the lookup
sent after it on the same connection are not ordered there, and a lookup
resets its session's record of what the previous one locked. A host is
therefore re-asked only once its release has replied. Each step is a request
whose reply the lookup client already polls for (a status query that answers
None is asked again), so nothing here waits on its own: narrowing a host takes
three more polls -- release, re-ask, status.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom")

# Rounds of narrowing before a disagreement is given up on. One normally
# settles it; a second is needed only when a re-asked host answers shorter
# than the hit it was cut at.
_MAX_NARROWING_ROUNDS = 3


def _is_power_of_two(count: int) -> bool:
    """Whether to log this occurrence: the 1st, 2nd, 4th, 8th, ... of them."""
    return count > 0 and count & (count - 1) == 0


class _Answered:
    """A finished future holding a value already received."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def query(self) -> bool:
        return True

    def result(self, timeout: float | None = None) -> Any:
        del timeout
        return self._value


class _AllHosts:
    """One future per host; done when all are, its result combines theirs once.

    ``on_error`` runs when a host's reply, or the combination, raises -- before
    the exception reaches the lookup client.
    """

    def __init__(
        self,
        futures: Sequence[Any],
        combine: Callable[[list], Any],
        on_error: Callable[[], None] | None = None,
    ) -> None:
        self._futures = list(futures)
        self._combine = combine
        self._on_error = on_error
        self._done = False
        self._value: Any = None

    def query(self) -> bool:
        return all(future.query() for future in self._futures)

    def result(self, timeout: float | None = None) -> Any:
        if not self._done:
            deadline = None if timeout is None else time.monotonic() + timeout
            try:
                results = []
                for future in self._futures:
                    left = (
                        None
                        if deadline is None
                        else max(0.0, deadline - time.monotonic())
                    )
                    results.append(future.result(timeout=left))
                # Combined once: combining can send requests (releases, re-asks).
                value = self._combine(results)
            except Exception:
                if self._on_error is not None:
                    self._on_error()
                raise
            self._value = value
            self._done = True
        return self._value


@dataclass(frozen=True)
class _HostsKey:
    """One lookup key per host, with the prompt and request they were made for.

    The prompt is kept here rather than read back from a host's key: LMCache's
    key carries it packed (``token_bytes``), and a release or a re-ask needs
    the token IDs themselves.
    """

    token_ids: Sequence[int]
    request_id: str
    keys: tuple


@dataclass
class _HostPart:
    """One host's part of a lookup.

    ``answer`` is its hit in chunks once it replied. While ``waiting`` is set,
    a release (``step == "release"``) or a re-ask (``step == "reask"``) is in
    flight on that host, and its status is not asked.
    """

    answer: int | None = None
    step: str = "status"
    waiting: Any = None
    # Chunks a release is followed by a re-ask for: the round's shortest hit.
    cut: int = 0


@dataclass
class _Lookup:
    """A lookup sent to every host, until their answers agree."""

    token_ids: Sequence[int]
    parts: list[_HostPart] = field(default_factory=list)
    rounds: int = 0
    # The longest first answer, once the hosts disagreed.
    longest: int | None = None


class _HostsClient:
    """The message-queue client surface `_MPLookupClient` uses, over all hosts."""

    def __init__(self, owner: TPHostsSchedulerAdapter) -> None:
        self._owner = owner

    def lookup(self, key: _HostsKey, tp_size: int) -> _AllHosts:
        del tp_size  # each host's own key and parallel config decide
        return self._owner._submit(key)

    def query_prefetch_status(self, request_id: str) -> _AllHosts:
        return self._owner._status(request_id)


class TPHostsSchedulerAdapter:
    """One `AtomMPSchedulerAdapter` per host, answering as one."""

    def __init__(self, hosts: Sequence[Any], *, mq_timeout: float = 300.0) -> None:
        if len(hosts) < 2:
            raise ValueError("a TP group spanning hosts needs one adapter per host")
        chunks = {int(host.lmcache_tokens_per_chunk) for host in hosts}
        if len(chunks) != 1:
            raise ValueError(
                "the hosts' LMCache MP servers disagree on the chunk size: "
                f"{sorted(chunks)}"
            )
        self._hosts = list(hosts)
        self.lmcache_tokens_per_chunk = chunks.pop()
        self._parallel = self._hosts[0]._parallel
        # `lmcache.mp.mq_timeout`, the wait each host's adapter was given.
        self._mq_timeout = float(mq_timeout)
        self._client = _HostsClient(self)
        self._pending_lookups: set[str] = set()
        self._lookup_results: dict[str, int] = {}
        # request_id -> the lookup in flight, until its hosts' answers agree
        self._lookups: dict[str, _Lookup] = {}
        # Cumulative; the scheduler reports them as `tp_hosts_*` counters.
        self.tp_hosts_stats = {
            # lookups whose hosts' answers were combined
            "answered": 0,
            # ... of which the hosts first answered differently
            "disagreed": 0,
            # host lookups sent again, cut at a shorter hit
            "reasks": 0,
            # disagreements given up after `_MAX_NARROWING_ROUNDS`
            "abandoned": 0,
            # chunks the disagreements cost: longest first answer - final hit
            "dropped_chunks": 0,
        }

    # -- the internals `_MPLookupClient` drives ------------------------------

    def _create_key(
        self,
        token_ids: Sequence[int],
        start: int,
        end: int,
        request_id: str,
        worker_id: int | None,
    ) -> _HostsKey:
        return _HostsKey(
            token_ids=token_ids,
            request_id=request_id,
            keys=tuple(
                host._create_key(
                    token_ids,
                    start=start,
                    end=end,
                    request_id=request_id,
                    worker_id=worker_id,
                )
                for host in self._hosts
            ),
        )

    def _submit(self, key: _HostsKey) -> _AllHosts:
        if not isinstance(key, _HostsKey) or len(key.keys) != len(self._hosts):
            raise ValueError(
                "a TP-hosts lookup needs one key per host, from "
                "TPHostsSchedulerAdapter._create_key"
            )
        futures = [
            host._client.lookup(host_key, host._parallel.tp_size)
            for host, host_key in zip(self._hosts, key.keys)
        ]
        lookup = _Lookup(
            token_ids=key.token_ids,
            parts=[_HostPart() for _ in self._hosts],
        )
        self._lookups[key.request_id] = lookup
        return _AllHosts(
            futures,
            lambda _results: None,
            on_error=lambda: self._forget(key.request_id, lookup),
        )

    def _status(self, request_id: str) -> _AllHosts:
        lookup = self._lookups.get(request_id)
        if lookup is None:
            # Not sent through `_submit`, or already combined; a server
            # answers 0 for a lookup it has already reported.
            futures = [
                host._client.query_prefetch_status(request_id) for host in self._hosts
            ]
            return _AllHosts(
                futures, lambda results: self._combine_unrecorded(request_id, results)
            )
        futures = []
        for host, part in zip(self._hosts, lookup.parts):
            if part.answer is not None:
                futures.append(_Answered(part.answer))
            elif part.waiting is not None:
                futures.append(part.waiting)
            else:
                futures.append(host._client.query_prefetch_status(request_id))
        return _AllHosts(
            futures,
            lambda results: self._settle(request_id, lookup, results),
            on_error=lambda: self._forget(request_id, lookup),
        )

    def _forget(self, request_id: str, lookup: _Lookup) -> None:
        """Drop a lookup whose step failed; what it locked is left to the TTL.

        As with one host, a lookup that raised is not retried from where it
        stopped: the client asks the servers again, which then report it as
        consumed (0) or answer what is still pending.
        """
        if self._lookups.get(request_id) is lookup:
            del self._lookups[request_id]

    def _settle(self, request_id: str, lookup: _Lookup, results: list) -> int | None:
        """Take one round of replies; the agreed hit in chunks, or None."""

        if self._lookups.get(request_id) is not lookup:
            # Cleaned up, or replaced by a newer lookup, since this was asked.
            return None
        for index, (part, result) in enumerate(zip(lookup.parts, results)):
            if part.answer is not None:
                continue
            if part.waiting is None:
                # A status reply; None while that host still prefetches.
                if result is not None:
                    part.answer = int(result)
                continue
            part.waiting = None
            if part.step == "release":
                self._reask(request_id, lookup, index)
            else:
                # The re-ask was acknowledged; its status is asked next.
                part.step = "status"
        if any(part.answer is None for part in lookup.parts):
            return None
        answers = [int(part.answer) for part in lookup.parts]
        if len(set(answers)) == 1:
            return self._answer(request_id, lookup, answers[0])
        return self._narrow(request_id, lookup, answers)

    def _narrow(
        self, request_id: str, lookup: _Lookup, answers: list[int]
    ) -> int | None:
        """Release the longer hosts' locks to ask them again; or give up."""

        shortest = min(answers)
        stats = self.tp_hosts_stats
        if lookup.longest is None:
            lookup.longest = max(answers)
            stats["disagreed"] += 1
            if _is_power_of_two(stats["disagreed"]):
                logger.warning(
                    "LMCache MP: the hosts' servers hold different prefixes for "
                    "%s (chunks per host: %s); narrowing every host to %d "
                    "(%d such lookups so far)",
                    request_id,
                    answers,
                    shortest,
                    stats["disagreed"],
                )
        if shortest == 0 or lookup.rounds >= _MAX_NARROWING_ROUNDS:
            if shortest:
                stats["abandoned"] += 1
                if _is_power_of_two(stats["abandoned"]):
                    logger.warning(
                        "LMCache MP: the hosts still disagree for %s after %d "
                        "rounds (chunks per host: %s); not loading it (%d such "
                        "lookups so far)",
                        request_id,
                        lookup.rounds,
                        answers,
                        stats["abandoned"],
                    )
            self._release_all(request_id, lookup.token_ids, answers)
            return self._answer(request_id, lookup, 0)
        lookup.rounds += 1
        chunk = self.lmcache_tokens_per_chunk
        for host, part, answer in zip(self._hosts, lookup.parts, answers):
            if answer == shortest:
                continue
            part.answer = None
            part.step = "release"
            part.cut = shortest
            # Its reply is awaited before the re-ask: see the module docstring.
            part.waiting = host._client.free_lookup_locks(
                host._create_key(
                    lookup.token_ids,
                    start=0,
                    end=answer * chunk,
                    request_id=request_id,
                    worker_id=None,
                ),
                host._parallel.tp_size,
            )
        return None

    def _reask(self, request_id: str, lookup: _Lookup, index: int) -> None:
        """Ask one host again, for the prompt cut at the round's shortest hit."""

        host, part = self._hosts[index], lookup.parts[index]
        end = part.cut * self.lmcache_tokens_per_chunk
        # The query is cut, not its answer: a state image's lock belongs to
        # the endpoint the lookup asked for.
        token_ids = list(lookup.token_ids[:end])
        part.step = "reask"
        part.waiting = host._client.lookup(
            host._create_key(
                token_ids, start=0, end=end, request_id=request_id, worker_id=None
            ),
            host._parallel.tp_size,
        )
        self.tp_hosts_stats["reasks"] += 1

    def _answer(self, request_id: str, lookup: _Lookup, chunks: int) -> int:
        del self._lookups[request_id]
        stats = self.tp_hosts_stats
        stats["answered"] += 1
        if lookup.longest is not None:
            stats["dropped_chunks"] += lookup.longest - chunks
        return chunks

    def _release_all(
        self, request_id: str, token_ids: Sequence[int], answers: list[int]
    ) -> None:
        chunk = self.lmcache_tokens_per_chunk
        for host, answer in zip(self._hosts, answers):
            if not answer:
                continue
            try:
                host.free_lookup_locks(
                    token_ids=token_ids,
                    start=0,
                    end=answer * chunk,
                    request_id=request_id,
                )
            except Exception:
                logger.warning(
                    "LMCache MP: releasing a host's locks for %s failed; they "
                    "are left to the read TTL",
                    request_id,
                    exc_info=True,
                )

    @staticmethod
    def _combine_unrecorded(request_id: str, results: list) -> int | None:
        if any(result is None for result in results):
            return None
        answers = [int(result) for result in results]
        if len(set(answers)) == 1:
            return answers[0]
        logger.warning(
            "LMCache MP: the hosts answered %s for %s, a lookup with no record; "
            "not loading it (its locks are left to the read TTL)",
            answers,
            request_id,
        )
        return 0

    # -- the public adapter surface -----------------------------------------

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int]
    ) -> None:
        if request_id in self._pending_lookups:
            return
        chunk = self.lmcache_tokens_per_chunk
        aligned_end = (len(token_ids) // chunk) * chunk
        if aligned_end == 0:
            self._lookup_results[request_id] = 0
            return
        key = self._create_key(
            token_ids, start=0, end=aligned_end, request_id=request_id, worker_id=None
        )
        self._client.lookup(key, self._parallel.tp_size).result(
            timeout=self._mq_timeout
        )
        self._pending_lookups.add(request_id)

    def check_lookup_result(self, request_id: str) -> int | None:
        if request_id in self._lookup_results:
            return self._lookup_results[request_id]
        if request_id not in self._pending_lookups:
            return 0
        result = self._client.query_prefetch_status(request_id).result(
            timeout=self._mq_timeout
        )
        if result is None:
            return None
        matched = int(result) * self.lmcache_tokens_per_chunk
        self._lookup_results[request_id] = matched
        return matched

    def free_lookup_locks(
        self, token_ids: list[int], start: int, end: int, request_id: str
    ) -> None:
        for host in self._hosts:
            host.free_lookup_locks(
                token_ids=token_ids, start=start, end=end, request_id=request_id
            )

    def cleanup_lookup_result(self, request_id: str) -> None:
        self._pending_lookups.discard(request_id)
        self._lookup_results.pop(request_id, None)
        self._lookups.pop(request_id, None)
        for host in self._hosts:
            host.cleanup_lookup_result(request_id)

    def end_session(self, request_id: str) -> None:
        for host in self._hosts:
            host.end_session(request_id)

    def shutdown(self) -> None:
        for host in self._hosts:
            try:
                host.shutdown()
            except Exception:
                logger.warning(
                    "LMCache MP: closing a host's scheduler adapter failed",
                    exc_info=True,
                )


__all__ = ["TPHostsSchedulerAdapter"]
