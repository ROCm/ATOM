# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler-side lookups for a TP group whose ranks span several hosts.

An LMCache MP server moves KV through GPU IPC, so it can serve only its own
host's ranks: each host runs one server, and its ranks store and retrieve their
shards there as a host-local world. A prefix is loadable only if every rank can
retrieve its shard, so the engine's scheduler, which runs on one host, asks
every host's server and takes the answer they all give.

``TPHostsSchedulerAdapter`` presents those per-host scheduler adapters as one,
with the attributes and methods `lookup._MPLookupClient` drives. Hosts store in
lockstep (every rank saves the same chunks at the same boundaries), so their
answers normally agree. When they do not, the prefix is reported as not held:
a host whose hit ends later holds no read lock on the STATE image at the
shorter hit's endpoint, and its retrieve would then release a lock it never
took. Every host's locks are released instead, and the request recomputes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger("atom")


class _Answered:
    """A finished future holding a value already received."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def query(self) -> bool:
        return True

    def wait(self, timeout: float | None = None) -> bool:
        return True

    def result(self, timeout: float | None = None) -> Any:
        return self._value


class _AllHosts:
    """One future per host; done when all are, its result combines theirs."""

    def __init__(self, futures: Sequence[Any], combine: Callable[[list], Any]) -> None:
        self._futures = list(futures)
        self._combine = combine
        self._done = False
        self._value: Any = None

    def query(self) -> bool:
        return all(future.query() for future in self._futures)

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        for future in self._futures:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not future.wait(left):
                return False
        return True

    def result(self, timeout: float | None = None) -> Any:
        if not self._done:
            deadline = None if timeout is None else time.monotonic() + timeout
            results = []
            for future in self._futures:
                left = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                results.append(future.result(timeout=left))
            # Combined once: the status combination releases locks.
            self._value = self._combine(results)
            self._done = True
        return self._value


class _HostsClient:
    """The message-queue client surface `_MPLookupClient` uses, over all hosts."""

    def __init__(self, owner: TPHostsSchedulerAdapter) -> None:
        self._owner = owner

    def lookup(self, keys: Sequence[Any], tp_size: int) -> _AllHosts:
        del tp_size  # each host's own key and parallel config decide
        return self._owner._submit(keys)

    def query_prefetch_status(self, request_id: str) -> _AllHosts:
        return self._owner._status(request_id)


class TPHostsSchedulerAdapter:
    """One `AtomMPSchedulerAdapter` per host, answering as one."""

    def __init__(self, hosts: Sequence[Any]) -> None:
        if len(hosts) < 2:
            raise ValueError("a TP group spanning hosts needs one adapter per host")
        chunks = {int(host.lmcache_tokens_per_chunk) for host in hosts}
        if len(chunks) != 1:
            raise ValueError(
                f"the hosts' LMCache MP servers disagree on the chunk size: {sorted(chunks)}"
            )
        self._hosts = list(hosts)
        self.lmcache_tokens_per_chunk = chunks.pop()
        self._parallel = self._hosts[0]._parallel
        self._client = _HostsClient(self)
        self._pending_lookups: set[str] = set()
        self._lookup_results: dict[str, int] = {}
        # request_id -> per-host hit in chunks (None until that host answers)
        self._answers: dict[str, list[int | None]] = {}
        # request_id -> the prompt a lookup was sent for, to release locks
        self._tokens: dict[str, list[int]] = {}
        self.host_mismatches = 0

    # -- the internals `_MPLookupClient` drives ------------------------------

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        worker_id: int | None,
    ) -> tuple:
        return tuple(
            host._create_key(
                token_ids,
                start=start,
                end=end,
                request_id=request_id,
                worker_id=worker_id,
            )
            for host in self._hosts
        )

    def _submit(self, keys: Sequence[Any]) -> _AllHosts:
        if len(keys) != len(self._hosts):
            raise ValueError("one lookup key per host is required")
        request_id = keys[0].request_id
        self._answers[request_id] = [None] * len(self._hosts)
        self._tokens[request_id] = list(keys[0].token_ids)
        futures = [
            host._client.lookup(key, host._parallel.tp_size)
            for host, key in zip(self._hosts, keys)
        ]
        return _AllHosts(futures, lambda _results: None)

    def _status(self, request_id: str) -> _AllHosts:
        answers = self._answers.get(request_id)
        if answers is None:
            # Not sent through `_submit` (or already combined): every host's
            # server answers 0 for a consumed job, which combines to 0.
            futures = [
                host._client.query_prefetch_status(request_id) for host in self._hosts
            ]
            return _AllHosts(futures, self._combine_unknown)
        futures = [
            (
                _Answered(answer)
                if answer is not None
                else host._client.query_prefetch_status(request_id)
            )
            for host, answer in zip(self._hosts, answers)
        ]
        return _AllHosts(futures, lambda results: self._combine(request_id, results))

    @staticmethod
    def _combine_unknown(results: list) -> int | None:
        if any(result is None for result in results):
            return None
        return min(int(result) for result in results)

    def _combine(self, request_id: str, results: list) -> int | None:
        answers = self._answers.get(request_id)
        if answers is None:
            return self._combine_unknown(results)
        for index, result in enumerate(results):
            if result is not None:
                answers[index] = int(result)
        if any(answer is None for answer in answers):
            return None
        del self._answers[request_id]
        token_ids = self._tokens.pop(request_id, None)
        if len(set(answers)) == 1:
            return answers[0]
        self.host_mismatches += 1
        logger.warning(
            "LMCache MP: the hosts' servers hold different prefixes for %s "
            "(chunks per host: %s); not loading it (%d such lookups so far)",
            request_id,
            answers,
            self.host_mismatches,
        )
        chunk = self.lmcache_tokens_per_chunk
        for host, answer in zip(self._hosts, answers):
            if answer and token_ids is not None:
                host.free_lookup_locks(
                    token_ids=token_ids,
                    start=0,
                    end=answer * chunk,
                    request_id=request_id,
                )
        return 0

    # -- the public adapter surface -----------------------------------------

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int]
    ) -> None:
        if request_id in self._pending_lookups or request_id in self._lookup_results:
            return
        chunk = self.lmcache_tokens_per_chunk
        aligned_end = (len(token_ids) // chunk) * chunk
        if aligned_end == 0:
            self._lookup_results[request_id] = 0
            return
        key = self._create_key(
            token_ids, start=0, end=aligned_end, request_id=request_id, worker_id=None
        )
        timeout = max(float(host._mq_timeout) for host in self._hosts)
        self._client.lookup(key, self._parallel.tp_size).result(timeout=timeout)
        self._pending_lookups.add(request_id)

    def check_lookup_result(self, request_id: str) -> int | None:
        if request_id in self._lookup_results:
            return self._lookup_results[request_id]
        if request_id not in self._pending_lookups:
            return 0
        timeout = max(float(host._mq_timeout) for host in self._hosts)
        result = self._client.query_prefetch_status(request_id).result(timeout=timeout)
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
        self._answers.pop(request_id, None)
        self._tokens.pop(request_id, None)
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
