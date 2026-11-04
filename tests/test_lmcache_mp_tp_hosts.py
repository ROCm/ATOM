# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache MP lookups for a TP group spanning hosts (`mp/tp_hosts.py`).

Each host's server is modelled as far as a lookup's read locks go, so the
tests check what every host locked and released, not only the hit returned.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from test_lmcache_mp import _config

from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import tp_hosts
from atom.kv_transfer.offload.mp.lookup_driver import _LockedAdapter
from atom.kv_transfer.offload.mp.tp_hosts import TPHostsSchedulerAdapter

CHUNK = 16


def _sid(lookup_id: str) -> str:
    """The MP session `_MPLookupClient` scopes a lookup to (`_config()`: DP 0)."""
    return f"atom-offload-dp0:{lookup_id}"


class _Future:
    """An LMCache MessagingFuture, done when created unless `done=False`."""

    def __init__(self, value=None, *, error: Exception | None = None, done=True):
        self._value = value
        self._error = error
        self.done = done

    def query(self):
        return self.done

    def result(self, timeout=None):
        if not self.done:
            raise TimeoutError("not done")
        if self._error is not None:
            raise self._error
        return self._value


@dataclass(frozen=True)
class _Key:
    """Like LMCache's IPCCacheServerKey: the prompt is packed, no `token_ids`."""

    token_bytes: tuple
    start: int
    end: int
    request_id: str


class _Server:
    """One host's LMCache MP server, as far as a lookup's read locks go.

    It holds the first `page` PAGE chunks of every prompt and a state image at
    every chunk count in `images`. A lookup's hit is the longest prefix with an
    image at its end; it locks the PAGE chunks under the hit and the image at
    its end. A release, or a retrieve, of `[start, end)` drops what the
    session's last lookup locked in that range -- what LMCache's
    `resolve_prefetched_obj_keys` resolves (the image group's window is one
    chunk, so its lock is the one at the hit's last chunk).
    """

    def __init__(self, *, page: int, images, pending: int = 0) -> None:
        self.page = page
        self.images = set(images)
        self.pending = pending
        self.page_locks: Counter = Counter()
        self.image_locks: Counter = Counter()
        self.sessions: dict[str, dict] = {}
        self.requests: list[tuple[str, int | None]] = []
        self.fail_on: str | None = None

    def _check(self, kind: str) -> None:
        if self.fail_on == kind:
            raise RuntimeError(f"server failed a {kind}")

    def lookup(self, key: _Key) -> None:
        self._check("lookup")
        self.requests.append(("lookup", key.end))
        reach = min(self.page, key.end // CHUNK, len(key.token_bytes) // CHUNK)
        hit = max((end for end in self.images if end <= reach), default=0)
        self.sessions[key.request_id] = {
            "hit": hit,
            "polls": self.pending,
            "reported": False,
        }
        for chunk in range(hit):
            self.page_locks[chunk] += 1
        if hit:
            self.image_locks[hit] += 1

    def status(self, request_id: str) -> int | None:
        self._check("status")
        self.requests.append(("status", None))
        session = self.sessions.get(request_id)
        if session is None or session["reported"]:
            return 0
        if session["polls"]:
            session["polls"] -= 1
            return None
        session["reported"] = True
        return session["hit"]

    def release(self, start: int, end: int, request_id: str) -> None:
        self._check("free")
        self.requests.append(("free", end))
        hit = self.sessions[request_id]["hit"]
        first, last = start // CHUNK, end // CHUNK
        for chunk in range(first, min(last, hit)):
            self.page_locks[chunk] -= 1
        if hit and first <= hit - 1 < last:
            self.image_locks[hit] -= 1
        assert min(self.page_locks.values(), default=0) >= 0
        assert min(self.image_locks.values(), default=0) >= 0

    def retrieve(self, start: int, end: int, request_id: str) -> None:
        """A worker's retrieve: it reads the image at `end`, so that is locked."""
        assert self.image_locks[end // CHUNK] > 0, "retrieve of an unlocked image"
        self.release(start, end, request_id)

    def holds_no_locks(self) -> bool:
        return not +self.page_locks and not +self.image_locks


class _ServerClient:
    """A host adapter's message-queue client (LMCache `RequestClient`)."""

    def __init__(self, server: _Server) -> None:
        self.server = server

    def lookup(self, key: _Key, tp_size: int) -> _Future:
        assert tp_size == 4
        self.server.lookup(key)
        return _Future(None)

    def query_prefetch_status(self, request_id: str) -> _Future:
        return _Future(self.server.status(request_id))

    def free_lookup_locks(self, key: _Key, tp_size: int) -> _Future:
        assert tp_size == 4
        self.server.release(key.start, key.end, key.request_id)
        return _Future(None)


class _Host:
    """One host's scheduler adapter (LMCache `AtomMPSchedulerAdapter`)."""

    def __init__(self, server: _Server) -> None:
        self.server = server
        self.lmcache_tokens_per_chunk = CHUNK
        self._parallel = SimpleNamespace(tp_size=4)
        self._mq_timeout = 1.0
        self._client = _ServerClient(server)
        self.cleaned: list[str] = []
        self.ended: list[str] = []
        self.closed = False

    def _create_key(self, token_ids, start, end, request_id, worker_id):
        assert worker_id is None
        return _Key(tuple(token_ids), start, end, request_id)

    def free_lookup_locks(self, token_ids, start, end, request_id):
        if start < end:
            self._client.free_lookup_locks(
                self._create_key(token_ids, start, end, request_id, None), 4
            )

    def cleanup_lookup_result(self, request_id):
        self.cleaned.append(request_id)

    def end_session(self, request_id):
        self.ended.append(request_id)

    def shutdown(self):
        self.closed = True


def _hosts_client(*servers: _Server, **kwargs):
    adapter = TPHostsSchedulerAdapter([_Host(server) for server in servers])
    client = mp_lookup._MPLookupClient(
        adapter,
        config=_config(),
        timeout=5.0,
        poll_interval=0.0001,
        **kwargs,
    )
    return client, adapter


def _kinds(server: _Server) -> list[str]:
    return [kind for kind, _ in server.requests]


def _load(client, servers, lookup_id: str, start: int, end: int) -> None:
    """Hand `[start, end)` of the hit to the workers, as a load would."""
    client.prepare_retrieve(lookup_id, start, end)
    for server in servers:
        server.retrieve(start, end, _sid(lookup_id))
    client.complete_retrieve(lookup_id, succeeded=True)


TOKENS = list(range(5 * CHUNK + 3))


def test_hosts_that_agree_are_answered_with_one_round_trip_each():
    servers = [_Server(page=4, images={3}), _Server(page=4, images={3}, pending=1)]
    client, adapter = _hosts_client(*servers)

    assert client.lookup(TOKENS, "r") == 3 * CHUNK
    # Host 1 was still prefetching: only it is asked again.
    assert _kinds(servers[0]) == ["lookup", "status"]
    assert _kinds(servers[1]) == ["lookup", "status", "status"]
    assert servers[0].requests[0] == ("lookup", 5 * CHUNK)

    _load(client, servers, "r", 0, 3 * CHUNK)
    assert all(server.holds_no_locks() for server in servers)
    assert adapter.tp_hosts_stats["answered"] == 1
    assert adapter.tp_hosts_stats["disagreed"] == 0


def test_a_longer_host_is_narrowed_to_the_hit_every_host_holds():
    # Host 0 also holds chunk 3 and its image; host 1 only up to chunk 2.
    servers = [_Server(page=4, images={2, 3}), _Server(page=2, images={2})]
    client, adapter = _hosts_client(*servers)

    assert client.lookup(TOKENS, "r") == 2 * CHUNK
    # Host 0 releases what its first lookup locked, and only once that has
    # replied is it asked again, for the prompt cut at 2 chunks.
    assert servers[0].requests == [
        ("lookup", 5 * CHUNK),
        ("status", None),
        ("free", 3 * CHUNK),
        ("lookup", 2 * CHUNK),
        ("status", None),
    ]
    assert _kinds(servers[1]) == ["lookup", "status"]
    for server in servers:
        assert +server.page_locks == Counter({0: 1, 1: 1})
        assert +server.image_locks == Counter({2: 1})

    # Every host's retrieve finds the image at the end of the load locked.
    _load(client, servers, "r", CHUNK, 2 * CHUNK)
    assert all(server.holds_no_locks() for server in servers)
    assert adapter.tp_hosts_stats == {
        "answered": 1,
        "disagreed": 1,
        "reasks": 1,
        "abandoned": 0,
        "dropped_chunks": 1,
    }


def test_narrowing_continues_while_a_reasked_host_answers_shorter():
    # Host 0 has no image at 3 chunks, so cut there it answers 2; host 1 is
    # then cut at 2 as well.
    servers = [_Server(page=5, images={2, 4}), _Server(page=3, images={2, 3})]
    client, adapter = _hosts_client(*servers)

    assert client.lookup(TOKENS, "r") == 2 * CHUNK
    assert ("lookup", 3 * CHUNK) in servers[0].requests
    assert ("lookup", 2 * CHUNK) in servers[1].requests
    assert adapter.tp_hosts_stats["reasks"] == 2
    assert adapter.tp_hosts_stats["dropped_chunks"] == 2

    _load(client, servers, "r", 0, 2 * CHUNK)
    assert all(server.holds_no_locks() for server in servers)


def test_a_host_holding_nothing_releases_the_others_without_a_reask():
    servers = [_Server(page=4, images={3}), _Server(page=0, images=())]
    client, adapter = _hosts_client(*servers)

    assert client.lookup(TOKENS, "r") == 0
    assert _kinds(servers[0]) == ["lookup", "status", "free"]
    assert all(server.holds_no_locks() for server in servers)
    assert adapter.tp_hosts_stats["reasks"] == 0
    assert adapter.tp_hosts_stats["abandoned"] == 0


def test_a_disagreement_is_given_up_after_the_narrowing_rounds():
    # Every cut lands one chunk short of the other host's image.
    servers = [
        _Server(page=10, images={4, 2}),
        _Server(page=10, images={5, 3, 1}),
    ]
    client, adapter = _hosts_client(*servers)

    rounds = tp_hosts._MAX_NARROWING_ROUNDS
    assert rounds == 3
    assert client.lookup(TOKENS, "r") == 0
    assert all(server.holds_no_locks() for server in servers)
    assert adapter.tp_hosts_stats["reasks"] == rounds
    assert adapter.tp_hosts_stats["abandoned"] == 1
    assert adapter.tp_hosts_stats["dropped_chunks"] == 5


def test_a_lookup_that_is_not_loaded_releases_every_hosts_locks():
    servers = [_Server(page=4, images={2, 3}), _Server(page=2, images={2})]
    client, _ = _hosts_client(*servers)

    assert client.lookup(TOKENS, "r") == 2 * CHUNK
    client.clear_lookup_status("r")
    assert all(server.holds_no_locks() for server in servers)


def test_the_async_lookup_client_narrows_without_waiting():
    servers = [_Server(page=4, images={2, 3}), _Server(page=2, images={2})]
    client, _ = _hosts_client(*servers)

    assert client.supports_async()
    assert client.submit(TOKENS, "r")
    polls = 1
    while not client.poll("r"):
        polls += 1
        assert polls < 20
    # (ack, status: disagree), release replied, re-ask acked, status
    assert polls == 4
    assert client.lookup(TOKENS, "r") == 2 * CHUNK
    _load(client, servers, "r", CHUNK, 2 * CHUNK)
    assert all(server.holds_no_locks() for server in servers)


def test_the_lookup_driver_thread_narrows_an_offered_lookup():
    servers = [_Server(page=4, images={2, 3}), _Server(page=2, images={2})]
    adapter = TPHostsSchedulerAdapter([_Host(server) for server in servers])
    lock = threading.RLock()
    client = mp_lookup._MPLookupClient(
        _LockedAdapter(adapter, lock),
        config=_config(),
        timeout=5.0,
        poll_interval=0.001,
        lock=lock,
        driver_window=4,
    )
    try:
        assert client.offer("r", lambda: list(TOKENS))
        deadline = time.monotonic() + 5.0
        while "r" not in client._async_hits and time.monotonic() < deadline:
            time.sleep(0.001)
        assert client._async_hits.get("r") == 2 * CHUNK
        assert client.lookup(TOKENS, "r") == 2 * CHUNK
        _load(client, servers, "r", 0, 2 * CHUNK)
    finally:
        client.close()
    assert all(server.holds_no_locks() for server in servers)
    assert adapter.tp_hosts_stats["reasks"] == 1


def test_a_failed_narrowing_step_drops_the_lookup_and_asks_afresh():
    servers = [_Server(page=4, images={2, 3}), _Server(page=2, images={2})]
    client, adapter = _hosts_client(*servers)
    servers[0].fail_on = "free"

    with pytest.raises(RuntimeError, match="failed a free"):
        client.lookup(TOKENS, "r")
    # Nothing of the failed lookup is left to poll again ...
    assert _sid("r") not in adapter._lookups
    # ... and cleanup asks the servers, which report the lookup consumed.
    servers[0].fail_on = None
    client.clear_lookup_status("r")
    assert all(host.cleaned for host in adapter._hosts)


def test_the_lookup_keeps_the_prompt_beside_the_hosts_keys():
    # LMCache's key holds the prompt packed; a release or a re-ask cannot read
    # token IDs back from it.
    servers = [_Server(page=4, images={2, 3}), _Server(page=2, images={2})]
    adapter = TPHostsSchedulerAdapter([_Host(server) for server in servers])

    key = adapter._create_key(TOKENS, 0, 5 * CHUNK, _sid("r"), None)
    assert not hasattr(key.keys[0], "token_ids")
    assert key.token_ids is TOKENS

    adapter.maybe_submit_lookup_request(_sid("r"), TOKENS)
    answers = [adapter.check_lookup_result(_sid("r")) for _ in range(4)]
    assert answers == [None, None, None, 2 * CHUNK]


def test_hosts_must_share_one_chunk_size():
    hosts = [_Host(_Server(page=1, images=())), _Host(_Server(page=1, images=()))]
    hosts[1].lmcache_tokens_per_chunk = 2 * CHUNK

    with pytest.raises(ValueError, match="disagree on the chunk size"):
        TPHostsSchedulerAdapter(hosts)
    with pytest.raises(ValueError, match="one adapter per host"):
        TPHostsSchedulerAdapter(hosts[:1])


def test_cleanup_and_sessions_fan_out_to_every_host():
    servers = [_Server(page=4, images={3}), _Server(page=4, images={3})]
    adapter = TPHostsSchedulerAdapter([_Host(server) for server in servers])

    adapter.maybe_submit_lookup_request("r", TOKENS)
    assert adapter.check_lookup_result("r") == 3 * CHUNK
    adapter.free_lookup_locks(token_ids=TOKENS, start=0, end=3 * CHUNK, request_id="r")
    adapter.cleanup_lookup_result("r")
    adapter.end_session("r")
    adapter.shutdown()

    assert all(server.holds_no_locks() for server in servers)
    assert all(host.cleaned == ["r"] for host in adapter._hosts)
    assert all(host.ended == ["r"] for host in adapter._hosts)
    assert all(host.closed for host in adapter._hosts)
    assert "r" not in adapter._lookup_results
