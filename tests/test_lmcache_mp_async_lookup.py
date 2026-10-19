# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Asynchronous LMCache MP lookups submitted ahead of admission."""

from __future__ import annotations

from types import SimpleNamespace

from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import transfer

CHUNK = 4


class _Future:
    def __init__(self, value=None, done=True):
        self.value = value
        self.done = done

    def query(self):
        return self.done

    def result(self, timeout=None):
        assert self.done
        return self.value


class _Client:
    """The MQ client: `lookup` acks, then `query_prefetch_status` answers."""

    def __init__(self, chunks, *, lookup_done=True, status=None):
        self.chunks = chunks
        self.lookup_future = _Future(None, lookup_done)
        self.status = list(status) if status is not None else [chunks]
        self.lookups = []
        self.status_queries = []

    def lookup(self, key, tp_size):
        self.lookups.append((key, tp_size))
        return self.lookup_future

    def query_prefetch_status(self, request_id):
        self.status_queries.append(request_id)
        value = self.status.pop(0) if self.status else self.chunks
        return _Future(value)


class _Adapter:
    """The real adapter's lookup bookkeeping, over a fake client."""

    lmcache_tokens_per_chunk = CHUNK

    def __init__(self, client):
        self._client = client
        self._parallel = SimpleNamespace(tp_size=1)
        self._pending_lookups: set[str] = set()
        self._lookup_results: dict[str, int] = {}
        self.freed = []
        self.cleaned = []

    def _create_key(self, token_ids, start, end, request_id, worker_id):
        return (tuple(token_ids[start:end]), request_id)

    def maybe_submit_lookup_request(self, request_id, token_ids):
        if request_id in self._pending_lookups:
            return
        raise AssertionError("consumption must not send a second lookup")

    def check_lookup_result(self, request_id):
        return self._lookup_results.get(request_id)

    def free_lookup_locks(self, **kwargs):
        self.freed.append(kwargs)

    def cleanup_lookup_result(self, request_id):
        self.cleaned.append(request_id)
        self._pending_lookups.discard(request_id)
        self._lookup_results.pop(request_id, None)


def _client(adapter):
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector_extra_config": {}},
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    return mp_lookup._MPLookupClient(
        adapter, config=config, timeout=10.0, poll_interval=0.01
    )


def _rid(client, sid):
    return mp_lookup._mp_session_id(client._config, sid)


def test_submitted_lookup_is_consumed_without_a_second_round_trip(monkeypatch):
    monkeypatch.setattr(transfer.time, "sleep", lambda _s: None)
    fake = _Client(chunks=2)
    adapter = _Adapter(fake)
    client = _client(adapter)

    assert client.submit(list(range(10)), "req")
    assert not client.submit(list(range(10)), "req")
    client.pump()

    assert client.lookup(list(range(10)), "req") == 2 * CHUNK
    assert len(fake.lookups) == 1
    assert fake.lookups[0][0][0] == tuple(range(8))
    assert client.hit_tokens("req") == 2 * CHUNK
    assert adapter.freed == []


def test_lookup_waits_for_an_in_flight_submission(monkeypatch):
    monkeypatch.setattr(transfer.time, "sleep", lambda _s: None)
    fake = _Client(chunks=1, lookup_done=False)
    adapter = _Adapter(fake)
    client = _client(adapter)
    client.submit(list(range(8)), "req")
    client.pump()
    assert "req" in client._async

    original = client._advance

    def finish_then_advance(lookup_id):
        fake.lookup_future.done = True
        return original(lookup_id)

    monkeypatch.setattr(client, "_advance", finish_then_advance)
    assert client.lookup(list(range(8)), "req") == CHUNK
    assert len(fake.lookups) == 1


def test_pending_status_is_requeried():
    fake = _Client(chunks=3, status=[None, None, 3])
    adapter = _Adapter(fake)
    client = _client(adapter)
    client.submit(list(range(12)), "req")
    client.pump()
    client.pump()
    assert "req" in client._async
    client.pump()
    assert "req" not in client._async
    assert adapter._lookup_results[_rid(client, "req")] == 3 * CHUNK


def test_discarded_answered_lookup_releases_its_locks():
    fake = _Client(chunks=2)
    adapter = _Adapter(fake)
    client = _client(adapter)
    client.submit(list(range(8)), "req")
    client.pump()

    client.discard("req")

    assert [(c["start"], c["end"]) for c in adapter.freed] == [(0, 2 * CHUNK)]
    assert adapter.cleaned == [_rid(client, "req")]
    client.discard("req")
    assert len(adapter.freed) == 1


def test_discarded_in_flight_lookup_releases_once_it_answers():
    fake = _Client(chunks=2, lookup_done=False)
    adapter = _Adapter(fake)
    client = _client(adapter)
    client.submit(list(range(8)), "req")

    client.discard("req")
    client.pump()
    assert adapter.freed == []

    fake.lookup_future.done = True
    client.pump()
    assert [(c["start"], c["end"]) for c in adapter.freed] == [(0, 2 * CHUNK)]
    assert client._orphans == set()


def test_short_prompt_is_not_submitted():
    fake = _Client(chunks=0)
    client = _client(_Adapter(fake))
    assert not client.submit(list(range(CHUNK - 1)), "req")
    assert fake.lookups == []


def test_poll_answers_without_blocking():
    fake = _Client(chunks=2, lookup_done=False)
    adapter = _Adapter(fake)
    client = _client(adapter)
    client.submit(list(range(8)), "req")

    assert client.is_pending("req")
    assert not client.poll("req")
    fake.lookup_future.done = True
    assert client.poll("req")
    assert not client.is_pending("req")
    assert client.poll("never-submitted")


def test_adapter_without_async_internals_stays_synchronous(monkeypatch):
    monkeypatch.setattr(transfer.time, "sleep", lambda _s: None)

    class _SyncOnly:
        lmcache_tokens_per_chunk = CHUNK

        def __init__(self):
            self.submitted = []

        def maybe_submit_lookup_request(self, request_id, token_ids):
            self.submitted.append(request_id)

        def check_lookup_result(self, request_id):
            return CHUNK

        def cleanup_lookup_result(self, request_id):
            pass

    adapter = _SyncOnly()
    client = _client(adapter)
    assert not client.submit(list(range(8)), "req")
    assert not client.is_pending("req")
    client.pump()
    assert client.lookup(list(range(8)), "req") == CHUNK
    assert adapter.submitted == [_rid(client, "req")]


def test_discard_after_status_cleanup_still_releases_locks():
    # `request_finished` clears the lookup status before it discards; that
    # cleanup empties the adapter's result cache for this request ID.
    fake = _Client(chunks=2)
    adapter = _Adapter(fake)
    client = _client(adapter)
    client.submit(list(range(8)), "req")
    client.pump()

    client.clear_lookup_status("req")
    assert _rid(client, "req") not in adapter._lookup_results
    client.discard("req")

    assert [(c["start"], c["end"]) for c in adapter.freed] == [(0, 2 * CHUNK)]


def test_answered_lookup_survives_status_cleanup(monkeypatch):
    monkeypatch.setattr(transfer.time, "sleep", lambda _s: None)
    fake = _Client(chunks=2)
    adapter = _Adapter(fake)
    client = _client(adapter)
    client.submit(list(range(8)), "req")
    client.pump()
    client.clear_lookup_status("req")

    assert not client.submit(list(range(8)), "req")
    assert client.lookup(list(range(8)), "req") == 2 * CHUNK
    assert len(fake.lookups) == 1
    assert adapter.freed == []


class _RaisingFuture(_Future):
    def result(self, timeout=None):
        raise RuntimeError("status query failed")


def test_failed_lookup_warns_its_locks_are_left_to_the_ttl(caplog):
    fake = _Client(chunks=2)
    fake.query_prefetch_status = lambda _rid: _RaisingFuture()
    adapter = _Adapter(fake)
    client = _client(adapter)
    client.submit(list(range(8)), "req")

    with caplog.at_level("WARNING", logger=mp_lookup.logger.name):
        client.pump()

    assert not client.is_pending("req")
    assert "read TTL" in caplog.text and "phase status" in caplog.text
