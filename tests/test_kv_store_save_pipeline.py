# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The opt-in save pipeline on the real worker, pool and Store client (CPU fakes)."""

import logging
import signal
import threading
import time
from types import SimpleNamespace

import kv_store_test_support
import pytest
import test_mooncake_store_offload as offload

from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL,
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.kv_store import save_pipeline
from atom.kv_transfer.offload.kv_store import worker as worker_mod
from atom.kv_transfer.offload.kv_store.client import CallRefused
from atom.kv_transfer.offload.kv_store.pool import SlotPoolExhausted
from atom.kv_transfer.offload.kv_store.save_pipeline import StreamStopped

aiter_arch = offload.aiter_arch
cluster = offload.cluster
make_worker = offload.make_worker
KEYS = 256  # tokens per Store chunk in these tests


def _hung(*_):
    raise TimeoutError("the test hung")


@pytest.fixture(autouse=True)
def pipeline_chunks(monkeypatch):
    # The mooncake_store test module's own settings and the shared helpers
    # its requests and completions come from (`kv_store_test_support`).
    for module in (offload, kv_store_test_support):
        monkeypatch.setattr(module, "CHUNK", KEYS)
        monkeypatch.setattr(module, "CHUNK_BYTES", 1024)
    monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", "1")
    previous = signal.signal(signal.SIGALRM, _hung)
    signal.alarm(20)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def _channels(worker):
    return worker.get_finished().connector_completions


def _source(gpu, req, index):
    """The bytes chunk ``index`` of ``req`` holds on the (fake) GPU."""
    blocks = req.block_ids[index * KEYS // offload.BLOCK :][: KEYS // offload.BLOCK]
    return b"".join(gpu.kv.get(b, gpu.pattern(b)) for b in blocks)


def _packed(gpu):
    return sorted(s for kind, starts in gpu.calls if kind == "from" for s in starts)


def _reports(worker, gpu):
    reported = []
    report = worker._source_group_safe

    def counting(identity):
        [(first, _end)] = identity.ranges
        reported.append(first // KEYS)
        report(identity)

    worker._source_group_safe = counting
    gpu.source_safe = counting
    return reported


@pytest.mark.parametrize(
    "tokens, value, expected",
    [(256, None, 0), (256, "", 0), (256, "0", 0), (256, "1", 1), (256, "64", 64)]
    + [(256, "-1", ValueError), (256, "65", ValueError), (8, "2", 2), (512, "64", 64)],
)
def test_pipeline_is_opt_in_and_validated(monkeypatch, tokens, value, expected):
    monkeypatch.delenv("ATOM_STORE_SAVE_BATCH_CHUNKS")
    if value is not None:
        monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", value)
    config = offload._config(extra=offload._extra(chunk_tokens=tokens))
    if expected is ValueError:
        with pytest.raises(ValueError):
            offload.MooncakeStoreOffloadConnector(config)
        return
    worker = offload.MooncakeStoreOffloadConnector(config)
    worker.close()
    assert worker._save_batch_chunks == expected
    assert (worker._sender is not None) == bool(expected)
    assert worker._sender is None or not worker._sender.thread.is_alive()


def test_pipeline_needs_one_save_worker(monkeypatch):
    monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", "64")
    monkeypatch.setenv("OFFLOAD_COPY_WORKERS", "2")
    with pytest.raises(ValueError, match="one save thread"):
        offload.MooncakeStoreOffloadConnector(offload._config())


def test_batches_reuse_the_window_lookup_tail_first():
    segments = [
        (range(4, 9), False, list("45678"), 3.5),
        (range(4), True, list("0123"), None),
    ]
    reserved = []
    got = []
    for r, h, k, ms in worker_mod._stream_chunks(
        segments, 2, lambda: reserved.append(len(got))
    ):
        got.append((list(r), h, k, ms))
    assert reserved == [0, 1, 2]  # one credit per batch the store lacks, before it
    assert got == [
        ([7, 8], False, ["7", "8"], 3.5),
        ([5, 6], False, ["5", "6"], None),
        ([4], False, ["4"], None),
        ([2, 3], True, ["2", "3"], None),
        ([0, 1], True, ["0", "1"], None),
    ]


def _sender_of(make_worker, name):
    worker, _ = make_worker(save_slots=32)
    req = offload._save_req(name, 8 * KEYS)
    worker._sender.begin_request(req)
    return worker, worker._sender, req


def _submit(worker, req, *, first=0, n=2):
    worker._sender.reserve(req)
    slots = worker._pool.acquire("save", n)
    run = range(first, first + n)
    worker._sender.submit(req, run, worker._window_keys(req, run), slots)
    return slots


def _block_native_put(monkeypatch, worker):
    store = worker._client._store
    entered, release = threading.Event(), threading.Event()
    put = store.batch_put_from

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release the native put"
        return put(*args, **kwargs)

    monkeypatch.setattr(store, "batch_put_from", blocked)
    return entered, release


def test_two_credits_bound_preparation_while_a_put_runs(monkeypatch, make_worker):
    worker, sender, req = _sender_of(make_worker, "credits")
    entered, release = _block_native_put(monkeypatch, worker)
    _submit(worker, req)
    assert entered.wait(1)
    _submit(worker, req, first=2)
    acquired = threading.Event()

    def third():
        sender.reserve(req)
        acquired.set()
        sender.release_preparing()

    thread = threading.Thread(target=third)
    thread.start()
    try:
        assert not acquired.wait(0.05) and sender.pending == 2
    finally:
        release.set()
        thread.join(2)
    assert acquired.is_set()
    result = sender.drain_request(req)
    assert (result.stored_chunks, result.problem, sender.pending) == (4, None, 0)


def _raise(error):
    def fail(*args):
        raise error

    return fail


def _partial_hook(pool, slots, answer):
    pool.release(slots[:1])
    pool.quarantine(slots[1:], reason="the native side still owns it")
    return "partial settlement"


def _hook_that_raises(pool, slots, answer):
    pool.release(slots[:1])
    raise RuntimeError("hook blew up")


@pytest.mark.parametrize(
    "patch, problem, quarantined, kind",
    [
        # The hook's release was only recorded, never applied: both slots stay out.
        (("_settle_put", _hook_that_raises), "hook blew up", 2, "failed"),
        (("_settle_put", _partial_hook), "partial settlement", 1, "failed"),
        (("_put_chunks", _raise(CallRefused("unsent"))), "refused", 0, "refused"),
        # Not an Exception, still the put's own failure: labelled as one.
        (("_put_chunks", _raise(SystemExit("gone"))), "native exception", 2, "raised"),
        (("_distrust_store", _raise(OSError())), "distrust failed", 2, "raised"),
    ],
)
def test_settlement_applies_only_what_the_hook_settled(
    monkeypatch, make_worker, patch, problem, quarantined, kind
):
    worker, sender, req = _sender_of(make_worker, "settlement")
    worker._client._store.raise_on = "put" if "distrust" in problem else None
    monkeypatch.setattr(worker, *patch)
    slots = _submit(worker, req)
    done = sender.drain_request(req)
    assert problem in done.problem and done.kind == f"put_{kind}"
    assert [slot.leased for slot in slots] == [False, False]
    assert worker._pool.quarantined("save") == quarantined
    assert sender.thread.is_alive()


def test_a_batch_behind_a_failed_put_is_released_not_put(
    monkeypatch, make_worker, caplog
):
    worker, sender, req = _sender_of(make_worker, "behind-failed")
    entered, release = _block_native_put(monkeypatch, worker)
    _submit(worker, req)
    assert entered.wait(1)
    behind = _submit(worker, req, first=2)
    worker._client._store.raise_on = "put"
    release.set()
    assert "put blew up" in sender.drain_request(req).problem
    assert not any(slot.leased for slot in behind)
    assert worker._pool.quarantined("save") == 2  # the failed put's, not behind's
    [record] = [r for r in caplog.records if "a native put raised" in r.getMessage()]
    assert record.exc_info and "a put raised" in worker._store_distrusted


def test_a_batch_preparing_during_close_is_still_put(make_worker):
    worker, sender, req = _sender_of(make_worker, "closing")
    sender.reserve(req)
    slots = worker._pool.acquire("save", 2)
    assert not sender.close(timeout=0.05), "close left a batch preparing behind"
    sender.submit(req, range(2), worker._window_keys(req, range(2)), slots)
    assert sender.drain_request(req).stored_chunks == 2 and sender.close(timeout=2)


@pytest.mark.parametrize("stop", ["source_read_deadline", "pool_exhausted"])
def test_a_break_after_reserve_returns_its_credit(monkeypatch, make_worker, stop):
    worker, _ = make_worker(save_slots=32)
    sender = worker._sender
    pool = worker._pool
    acquire = pool.acquire
    real_deadline = worker._source_read_deadline
    if stop == "source_read_deadline":
        monkeypatch.setattr(
            worker, "_source_read_deadline", lambda req: time.monotonic() - 1
        )
    else:

        def exhausted(*args, **kwargs):
            raise SlotPoolExhausted("test")

        monkeypatch.setattr(pool, "acquire", exhausted)
    for name in ("first", "second"):  # more breaks than credits
        worker._do_save_req(offload._save_req(name, 2 * KEYS))
        assert sender.pending == 0 and sender.credits._value == save_pipeline.CREDITS
    monkeypatch.setattr(worker, "_source_read_deadline", real_deadline)
    monkeypatch.setattr(pool, "acquire", acquire)
    req = offload._save_req("after", 2 * KEYS)
    worker._do_save_req(req)  # blocks in `reserve` if a credit leaked
    assert offload._store(req.save_operation, True) in _channels(worker)
    assert sender.pending == 0 and sender.close(timeout=2)


def test_reserve_checks_stop_after_taking_a_credit(make_worker):
    _, sender, req = _sender_of(make_worker, "close-in-acquire")
    credits = sender.credits

    class ClosingAfterAcquire:
        _value = property(lambda self: credits._value)

        def acquire(self, timeout=None):
            got = credits.acquire(timeout=timeout)
            sender.close()  # after the credit is taken, before `reserve` checks
            return got

        def release(self):
            credits.release()

    sender.credits = ClosingAfterAcquire()
    with pytest.raises(StreamStopped):
        sender.reserve(req)  # without the check this queues for no thread
    assert sender.pending == 0 and credits._value == save_pipeline.CREDITS
    assert not sender.thread.is_alive()


def test_close_waits_for_a_queued_lookup(monkeypatch, make_worker):
    worker, sender, _ = _sender_of(make_worker, "close-lookup")
    entered, release = threading.Event(), threading.Event()
    exists = worker._client.exists

    def blocked(keys):
        entered.set()
        assert release.wait(5), "test did not release the lookup"
        return exists(keys)

    monkeypatch.setattr(worker._client, "exists", blocked)
    lookup = sender.start_lookup(["k"])
    assert entered.wait(1)
    assert not sender.close(timeout=0.05), "close must wait for the lookup"
    release.set()
    sender.finish_lookup(lookup)
    assert sender.close(timeout=2)


def test_a_closed_sender_queues_no_lookup(make_worker):
    _, sender, _ = _sender_of(make_worker, "closed-lookup")
    assert sender.close(timeout=1)
    # A lookup queued behind the stop sentinel would never run, and a later
    # close would wait for it forever.
    with pytest.raises(StreamStopped):
        sender.start_lookup(["k"])
    assert sender.jobs.unfinished_tasks == 0


def test_flag_off_save_looks_up_directly_without_wrapper_or_lookahead(
    monkeypatch, make_worker
):
    monkeypatch.delenv("ATOM_STORE_SAVE_BATCH_CHUNKS")
    worker, _ = make_worker(save_slots=4)
    assert worker._sender is None
    # The serial save is the class's own `_do_save_req`: no wrapper in front.
    assert "_do_save_req" not in vars(worker)
    events, built = [], []
    window_keys = worker._window_keys
    exists = worker._client.exists

    def keys_of(req, window):
        keys = window_keys(req, window)
        built.append(keys)
        events.append("keys")
        return keys

    def lookup(keys):
        assert any(keys is made for made in built), "keys were copied"
        events.append("exists")
        return exists(keys)

    monkeypatch.setattr(worker, "_window_keys", keys_of)
    monkeypatch.setattr(worker._client, "exists", lookup)
    req = offload._save_req("flag-off", 12 * KEYS)
    worker._do_save_req(req)
    assert offload._store(req.save_operation, True) in _channels(worker)
    assert len(built) >= 2
    # No lookahead: each window's keys are built right before its lookup.
    assert events == ["keys", "exists"] * len(built)


def test_held_chunks_are_never_packed_or_put(cluster, make_worker):
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("mixed", 4 * KEYS)
    held = {offload._key(req.chunk_hashes, i) for i in (0, 2)}
    objects = cluster.objects.setdefault(offload.MASTER, {})
    for key in held:
        objects[key] = b"already present"
    worker._do_save_req(req)
    assert _packed(gpu) == [KEYS, 3 * KEYS]
    calls = worker._client._store.calls
    puts = {key for call in calls if call[0] == "put" for key in call[1]}
    assert puts == {offload._key(req.chunk_hashes, i) for i in (1, 3)}
    for i in (1, 3):  # each stored key holds exactly its own chunk's source bytes
        assert objects[offload._key(req.chunk_hashes, i)] == _source(gpu, req, i)
    assert all(objects[key] == b"already present" for key in held)
    done = _channels(worker)
    assert {offload._safe(req.save_operation, i) for i in range(4)} <= done
    assert offload._store(req.save_operation) in done


def test_lookup_failure_puts_every_chunk(make_worker, caplog):
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("lookup-failure", 3 * KEYS)
    worker._client._store.raise_on = "exists"
    worker._do_save_req(req)
    assert _packed(gpu) == [0, KEYS, 2 * KEYS]
    assert offload._store(req.save_operation) in _channels(worker)
    assert "lookup-failure" in caplog.text and "lookup failed" in caplog.text


def test_a_fatal_lookup_reaches_the_waiter_and_the_sender_survives(make_worker):
    worker, sender, _ = _sender_of(make_worker, "fatal")
    worker._client.exists = _raise(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        sender.finish_lookup(sender.start_lookup(["k"]))
    assert sender.thread.is_alive()


def test_a_stopped_sender_is_not_a_lookup_failure(make_worker, caplog):
    worker, gpu = make_worker(save_slots=32)
    sender = worker._sender
    req = offload._save_req("stopped", 3 * KEYS)
    assert sender.close(timeout=1)
    worker._do_save_req(req)
    assert gpu.calls == []
    assert offload._store(req.save_operation, False) in _channels(worker)
    assert "lookup failed" not in caplog.text


def test_a_lookahead_stopped_at_close_ends_the_save_like_reserve(
    monkeypatch, make_worker, caplog
):
    worker, gpu = make_worker(save_slots=8)  # windows of 2 chunks
    req = offload._save_req("ahead-stop", 6 * KEYS)
    start, calls = worker._sender.start_lookup, []

    def stop_the_lookahead(keys):
        calls.append(keys)
        # Window 2's lookup, queued ahead at window 1 (window 0 submitted).
        if len(calls) == 3:
            raise StreamStopped("closed")
        return start(keys)

    monkeypatch.setattr(worker._sender, "start_lookup", stop_the_lookahead)
    reported = _reports(worker, gpu)
    worker._do_save_req(req)
    assert reported[:2] == [5, 4] and sorted(reported) == list(range(6))
    # Window 0 (chunks 4, 5) was packed and put, and is counted.
    assert _packed(gpu) == [4 * KEYS, 5 * KEYS] and len(calls) == 3
    assert worker._save_counts["put_chunks"] == 2
    assert (
        "stream_stopped" in caplog.text and "pipelined save failed" not in caplog.text
    )
    assert offload._store(req.save_operation, False) in _channels(worker)


def test_a_registration_error_in_put_leaves_the_sender_draining(make_worker):
    worker, sender, req = _sender_of(make_worker, "registered")
    registered = worker._registered
    worker._registered = _raise(RuntimeError("gone"))  # before the put can run
    _submit(worker, req)
    assert "gone" in sender.drain_request(req).problem
    worker._registered = registered
    assert sender.thread.is_alive()
    assert worker._store_distrusted == "put handling raised RuntimeError('gone')"


@pytest.mark.parametrize("raising_hook", [True, False])
def test_only_a_raising_put_handler_distrusts_the_store(
    monkeypatch, make_worker, raising_hook
):
    worker, sender, req = _sender_of(make_worker, "distrust")
    patch = _hook_that_raises if raising_hook else _raise(CallRefused("unsent"))
    monkeypatch.setattr(worker, "_settle_put" if raising_hook else "_put_chunks", patch)
    _submit(worker, req)
    sender.drain_request(req)
    expected = "put handling raised RuntimeError('hook blew up')"
    assert worker._store_distrusted == (expected if raising_hook else None)


def test_an_early_raise_claims_every_unread_chunk(monkeypatch, make_worker):
    worker, _ = make_worker(save_slots=32)
    req = offload._save_req("early", 3 * KEYS)
    monkeypatch.setattr(worker, "_source_read_deadline", _raise(RuntimeError("x")))
    aborted = []
    monkeypatch.setattr(worker, "_abort_save", lambda r, chunks: aborted.append(chunks))
    worker._do_save_req(req)
    assert aborted == [range(3)]


class Fatal(BaseException):
    """Not an Exception: nothing may swallow it."""


@pytest.mark.parametrize(
    "raised, fence_fails",
    [
        (RuntimeError("second pack failed"), False),
        (RuntimeError("second pack failed"), True),
        (Fatal(), False),
    ],
)
def test_copy_failure_drains_native_before_the_terminal(
    make_worker, monkeypatch, caplog, cluster, raised, fence_fails
):
    monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", "2")
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("copy-failure", 8 * KEYS)
    objects = cluster.objects.setdefault(offload.MASTER, {})
    for i in (4, 5):
        objects[offload._key(req.chunk_hashes, i)] = b"already present"
    entered, release = _block_native_put(monkeypatch, worker)
    copy_failed = threading.Event()
    copy = gpu.batched_from_gpu
    reported = _reports(worker, gpu)

    def fail_second_copy(*args, **kwargs):
        if gpu.calls and not copy_failed.is_set():
            assert entered.wait(2)
            copy_failed.set()
            raise raised
        return copy(*args, **kwargs)

    def fence(device):
        if fence_fails:
            raise RuntimeError("copy fence failed")

    monkeypatch.setattr(gpu, "batched_from_gpu", fail_second_copy)
    monkeypatch.setattr(worker_mod, "_synchronize", fence)
    errors = []
    draining = threading.Event()
    drain = worker._sender.drain_request

    def drain_noted(request):
        draining.set()
        return drain(request)

    monkeypatch.setattr(worker._sender, "drain_request", drain_noted)

    def save():
        try:
            worker._do_save_req(req)
        except BaseException as exc:  # noqa: BLE001 - surface thread failures
            errors.append(exc)

    thread = threading.Thread(target=save)
    thread.start()
    try:
        assert copy_failed.wait(2) and draining.wait(2)
        thread.join(0.1)  # a save that skipped the drain ends meanwhile
        before = _channels(worker)
        assert before == {offload._safe(req.save_operation, i) for i in range(4, 8)}
        assert thread.is_alive() and worker._sender.pending > 0
    finally:
        release.set()
        thread.join(5)
    sender = worker._sender
    if isinstance(raised, Fatal):
        assert errors == [raised] and not sender.results and sender.pending == 0
        later = offload._save_req("after-fatal", KEYS)
        worker._do_save_req(later)
        assert offload._store(later.save_operation) in _channels(worker)
        return
    assert not thread.is_alive() and not errors
    after = before | _channels(worker)
    stores = [c for c in after if c.channel == DENSE_PAGE_STORE_CHANNEL]
    assert stores == [offload._store(req.save_operation, False)]
    safe = {c for c in after if c.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL}
    quiescent = {c for c in after if c.channel == DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL}
    if fence_fails:
        assert safe == before and not quiescent
        assert worker._pool.quarantined("save") == 2
    else:
        assert safe == {offload._safe(req.save_operation, i) for i in range(8)}
        assert quiescent == {offload._quiescent(req.save_operation)}
        assert worker._pool.quarantined("save") == 0
        assert sorted(reported) == list(range(8))
    assert "copy-failure" in caplog.text and "second pack failed" in caplog.text
    assert "Traceback" in caplog.text


def test_an_earlier_put_failure_stops_preparation_and_reports_every_chunk_once(
    make_worker, monkeypatch, caplog
):
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("reserve-stopped", 2 * KEYS)
    worker._client._store.raise_on = "put"
    sender = worker._sender
    reserve = sender.reserve
    reported = _reports(worker, gpu)
    calls, failure_seen = 0, []

    def reserve_after_failure(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            with sender.condition:
                failure_seen.append(
                    sender.condition.wait_for(
                        lambda: sender.results[request.save_operation].problem
                        is not None,
                        timeout=2,
                    )
                )
        return reserve(request)

    monkeypatch.setattr(sender, "reserve", reserve_after_failure)
    worker._do_save_req(req)
    assert calls == 2 and failure_seen == [True]
    assert gpu.calls == [("from", [KEYS])]
    done = _channels(worker)
    assert {c for c in done if c.channel == DENSE_PAGE_STORE_CHANNEL} == {
        offload._store(req.save_operation, False)
    }
    assert {offload._safe(req.save_operation, i) for i in range(2)} <= done
    assert offload._quiescent(req.save_operation) in done
    assert sorted(reported) == [0, 1]
    assert worker._pool.quarantined("save") == 1 and worker._store_distrusted
    assert "reserve-stopped" in caplog.text and "put blew up" in caplog.text
    # The put's own failure, not the stop it caused, is the one reported.
    assert "a save failed (put_raised:" in caplog.text
    assert "stream_stopped" not in caplog.text


def test_next_window_lookup_does_not_wait_behind_the_current_put(
    make_worker, monkeypatch
):
    monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", "64")
    worker, gpu = make_worker(save_slots=32)
    window = worker._pool.window("save", worker.save_workers)
    assert window < 64, "each window must be one put at batch size 64"
    req = offload._save_req("overlap", 3 * window * KEYS)
    store = worker._client._store
    events = []
    entered, release = threading.Event(), threading.Event()
    put, copy, exists = worker._put_chunks, gpu.batched_from_gpu, store.batch_is_exist

    def blocked_put(*args, **kwargs):
        events.append("put")
        entered.set()
        assert release.wait(5), "test did not release the native put"
        return put(*args, **kwargs)

    def logged_copy(*args, **kwargs):
        events.append("pack")
        return copy(*args, **kwargs)

    def logged_exists(keys):
        events.append("exists")
        return exists(keys)

    monkeypatch.setattr(worker, "_put_chunks", blocked_put)
    monkeypatch.setattr(gpu, "batched_from_gpu", logged_copy)
    monkeypatch.setattr(store, "batch_is_exist", logged_exists)
    thread = threading.Thread(target=worker._do_save_req, args=(req,))
    thread.start()
    try:
        assert entered.wait(2)
        deadline = time.monotonic() + 2
        while events.count("pack") < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not release.is_set() and events.count("pack") >= 2, events
        assert events[: events.index("put")].count("exists") == 2, events
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert offload._store(req.save_operation) in _channels(worker)


def test_close_keeps_the_owners_while_a_lookahead_lookup_is_blocked(
    monkeypatch, cluster, make_worker
):
    # Windows [4,6), [2,4), [0,2), one chunk a batch. The first put is held
    # until the second window's `exists` has queued the lowest window's
    # lookup, then raises; the two credits are both out after the first
    # window, so the second window's reserve waits for that put and the save
    # stops there. The lowest window's lookup then blocks in `exists`.
    worker, gpu = make_worker(save_slots=8)
    pool, client, _ = worker._registered()
    store = cluster.stores[-1]
    exists = worker._client.exists
    entered, release = threading.Event(), threading.Event()
    req = offload._save_req("blocked-lookup", 6 * KEYS)
    lowest = worker._window_keys(req, range(2))

    def blocking(keys):
        if keys == lowest:
            entered.set()
            assert release.wait(5), "test did not release the lookup"
        return exists(keys)

    monkeypatch.setattr(worker._client, "exists", blocking)
    queued, waited = threading.Event(), []
    start_lookup = worker._sender.start_lookup

    def noting(keys):
        future = start_lookup(keys)
        if list(keys) == lowest:
            queued.set()
        return future

    def held_then_raising(*args, **kwargs):
        waited.append(queued.wait(2))
        raise RuntimeError("put blew up")

    monkeypatch.setattr(worker._sender, "start_lookup", noting)
    monkeypatch.setattr(worker, "_put_chunks", held_then_raising)
    close = worker._sender.close
    monkeypatch.setattr(worker._sender, "close", lambda: close(timeout=0.02))
    worker._do_save_req(req)  # stopped at its next batch, its puts drained
    assert waited == [True]  # one put ran, after the lowest lookup was queued
    assert offload._store(req.save_operation, False) in _channels(worker)
    assert entered.wait(1)
    started = time.monotonic()
    worker.close()  # the one close the exit path makes
    assert time.monotonic() - started < 1, "close must give up, not wait"
    assert worker._pool is pool and worker._client is client
    assert not gpu.closed and not store.closed
    with pytest.raises(RuntimeError, match="shutdown"):
        worker._save_executor.submit(lambda: None)
    release.set()
    worker.close()  # a second close releases everything
    assert gpu.closed and store.closed
    assert not worker._sender.thread.is_alive()


def test_close_joins_a_save_that_waits_on_its_put(monkeypatch, cluster, make_worker):
    worker, gpu = make_worker(save_slots=32)
    store = cluster.stores[-1]
    entered, release = _block_native_put(monkeypatch, worker)
    close = worker._sender.close
    monkeypatch.setattr(worker._sender, "close", lambda: close(timeout=0.02))
    req = offload._save_req("joined-close", KEYS)

    def save():
        worker._sender.begin_request(req)
        _submit(worker, req, n=1)
        return worker._sender.drain_request(req)

    future = worker._save_executor.submit(save)
    assert entered.wait(1)
    threading.Timer(0.2, release.set).start()
    worker.close()
    # The executor was joined: no save thread outlives close.
    assert future.done() and future.result().stored_chunks == 1
    assert gpu.closed and store.closed
    assert not worker._sender.thread.is_alive()


def test_a_pipelined_save_stores_each_chunks_own_bytes(
    monkeypatch, cluster, make_worker
):
    monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", "2")  # keys map within a batch
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("bytes", 5 * KEYS)
    worker._do_save_req(req)
    objects = cluster.objects[offload.MASTER]
    for index in range(5):
        stored = objects[offload._key(req.chunk_hashes, index)]
        assert stored == _source(gpu, req, index)
    assert worker._sender.results == {}
    assert offload._store(req.save_operation) in _channels(worker)


def test_close_returns_true_when_the_last_put_returns_just_before_the_deadline(
    monkeypatch, make_worker
):
    worker, sender, req = _sender_of(make_worker, "late-drain")
    entered, release = _block_native_put(monkeypatch, worker)
    _submit(worker, req)
    assert entered.wait(1)
    now, gate = [0.0], threading.Event()
    task_done = sender.jobs.task_done

    def held_exit():
        if sender.stop_sent:
            assert gate.wait(5), "test did not open the gate"
        task_done()

    def sleep(_):
        release.set()
        while sender.jobs.unfinished_tasks:
            time.sleep(0.001)
        now[0] = 0.999
        threading.Timer(0.2, gate.set).start()

    monkeypatch.setattr(sender.jobs, "task_done", held_exit)
    fake = SimpleNamespace(
        monotonic=lambda: now[0], sleep=sleep, perf_counter=time.perf_counter
    )
    monkeypatch.setattr(save_pipeline, "time", fake)
    assert sender.close(timeout=1.0) is True and not sender.thread.is_alive()


def test_each_lookup_is_queued_once_for_its_own_window_in_order(
    make_worker, monkeypatch
):
    worker, gpu = make_worker(save_slots=8)  # windows of 2 chunks
    req = offload._save_req("lookup-keys", 6 * KEYS)
    queued = []
    start = worker._sender.start_lookup

    def noted(keys):
        queued.append(list(keys))
        return start(keys)

    monkeypatch.setattr(worker._sender, "start_lookup", noted)
    worker._do_save_req(req)
    windows = [range(4, 6), range(2, 4), range(2)]
    assert queued == [worker._window_keys(req, w) for w in windows]
    assert _packed(gpu) == [i * KEYS for i in range(6)]


def test_a_held_window_is_found_through_the_lookahead(make_worker, cluster):
    worker, gpu = make_worker(save_slots=8)
    req = offload._save_req("lookahead-held", 6 * KEYS)
    objects = cluster.objects.setdefault(offload.MASTER, {})
    for i in (2, 3):  # the middle window: queued when the first is looked up
        objects[offload._key(req.chunk_hashes, i)] = b"already present"
    worker._do_save_req(req)
    assert _packed(gpu) == [i * KEYS for i in (0, 1, 4, 5)]
    assert objects[offload._key(req.chunk_hashes, 2)] == b"already present"


def test_drain_waits_for_the_last_put_despite_a_wakeup(make_worker, monkeypatch):
    worker, sender, req = _sender_of(make_worker, "drain-wakeup")
    entered, release = _block_native_put(monkeypatch, worker)
    _submit(worker, req)
    assert entered.wait(2)
    returned = threading.Event()

    def drain():
        sender.drain_request(req)
        returned.set()

    thread = threading.Thread(target=drain)
    thread.start()
    try:
        time.sleep(0.05)
        with sender.condition:
            sender.condition.notify_all()  # not a put's return
        assert not returned.wait(0.1), "drain returned with a put still running"
    finally:
        release.set()
        thread.join(5)
    assert returned.is_set()
    assert sender.results[req.save_operation].outstanding == 0


def test_a_failed_second_copy_reports_the_first_batch_once(make_worker, monkeypatch):
    monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", "2")
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("second-copy", 4 * KEYS)
    reported = _reports(worker, gpu)
    copy = gpu.batched_from_gpu

    def fail_second_copy(*args, **kwargs):
        if gpu.calls:
            raise RuntimeError("second copy failed")
        return copy(*args, **kwargs)

    monkeypatch.setattr(gpu, "batched_from_gpu", fail_second_copy)
    monkeypatch.setattr(worker_mod, "_synchronize", lambda device: None)
    worker._do_save_req(req)
    assert sorted(reported) == [0, 1, 2, 3]  # chunks 2, 3 once: read, then no abort
    assert offload._store(req.save_operation, False) in _channels(worker)


def test_a_later_batch_keeps_the_first_put_failure_kind(monkeypatch, make_worker):
    worker, sender, req = _sender_of(make_worker, "first-kind")
    entered, release = _block_native_put(monkeypatch, worker)
    store = worker._client._store
    blocked = store.batch_put_from

    def raising(*args, **kwargs):
        blocked(*args, **kwargs)
        raise RuntimeError("the NIC went away")

    monkeypatch.setattr(store, "batch_put_from", raising)
    _submit(worker, req, first=0)
    assert entered.wait(1)
    _submit(worker, req, first=2)  # queued behind the put that will raise
    release.set()
    result = sender.drain_request(req)
    assert result.kind == "put_raised"
    assert "the NIC went away" in result.problem


def test_a_failed_copy_after_a_skip_reports_only_the_saved_range(
    make_worker, monkeypatch
):
    monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", "2")
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("skip-copy", 4 * KEYS, skip=KEYS)  # chunk 0 is skipped
    reported = _reports(worker, gpu)

    def fail(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(gpu, "batched_from_gpu", fail)
    monkeypatch.setattr(worker_mod, "_synchronize", lambda device: None)
    worker._do_save_req(req)
    assert sorted(reported) == [1, 2, 3]  # never the skipped chunk 0
    assert offload._store(req.save_operation, False) in _channels(worker)


def test_the_pipelined_profile_counts_only_the_bytes_stored(
    monkeypatch, cluster, make_worker, caplog
):
    worker, _ = make_worker(save_slots=8)
    req = offload._save_req("put-bytes", 2 * KEYS)  # one window, two batches
    store = cluster.store_of(offload.MASTER)
    # Tail first: chunk 1 is put and stored, chunk 0's put returns refused.
    store.put_codes[offload._key(req.chunk_hashes, 0)] = (
        offload.store_client.NO_AVAILABLE_HANDLE
    )
    monkeypatch.setattr(worker, "_profile_enabled", lambda: True)
    with caplog.at_level(logging.INFO, logger="atom"):
        worker._do_save_req(req)
    [profile] = [
        r.getMessage() for r in caplog.records if "SAVE-PROF" in r.getMessage()
    ]
    # Both puts returned; only the settled one's bytes count.
    assert "put_chunks=2 " in profile
    assert f"put_bytes={worker._chunk_bytes} " in profile


def test_a_failed_save_stops_looking_up_held_windows(cluster, make_worker):
    worker, _ = make_worker(save_slots=8)  # eight windows of two chunks
    req = offload._save_req("stop-lookups", 16 * KEYS)
    objects = cluster.objects.setdefault(offload.MASTER, {})
    for i in range(14):  # every window below the first is already stored
        objects[offload._key(req.chunk_hashes, i)] = b"already present"
    worker._client._store.raise_on = "put"  # the first window's puts fail
    windows = {
        tuple(worker._window_keys(req, range(i, i + 2))) for i in range(0, 16, 2)
    }
    asked = []
    exists = worker._client.exists

    def counting(keys):
        if tuple(keys) in windows:
            asked.append(keys)
        return exists(keys)

    worker._client.exists = counting
    worker._do_save_req(req)
    assert offload._store(req.save_operation, False) in _channels(worker)
    # Held windows take no credit, so only the lookahead can see the failed
    # put: once it has, no lookup is queued (the serial save asks once).
    assert len(asked) <= 4


def test_recovery_never_quarantines_a_slot_the_next_batch_leased(
    monkeypatch, make_worker
):
    # Two slots in all: the hook releases one and quarantines the other, and
    # that quarantine raises after the release. Another thread (the save
    # thread's next batch) asks for a slot meanwhile; the free one is the slot
    # just released. The recovery must leave it to that batch.
    worker, _ = make_worker(save_slots=2)
    sender, pool = worker._sender, worker._pool
    req = offload._save_req("settle-race", 2 * KEYS)
    sender.begin_request(req)
    taken, quarantine, calls = [], pool.quarantine, []

    def next_batch():
        taken.extend(pool.acquire("save", 1))

    def raising_once(slots, **kwargs):
        calls.append(list(slots))
        if len(calls) > 1:  # the recovery's own quarantine goes through
            return quarantine(slots, **kwargs)
        other = threading.Thread(target=next_batch)
        other.start()
        other.join(0.3)  # the next batch's lease lands now, if it can
        raise RuntimeError("quarantine blew up")

    def hook(settlement, slots, answer):
        settlement.release(slots[:1])
        settlement.quarantine(slots[1:])
        return "partial settlement"

    monkeypatch.setattr(pool, "quarantine", raising_once)
    monkeypatch.setattr(worker, "_settle_put", hook)
    _submit(worker, req, n=2)
    assert sender.drain_request(req).problem == "partial settlement"
    deadline = time.monotonic() + 2
    while not taken:
        assert time.monotonic() < deadline, "the next batch never got its slot"
        time.sleep(0.01)
    [slot] = taken
    assert slot.leased, "the next batch's slot was quarantined"
    assert pool.quarantined("save") == 1
    distrusted = "put handling raised RuntimeError('quarantine blew up')"
    assert worker._store_distrusted == distrusted
    pool.release(taken)
