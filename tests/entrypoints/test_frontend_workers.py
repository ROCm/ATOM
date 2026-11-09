# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``--api-server-count``: the hub / worker plumbing in
``atom.entrypoints.openai.frontend_workers``, over real ZMQ sockets in one
process, with a stand-in for ``CoreManager``. No GPU, no engine."""

from __future__ import annotations

import socket
import sys
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeout

import pytest

from atom.entrypoints.openai import frontend_workers as fw
from atom.model_engine.request import RequestOutput

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="ipc:// sockets and SO_REUSEPORT"
)


class _Seq:
    """What ``RemoteCoreManager.add_request`` / ``_add_frame`` touch on a
    Sequence: an id, a stream callback and a shipped prompt chain."""

    def __init__(self, seq_id):
        self.id = seq_id
        self.stream_callback = None
        self.prompt_hash_chain = None


class _FakeCoreManager:
    """Answers each request with one chunk and a finished output, the way the
    engine's output thread drives the callbacks and then flushes."""

    max_pool_tokens = 1234

    def __init__(self):
        self.added = []
        self.aborted = []
        self.hub = None
        self.reject = False

    def add_request(self, seqs):
        if self.reject:
            raise ValueError("bad data_parallel_rank")
        self.added.extend(s.id for s in seqs)
        for seq in seqs:
            seq.stream_callback(RequestOutput(seq.id, [7], False))
            seq.stream_callback(RequestOutput(seq.id, [8], True, "stop"))
        self.hub.flush()

    def abort_request(self, req_id):
        self.aborted.append(req_id)


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def hub_and_worker():
    core = _FakeCoreManager()
    hub = fw.FrontendHub(core, num_workers=2)
    core.hub = hub
    hub.start()
    worker = fw.RemoteCoreManager(hub.connection_info(1))
    yield core, hub, worker
    worker.close()
    hub.close()


def test_outputs_reach_the_worker_that_sent_the_request(hub_and_worker):
    core, _hub, worker = hub_and_worker
    got = []
    flushes = []
    worker._flush_stream_batch_fn = lambda: flushes.append(1)

    seq = _Seq(fw.worker_seq_id_base(1) + 5)
    seq.stream_callback = got.append
    worker.add_request([seq])

    assert _wait(lambda: len(got) == 2)
    assert core.added == [seq.id]
    assert [o.output_tokens for o in got] == [[7], [8]]
    assert got[-1].finished
    assert flushes, "worker must flush its stream batch after each hub message"
    # The finished output released the worker's callback.
    assert seq.id not in worker._seq_id_to_callback
    assert worker.max_pool_tokens == 1234


def test_abort_is_forwarded_and_drops_the_local_callback(hub_and_worker):
    core, _hub, worker = hub_and_worker
    worker._seq_id_to_callback[42] = lambda _o: None
    worker.abort_request(42)
    assert _wait(lambda: core.aborted == [42])
    assert 42 not in worker._seq_id_to_callback


def test_a_rejected_request_still_ends(hub_and_worker):
    # core_mgr.add_request raising (e.g. an invalid DP-rank hint) must not
    # leave the HTTP request waiting forever.
    core, _hub, worker = hub_and_worker
    core.reject = True
    got = []
    seq = _Seq(fw.worker_seq_id_base(1) + 9)
    seq.stream_callback = got.append
    worker.add_request([seq])
    assert _wait(lambda: len(got) == 1)
    assert got[0].finished and got[0].finish_reason == "abort"


def test_sequence_id_ranges_do_not_overlap():
    bases = [0] + [fw.worker_seq_id_base(i) for i in range(8)]
    assert bases == sorted(set(bases))
    # 16M ids per worker window, and a 127th worker still fits an int32.
    assert fw.worker_seq_id_base(0) == 1 << 24
    assert fw.worker_seq_id_base(126) + (1 << 24) - 1 < 2**31


def test_workers_can_share_one_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    first = fw.bind_shared_listener("127.0.0.1", port)
    second = fw.bind_shared_listener("127.0.0.1", port)
    try:
        assert first.getsockname()[1] == second.getsockname()[1] == port
    finally:
        first.close()
        second.close()


def test_hub_flush_sends_one_message_per_worker_per_step(hub_and_worker):
    _core, hub, _worker = hub_and_worker

    class _Counting:
        def __init__(self, sock):
            self.sock, self.frames = sock, 0

        def send_multipart(self, parts, copy=False):
            self.frames += 1
            return self.sock.send_multipart(parts, copy=copy)

    counting = _Counting(hub._output_sockets[1])
    hub._output_sockets[1] = counting
    callback = hub._callbacks[1]

    def step():
        for i in range(3):
            callback(RequestOutput(fw.worker_seq_id_base(1) + i, [i], False))
        hub.flush()

    t = threading.Thread(target=step)
    t.start()
    t.join()
    hub._output_sockets[1] = counting.sock
    assert counting.frames == 1


def test_sequence_ids_wrap_inside_the_worker_window():
    mask = (1 << 24) - 1
    base = fw.worker_seq_id_base(2)
    ids = fw.worker_seq_ids(2, start=mask - 1)
    assert [next(ids) for _ in range(3)] == [base + mask - 1, base + mask, base]


class _Proc:
    def __init__(self):
        self.exitcode = None
        self.terminated = False

    def terminate(self):
        self.terminated = True


def test_hub_waits_until_every_worker_serves(hub_and_worker):
    _core, hub, worker = hub_and_worker
    other = fw.RemoteCoreManager(hub.connection_info(0))
    try:
        procs = [_Proc(), _Proc()]
        threading.Timer(0.2, worker.announce_ready).start()
        threading.Timer(0.3, other.announce_ready).start()
        hub.wait_for_workers(procs, timeout=5)
        assert hub.dead_workers() == []
        procs[1].exitcode = -9
        assert hub.dead_workers() == [1]
    finally:
        other.close()


def test_hub_fails_the_launch_if_a_worker_exits_before_serving(hub_and_worker):
    _core, hub, worker = hub_and_worker
    procs = [_Proc(), _Proc()]
    worker.announce_ready()  # worker 1 is fine
    threading.Timer(0.2, lambda: setattr(procs[0], "exitcode", 1)).start()
    with pytest.raises(RuntimeError, match="worker 0 exited with code 1"):
        hub.wait_for_workers(procs, timeout=5)


def test_hub_fails_the_launch_if_a_worker_never_serves(hub_and_worker):
    _core, hub, _worker = hub_and_worker
    with pytest.raises(RuntimeError, match=r"\[0, 1\] not serving"):
        hub.wait_for_workers([_Proc(), _Proc()], timeout=0.3)


def test_metrics_cover_every_worker():
    # The hub's /metrics: the most starved stream across all processes, and
    # each worker's GC counters under its own `process` label.
    from atom.metrics.exporter import AtomMetricsExporter

    board = fw.FrontendStatsBoard(2)
    board.publish(0, None)
    board.publish(1, time.monotonic() - 30.0)
    assert 29.0 < board.longest_silence() < 60.0

    exporter = AtomMetricsExporter(stream_silence=lambda: 1.0)
    text = exporter.render().decode()
    assert 'process="' not in text  # one process: unchanged
    exporter.attach_frontend_board(board)
    text = exporter.render().decode()
    silence = next(
        line
        for line in text.splitlines()
        if line.startswith("atom:stream_longest_silence_seconds ")
    )
    assert float(silence.split()[1]) >= 29.0
    for process in ("hub", "worker0", "worker1"):
        assert (
            f'atom:gc_collections_total{{generation="0",process="{process}"}}' in text
        )
        assert f'atom:gc_threshold{{generation="2",process="{process}"}}' in text


def test_hub_fails_the_launch_if_a_worker_dies_after_serving(hub_and_worker):
    _core, hub, worker = hub_and_worker
    other = fw.RemoteCoreManager(hub.connection_info(0))
    try:
        procs = [_Proc(), _Proc()]
        other.announce_ready()
        worker.announce_ready()
        assert _wait(lambda: len(hub._ready) == 2)
        procs[0].exitcode = 1  # announced, then died
        with pytest.raises(RuntimeError, match=r"\[0\] exited after starting"):
            hub.wait_for_workers(procs, timeout=5)
    finally:
        other.close()


def test_hub_close_stops_the_workers():
    core = _FakeCoreManager()
    hub = fw.FrontendHub(core, num_workers=1)
    hub.start()
    proc = _Proc()
    hub._processes = [proc]
    hub.close()
    assert proc.terminated


def test_sequence_ids_are_safe_to_draw_from_several_threads():
    # Preprocessing runs on a pool; a generator raised "generator already
    # executing" when two threads drew at once.
    ids = fw.worker_seq_ids(3)
    drawn = []
    lock = threading.Lock()

    def draw():
        local = [next(ids) for _ in range(5000)]
        with lock:
            drawn.extend(local)

    threads = [threading.Thread(target=draw) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(drawn) == len(set(drawn)) == 40000
    base = fw.worker_seq_id_base(3)
    assert all(base <= i < base + (1 << 24) for i in drawn)


class _Params:
    def __init__(self, schema):
        self.json_schema = schema


def test_structured_admission_is_answered_by_the_hub_engine():
    # --api-server-count > 1: a worker has no engine, so json_schema /
    # json_object admission goes to the hub's prepare_structured_request,
    # and its error comes back to the request as the engine raised it.
    seen = []

    def prepare(params, kv_transfer, multimodal):
        seen.append((params.json_schema, kv_transfer, multimodal))
        if params.json_schema == "bad":
            raise ValueError("Invalid JSON Schema: bad")

    core = _FakeCoreManager()
    hub = fw.FrontendHub(core, num_workers=1, prepare_structured=prepare)
    core.hub = hub
    hub.start()
    worker = fw.RemoteCoreManager(hub.connection_info(0))
    try:
        worker.prepare_structured_request(_Params('{"type": "object"}'), None, False)
        with pytest.raises(ValueError, match="Invalid JSON Schema"):
            worker.prepare_structured_request(_Params("bad"), {"k": 1}, True)
        # Plain requests never leave the worker.
        worker.prepare_structured_request(_Params(None))
        assert seen == [('{"type": "object"}', None, False), ("bad", {"k": 1}, True)]
        assert worker._calls == {}
        # Outputs still flow on the same socket after replies.
        got = []
        seq = _Seq(fw.worker_seq_id_base(0) + 1)
        seq.stream_callback = got.append
        worker.add_request([seq])
        assert _wait(lambda: len(got) == 2)
    finally:
        worker.close()
        hub.close()


def test_strict_tools_alone_are_admitted_by_the_hub_engine():
    """Strict tool arguments constrain decoding without a JSON Schema answer,
    so such a request goes to the hub's admission too."""
    seen = []
    core = _FakeCoreManager()
    hub = fw.FrontendHub(
        core,
        num_workers=1,
        prepare_structured=lambda params, *_: seen.append(
            params.structured_tool_format
        ),
    )
    core.hub = hub
    hub.start()
    worker = fw.RemoteCoreManager(hub.connection_info(0))
    try:
        params = _Params(None)
        params.structured_tool_format = '{"tools": []}'
        worker.prepare_structured_request(params)
        assert seen == ['{"tools": []}']
    finally:
        worker.close()
        hub.close()


def test_hub_without_an_engine_admission_refuses_structured_requests():
    core = _FakeCoreManager()
    hub = fw.FrontendHub(core, num_workers=1)
    hub.start()
    worker = fw.RemoteCoreManager(hub.connection_info(0))
    try:
        with pytest.raises(ValueError, match="requires engine admission"):
            worker.prepare_structured_request(_Params("{}"))
    finally:
        worker.close()
        hub.close()


def test_hub_close_does_not_wait_for_a_reply_in_flight():
    # A grammar preflight still running at shutdown must not hold close(),
    # and its late reply is dropped instead of touching a closed socket.
    started = threading.Event()
    release = threading.Event()

    def prepare(params, kv_transfer, multimodal):
        started.set()
        release.wait(5)

    core = _FakeCoreManager()
    hub = fw.FrontendHub(core, num_workers=1, prepare_structured=prepare)
    hub.start()
    worker = fw.RemoteCoreManager(hub.connection_info(0))
    outcome = []

    def call():
        try:
            worker.prepare_structured_request(_Params("{}"))
            outcome.append(None)
        except Exception as exc:  # noqa: BLE001 - asserted in the main thread
            outcome.append(exc)

    caller = threading.Thread(target=call, daemon=True)
    fw._STRUCTURED_TIMEOUT_S, saved = 1.0, fw._STRUCTURED_TIMEOUT_S
    try:
        caller.start()
        assert started.wait(5)
        t0 = time.monotonic()
        hub.close()
        assert time.monotonic() - t0 < 2.0
        release.set()  # the late reply finds the sockets closed
        caller.join(5)
        # The caller gave up on its own timeout; no reply ever reached it.
        assert len(outcome) == 1 and isinstance(outcome[0], FuturesTimeout)
        # The drop half: a send after close neither raises nor touches the
        # closed socket (without the guard this is a ZMQError).
        hub._send_to_worker(0, b"R", b"0", b"x")
    finally:
        fw._STRUCTURED_TIMEOUT_S = saved
        release.set()
        worker.close()


def test_a_send_to_a_worker_that_stopped_reading_is_dropped(caplog):
    # SNDTIMEO bounds every hub send; on timeout the message is dropped and
    # logged instead of blocking an engine output thread.
    import zmq

    core = _FakeCoreManager()
    hub = fw.FrontendHub(core, num_workers=1)
    try:
        assert hub._output_sockets[0].getsockopt(zmq.SNDTIMEO) == fw._SEND_TIMEOUT_MS
        assert hub._output_sockets[0].getsockopt(zmq.SNDHWM) == fw._SEND_HWM > 0

        class _Stuck:
            def send_multipart(self, parts, copy=False):
                raise zmq.Again()

        real, hub._output_sockets[0] = hub._output_sockets[0], _Stuck()
        with caplog.at_level("WARNING"):
            hub._send_to_worker(0, b"O", b"frame")
        assert "message dropped" in caplog.text
        hub._output_sockets[0] = real
    finally:
        hub.close()
