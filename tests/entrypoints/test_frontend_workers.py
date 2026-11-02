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
    # 16M ids per process, and a 127th worker still fits an int32.
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

        def send(self, frame, copy=False):
            self.frames += 1
            return self.sock.send(frame, copy=copy)

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
