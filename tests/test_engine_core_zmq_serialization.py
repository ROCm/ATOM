# SPDX-License-Identifier: MIT

import pickle
import queue
import threading
import time

from atom.model_engine.engine_core_mgr import CoreManager


class _ContentionDetectingSocket:
    closed = False

    def __init__(self):
        self._state_lock = threading.Lock()
        self.active_sends = 0
        self.max_active_sends = 0
        self.messages = []

    def send_multipart(self, frames, copy=False):
        del copy
        with self._state_lock:
            self.active_sends += 1
            self.max_active_sends = max(self.max_active_sends, self.active_sends)
        # Release the GIL so concurrent callers reliably overlap without the
        # CoreManager send lock.
        time.sleep(0.002)
        self.messages.append(frames)
        with self._state_lock:
            self.active_sends -= 1


def _bare_manager(num_ranks=1):
    mgr = CoreManager.__new__(CoreManager)
    mgr.label = "Engine Core Mgr"
    mgr.local_engine_count = num_ranks
    mgr.engine_core_identities = [f"rank-{rank}".encode() for rank in range(num_ranks)]
    mgr.input_sockets = [_ContentionDetectingSocket() for _ in range(num_ranks)]
    mgr._input_send_lock = threading.Lock()
    mgr._utility_command_lock = threading.Lock()
    mgr.utility_response_queue = queue.Queue()
    return mgr


def test_concurrent_input_messages_are_serialized():
    mgr = _bare_manager()
    barrier = threading.Barrier(8)

    def send(worker):
        barrier.wait()
        for sequence in range(10):
            mgr._send_input_message(0, pickle.dumps((worker, sequence)), copy=False)

    threads = [threading.Thread(target=send, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    socket = mgr.input_sockets[0]
    assert all(not thread.is_alive() for thread in threads)
    assert socket.max_active_sends == 1
    assert len(socket.messages) == 80
    assert all(frames[0] == b"rank-0" for frames in socket.messages)
    assert {pickle.loads(frames[1]) for frames in socket.messages} == {
        (worker, sequence) for worker in range(8) for sequence in range(10)
    }


def test_synchronous_utility_transactions_do_not_overlap():
    mgr = _bare_manager(num_ranks=2)
    state_lock = threading.Lock()
    active_transactions = 0
    max_active_transactions = 0

    def fake_broadcast(cmd, **kwargs):
        nonlocal active_transactions, max_active_transactions
        del kwargs
        with state_lock:
            active_transactions += 1
            max_active_transactions = max(max_active_transactions, active_transactions)
        time.sleep(0.01)
        for rank in range(mgr.local_engine_count):
            mgr.utility_response_queue.put({"cmd": cmd, "rank": rank})
        with state_lock:
            active_transactions -= 1

    mgr.broadcast_utility_command = fake_broadcast
    barrier = threading.Barrier(2)
    results = {}

    def call(cmd):
        barrier.wait()
        results[cmd] = mgr.broadcast_utility_command_sync(cmd, timeout=1)

    threads = [
        threading.Thread(target=call, args=(cmd,)) for cmd in ("metrics", "cache")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert max_active_transactions == 1
    assert {
        cmd: {(response["cmd"], response["rank"]) for response in responses}
        for cmd, responses in results.items()
    } == {
        "metrics": {("metrics", 0), ("metrics", 1)},
        "cache": {("cache", 0), ("cache", 1)},
    }
