# SPDX-License-Identifier: MIT
"""An idle `EngineCore.busy_loop` sleeps on the event its input thread sets after each
put, rather than spinning on the GIL that thread needs; a request queued wakes it."""

import pickle
import queue
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

LONG_WAIT_S = 3600.0  # the idle loop's timeout here: only a wake ends its wait


@pytest.fixture
def idle(monkeypatch):
    """An `EngineCore` with nothing to run, whose scheduler steps what it is given;
    `waits` gets each wait's timeout and whether that wait could return at once."""
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine import engine_core as core
    monkeypatch.setattr(core, "IDLE_WAIT_S", LONG_WAIT_S)
    engine = core.EngineCore.__new__(core.EngineCore)
    waits, pending, stepped = queue.Queue(), [], []

    class Event(threading.Event):
        def wait(self, timeout=None):
            queued = not engine.input_queue.empty() or not engine.utility_queue.empty()
            waits.put((timeout, self.is_set() or queued))
            return super().wait(timeout)

    def step():
        stepped.append([seq.id for seq in pending])
        pending.clear()
        return True

    engine.scheduler = Mock(extend=pending.extend, is_finished=lambda: not pending)
    engine.utility_handler = Mock()
    engine.label = "test"
    engine.kv_transfer_enabled = False
    engine._ingested_seqs = 0
    engine._is_rl_weights_offloaded = False
    engine._weight_sync_fenced = False
    engine._generation_paused = False
    engine.input_queue = queue.Queue()
    engine.utility_queue = queue.Queue()
    engine._kv_progress_event = Event()
    engine._process_engine_step = step
    return core, engine, waits, stepped


def test_a_request_from_the_input_thread_wakes_an_idle_loop(idle, monkeypatch):
    core, engine, waits, stepped = idle
    frames = queue.Queue()  # what the input thread's sockets receive
    sock = SimpleNamespace(send=lambda frame: None, recv=lambda **_: frames.get())
    poller = SimpleNamespace(
        register=lambda *args: None, poll=lambda: [(sock, core.zmq.POLLIN)]
    )
    monkeypatch.setattr(core, "make_zmq_socket", lambda *a, **k: nullcontext(sock))
    monkeypatch.setattr(core.zmq, "Poller", lambda: poller)
    seq = core.Sequence([1, 2, 3], block_size=4)
    inputs = threading.Thread(
        target=engine.process_input_sockets, args=("input", "control"), daemon=True
    )
    loop = threading.Thread(target=engine.busy_loop, daemon=True)
    inputs.start()
    loop.start()
    try:
        assert waits.get(timeout=5) == (LONG_WAIT_S, False)  # idle: asleep
        frames.put(pickle.dumps((core.EngineCoreRequestType.ADD, [seq])))
        assert waits.get(timeout=5) == (LONG_WAIT_S, False)  # stepped, asleep again
        assert stepped == [[seq.id]]
    finally:
        frames.put(pickle.dumps((core.EngineCoreRequestType.SHUTDOWN, None)))
        inputs.join(5)
        loop.join(5)
    assert not inputs.is_alive() and not loop.is_alive()


def test_a_request_queued_before_the_wait_is_not_slept_on(idle):
    core, engine, waits, stepped = idle
    seq = core.Sequence([1, 2, 3], block_size=4)
    stats = engine.scheduler.engine_stats

    def arrive(now):  # the input thread's put and set, after the pull, before the wait
        stats.note_idle = lambda now: None
        engine.input_queue.put_nowait([seq])
        engine._kv_progress_event.set()

    stats.note_idle = arrive
    loop = threading.Thread(target=engine.busy_loop, daemon=True)
    loop.start()
    try:
        assert waits.get(timeout=5) == (LONG_WAIT_S, False)  # only after its step
        assert stepped == [[seq.id]]
    finally:
        engine.input_queue.put_nowait([core.get_exit_sequence()])
        engine._kv_progress_event.set()
        loop.join(5)
    assert not loop.is_alive()
