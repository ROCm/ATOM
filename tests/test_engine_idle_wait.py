# SPDX-License-Identifier: MIT
"""An idle engine loop blocks instead of spinning, so its socket threads get
the GIL as soon as they have something to send.

The step that finishes a request is the one after which the loop goes idle.
A loop spinning in Python hands the GIL to the output thread only when forced,
once per `sys.getswitchinterval()` (5ms), so the last token chunk of every
request reached the client ~5ms late. These tests drive the real busy loop and
the real output thread over a real ZMQ socket, with no GPU.
"""

from __future__ import annotations

import pickle
import queue
import statistics
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import zmq
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine import engine_core as core

from atom.model_engine.sequence import get_exit_sequence


def _bare_engine():
    engine = core.EngineCore.__new__(core.EngineCore)
    engine.label = "idle-wait-test"
    engine._work_arrived = threading.Event()
    engine._kv_progress_event = threading.Event()
    engine.input_queue = queue.Queue()
    engine.utility_queue = queue.Queue()
    engine.output_queue = queue.Queue()
    return engine


def test_wait_returns_at_once_when_input_is_already_queued():
    engine = _bare_engine()
    engine.input_queue.put_nowait(["req"])
    start = time.perf_counter()
    engine._wait_for_work(timeout=1.0)
    assert time.perf_counter() - start < 0.1


def test_wait_returns_at_once_when_a_utility_command_is_queued():
    engine = _bare_engine()
    engine.utility_queue.put_nowait(("cmd", {}))
    start = time.perf_counter()
    engine._wait_for_work(timeout=1.0)
    assert time.perf_counter() - start < 0.1


def test_wait_is_bounded_by_its_timeout():
    engine = _bare_engine()
    start = time.perf_counter()
    engine._wait_for_work(timeout=0.02)
    assert 0.015 <= time.perf_counter() - start < 0.5


def test_a_stale_signal_does_not_skip_the_wait():
    """A wake-up already consumed must not turn the next idle pass into a spin."""
    engine = _bare_engine()
    engine._work_arrived.set()
    start = time.perf_counter()
    engine._wait_for_work(timeout=0.02)
    assert time.perf_counter() - start >= 0.015


def test_an_arrival_wakes_the_wait():
    engine = _bare_engine()

    def arrive():
        time.sleep(0.02)
        engine.input_queue.put_nowait(["req"])
        engine._work_arrived.set()

    threading.Thread(target=arrive, daemon=True).start()
    start = time.perf_counter()
    engine._wait_for_work(timeout=5.0)
    assert time.perf_counter() - start < 1.0


class _IdleScheduler:
    """Nothing to run: every pass of the loop takes the idle branch."""

    engine_stats = SimpleNamespace(
        note_idle=lambda now: None,
        note_busy=lambda now: None,
        bump_step=lambda: None,
    )

    def __init__(self):
        self.passes = 0

    def heartbeat_throughput(self, now):
        self.passes += 1

    def is_finished(self):
        return True

    def extend(self, seqs):
        raise AssertionError("no request was scripted")

    def publish_kv_events(self):
        pass

    def shutdown_kv_events(self):
        pass


def test_idle_loop_lets_the_output_thread_send_the_last_output_promptly():
    """put -> socket latency of an output enqueued just before the loop idles.

    Spinning, this waits for the GIL switch interval (measured 10ms median on
    the same harness with the wait removed); blocking, ~0.1ms.
    """
    engine = _bare_engine()
    engine.scheduler = _IdleScheduler()
    engine.utility_handler = SimpleNamespace(
        process_queue=lambda *a: None, push_metrics=lambda: None
    )
    engine._has_pending_utility = False
    engine._is_rl_weights_offloaded = False
    engine._weight_sync_fenced = False
    engine._generation_paused = False
    engine.kv_transfer_enabled = False

    ctx = zmq.Context()
    address = f"ipc:///tmp/atom-idle-wait-{id(engine)}.ipc"
    pull = ctx.socket(zmq.PULL)
    pull.bind(address)
    output = threading.Thread(
        target=engine.process_output_sockets, args=(address,), daemon=True
    )
    output.start()
    loop = threading.Thread(target=engine.busy_loop, daemon=True)
    loop.start()
    latencies = []
    try:
        # Warm the socket pair up before measuring.
        engine.output_queue.put_nowait(("METRICS", time.perf_counter()))
        assert pull.poll(5000)
        pull.recv()
        for _ in range(30):
            time.sleep(0.02)  # the loop is idle in between
            engine.output_queue.put_nowait(("METRICS", time.perf_counter()))
            assert pull.poll(5000)
            _, sent_at = pickle.loads(pull.recv())
            latencies.append(time.perf_counter() - sent_at)
    finally:
        engine.input_queue.put_nowait([get_exit_sequence()])
        engine._work_arrived.set()
        loop.join(timeout=5)
        engine.output_queue.put_nowait([get_exit_sequence()])  # ends the thread
        output.join(timeout=5)
        pull.close(linger=0)
        ctx.term()
    assert not loop.is_alive()
    assert statistics.median(latencies) < sys.getswitchinterval() / 2, latencies
    # The loop really idled through the measurement, at a bounded pace rather
    # than a spin (one pass per `IDLE_WAIT_S`, give or take).
    assert engine.scheduler.passes < 30 * 0.02 / core.IDLE_WAIT_S * 3


@pytest.mark.parametrize("held", [False, True])
def test_an_idle_loop_still_exits_on_shutdown(held):
    engine = _bare_engine()
    engine.scheduler = _IdleScheduler()
    engine.utility_handler = SimpleNamespace(
        process_queue=lambda *a: None, push_metrics=lambda: None
    )
    engine._is_rl_weights_offloaded = held
    engine._weight_sync_fenced = False
    engine._generation_paused = False
    engine.kv_transfer_enabled = False
    engine._report_held_kv_work = lambda: None
    loop = threading.Thread(target=engine.busy_loop, daemon=True)
    loop.start()
    time.sleep(0.05)
    engine.input_queue.put_nowait([get_exit_sequence()])
    engine._work_arrived.set()
    loop.join(timeout=5)
    assert not loop.is_alive()
