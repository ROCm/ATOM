# SPDX-License-Identifier: MIT
"""Idle input/utility wakeups, including the clear/check/wait races."""

import queue
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def idle_engine(monkeypatch):
    async_proc = ModuleType("atom.model_engine.async_proc")
    async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.async_proc", async_proc)
    from atom.model_engine.engine_core import EngineCore

    engine = EngineCore.__new__(EngineCore)
    engine.input_queue = queue.Queue()
    engine.utility_queue = queue.Queue()
    engine._kv_progress_event = threading.Event()
    return engine


@pytest.mark.parametrize("queue_name", ["input_queue", "utility_queue"])
def test_queued_arrival_survives_clearing_old_event(idle_engine, queue_name):
    message = object()
    getattr(idle_engine, queue_name).put_nowait(message)
    idle_engine._kv_progress_event.set()
    # A wait would block this test; queued work must return without waiting.
    idle_engine._kv_progress_event.wait = lambda _: pytest.fail("queued work slept")

    idle_engine._wait_for_input()

    assert getattr(idle_engine, queue_name).get_nowait() is message


def test_arrival_between_queue_check_and_wait_stays_latched(idle_engine):
    event = idle_engine._kv_progress_event
    messages = queue.Queue()

    def checked_empty():
        # The input thread runs after the last empty check but before wait.
        messages.put_nowait("shutdown")
        event.set()
        return True

    idle_engine.utility_queue = SimpleNamespace(empty=checked_empty)
    idle_engine._wait_for_input()

    assert event.is_set()
    assert messages.get_nowait() == "shutdown"


@pytest.mark.parametrize("queue_name", ["input_queue", "utility_queue", None])
def test_blocked_idle_engine_wakes_for_request_control_or_kv(idle_engine, queue_name):
    entered_wait = threading.Event()
    resume = threading.Event()
    done = threading.Event()
    failures = []
    real_event = idle_engine._kv_progress_event
    real_wait = real_event.wait

    def observed_wait(timeout):
        assert timeout > 0
        entered_wait.set()
        # Longer test timeout avoids requiring an OS thread to run within 10ms.
        assert real_wait(5), "arrival did not wake the engine"

    real_event.wait = observed_wait

    def run():
        try:
            idle_engine._wait_for_input()
            resume.set()
        except BaseException as exc:
            failures.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered_wait.wait(5)
        if queue_name:
            getattr(idle_engine, queue_name).put_nowait("arrival")
        real_event.set()
        assert done.wait(5)
        assert not failures
        assert resume.is_set()
        if queue_name:
            assert getattr(idle_engine, queue_name).get_nowait() == "arrival"
    finally:
        real_event.set()
        thread.join(5)
