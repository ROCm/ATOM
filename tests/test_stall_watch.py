# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
import threading
import time

from atom.utils import stall_watch as sw


def _wait_for(predicate, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _stall_headers(caplog):
    return [
        r.getMessage() for r in caplog.records if "one step has run" in r.getMessage()
    ]


def test_the_watch_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("ATOM_STALL_DUMP_S", raising=False)
    assert sw.stall_watch("PP0TP0") is sw.NO_STALL_WATCH


def test_the_watch_starts_when_asked(monkeypatch):
    monkeypatch.setenv("ATOM_STALL_DUMP_S", "60")
    watch = sw.stall_watch("PP3TP0")
    assert isinstance(watch, sw.StallWatch)
    assert watch.seconds == 60.0


def test_a_step_past_the_limit_dumps_every_thread_once(caplog):
    caplog.set_level(logging.WARNING, logger="atom")
    release = threading.Event()

    def stuck_in_a_save():
        release.wait()

    worker = threading.Thread(target=stuck_in_a_save, name="offload-save")
    worker.start()
    try:
        watch = sw.StallWatch("PP2TP0", 0.2, poll_s=0.02)
        watch.arm()
        assert _wait_for(lambda: _stall_headers(caplog))
        time.sleep(0.3)
        assert len(_stall_headers(caplog)) == 1
        assert "PP2TP0" in _stall_headers(caplog)[0]
        stacks = [
            r.getMessage() for r in caplog.records if "offload-save" in r.getMessage()
        ]
        assert stacks and "stuck_in_a_save" in stacks[0]
    finally:
        release.set()
        worker.join()


def test_each_new_step_gets_its_own_dump(caplog):
    caplog.set_level(logging.WARNING, logger="atom")
    watch = sw.StallWatch("EngineCore_PP2", 0.1, poll_s=0.02)
    watch.arm()
    assert _wait_for(lambda: len(_stall_headers(caplog)) == 1)
    watch.arm()
    assert _wait_for(lambda: len(_stall_headers(caplog)) == 2)


def test_a_step_that_finishes_in_time_logs_nothing(caplog):
    caplog.set_level(logging.WARNING, logger="atom")
    watch = sw.StallWatch("PP0TP0", 0.2, poll_s=0.02)
    for _ in range(10):
        watch.arm()
        time.sleep(0.03)
        watch.disarm()
    time.sleep(0.3)
    assert _stall_headers(caplog) == []
