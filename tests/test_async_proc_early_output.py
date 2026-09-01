# SPDX-License-Identifier: MIT

"""Contracts for releasing deferred model output before a forward returns."""

import queue

import pytest

pytest.importorskip("zmq")
pytest.importorskip("aiter.dist.shm_broadcast", exc_type=ImportError)

from atom.model_engine.async_proc import AsyncIOProc


class _Runner:
    def __init__(self):
        self.output_sink = None
        self.sink_history = []

    def _set_forward_output_sink(self, sink):
        self.output_sink = sink
        self.sink_history.append(sink)

    def forward(self):
        assert self.output_sink is not None
        self.output_sink("early")

    def dummy_execution(self):
        # Nested/self-driven forwards must not publish an unsolicited response.
        assert self.output_sink is None
        return "done"


def _proc():
    proc = object.__new__(AsyncIOProc)
    proc.io_addrs = (None, "primary-output")
    proc.io_queues = (queue.Queue(), queue.Queue())
    return proc


def test_top_level_forward_gets_an_early_output_sink():
    proc = _proc()
    runner = _Runner()

    result = proc._invoke_runner(runner, "forward", [])

    assert result is None
    assert proc.io_queues[1].get_nowait() == "early"
    assert runner.output_sink is None
    assert runner.sink_history[0] is not None
    assert runner.sink_history[-1] is None


def test_non_forward_rpc_never_gets_the_sink():
    proc = _proc()
    runner = _Runner()

    result = proc._invoke_runner(runner, "dummy_execution", [])

    assert result == "done"
    assert proc.io_queues[1].empty()
    assert runner.sink_history == [None, None]
