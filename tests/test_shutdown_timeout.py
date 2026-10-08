# SPDX-License-Identifier: MIT
# Grace period before shutdown terminates child processes (GPU-free).

"""Each shutdown path waits ``ATOM_SHUTDOWN_TIMEOUT_S`` for its children to
exit on their own before terminating them, and the children of one process
share that deadline: a slow first child shortens the wait for the next one
instead of every child getting its own full timeout.
"""

import queue
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine.async_proc import AsyncIOProcManager
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.engine_core_mgr import CoreManager


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _Child:
    """A child process that exits on its own ``exits_after`` seconds into the
    shutdown (never if None), or when terminated. ``join`` advances the fake
    clock by the time it would have blocked."""

    def __init__(self, clock, pid, exits_after=None):
        self.clock = clock
        self.pid = pid
        self.exits_at = None if exits_after is None else clock.now + exits_after
        self.alive = True
        self.terminated = False
        self.joins = []

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.joins.append(timeout)
        if not self.alive:
            return
        if self.exits_at is not None and self.exits_at <= self.clock.now + timeout:
            self.clock.now = max(self.clock.now, self.exits_at)
            self.alive = False
        else:
            self.clock.now += timeout

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.alive = False

    def close(self):
        pass


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("time.monotonic", clock)
    return clock


def _core_manager(children):
    mgr = object.__new__(CoreManager)
    mgr._closed = False
    mgr.label = "test"
    mgr.input_sockets = []
    mgr.control_sockets = []
    mgr.shutdown_paths = []
    mgr.output_threads = []
    mgr.engine_core_processes = children
    mgr._distributed_stores = []
    return mgr.close


def _engine_core(children):
    core = object.__new__(EngineCore)
    core.still_running = True
    core.label = "test"
    core.runner_mgr = SimpleNamespace(
        keep_monitoring=True, call_func=lambda _name: None, procs=children
    )
    core._send_engine_dead = lambda: None
    return core.exit


def _runner_manager(children):
    mgr = object.__new__(AsyncIOProcManager)
    mgr.still_running = True
    mgr.label = "test"
    mgr.procs = children
    mgr.output_thread = SimpleNamespace(join=lambda timeout=None: None)
    mgr.kv_output_threads = []
    mgr.outputs_queue = queue.Queue()
    mgr.parent_finalizer = lambda: None
    return mgr.exit


SHUTDOWN_PATHS = pytest.mark.parametrize(
    "make_shutdown",
    [_core_manager, _engine_core, _runner_manager],
    ids=["server->EngineCore", "EngineCore->ModelRunner", "runner manager"],
)


@SHUTDOWN_PATHS
def test_the_default_grace_period_is_five_seconds(monkeypatch, clock, make_shutdown):
    monkeypatch.delenv("ATOM_SHUTDOWN_TIMEOUT_S", raising=False)
    child = _Child(clock, pid=1)
    make_shutdown([child])()
    assert child.joins[0] == 5
    assert child.terminated


@SHUTDOWN_PATHS
def test_a_child_that_exits_within_the_timeout_is_not_terminated(
    monkeypatch, clock, make_shutdown
):
    monkeypatch.setenv("ATOM_SHUTDOWN_TIMEOUT_S", "600")
    child = _Child(clock, pid=1, exits_after=300)
    make_shutdown([child])()
    assert child.joins[0] == 600
    assert not child.terminated


@SHUTDOWN_PATHS
def test_children_share_one_deadline(monkeypatch, clock, make_shutdown):
    monkeypatch.setenv("ATOM_SHUTDOWN_TIMEOUT_S", "30")
    start = clock.now
    slow = _Child(clock, pid=1, exits_after=20)
    stuck = _Child(clock, pid=2)
    late = _Child(clock, pid=3)
    make_shutdown([slow, stuck, late])()
    assert not slow.terminated
    assert stuck.joins[0] == 10
    assert late.joins[0] == 0
    assert stuck.terminated and late.terminated
    # 30s of grace in total, plus whatever the post-terminate joins blocked.
    assert clock.now - start == 30
