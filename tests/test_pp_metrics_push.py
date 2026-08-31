# SPDX-License-Identifier: MIT
# The PP head's metrics publication (GPU-free).

"""``/metrics`` is served from a cache the API server refreshes out of the
snapshots each EngineCore pushes, so a busy loop that never pushes leaves every
gauge at zero with no error to show for it. ``PPEngineCoreProc`` split the base
loop in two and neither half carried the push, which blanked the whole prefill
role on any ``-pp > 1`` deployment. The head owns the Scheduler and publishes;
downstream stages share its output socket and dp_rank, so their push would land
on the same slot and blank what the head just reported.
"""

from types import SimpleNamespace

from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine import pp_engine_core as pp_mod
    from atom.model_engine.pp_engine_core import PPEngineCoreProc


class FakeUtilityHandler:
    def __init__(self):
        self.pushes = 0

    def process_queue(self, utility_queue, engine):
        pass

    def push_metrics(self):
        self.pushes += 1


class FakeScheduler:
    def __init__(self):
        self.events_published = 0
        self.events_shutdown = 0

    def is_finished(self):
        return True

    def publish_kv_events(self):
        self.events_published += 1

    def shutdown_kv_events(self):
        self.events_shutdown += 1


class FakeRunnerMgr:
    def call_func(self, name, *args, **kwargs):
        return None

    def call_func_with_aggregation(self, name):
        return None


def _stage(*, shutdown_after):
    """A stage whose input queue signals shutdown on the Nth poll."""
    proc = PPEngineCoreProc.__new__(PPEngineCoreProc)
    proc.label = "test-stage"
    proc.kv_transfer_enabled = False
    proc.pp_size = 4
    proc.is_last = False
    proc._in_flight = []
    proc._pp_kv_aggregator = None
    proc._held_sending = {}
    proc._next_idle_kv_drain = 0.0
    proc._is_rl_weights_offloaded = False
    proc.utility_handler = FakeUtilityHandler()
    proc.utility_queue = None
    proc.scheduler = FakeScheduler()
    proc.runner_mgr = FakeRunnerMgr()
    proc.pp_transport = SimpleNamespace(recv_metadata=lambda timeout_ms=0: None)

    polls = {"n": 0}

    def pull_and_process_input_queue():
        polls["n"] += 1
        return polls["n"] >= shutdown_after

    proc.pull_and_process_input_queue = pull_and_process_input_queue
    proc.has_pending_kv_work = lambda: False
    return proc


def _pin_clock(monkeypatch, ticks):
    remaining = list(ticks)
    monkeypatch.setattr(pp_mod.time, "monotonic", lambda: remaining.pop(0))


def test_head_publishes_before_it_touches_the_input_queue():
    proc = _stage(shutdown_after=1)
    proc._head_busy_loop()
    assert proc.utility_handler.pushes == 1


def test_head_paces_pushes_at_the_publish_interval(monkeypatch):
    _pin_clock(monkeypatch, [0.0, 1.0, pp_mod.METRICS_PUSH_INTERVAL_S + 0.5])
    proc = _stage(shutdown_after=3)
    proc._head_busy_loop()
    # First iteration and the one past the interval; not the one 1s in.
    assert proc.utility_handler.pushes == 2


def test_downstream_stage_leaves_the_snapshot_to_the_head():
    proc = _stage(shutdown_after=3)
    proc._downstream_busy_loop()
    assert proc.utility_handler.pushes == 0
