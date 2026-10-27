# SPDX-License-Identifier: MIT
"""The two busy loops, driven over a scripted schedule against real stats.

Step-mode throughput has state that only the loop can get wrong: `bump_step`
must see every forward and `note_idle` / `note_busy` must bracket exactly the
passes where none ran. Asserting that on an `EngineStats` built by hand proves
nothing about either loop, and the DP loop in particular has a shape that
defeats naive wiring -- `engines_running` starts True, so the first pass runs a
dummy and counts it before the idle branch is ever reachable. A guard keyed on
the step count would have been dead there and no unit test on `EngineStats`
could have seen it.

Executed by AST extraction, so the loops under test are the real ones with no
torch, aiter or GPU present.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.model_engine.engine_stats import EngineStats

ENGINE_CORE = Path(__file__).resolve().parents[1] / "atom/model_engine/engine_core.py"

INTERVAL_STEPS = 4


def _extract(class_name: str, method: str):
    """The real method, compiled with stubs for the module globals it reads."""
    tree = ast.parse(ENGINE_CORE.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    func = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    scope: dict = {
        "envs": SimpleNamespace(ATOM_METRICS_UPDATE_INTERVAL_S=1e9),
        "logger": SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a: None),
        "tracing": SimpleNamespace(shutdown=lambda: None),
        "time": SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _: None),
    }
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module([func], []), str(ENGINE_CORE), "exec"), scope
    )
    return scope[method], scope


class _Engine:
    """Enough of an engine for one busy loop, driven by a pass schedule.

    `schedule` is one `(now, has_work)` per pass; the loop exits when it runs
    out. `has_work` drives both `is_finished` and the DP all-reduce, i.e. it is
    engine-wide, which is the input contract `note_idle` documents.
    """

    def __init__(self, schedule, interval_steps=INTERVAL_STEPS):
        self._schedule = list(schedule)
        self._pass = -1
        self.stats = EngineStats(
            enable_log_stats=True,
            engine_index=0,
            throughput_log_interval_s=1e9,
            throughput_log_interval_steps=interval_steps,
        )
        self.stats._throughput_last_log_time = 0.0
        self.scheduler = SimpleNamespace(
            engine_stats=self.stats,
            heartbeat_throughput=lambda now: None,
            is_finished=lambda: not self._has_work,
            prefill_delayer=None,
            publish_kv_events=lambda: None,
            shutdown_kv_events=lambda: None,
        )
        self.utility_queue = None
        self.utility_handler = SimpleNamespace(
            process_queue=lambda *a: self._advance(),
            push_metrics=lambda: None,
        )
        self.label = "test"
        self.engines_running = True
        self._is_rl_weights_offloaded = False
        self._weight_sync_fenced = False
        self.forwards = 0
        self.dummies = 0

    def _advance(self):
        self._pass += 1
        if self._pass >= len(self._schedule):
            raise StopIteration
        self._now, self._has_work = self._schedule[self._pass]

    # -- collaborators the loop bodies call ---------------------------------

    def now(self):
        return self._now

    def pull_and_process_input_queue(self):
        return False

    def _is_idle_rl_weights_offloaded(self):
        return False

    def _sync_dp_state(self, local_unfinished, shutdown, offloaded):
        return self._has_work, False, False

    def _process_engine_step(self):
        if not self._has_work:
            return False
        self.forwards += 1
        return True

    def _execute_dummy_batch(self):
        self.dummies += 1

    def has_pending_kv_work(self):
        return False

    def _advance_idle_kv_transfer(self):
        raise AssertionError("no KV work was scripted")

    def _drain_kv_work_at_exit(self):
        return None


def _run(class_name, schedule, **kw):
    loop, scope = _extract(class_name, "busy_loop")
    engine = _Engine(schedule, **kw)
    scope["time"].monotonic = engine.now
    with pytest.raises(StopIteration):
        loop(engine)
    return engine


# A burst, a long silence, a second burst. The window stays open across all of
# it -- it closes on the step count, which the silence does not advance -- so
# the silence is exactly what must not reach the divisor.
BURST_IDLE_BURST = [
    (0.0, True),
    (1.0, True),
    (2.0, False),  # work has drained
    (3.0, False),
    (603.0, True),
    (604.0, True),
]


@pytest.mark.parametrize(
    "class_name, origin, work_seconds",
    # The loops differ by one pass, and legitimately: at 2.0 the base loop has
    # nothing to run, while the DP loop pads the pass with a dummy for its
    # peers. A forward ran, so that second is work on one and silence on the
    # other. Both drop the 600 in the middle.
    [("EngineCore", 601.0, 3.0), ("DPEngineCoreProc", 600.0, 4.0)],
)
def test_a_silent_stretch_is_not_charged_to_the_window(
    class_name, origin, work_seconds
):
    stats = _run(class_name, BURST_IDLE_BURST).stats
    assert stats._throughput_last_log_time == pytest.approx(origin)
    assert 604.0 - stats._throughput_last_log_time == pytest.approx(work_seconds)


@pytest.mark.parametrize("class_name", ["EngineCore", "DPEngineCoreProc"])
def test_every_pass_with_a_forward_is_counted(class_name):
    """And no pass without one is.

    The DP loop pads an empty schedule with a dummy and counts that; the base
    loop has no dummy, so its idle passes count nothing. Both must agree that
    a silent pass is not a step.
    """
    engine = _run(class_name, BURST_IDLE_BURST)
    forwards = engine.forwards + engine.dummies
    assert engine.stats._throughput_steps == forwards
    assert engine.forwards == 4  # the four passes that had work


def test_the_dp_loop_reaches_its_idle_branch_with_a_dummy_already_counted():
    """The shape that makes a step-count guard on the idle path dead.

    `engines_running` is True out of `__init__`, so pass one runs a dummy and
    counts it; the branch that notices idleness is first reachable on pass two,
    by which time the counter is non-zero. Idle accounting must not depend on
    it.
    """
    engine = _run("DPEngineCoreProc", [(0.0, False), (1.0, False), (601.0, True)])
    assert engine.dummies == 1  # pass one, before the idle branch was reachable
    assert engine.stats._throughput_steps == 2  # that dummy, plus pass three
    assert engine.stats._throughput_last_log_time == pytest.approx(600.0)


@pytest.mark.parametrize("class_name", ["EngineCore", "DPEngineCoreProc"])
def test_time_mode_loops_leave_the_origin_alone(class_name):
    """The default path: every deployment that never sets the step interval."""
    engine = _run(class_name, BURST_IDLE_BURST, interval_steps=0)
    assert engine.stats._throughput_last_log_time == 0.0
