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

WORK, STALLED, IDLE, OFFLOAD = "work", "stalled", "idle", "offload"


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

    `schedule` is one `(now, state)` per pass; the loop exits when it runs out.
    The three states are separate on purpose -- collapsing them onto one flag
    is what hides the passes where a forward does not run:

    - `WORK`     requests exist and a batch is scheduled: a forward runs.
    - `STALLED`  requests exist but none can be scheduled, so `is_finished()`
                 is False while `_process_engine_step()` returns False. Real:
                 a deferred offload save, or decode spinning for PrefillDone.
    - `IDLE`     nothing to run. Engine-wide, which `note_idle` requires.
    - `OFFLOAD`  RL rollout parked the weights; both loops skip the pass.
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
            is_finished=lambda: self._state is IDLE,
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
        self._now, self._state = self._schedule[self._pass]

    # -- collaborators the loop bodies call ---------------------------------

    def now(self):
        return self._now

    def pull_and_process_input_queue(self):
        return False

    def _is_idle_rl_weights_offloaded(self):
        # The real one sleeps 10ms and returns True; the base loop reads it as
        # "skip this pass", which is all the schedule needs to express.
        return self._state is OFFLOAD

    def _sync_dp_state(self, local_unfinished, shutdown, offloaded):
        return self._state is WORK, False, self._state is OFFLOAD

    def _process_engine_step(self):
        if self._state is not WORK:
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
    (0.0, WORK),
    (1.0, WORK),
    (2.0, IDLE),  # work has drained
    (3.0, IDLE),
    (603.0, WORK),
    (604.0, WORK),
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
    engine = _run("DPEngineCoreProc", [(0.0, IDLE), (1.0, IDLE), (601.0, WORK)])
    assert engine.dummies == 1  # pass one, before the idle branch was reachable
    assert engine.stats._throughput_steps == 2  # that dummy, plus pass three
    assert engine.stats._throughput_last_log_time == pytest.approx(600.0)


def test_requests_that_cannot_be_scheduled_are_not_work():
    """`is_finished()` is False and no forward runs: the base loop's own case.

    A deferred offload save keeps `deferred_free_blocks` non-empty while
    `_schedule` has nothing to return, and `DecodeEngineCore` spins the same
    way waiting for PrefillDone. Keying the window on "has requests" charges
    those seconds to tokens no forward produced.
    """
    engine = _run(
        "EngineCore",
        [(0.0, WORK), (1.0, STALLED), (301.0, STALLED), (601.0, WORK), (602.0, WORK)],
    )
    assert engine.forwards == 3
    assert engine.stats._throughput_steps == 3  # the stalled passes are not steps
    assert engine.stats._throughput_last_log_time == pytest.approx(600.0)


@pytest.mark.parametrize(
    "class_name, origin",
    # Base skips the pass via `_is_idle_rl_weights_offloaded`, DP via the
    # `global_offloaded` branch; both leave the window open across it.
    [("EngineCore", 600.0), ("DPEngineCoreProc", 600.0)],
)
def test_parked_weights_are_not_charged_to_the_window(class_name, origin):
    """An RL rollout offloads the weights for minutes between phases."""
    engine = _run(
        class_name,
        [(0.0, WORK), (1.0, OFFLOAD), (301.0, OFFLOAD), (601.0, WORK), (602.0, WORK)],
    )
    assert engine.forwards == 3
    assert engine.stats._throughput_last_log_time == pytest.approx(origin)


@pytest.mark.parametrize("class_name", ["EngineCore", "DPEngineCoreProc"])
def test_time_mode_loops_leave_the_origin_alone(class_name):
    """The default path: every deployment that never sets the step interval."""
    engine = _run(class_name, BURST_IDLE_BURST, interval_steps=0)
    assert engine.stats._throughput_last_log_time == 0.0
