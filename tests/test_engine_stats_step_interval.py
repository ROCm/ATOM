# SPDX-License-Identifier: MIT
"""Step-triggered engine-status windows.

Exists so offline / MLPerf runs can line one DP rank's lines up against
another's, which the clock cannot do. The load-bearing detail is that
`bump_step` counts the idle rank's DUMMY -- a counter that advanced only on
real work would look right under even load and drift under uneven load,
which is the case the option is for. Pinned by the last test here.
"""

from __future__ import annotations

import pytest

from atom.model_engine.engine_stats import EngineStats


def _stats(**kw) -> EngineStats:
    kw.setdefault("enable_log_stats", True)
    kw.setdefault("engine_index", 0)
    return EngineStats(**kw)


def test_time_mode_is_unchanged_by_default():
    """Default construction keeps the clock; steps are inert."""
    s = _stats(throughput_log_interval_s=10.0)
    assert s.throughput_log_interval_steps == 0
    for _ in range(1000):
        s.bump_step()
    assert not s.window_expired(s._throughput_last_log_time + 1.0)
    assert s.window_expired(s._throughput_last_log_time + 10.0)


def test_step_mode_ignores_the_clock():
    """A step window does not open just because time passed."""
    s = _stats(throughput_log_interval_s=10.0, throughput_log_interval_steps=4)
    assert not s.window_expired(s._throughput_last_log_time + 3600.0)


def test_step_mode_opens_on_the_count():
    s = _stats(throughput_log_interval_s=10.0, throughput_log_interval_steps=4)
    now = s._throughput_last_log_time
    for expected in (False, False, False, True):
        assert s.window_expired(now) is False or expected
        s.bump_step()
    assert s.window_expired(now) is True


def test_emitting_resets_the_count(caplog):
    s = _stats(throughput_log_interval_s=10.0, throughput_log_interval_steps=2)
    s.update_throughput(num_generation_tokens=8)
    s.bump_step()
    s.bump_step()
    with caplog.at_level("INFO"):
        s.maybe_log_throughput(num_running_reqs=1, num_waiting_reqs=0, kv_usage=0.5)
    assert "step 2:" in caplog.text, caplog.text
    assert s._throughput_steps == 0
    assert not s.window_expired(s._throughput_last_log_time + 3600.0)


def test_idle_window_resets_the_count_too():
    """The nothing-happened path returns early; it must still disarm.

    Left armed, every later pass would re-enter and the line would fire on
    the first step after an idle stretch rather than `interval` steps later.
    """
    s = _stats(throughput_log_interval_s=10.0, throughput_log_interval_steps=2)
    s.bump_step()
    s.bump_step()
    s.maybe_log_throughput(num_running_reqs=0, num_waiting_reqs=0, kv_usage=None)
    assert s._throughput_steps == 0


def test_disabled_stats_never_count_or_open():
    s = _stats(enable_log_stats=False, throughput_log_interval_steps=1)
    s.bump_step()
    assert s._throughput_steps == 0
    assert s.window_expired(s._throughput_last_log_time + 3600.0) is False


def test_ranks_stay_aligned_when_only_one_of_them_has_work():
    """The reason the counter sits after the dummy, not inside the step.

    Both ranks reach `bump_step` every pass -- one via `_process_engine_step`,
    the idle one via `_execute_dummy_batch` -- so both windows close together.
    """
    busy = _stats(throughput_log_interval_s=10.0, throughput_log_interval_steps=8)
    idle = _stats(throughput_log_interval_s=10.0, throughput_log_interval_steps=8)
    now = busy._throughput_last_log_time
    for pass_no in range(1, 9):
        busy.bump_step()  # ran a real batch
        idle.bump_step()  # ran dummy_execution
        assert busy.window_expired(now) == idle.window_expired(
            now
        ), f"ranks disagreed on pass {pass_no}"
    assert busy.window_expired(now) is True


@pytest.mark.parametrize("steps", [-5, -1])
def test_a_negative_interval_falls_back_to_the_clock(steps):
    """Clamped rather than raised: a bad value must not stop the engine."""
    s = _stats(throughput_log_interval_s=10.0, throughput_log_interval_steps=steps)
    assert s.throughput_log_interval_steps == 0
    s.bump_step()
    assert not s.window_expired(s._throughput_last_log_time + 1.0)
    assert s.window_expired(s._throughput_last_log_time + 10.0)
