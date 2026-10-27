# SPDX-License-Identifier: MIT
"""Step-triggered engine-status windows.

Exists so offline / MLPerf runs can line one DP rank's lines up against
another's, which the clock cannot do. The load-bearing detail is that
`bump_step` counts the idle rank's DUMMY -- a counter that advanced only on
real work would look right under even load and drift under uneven load,
which is the case the option is for. Pinned by the last test here.
"""

from __future__ import annotations

import time

import pytest

from atom.model_engine import engine_stats
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


def test_idle_before_any_work_is_not_charged_to_the_first_window(monkeypatch, caplog):
    """The startup case: model load and the wait for request one.

    Both land inside window one, whose origin is stamped in `__init__`. Read
    off the emitted line, so the divisor is the one users see.
    """
    clock = [0.0]
    monkeypatch.setattr(engine_stats.time, "monotonic", lambda: clock[0])
    s = _stats(throughput_log_interval_steps=2)
    s._throughput_last_log_time = 0.0
    # The loop's first pass is already idle and lands on the origin stamped in
    # `__init__`; the rest of the stretch is the same one, re-noted each pass.
    for tick in (0.0, 30.0, 60.0):
        clock[0] = tick
        s.note_idle(tick)
        assert not s.window_expired(tick)
    s.note_busy(60.0)
    s.update_throughput(num_generation_tokens=10)
    s.bump_step()
    s.bump_step()
    clock[0] = 61.0
    with caplog.at_level("INFO"):
        s.maybe_log_throughput(num_running_reqs=1, num_waiting_reqs=0, kv_usage=0.5)
    assert "step 2: " in caplog.text
    # 10 tokens over the 1s a forward ran, not over the 61s since startup.
    assert "Avg generation throughput: 10.0 tokens/s" in caplog.text


def test_a_half_filled_window_keeps_its_work_time_and_drops_the_idle():
    """The case a reset-to-now cannot express, and the reason for `+=`.

    Burst A ran 2s, then 600s of nothing, then burst B ran 3s. The window
    closes on the step count with 5s of forwards in it. Resetting the origin
    at the end of the idle stretch would divide A's tokens by B's 3s and
    report a rate the engine never reached.
    """
    s = _stats(throughput_log_interval_steps=4)
    s._throughput_last_log_time = 0.0
    s.update_throughput(num_prompt_tokens=400)  # burst A, t=0..2
    s.bump_step()
    s.bump_step()
    s.note_idle(2.0)
    s.note_busy(602.0)
    assert s._throughput_last_log_time == 600.0  # origin moved by the idle span
    assert s._throughput_steps == 2  # A's forwards survive
    assert s.num_prompt_tokens == 400
    s.update_throughput(num_generation_tokens=600)  # burst B, t=602..605
    s.bump_step()
    s.bump_step()
    assert s.window_expired(605.0)
    assert (s.num_prompt_tokens + s.num_generation_tokens) / (
        605.0 - s._throughput_last_log_time
    ) == 200.0  # 1000 tokens over the 5s of forwards


def test_a_window_closing_while_idle_re_anchors_the_stretch(monkeypatch):
    """A window can close after the idle stretch opened, and does.

    `note_idle` takes the loop's `now`; the close a few lines later inside
    `maybe_log_throughput` reads the clock again and stamps the later value as
    the new origin. An anchor left at the earlier one is then subtracted from
    a window that already skipped it, putting the origin past `now` and
    `elapsed` below zero. The real gap is one clock read, so it costs a
    skipped line rather than a hang -- the invariant is still worth holding
    unconditionally, which is what the span below is sized to show.
    """
    clock = [100.0]
    monkeypatch.setattr(engine_stats.time, "monotonic", lambda: clock[0])
    s = _stats(throughput_log_interval_steps=2)
    s._throughput_last_log_time = 0.0
    s.bump_step()
    s.bump_step()
    s.note_idle(5.0)
    # Nothing ran and nothing is queued: the empty-window path, not a line.
    s.maybe_log_throughput(num_running_reqs=0, num_waiting_reqs=0, kv_usage=None)
    assert s._throughput_last_log_time == 100.0
    s.note_busy(605.0)
    # 605 - 100, not 605 - 5: 0..100 is behind the origin already.
    assert s._throughput_last_log_time == 605.0


def test_an_anchor_never_lands_behind_the_window_origin(monkeypatch):
    """The loop reads `now` before the heartbeat that may close the window.

    `_open_throughput_window` then stamps a LATER clock read as the origin, so
    the `now` the loop still carries is already behind it. Subtracting from
    there removes a span the window never contained: the origin ends up past
    `now` and the window reports a rate higher than the engine reached, or an
    `elapsed` below zero that skips the line without resetting the window.
    """
    clock = [0.006]  # emitting the line cost 2ms after the loop read 0.004
    monkeypatch.setattr(engine_stats.time, "monotonic", lambda: clock[0])
    s = _stats(throughput_log_interval_steps=4)
    s._throughput_last_log_time = 0.0
    for _ in range(4):
        s.update_throughput(num_generation_tokens=10)
        s.bump_step()
    s.maybe_log_throughput(num_running_reqs=1, num_waiting_reqs=0, kv_usage=0.5)
    assert s._throughput_last_log_time == 0.006
    s.note_idle(0.004)  # the loop's `now`, from before the close
    s.note_busy(180.005)
    assert s._throughput_last_log_time <= 180.005, "origin moved past now"
    # 4 x 1ms of forwards, not the 2ms left after over-subtracting.
    assert 180.009 - s._throughput_last_log_time == pytest.approx(0.004)


def test_note_idle_keeps_the_first_anchor():
    """Called on every idle pass of the loop, not once per stretch."""
    s = _stats(throughput_log_interval_steps=4)
    s._throughput_last_log_time = 0.0
    for tick in (10.0, 20.0, 30.0):
        s.note_idle(tick)
    s.note_busy(40.0)
    assert s._throughput_last_log_time == 30.0  # idle ran 10..40, not 30..40


@pytest.mark.parametrize(
    "options",
    [{"throughput_log_interval_steps": 0}, {"enable_log_stats": False}],
    ids=["time_mode", "disabled"],
)
def test_idle_accounting_is_inert_outside_step_mode(options):
    """Time mode's window is already bounded by its interval.

    Shifting its origin would redefine the default status line's throughput
    from per-interval to per-active-second for every existing deployment.
    """
    s = _stats(**options)
    start = s._throughput_last_log_time
    s.note_idle(start + 60.0)
    s.note_busy(start + 120.0)
    assert s._throughput_last_log_time == start


def test_idle_moves_the_divisor_not_the_trigger():
    """One rank idles mid-window, the other does not; both still close together.

    Idle accounting touches `elapsed` only. Wiring it into `window_expired`
    would desynchronize the ranks -- the thing the step counter exists for.
    """
    busy = _stats(throughput_log_interval_steps=4)
    idle = _stats(throughput_log_interval_steps=4)
    for s in (busy, idle):
        s._throughput_last_log_time = 0.0
    idle.note_idle(1.0)
    idle.note_busy(601.0)
    for pass_no in range(1, 5):
        busy.bump_step()
        idle.bump_step()
        assert busy.window_expired(601.0) == idle.window_expired(
            601.0
        ), f"ranks disagreed on pass {pass_no}"
    assert busy._throughput_last_log_time != idle._throughput_last_log_time


def test_the_printed_step_advances_across_lines(caplog):
    """The prefix is the whole point, and it must not be the interval.

    `_throughput_steps` resets every close and `window_expired` fires exactly
    at the interval, so printing it prints the same number on every line --
    which lines up nothing. The cumulative count is what two ranks share.
    """
    s = _stats(throughput_log_interval_steps=4)
    with caplog.at_level("INFO"):
        for _ in range(12):
            if s.window_expired(0.0):
                s.maybe_log_throughput(
                    num_running_reqs=2, num_waiting_reqs=0, kv_usage=0.5
                )
            s.update_throughput(num_generation_tokens=20)
            s.bump_step()
    printed = [
        line.split("step ")[1].split(":")[0]
        for line in caplog.text.splitlines()
        if "step " in line  # any other INFO line would IndexError here
    ]
    assert printed == ["4", "8"], printed


def test_a_rank_that_only_ran_dummies_still_emits_its_line(caplog):
    """DP, unbalanced load -- the case the whole option exists for.

    This rank has no tokens and no requests: it spent the window on dummies so
    its busy peers could run. Time mode suppresses such a window because the
    clock would repeat it forever while idle. In step mode the window closed,
    so `interval` forwards did run, and suppressing it removes the line for
    exactly the rank the other one is being compared against.
    """
    s = _stats(throughput_log_interval_steps=4)
    for _ in range(4):
        s.bump_step()
    with caplog.at_level("INFO"):
        s.maybe_log_throughput(num_running_reqs=0, num_waiting_reqs=0, kv_usage=0.0)
    assert "step 4: " in caplog.text
    assert "Avg generation throughput: 0.0 tokens/s" in caplog.text
    assert "Running: 0 reqs" in caplog.text
    assert s._throughput_steps == 0  # and the window still closed


def test_time_mode_still_suppresses_an_empty_window(caplog):
    """The default path: no line per interval while the engine sits idle."""
    s = _stats(throughput_log_interval_s=10.0)
    # The branch under test sits behind `window_expired`. Without this the
    # call returns at that guard and the assertion passes for the wrong
    # reason -- it stays green with the suppression removed entirely.
    s._throughput_last_log_time -= 20.0
    assert s.window_expired(time.monotonic())
    with caplog.at_level("INFO"):
        s.maybe_log_throughput(num_running_reqs=0, num_waiting_reqs=0, kv_usage=0.0)
    assert caplog.text == ""


def test_step_mode_opens_on_the_count():
    s = _stats(throughput_log_interval_s=10.0, throughput_log_interval_steps=4)
    now = s._throughput_last_log_time
    for bumps, expected in enumerate((False, False, False, True)):
        assert s.window_expired(now) is False, f"opened after {bumps} forwards"
        s.bump_step()
        assert s.window_expired(now) is expected


def test_emitting_resets_the_count(caplog):
    s = _stats(throughput_log_interval_s=10.0, throughput_log_interval_steps=2)
    s.update_throughput(num_generation_tokens=8)
    s.bump_step()
    s.bump_step()
    with caplog.at_level("INFO"):
        s.maybe_log_throughput(num_running_reqs=1, num_waiting_reqs=0, kv_usage=0.5)
    assert "step 2:" in caplog.text, caplog.text
    assert s._throughput_steps == 0  # the window's count, reset
    assert s._throughput_total_steps == 2  # the printed one, never reset
    assert not s.window_expired(s._throughput_last_log_time + 3600.0)


def test_a_window_with_no_tokens_still_disarms():
    """Whichever way it ends -- emitted in step mode, suppressed in time mode.

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
