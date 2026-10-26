"""Unit tests for PrefillGroupDelayer -- the per-rank MIN barrier on
PrefillScheduler itself.

A different problem from PrefillDelayer (test_prefill_delayer.py), which gates
when a NEW request becomes visible to prefill. This gates whether prefill
fires a forward THIS TICK for sequences it already holds, releasing every
rank together only once each one individually clears `min_tokens` -- not the
cross-rank SUM/average `PrefillDelayer` uses.

Same no-GPU harness: patch `torch.distributed.all_reduce` so a single test
process behaves as one DP rank whose local values ARE the global sum
(dp_size=1), and inject extra ranks explicitly where alignment matters.
"""

from __future__ import annotations

import ast
import pathlib
from unittest import mock

from atom.model_engine.prefill_delayer import PrefillGroupDelayer

MIN_TOKENS = 16384

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEDULER = ROOT / "atom/model_engine/scheduler.py"
CORE = ROOT / "atom/model_engine/engine_core.py"


def _method(path: pathlib.Path, owner: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = [
        fn
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == owner
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == name
    ]
    assert len(found) == 1, f"{owner}.{name} is defined {len(found)} times"
    return found[0]


def _src(path: pathlib.Path, owner: str, name: str) -> str:
    out = ast.get_source_segment(path.read_text(), _method(path, owner, name))
    assert out is not None
    return out


def _noop_all_reduce(tensor, op=None, group=None):
    return tensor


def _add_rank(slots):
    """all_reduce mock that adds one more rank's contribution to the buffer.

    `slots` maps buffer index -> that sibling rank's value (indices per
    PrefillGroupDelayer._reduce_buf: 0=any_ready, 1=cleared, 2=pending_tokens).
    """

    def f(tensor, op=None, group=None):
        for i, v in slots.items():
            tensor[i] += v
        return tensor

    return f


def make_delayer(**kw):
    defaults = {
        "dp_size": 1,
        "cpu_group": None,
        "min_tokens": MIN_TOKENS,
        "ttft_max_ticks": 30,
        "stall_ticks": 3,
    }
    defaults.update(kw)
    # Cross-DP all_reduce only runs when cpu_group is not None; multi-rank
    # tests mock it to inject sibling contributions, so they need a non-None
    # group for that branch to execute. Real dp>1 always has a cpu_group.
    if defaults["dp_size"] > 1 and defaults["cpu_group"] is None:
        defaults["cpu_group"] = object()  # sentinel; all_reduce is mocked in call()
    return PrefillGroupDelayer(**defaults)


def call(d, *, any_ready=True, pending_tokens=0, reduce=_noop_all_reduce):
    with mock.patch("torch.distributed.all_reduce", reduce):
        return d.should_allow_prefill(any_ready, pending_tokens)


def skip_first(d):
    assert call(d, pending_tokens=0) is True


class TestSkipFirst:
    def test_first_call_fires(self):
        d = make_delayer()
        assert call(d, pending_tokens=0) is True
        assert d._first is False


class TestParamValidation:
    def test_min_tokens_must_be_positive(self):
        import pytest

        with pytest.raises(ValueError):
            make_delayer(min_tokens=0)
        with pytest.raises(ValueError):
            make_delayer(min_tokens=-1)

    def test_dp_gt_1_requires_cpu_group(self):
        import pytest

        with pytest.raises(ValueError):
            PrefillGroupDelayer(dp_size=2, cpu_group=None, min_tokens=MIN_TOKENS)


class TestSingleRankThreshold:
    def test_holds_below_min_tokens(self):
        d = make_delayer()
        skip_first(d)
        assert call(d, pending_tokens=MIN_TOKENS - 1) is False
        assert d._stat_hold == 1

    def test_fires_at_min_tokens(self):
        d = make_delayer()
        skip_first(d)
        assert call(d, pending_tokens=MIN_TOKENS) is True
        assert d._stat_fire_min_tokens == 1

    def test_fires_above_min_tokens(self):
        d = make_delayer()
        skip_first(d)
        assert call(d, pending_tokens=MIN_TOKENS * 3) is True
        assert d._stat_fire_min_tokens == 1

    def test_refires_only_after_reaccumulating(self):
        """The literal ask: fire, then don't fire again until the SAME
        condition -- every rank clearing min_tokens -- is met again."""
        d = make_delayer()
        skip_first(d)
        assert call(d, pending_tokens=MIN_TOKENS) is True  # fire #1
        # Pending drops back down (the fire drained `ready`) -- must hold
        # again even though it was just released.
        assert call(d, pending_tokens=100) is False
        assert call(d, pending_tokens=MIN_TOKENS - 1) is False
        assert call(d, pending_tokens=MIN_TOKENS) is True  # fire #2
        assert d._stat_fire_min_tokens == 2


class TestVacuous:
    def test_nothing_ready_fires(self):
        d = make_delayer()
        skip_first(d)
        assert call(d, any_ready=False, pending_tokens=0) is True
        assert d._stat_fire_vacuous == 1


class TestAlignmentGate:
    """Literally every one of dp_size ranks, not just the ones already ready
    -- same anti-skew idiom PrefillDelayer's `n_prefillable < dp_size` uses.
    A rank with nothing ready yet still blocks the release (it may be about
    to receive its BlockAssignment this very tick); the must-fire bounds, not
    this gate, are what rescue a rank that never clears min_tokens at all --
    see TestMustFireBounds."""

    def test_holds_while_a_ready_rank_is_still_short(self):
        d = make_delayer(dp_size=2, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        # Sibling is ready but short of the floor.
        short_sibling = _add_rank({0: 1, 1: 0, 2: 1000})
        assert call(d, pending_tokens=MIN_TOKENS, reduce=short_sibling) is False
        assert d._stat_hold == 1

    def test_fires_once_every_ready_rank_clears_the_floor(self):
        d = make_delayer(dp_size=2, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        cleared_sibling = _add_rank({0: 1, 1: 1, 2: MIN_TOKENS})
        assert call(d, pending_tokens=MIN_TOKENS, reduce=cleared_sibling) is True
        assert d._stat_fire_min_tokens == 1

    def test_holds_while_a_sibling_has_nothing_ready_yet(self):
        """A rank with NO ready work still blocks the release -- unlike the
        threshold check, the gate does not exclude an idle rank, because it
        might be one tick away from becoming ready and an uneven release
        would otherwise leave it out."""
        d = make_delayer(dp_size=2, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        idle_sibling = _add_rank({0: 0, 1: 0, 2: 0})
        assert call(d, pending_tokens=MIN_TOKENS, reduce=idle_sibling) is False
        assert d._stat_hold == 1

    def test_holds_while_some_rank_has_nothing_ready_and_others_do(self):
        """A MIXED group (one ready+cleared, one ready+short) still holds --
        i.e. the gate is not satisfied merely by n_any == dp_size without
        n_ready also reaching it."""
        d = make_delayer(dp_size=3, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        mixed = _add_rank({0: 2, 1: 1, 2: MIN_TOKENS + 500})
        # n_any = 1(self) + 2 = 3 == dp_size; n_ready = 1(self) + 1 = 2 < 3.
        assert call(d, pending_tokens=MIN_TOKENS, reduce=mixed) is False
        assert d._stat_hold == 1


class TestMustFireBounds:
    def test_ttft_bound_fires_even_when_unaligned(self):
        d = make_delayer(dp_size=2, ttft_max_ticks=3, stall_ticks=1000)
        skip_first(d)
        short_sibling = _add_rank({0: 1, 1: 0, 2: 1000})
        assert call(d, pending_tokens=MIN_TOKENS, reduce=short_sibling) is False
        assert call(d, pending_tokens=MIN_TOKENS, reduce=short_sibling) is False
        assert call(d, pending_tokens=MIN_TOKENS, reduce=short_sibling) is False
        assert call(d, pending_tokens=MIN_TOKENS, reduce=short_sibling) is True
        assert d._stat_fire_ttft == 1

    def test_stall_bound_fires_when_aggregate_stops_growing(self):
        d = make_delayer(dp_size=2, ttft_max_ticks=1000, stall_ticks=3)
        skip_first(d)
        short_sibling = _add_rank({0: 1, 1: 0, 2: 1000})
        assert call(d, pending_tokens=1000, reduce=short_sibling) is False  # prev set
        assert call(d, pending_tokens=1000, reduce=short_sibling) is False  # stall 1
        assert call(d, pending_tokens=1000, reduce=short_sibling) is False  # stall 2
        assert call(d, pending_tokens=1000, reduce=short_sibling) is True  # stall 3
        assert d._stat_fire_stall == 1

    def test_growth_resets_stall(self):
        d = make_delayer(dp_size=1, ttft_max_ticks=1000, stall_ticks=2)
        skip_first(d)
        assert call(d, pending_tokens=1000) is False  # prev set
        assert call(d, pending_tokens=1000) is False  # stall 1
        assert call(d, pending_tokens=2000) is False  # grew -> stall reset
        assert call(d, pending_tokens=2000) is False  # stall 1
        assert call(d, pending_tokens=2000) is True  # stall 2 -> fire
        assert d._stat_fire_stall == 1

    def test_must_fire_bounds_checked_before_alignment(self):
        """ttft/stall release even though the alignment gate alone would
        hold forever -- the give-up bounds are the rescue for a sibling that
        is ready but never clears the floor (n_any == dp_size stays true, so
        it is specifically the n_ready < n_any branch that would hold
        forever without this)."""
        d = make_delayer(dp_size=2, ttft_max_ticks=2, stall_ticks=1000)
        skip_first(d)
        never_clears = _add_rank({0: 1, 1: 0, 2: 1})
        assert call(d, pending_tokens=MIN_TOKENS, reduce=never_clears) is False  # 1
        assert call(d, pending_tokens=MIN_TOKENS, reduce=never_clears) is False  # 2
        assert call(d, pending_tokens=MIN_TOKENS, reduce=never_clears) is True  # >=2
        assert d._stat_fire_ttft == 1


class TestClampsAndLocal:
    def test_tick_bounds_below_one_are_clamped(self):
        d = make_delayer(ttft_max_ticks=0, stall_ticks=-5)
        assert d.ttft_max_ticks == 1
        assert d.stall_ticks == 1

    def test_is_local_when_dp_size_one_and_no_group(self):
        d = make_delayer(dp_size=1, cpu_group=None)
        assert d.is_local is True

    def test_not_local_when_dp_size_above_one(self):
        d = make_delayer(dp_size=2)
        assert d.is_local is False


# ── The one correctness-critical wiring invariant ─────────────────────────


class TestGateRunsBeforeHasRequests:
    """`should_fire_this_tick()` runs the all_reduce. `busy_loop`'s own
    `_sync_dp_state` guarantees `_process_engine_step` is reached on every
    prefill rank or none, each tick -- but ONLY this method call inherits
    that guarantee directly. `has_requests()` returns before `schedule()` is
    ever called on a rank with nothing locally ready, so a rank that reached
    this method must call it BEFORE that short-circuit, or an idle rank
    skips the collective a busy rank still makes, mismatching the call count
    and hanging the group. A future edit reordering these two lines breaks
    this silently -- no test exercises the real collective, so this is the
    one line standing between a refactor and a hang nobody can repro from a
    traceback.
    """

    def test_order_in_process_engine_step(self):
        src = _src(CORE, "PrefillEngineCore", "_process_engine_step")
        fire_gate = src.index("self.scheduler.should_fire_this_tick()")
        has_requests = src.index("self.scheduler.has_requests()")
        assert fire_gate < has_requests

    def test_schedule_scanned_predicate_matches_the_gates(self):
        """`_is_ready` is the one place "ready" is defined; `_schedule`'s own
        admission filter and `should_fire_this_tick`'s pending-token scan
        both have to call it rather than keep a second copy of the
        predicate."""
        schedule_src = _src(SCHEDULER, "PrefillScheduler", "_schedule")
        assert "self._is_ready(s)" in schedule_src
        gate_src = _src(SCHEDULER, "PrefillScheduler", "should_fire_this_tick")
        assert "self._is_ready(seq)" in gate_src

    def test_gate_defaults_to_true_with_no_delayer_attached(self):
        """An engine that never opts in sees zero behavior change."""
        import threading

        from atom.model_engine.scheduler import PrefillScheduler

        stub = PrefillScheduler.__new__(PrefillScheduler)
        stub.prefill_group_delayer = None
        stub.waiting = ()
        stub._pending_lock = threading.Lock()
        assert stub.should_fire_this_tick() is True


# ── Debug logging -- the live verification recipe reads these lines ──────


class TestDebugLogging:
    """`ATOM_PREFILL_GROUP_DELAYER_DEBUG=1` is what a live run checks: every
    FIRE/HOLD line carries n_ready/n_any so a trace shows not just whether a
    tick released, but which ranks were still short when it did or didn't.
    """

    @staticmethod
    def _logs(caplog, fn):
        import atom.model_engine.prefill_delayer as pd

        with (
            mock.patch.object(pd, "_DEBUG_GROUP", True),
            caplog.at_level("INFO", logger="atom.model_engine.prefill_delayer"),
        ):
            result = fn()
        return result, [r.message for r in caplog.records]

    def test_hold_logs_n_ready_over_n_any(self, caplog):
        d = make_delayer(dp_size=2, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        short_sibling = _add_rank({0: 1, 1: 0, 2: 1000})
        result, lines = self._logs(
            caplog, lambda: call(d, pending_tokens=MIN_TOKENS, reduce=short_sibling)
        )
        assert result is False
        (line,) = [m for m in lines if "HOLD" in m]
        assert "[PrefillGroupDelayer] HOLD" in line
        assert "n_ready=1/2" in line
        assert "dp_size=2" in line
        assert f"min_tokens={MIN_TOKENS}" in line

    def test_fire_logs_the_reason(self, caplog):
        d = make_delayer(dp_size=2, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        cleared_sibling = _add_rank({0: 1, 1: 1, 2: MIN_TOKENS})
        result, lines = self._logs(
            caplog, lambda: call(d, pending_tokens=MIN_TOKENS, reduce=cleared_sibling)
        )
        assert result is True
        (line,) = [m for m in lines if "FIRE" in m]
        assert "[PrefillGroupDelayer] FIRE (min_tokens)" in line
        assert "n_ready=2/2" in line

    def test_silent_by_default(self, caplog):
        """No env var, no mock.patch -- _DEBUG_GROUP reads False at import
        time, same as PrefillDelayer's own _DEBUG, so a run that never opts
        in produces none of this per-tick noise."""
        d = make_delayer()
        with caplog.at_level("INFO", logger="atom.model_engine.prefill_delayer"):
            skip_first(d)
            call(d, pending_tokens=1)
        assert not any(
            "FIRE" in r.message or "HOLD" in r.message for r in caplog.records
        )
