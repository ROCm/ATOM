"""Unit tests for PrefillDelayer — the prefill coalescer.

The delayer's only cross-process dependency is a single
``torch.distributed.all_reduce(SUM)`` over its internal buffer. We patch that so
a single test process behaves as one DP rank whose local values ARE the global
sum (dp_size=1), and inject extra ranks explicitly where alignment matters.

No GPU / real distributed group needed.
"""

from unittest import mock

import pytest

from atom.model_engine.prefill_delayer import PrefillDelayer

MAX_BATCHED = 16384


def _noop_all_reduce(tensor, op=None, group=None):
    # Single rank: SUM(x) == x, so leaving the buffer untouched is correct.
    return tensor


def _add_rank(slots):
    """all_reduce mock that adds one more rank's contribution to the buffer.

    `slots` maps buffer index → that sibling rank's value (indices per
    PrefillDelayer._reduce_buf: 0=prefillable, 1=pending, 2=running_decode).
    """

    def f(tensor, op=None, group=None):
        for i, v in slots.items():
            tensor[i] += v
        return tensor

    return f


def make_delayer(**kw):
    defaults = dict(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=MAX_BATCHED,
        target_fill=0.7,
        ttft_max_ticks=30,
        partial_max_ticks=8,
        stall_ticks=3,
        kv_high_watermark=0.9,
        token_usage_low_watermark=None,
    )
    defaults.update(kw)
    return PrefillDelayer(**defaults)


def call(
    d,
    *,
    prefillable=True,
    pending_tokens=0,
    running_decode_batch=64,
    kv_usage=0.5,
    has_partial=False,
    oldest_waiting_age_ms=0.0,
    reduce=_noop_all_reduce,
):
    with mock.patch("torch.distributed.all_reduce", reduce):
        return d.should_allow_prefill(
            prefillable=prefillable,
            pending_tokens=pending_tokens,
            running_decode_batch=running_decode_batch,
            kv_usage=kv_usage,
            has_partial=has_partial,
            oldest_waiting_age_ms=oldest_waiting_age_ms,
        )


def skip_first(d):
    # The first call always fires (seed initial decode batch); tests want steady.
    assert call(d, pending_tokens=0) is True


class TestSkipFirst:
    def test_first_call_fires(self):
        d = make_delayer()
        assert call(d, pending_tokens=0) is True  # one-time warmup seed
        assert d._first is False


class TestFillTarget:
    def test_holds_below_target(self):
        d = make_delayer()
        skip_first(d)
        # threshold = 0.7 * 16384 = 11468.8; 1000 < that, decode running → HOLD
        assert call(d, pending_tokens=1000) is False
        assert d._stat_hold == 1

    def test_fires_at_target(self):
        d = make_delayer()
        skip_first(d)
        assert call(d, pending_tokens=12000) is True  # >= 0.7 * budget
        assert d._stat_fire_fill == 1

    def test_fires_exactly_at_threshold(self):
        d = make_delayer()
        skip_first(d)
        assert call(d, pending_tokens=11469) is True  # ceil(0.7*16384)


class TestMustFireBounds:
    def test_no_decode_fires(self):
        d = make_delayer()
        skip_first(d)
        assert call(d, pending_tokens=1000, running_decode_batch=0) is True
        assert d._stat_fire_nodecode == 1

    def test_kv_high_fires(self):
        d = make_delayer()
        skip_first(d)
        assert call(d, pending_tokens=1000, kv_usage=0.95) is True
        assert d._stat_fire_kv == 1

    def test_kv_low_fires_when_watermark_set(self):
        d = make_delayer(token_usage_low_watermark=0.2)
        skip_first(d)
        assert call(d, pending_tokens=1000, kv_usage=0.1) is True
        assert d._stat_fire_kv == 1

    def test_kv_low_disabled_by_default(self):
        d = make_delayer()  # low watermark None
        skip_first(d)
        assert call(d, pending_tokens=1000, kv_usage=0.001) is False  # no kv fire

    def test_ttft_bound_fires(self):
        # Growing (no stall) but always below target → held until TTFT bound.
        d = make_delayer(ttft_max_ticks=3, stall_ticks=1000)
        skip_first(d)
        assert call(d, pending_tokens=1000) is False  # hold_ticks 1
        assert call(d, pending_tokens=2000) is False  # hold_ticks 2
        assert call(d, pending_tokens=3000) is False  # hold_ticks 3
        assert call(d, pending_tokens=4000) is True  # hold_ticks>=3 → ttft
        assert d._stat_fire_ttft == 1


class TestQueueAgeGuard:
    def test_disabled_by_default(self):
        # max_queue_ms=None → guard inactive even with an ancient waiting req.
        d = make_delayer()  # no max_queue_ms
        skip_first(d)
        assert call(d, pending_tokens=1000, oldest_waiting_age_ms=1e9) is False

    def test_fires_when_queue_age_exceeds_threshold(self):
        d = make_delayer(max_queue_ms=5000, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        # below target, below age → hold
        assert call(d, pending_tokens=1000, oldest_waiting_age_ms=1000) is False
        # aged past 5000ms → force release regardless of fill
        assert call(d, pending_tokens=1000, oldest_waiting_age_ms=6000) is True
        assert d._stat_fire_queue_ms == 1

    def test_age_guard_beats_alignment_gate(self):
        # Even when mixed (n_prefillable<dp_size), an over-age request releases.
        d = make_delayer(dp_size=2, max_queue_ms=5000, ttft_max_ticks=1000)
        skip_first(d)
        mixed = _add_rank({})  # sibling not prefillable → mixed
        assert (
            call(d, pending_tokens=1000, oldest_waiting_age_ms=6000, reduce=mixed)
            is True
        )
        assert d._stat_fire_queue_ms == 1

    def test_age_guard_gated_on_prefillable(self):
        # A non-prefillable rank (can't admit) does not trip the guard itself;
        # with dp_size=1 that means no release from age.
        d = make_delayer(max_queue_ms=5000, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        # not prefillable → vacuous allow (n_prefillable==0), not a queue_ms fire
        assert (
            call(d, prefillable=False, pending_tokens=0, oldest_waiting_age_ms=9000)
            is True
        )
        assert d._stat_fire_queue_ms == 0


class TestStall:
    def test_stall_fires_when_queue_not_growing(self):
        d = make_delayer(stall_ticks=3, ttft_max_ticks=1000)
        skip_first(d)
        assert call(d, pending_tokens=1000) is False  # prev set
        assert call(d, pending_tokens=1000) is False  # stall 1
        assert call(d, pending_tokens=1000) is False  # stall 2
        assert call(d, pending_tokens=1000) is True  # stall 3 → fire
        assert d._stat_fire_stall == 1

    def test_growth_resets_stall(self):
        d = make_delayer(stall_ticks=2, ttft_max_ticks=1000)
        skip_first(d)
        assert call(d, pending_tokens=1000) is False  # prev set
        assert call(d, pending_tokens=1000) is False  # stall 1
        assert call(d, pending_tokens=2000) is False  # grew → stall reset to 0
        assert call(d, pending_tokens=2000) is False  # stall 1
        assert call(d, pending_tokens=2000) is True  # stall 2 → fire


class TestPartial:
    def test_small_partial_does_not_fire_immediately(self):
        # A small partial (below fill target) must be held, not fired at once.
        d = make_delayer(partial_max_ticks=3, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        assert call(d, pending_tokens=500, has_partial=True) is False
        assert call(d, pending_tokens=500, has_partial=True) is False

    def test_partial_fires_after_partial_deadline(self):
        d = make_delayer(partial_max_ticks=2, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        assert call(d, pending_tokens=500, has_partial=True) is False  # hold 1
        assert call(d, pending_tokens=500, has_partial=True) is False  # hold 2
        assert call(d, pending_tokens=500, has_partial=True) is True  # >=2 → fire
        assert d._stat_fire_partial == 1

    def test_partial_deadline_tighter_than_ttft(self):
        # Same held duration fires via partial (2) well before ttft (1000).
        d = make_delayer(partial_max_ticks=2, ttft_max_ticks=1000, stall_ticks=1000)
        skip_first(d)
        call(d, pending_tokens=500, has_partial=True)
        call(d, pending_tokens=500, has_partial=True)
        assert call(d, pending_tokens=500, has_partial=True) is True
        assert d._stat_fire_ttft == 0


class TestVacuous:
    def test_no_prefillable_rank_allows(self):
        d = make_delayer()
        skip_first(d)
        assert call(d, prefillable=False, pending_tokens=0) is True
        assert d._stat_fire_vacuous == 1  # skip_first is not counted


class TestAlignmentGate:
    def test_mixed_holds_even_when_full(self):
        # dp_size=2, sibling rank NOT prefillable → n_prefillable(1) < 2 → HOLD,
        # even though local pending alone would exceed the fill target.
        d = make_delayer(dp_size=2, ttft_max_ticks=1000)
        skip_first(d)
        mixed = _add_rank({})  # sibling contributes nothing (prefillable=0)
        assert call(d, pending_tokens=MAX_BATCHED, reduce=mixed) is False
        assert d._stat_hold == 1

    def test_aligned_fires_when_full(self):
        # dp_size=2, sibling prefillable + full → aligned, aggregate fill high.
        d = make_delayer(dp_size=2, ttft_max_ticks=1000)
        skip_first(d)
        aligned = _add_rank({0: 1, 1: MAX_BATCHED, 2: 64})
        assert call(d, pending_tokens=MAX_BATCHED, reduce=aligned) is True
        assert d._stat_fire_fill == 1

    def test_mixed_released_by_ttft(self):
        d = make_delayer(dp_size=2, ttft_max_ticks=2, stall_ticks=1000)
        skip_first(d)
        mixed = _add_rank({})
        assert call(d, pending_tokens=MAX_BATCHED, reduce=mixed) is False  # hold 1
        assert call(d, pending_tokens=MAX_BATCHED, reduce=mixed) is False  # hold 2
        assert call(d, pending_tokens=MAX_BATCHED, reduce=mixed) is True  # ttft
        assert d._stat_fire_ttft == 1


class TestParamClamps:
    def test_target_fill_above_one_clamped(self):
        d = make_delayer(target_fill=1.5)
        assert d.target_fill == 1.0

    def test_target_fill_zero_clamped_positive(self):
        d = make_delayer(target_fill=0.0)
        assert 0.0 < d.target_fill <= 0.1

    def test_ticks_below_one_clamped(self):
        d = make_delayer(ttft_max_ticks=0, partial_max_ticks=-3, stall_ticks=0)
        assert d.ttft_max_ticks == 1
        assert d.partial_max_ticks == 1
        assert d.stall_ticks == 1


class TestStatsDistinct:
    def test_fill_and_stall_counted_separately(self):
        d = make_delayer(stall_ticks=2, ttft_max_ticks=1000)
        skip_first(d)
        # a stall fire
        call(d, pending_tokens=1000)
        call(d, pending_tokens=1000)
        assert call(d, pending_tokens=1000) is True  # stall
        # then a healthy fill fire
        assert call(d, pending_tokens=12000) is True
        assert d._stat_fire_stall == 1
        assert d._stat_fire_fill == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
