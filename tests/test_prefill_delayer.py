"""Unit tests for PrefillDelayer — the prefill coalescer.

The delayer's only cross-process dependency is a single
``torch.distributed.all_reduce(SUM)`` over its internal buffer. We patch that so
a single test process behaves as one DP rank whose local values ARE the global
sum (dp_size=1), and inject extra ranks explicitly where alignment matters.

No GPU / real distributed group needed.
"""

from unittest import mock

import pytest

from atom.model_engine.prefill_delayer import (
    _FIXED_SLOTS,
    PrefillDelayer,
    _Gate,
)

MAX_BATCHED = 16384
# reduce-buffer slot of the requests that arrived since the previous tick
ARRIVED = 8
# reduce-buffer slot of the ranks with waiting work they cannot admit
BLOCKED = 9
# reduce-buffer slot of the ranks without prefill that cannot align in time
UNREACHABLE = 10


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
    defaults = {
        "dp_size": 1,
        "cpu_group": None,
        "max_num_batched_tokens": MAX_BATCHED,
        "target_fill": 0.7,
        "ttft_max_ticks": 30,
        "partial_max_ticks": 8,
        "stall_ticks": 3,
        "kv_high_watermark": 0.9,
        "token_usage_low_watermark": None,
        "prefill_decode_interval": 0,
    }
    defaults.update(kw)
    # The cross-DP all_reduce only runs when cpu_group is not None (dp=1/TP-only
    # skips it — there is no sibling to align with). Multi-rank tests mock
    # all_reduce to inject sibling contributions, so they need a non-None group
    # for that branch to execute. Real dp>1 always has a cpu_group; mirror that.
    if defaults["dp_size"] > 1 and defaults["cpu_group"] is None:
        defaults["cpu_group"] = object()  # sentinel; all_reduce is mocked in call()
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
    **kw,
):
    with mock.patch("torch.distributed.all_reduce", reduce):
        return d.should_allow_prefill(
            prefillable=prefillable,
            pending_tokens=pending_tokens,
            running_decode_batch=running_decode_batch,
            kv_usage=kv_usage,
            has_partial=has_partial,
            oldest_waiting_age_ms=oldest_waiting_age_ms,
            **kw,
        ).allow


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

    def test_mixed_flat_queue_released_by_stall(self):
        # Nothing changes anywhere in the group: no arrival, no rank turning
        # prefillable, and the peer completes no decode in time. Waiting
        # longer cannot align it, so stall releases.
        d = make_delayer(dp_size=2, ttft_max_ticks=1000, stall_ticks=2)
        skip_first(d)
        mixed = _add_rank({2: 64, UNREACHABLE: 1})
        assert call(d, pending_tokens=500, reduce=mixed) is False  # first look
        assert call(d, pending_tokens=500, reduce=mixed) is False  # stall 1
        assert call(d, pending_tokens=500, reduce=mixed) is True  # stall 2
        assert d._stat_fire_stall == 1

    def test_low_concurrency_idle_ranks_release_on_stall(self):
        # DP8 at concurrency 4: three ranks decode, this rank just got a
        # prompt, four ranks are idle and will get no work. Default bounds.
        d = make_delayer(dp_size=8, ttft_max_ticks=200, stall_ticks=10)
        skip_first(d)
        mixed = _add_rank({2: 3, UNREACHABLE: 4})
        ticks = 0
        while not call(d, pending_tokens=800, running_decode_batch=0, reduce=mixed):
            ticks += 1
            assert ticks < 200
        assert ticks == 10
        assert d._stat_fire_stall == 1
        assert d._stat_fire_ttft == 0

    def test_arrivals_restart_the_stall_count(self):
        # Requests keep arriving in the group while the capped pending sum
        # stays flat: peers are still receiving work, so keep waiting.
        d = make_delayer(dp_size=2, ttft_max_ticks=6, stall_ticks=2)
        skip_first(d)
        arriving = _add_rank({2: 64, ARRIVED: 1})
        for _ in range(6):
            assert call(d, pending_tokens=500, reduce=arriving) is False
        assert call(d, pending_tokens=500, reduce=arriving) is True
        assert d._stat_fire_ttft == 1
        assert d._stat_fire_stall == 0

    def test_unaligned_ticks_do_not_stall_an_aligned_group(self):
        d = make_delayer(dp_size=2, ttft_max_ticks=1000, stall_ticks=3)
        skip_first(d)
        mixed = _add_rank({2: 64})
        assert call(d, pending_tokens=500, reduce=mixed) is False  # first look
        assert call(d, pending_tokens=500, reduce=mixed) is False  # stall 1
        assert call(d, pending_tokens=500, reduce=mixed) is False  # stall 2
        # The peer turns prefillable with a request estimated at 0 tokens:
        # the flat, underfilled aligned group restarts the count.
        aligned = _add_rank({0: 1, 2: 64})
        assert call(d, pending_tokens=500, reduce=aligned) is False
        assert d._stat_fire_stall == 0

    def test_kv_full_blocked_rank_does_not_release_every_tick(self):
        # DP8: this rank's head cannot allocate KV (prefillable=False, KV 0.95,
        # 4 waiting), one peer holds 300 tokens, the group decodes. The blocked
        # rank counts as aligned, so only the stall bound releases the
        # underfilled forward, never KV pressure on every tick.
        d = make_delayer(dp_size=8, ttft_max_ticks=200, stall_ticks=10)
        skip_first(d)
        peer = _add_rank({0: 1, 1: 300, 2: 7 * 16, UNREACHABLE: 6})
        fired = [
            call(d, prefillable=False, kv_usage=0.95, waiting_requests=4, reduce=peer)
            for _ in range(100)
        ]
        assert sum(fired) == 9
        assert d._stat_fire_kv == 0
        assert d._stat_fire_stall == 9

    def test_blocked_rank_does_not_hold_full_peers(self):
        # DP8: this rank is blocked below the watermark, the seven peers each
        # hold a full budget, and requests keep arriving on a peer. Waiting
        # cannot make this rank admittable, so the full peers fire at once.
        d = make_delayer(dp_size=8, ttft_max_ticks=200, stall_ticks=10)
        skip_first(d)
        peers = _add_rank({0: 7, 1: 7 * MAX_BATCHED, 2: 7 * 16, ARRIVED: 1})
        assert (
            call(d, prefillable=False, kv_usage=0.85, waiting_requests=4, reduce=peers)
            is True
        )
        assert d._stat_fire_fill == 1

    def test_kv_high_ignores_a_rank_that_cannot_prefill(self):
        d = make_delayer()
        skip_first(d)
        call(d, prefillable=False, kv_usage=0.95, waiting_requests=1)
        assert d._reduce_buf[3] == 0
        assert d._reduce_buf[BLOCKED] == 1

    def test_quiet_group_waits_for_a_straggler_completing_in_time(self):
        # c128-like: seven ranks hold prefill, the eighth decodes and frees a
        # slot within the TTFT bound. Nothing changes for many more ticks than
        # stall_ticks, yet the group waits and fires aligned.
        d = make_delayer(dp_size=8, ttft_max_ticks=200, stall_ticks=10)
        skip_first(d)
        mixed = _add_rank({0: 6, 1: 6 * MAX_BATCHED, 2: 7 * 16})
        for _ in range(50):
            assert call(d, pending_tokens=MAX_BATCHED, reduce=mixed) is False
        aligned = _add_rank({0: 7, 1: 6 * MAX_BATCHED, 2: 7 * 16, ARRIVED: 1})
        assert call(d, pending_tokens=MAX_BATCHED, reduce=aligned) is True
        assert d._stat_fire_fill == 1
        assert d._stat_fire_stall == 0

    def test_unreachable_flag_compares_to_the_remaining_hold(self):
        d = make_delayer(dp_size=2, ttft_max_ticks=200)
        skip_first(d)
        prefilling_peer = _add_rank({0: 1, 1: 500, 2: 64})

        def unreachable(**kw):
            call(d, prefillable=False, reduce=prefilling_peer, **kw)
            return d._reduce_buf[UNREACHABLE]

        assert unreachable(next_completion_steps=150) == 0
        # one held tick so far leaves 199 of the TTFT bound
        assert d._hold_ticks == 1
        assert unreachable(next_completion_steps=199) == 0
        assert unreachable(next_completion_steps=199) == 1
        assert unreachable() == 1  # no decode running
        # waiting work makes it blocked, not a straggler
        assert unreachable(waiting_requests=1) == 0
        assert d._reduce_buf[BLOCKED] == 1

    def test_capped_pending_waits_for_decoding_peer_to_refill(self):
        d = make_delayer(dp_size=8, ttft_max_ticks=8, stall_ticks=10)
        skip_first(d)
        # Seven ranks have prefill, one still decodes. The sum is already
        # above the fill target, but every local probe caps pending tokens.
        mixed = _add_rank({0: 6, 1: 6 * MAX_BATCHED, 2: 7 * 16})
        for _ in range(5):
            assert call(d, pending_tokens=MAX_BATCHED, reduce=mixed) is False
        # The peer completes and receives a replacement. Its admission flag
        # changes even though capped aggregate pending stays exactly flat.
        aligned = _add_rank({0: 7, 1: 6 * MAX_BATCHED, 2: 7 * 16})
        assert call(d, pending_tokens=MAX_BATCHED, reduce=aligned) is True
        assert d._stat_fire_fill == 1
        assert d._stat_fire_stall == 0

    def test_mixed_holds_while_growing(self):
        d = make_delayer(dp_size=2, ttft_max_ticks=1000, stall_ticks=2)
        skip_first(d)
        mixed = _add_rank({2: 64})
        for pending in (500, 1000, 1500, 2000):
            assert call(d, pending_tokens=pending, reduce=mixed) is False
        assert d._stat_fire_stall == 0


class _Rank0Group:
    def rank(self):
        return 0


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# dp_size=2 with costs and the cost gate: after the fixed slots the two
# ranks' chunk costs, then the gate's waiting / decode µs / prefill ps / timed
# ranks / arrivals per 1000 s on ranks without prefill
WAITING, DECODE_US, PREFILL_PS, TIMED, ARRIVALS = range(
    _FIXED_SLOTS + 2, _FIXED_SLOTS + 2 + len(_Gate._fields)
)
# The sibling decodes 1 seq with no prefill; it timed a 10 ms decode step and
# 100 ps a cost unit. Firing stalls 2 decodes for 3e8 x 100 ps = 30 ms: 60 ms
# of decode stall. Its decode does not complete within the TTFT bound.
TIMED_SIBLING = {2: 1, DECODE_US: 10_000, PREFILL_PS: 100, TIMED: 1, UNREACHABLE: 1}


def lone_request(d, reduce):
    # this rank: one waiting request of cost 3e8, one decode; the sibling has
    # no prefill
    return call(
        d,
        pending_tokens=500,
        running_decode_batch=1,
        prefill_cost=3e8,
        waiting_requests=1,
        reduce=reduce,
    )


def make_gated(**kw):
    return make_delayer(
        dp_size=2,
        cpu_group=_Rank0Group(),
        ttft_max_ticks=1000,
        stall_ticks=1000,
        exchange_prefill_costs=True,
        cost_gate=True,
        **kw,
    )


class TestCostGate:
    def test_releases_at_once_when_no_work_is_coming(self):
        # 1 arrival / 100 s on the sibling: a held tick's expected saving is
        # 60 ms x 1e-8 / µs, far below the waiting request it delays
        d = make_gated()
        skip_first(d)
        mixed = _add_rank({**TIMED_SIBLING, ARRIVALS: 10})
        assert lone_request(d, mixed) is True
        assert d._stat_fire_cost == 1

    def test_holds_while_work_is_coming_until_the_wait_costs_the_stall(self):
        # 100 arrivals / s: holding is expected to pay (60 ms x 1e-4 / µs = 6
        # > 1 request); the request pays 10 ms a held tick, so the 7th tick
        # (60 ms already paid) releases.
        d = make_gated()
        skip_first(d)
        mixed = _add_rank({**TIMED_SIBLING, ARRIVALS: 100_000})
        for _ in range(6):
            assert lone_request(d, mixed) is False
        assert lone_request(d, mixed) is True
        assert d._stat_fire_cost == 1

    def test_waits_for_a_sibling_completing_in_time(self):
        # Same sibling, but its decode completes within the TTFT bound: its
        # next request comes with that completion, so the gate does not judge.
        d = make_gated()
        skip_first(d)
        reachable = {**TIMED_SIBLING, ARRIVALS: 10}
        del reachable[UNREACHABLE]
        for _ in range(20):
            assert lone_request(d, _add_rank(reachable)) is False
        assert d._stat_fire_cost == 0

    def test_untimed_holds_for_the_stall_bound(self):
        d = make_gated()
        skip_first(d)
        mixed = _add_rank({2: 1})
        for _ in range(20):
            assert lone_request(d, mixed) is False
        assert d._stat_fire_cost == 0

    def test_arrival_rate_decays_while_nothing_comes(self):
        clock = _Clock()
        d = make_gated(clock=clock)
        skip_first(d)
        for _ in range(3):
            clock.now += 0.5
            call(d, prefillable=False, arrived_requests=1)
        assert d._arrival_rate(clock.now) == pytest.approx(2.0)
        clock.now += 4.0
        assert d._arrival_rate(clock.now) == pytest.approx(0.25)
        # a rank without prefill reports its rate (per 1000 s)
        call(d, prefillable=False)
        assert d._reduce_buf[ARRIVALS] == 250

    def test_an_idle_spell_is_not_timed_as_a_decode_step(self):
        """The DP group idle schedules nothing, so the next decision follows
        no step: 100 s of nothing must not average into a decode step."""
        clock = _Clock()
        d = make_gated(clock=clock)
        skip_first(d)
        sibling = _add_rank({0: 1, 1: MAX_BATCHED, 2: 1})
        for _ in range(3):
            clock.now += 0.010
            call(d, prefillable=False, running_decode_batch=1, reduce=sibling)
        d.notify_idle()
        clock.now += 100.0
        call(d, prefillable=False, running_decode_batch=1, reduce=sibling)
        assert d._decode_tick_us == pytest.approx(10_000)

    def test_times_decode_and_prefill_steps(self):
        clock = _Clock()
        d = make_gated(clock=clock)
        skip_first(d)
        sibling = _add_rank({0: 1, 1: MAX_BATCHED, 2: 1})
        # decode steps of 10 ms (no prefill ran)
        for _ in range(3):
            clock.now += 0.010
            call(d, prefillable=False, running_decode_batch=1, reduce=sibling)
        assert d._decode_tick_us == pytest.approx(10_000)
        # both ranks full: fill fires a 2e8-cost prefill that runs 20 ms
        clock.now += 0.010
        assert (
            call(
                d,
                pending_tokens=MAX_BATCHED,
                running_decode_batch=1,
                prefill_cost=2e8,
                reduce=sibling,
            )
            is True
        )
        d.notify_prefill_executed()
        clock.now += 0.020
        call(d, prefillable=False, running_decode_batch=1, reduce=sibling)
        assert d._prefill_ps_per_cost == pytest.approx(0.020e12 / 2e8)
        assert d._decode_tick_us == pytest.approx(10_000)


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

    def test_negative_prefill_decode_interval_rejected(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            make_delayer(prefill_decode_interval=-1)


class TestPrefillDecodeInterval:
    def test_protects_exact_number_of_decode_passes(self):
        d = make_delayer(
            prefill_decode_interval=3,
            ttft_max_ticks=1,
            stall_ticks=1,
            kv_high_watermark=0.1,
        )
        # FIRE only grants admission; the completed prefill forward arms it.
        assert call(d, pending_tokens=MAX_BATCHED, kv_usage=1.0) is True
        d.notify_prefill_executed()
        # The hard interval takes precedence over all ordinary coalescer
        # release conditions and does not consume the coalescer TTFT budget.
        for _ in range(3):
            assert call(d, pending_tokens=MAX_BATCHED, kv_usage=1.0) is False
        assert d._hold_ticks == 0
        assert d._stat_hold_decode_interval == 3
        # The next pass reaches the normal decision and fires on KV pressure.
        assert call(d, pending_tokens=MAX_BATCHED, kv_usage=1.0) is True

    def test_countdown_advances_when_no_prefill_is_waiting(self):
        d = make_delayer(prefill_decode_interval=2)
        assert call(d, pending_tokens=MAX_BATCHED) is True
        d.notify_prefill_executed()
        assert call(d, prefillable=False, pending_tokens=0) is True
        assert call(d, prefillable=False, pending_tokens=0) is True
        assert d._decode_interval_remaining == 0

    def test_no_decode_does_not_block_prefill_progress(self):
        d = make_delayer(prefill_decode_interval=10)
        assert call(d, pending_tokens=MAX_BATCHED, running_decode_batch=0) is True
        d.notify_prefill_executed()
        # A partial-only workload has no decode pass to protect, so it may
        # continue instead of producing ten empty scheduler iterations.
        assert call(d, pending_tokens=1000, running_decode_batch=0) is True

    def test_fire_without_executed_prefill_does_not_arm_interval(self):
        d = make_delayer(
            prefill_decode_interval=3,
            ttft_max_ticks=1,
            stall_ticks=1,
            kv_high_watermark=0.1,
        )
        assert call(d, pending_tokens=MAX_BATCHED, kv_usage=1.0) is True
        # Admission may still yield no prefill batch. Without an execution
        # notification, the next eligible prefill must not be delayed.
        assert call(d, pending_tokens=MAX_BATCHED, kv_usage=1.0) is True
        assert d._stat_hold_decode_interval == 0

    def test_arrival_in_window_restarts_the_stall_count(self):
        d = make_delayer(
            dp_size=2, prefill_decode_interval=2, ttft_max_ticks=1000, stall_ticks=3
        )
        skip_first(d)
        mixed = _add_rank({2: 64, UNREACHABLE: 1})
        assert call(d, pending_tokens=500, reduce=mixed) is False  # first look
        assert call(d, pending_tokens=500, reduce=mixed) is False  # stall 1
        assert call(d, pending_tokens=500, reduce=mixed) is False  # stall 2
        d.notify_prefill_executed()
        # A request arrives on the peer inside the decode-protection window.
        arriving = _add_rank({2: 64, ARRIVED: 1, UNREACHABLE: 1})
        assert call(d, pending_tokens=500, reduce=arriving) is False
        assert call(d, pending_tokens=500, reduce=mixed) is False
        assert d._stat_hold_decode_interval == 2
        assert d._stall_count == 0
        # The count restarts from the arrival instead of resuming at 2.
        assert call(d, pending_tokens=500, reduce=mixed) is False  # stall 1
        assert call(d, pending_tokens=500, reduce=mixed) is False  # stall 2
        assert call(d, pending_tokens=500, reduce=mixed) is True  # stall 3
        assert d._stat_fire_stall == 1

    def test_execution_on_peer_arms_all_ranks(self):
        d = make_delayer(dp_size=2, prefill_decode_interval=1)
        skip_first(d)
        peer_prefill_ran = _add_rank({7: 1})
        assert call(d, pending_tokens=1000, reduce=peer_prefill_ran) is False
        assert d._stat_hold_decode_interval == 1


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
