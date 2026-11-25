"""--prefill-admit-policy adaptive (rent-or-buy / EOQ prefill admission).

Single-rank (dp_size=1, no group) with an injected clock; a small plant model
drives the delayer so the fitted step costs and the resulting cycle length can
be checked against the closed-form EOQ optimum. No GPU / distributed group.
"""

import math
import random

import pytest

from atom.model_engine.prefill_delayer import (
    _COLD_MAX_TICKS,
    _DEC_HALF_LIFE,
    _DEC_MIN_N,
    _FIXED_SLOTS,
    _MAX_HOLD_MS,
    _MIN_TICKS,
    PrefillDelayer,
    _EwFit,
)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _run(a_us, t0_us, c_us, lam, ticks=6000, seed=0, b0=300):
    """Plant: B decode seqs, arrivals ~lam/tick, each FIRE admits all waiting.
    Returns (delayer, mean decode ticks between fires)."""
    rng = random.Random(seed)
    clk = _Clock()
    d = PrefillDelayer(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=16384,
        prefill_decode_interval=32,
        clock=clk,
        admit_policy="adaptive",
    )
    running, waiting, fires, decode_ticks = b0, 0, 0, 0
    for _ in range(ticks):
        k = int(lam) + (1 if rng.random() < lam - int(lam) else 0)
        waiting += k
        running -= k  # closed loop: as many finish as arrive
        dec = d.should_allow_prefill(
            prefillable=waiting > 0,
            pending_tokens=min(16384, 400 * waiting),
            running_decode_batch=running,
            waiting_requests=waiting,
            admittable_requests=waiting,
        )
        if dec.allow and waiting > 0:
            toks = 400 * waiting
            running += waiting
            waiting = 0
            d.notify_prefill_executed(toks)
            clk.t += (a_us + 2.0 * toks + rng.gauss(0, 500)) / 1e6
            fires += 1
        else:
            clk.t += (t0_us + c_us * running + rng.gauss(0, 300)) / 1e6
            decode_ticks += 1
    return d, decode_ticks / max(fires, 1)


def test_default_off_buffer_unchanged():
    d = PrefillDelayer(dp_size=1, cpu_group=None, max_num_batched_tokens=16384)
    assert d.admit_policy == "interval"
    assert d._reduce_buf.numel() == _FIXED_SLOTS


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        PrefillDelayer(
            dp_size=1, cpu_group=None, max_num_batched_tokens=16384, admit_policy=""
        )


def test_adaptive_ignores_interval():
    d = PrefillDelayer(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=16384,
        prefill_decode_interval=24,
        admit_policy="adaptive",
    )
    assert d.prefill_decode_interval == 0


def test_ewfit_recovers_line():
    f = _EwFit(500)
    rng = random.Random(1)
    for _ in range(3000):
        x = rng.uniform(50, 120)
        f.add(x, 20000 + 60 * x + rng.gauss(0, 200))
    a, b, _, _ = f.fit(64, 0.02)
    assert abs(a - 20000) < 600 and abs(b - 60) < 8
    g = _EwFit(500)
    for _ in range(200):
        g.add(85.0, 25000.0)
    assert g.fit(64, 0.02) is None  # constant x: not identifiable


def test_no_admittable_holds_without_regret():
    d = PrefillDelayer(
        dp_size=1, cpu_group=None, max_num_batched_tokens=16384, admit_policy="adaptive"
    )
    d.should_allow_prefill(prefillable=True, pending_tokens=1, running_decode_batch=10)
    for _ in range(50):
        dec = d.should_allow_prefill(
            prefillable=True,
            pending_tokens=1000,
            running_decode_batch=100,
            waiting_requests=5,
            admittable_requests=0,
        )
        assert not dec.allow
    assert d._adapt_regret == 0.0


def test_t0_is_fraction_of_mean():
    # decode step 25 ms at b~300; T0 = 0.75*mean, not the ~19 ms intercept
    d, _ = _run(60000, 19000, 20.0, 2.0)
    t0, a = d._adapt_local_fits()
    mean_dec = d._adapt_dec_fit.mean_y()
    assert abs(t0 - 0.75 * mean_dec) < 1e-6
    my = d._adapt_pf_fit.mean_y()
    assert 0.3 * my - 1e-6 <= a <= 1.5 * my + 1e-6


def test_a_clamped_low():
    # A tiny vs the per-token part: intercept below 0.3*mean -> clamped
    d, _ = _run(1000, 20000, 20.0, 2.0)
    _, a = d._adapt_local_fits()
    assert abs(a - 0.3 * d._adapt_pf_fit.mean_y()) < 1.0


def test_cycle_capped_by_budget():
    # huge A: EOQ would hold ~200 ticks; cap = _MAX_HOLD_MS / ~26 ms
    d, cyc = _run(400000, 20000, 20.0, 0.3)
    t0, _ = d._adapt_local_fits()
    cap = d._adapt_max_ticks(t0, True)
    budget = _MAX_HOLD_MS * 1000.0 / 26000.0
    assert 0.85 * budget <= cap <= 1.15 * budget, (cap, budget)
    # + the vacuous ticks with an empty queue right after a FIRE
    assert cyc <= cap + 2, (cyc, cap)
    assert d._stat_fire_adaptcap > 0


def test_min_ticks():
    # tiny A, big queue: EOQ wants to fire every tick -> held to MIN_TICKS
    _d, cyc = _run(1000, 20000, 20.0, 8.0)
    assert cyc >= _MIN_TICKS - 1, cyc  # n-1 held ticks before the n-th decision


def test_slower_prefill_not_shorter_and_bounded():
    d, fast = _run(30000, 20000, 20.0, 0.5)
    _, slow = _run(90000, 20000, 20.0, 0.5)
    cap = d._adapt_max_ticks(d._adapt_local_fits()[0], True)
    assert fast <= slow <= cap + 2, (fast, slow, cap)


@pytest.mark.parametrize("a_us,lam", [(40000, 0.5), (60000, 0.5)])
def test_cycle_matches_eoq_with_fitted_costs(a_us, lam):
    # between the MIN_TICKS floor and the TTFT cap the cycle follows EOQ
    # with the (bounded) fitted T0 and A
    d, cyc = _run(a_us, 20000, c_us=20.0, lam=lam)
    t0, a = d._adapt_local_fits()
    k_eoq = math.sqrt(2 * a * 300 / (lam * t0))
    assert _MIN_TICKS < k_eoq < d._adapt_max_ticks(t0, True)
    assert 0.7 * k_eoq < cyc < 1.4 * k_eoq, (cyc, k_eoq)


def _adaptive():
    return PrefillDelayer(
        dp_size=1, cpu_group=None, max_num_batched_tokens=16384, admit_policy="adaptive"
    )


def _decode_steps(d, n, step_s, running=100):
    zeros = [0] * d._reduce_buf.numel()
    for _ in range(n):
        d._adapt_observe(zeros, step_s, False, running)


def test_cap_not_below_cold_start_cap():
    # a slow decode mean must not shorten the TTFT cap below the cold-start
    # cap: a shorter cap admits prefill more often, which slows decode more
    d = _adaptive()
    budget_us = _MAX_HOLD_MS * 1000.0
    slow_t0 = 0.75 * 4 * (budget_us / _COLD_MAX_TICKS)
    assert d._adapt_max_ticks(slow_t0, True) == _COLD_MAX_TICKS
    fast_t0 = 0.75 * (budget_us / (2 * _COLD_MAX_TICKS))
    assert d._adapt_max_ticks(fast_t0, True) == 2 * _COLD_MAX_TICKS


def test_decode_step_after_prefill_not_in_t0():
    d = _adaptive()
    zeros = [0] * d._reduce_buf.numel()
    _decode_steps(d, _DEC_MIN_N + 1, 0.020)
    n = d._adapt_dec_fit.n
    d._adapt_observe(zeros, 0.090, True, 100)  # a prefill step
    _decode_steps(d, 1, 0.060)  # the decode step right after it
    assert d._adapt_dec_fit.n == n
    _decode_steps(d, 1, 0.020)  # the next decode step counts
    assert d._adapt_dec_fit.n == n + 1
    assert abs(d._adapt_dec_fit.mean_y() - 20000.0) < 1e-6


def test_decode_spike_clipped_at_twice_mean():
    clipped, ref = _adaptive(), _EwFit(_DEC_HALF_LIFE)
    _decode_steps(clipped, _DEC_MIN_N + 1, 0.020)
    for _ in range(_DEC_MIN_N):
        ref.add(100.0, 20000.0)
    _decode_steps(clipped, 1, 2.0)  # a stall
    ref.add(100.0, 40000.0)
    assert abs(clipped._adapt_dec_fit.mean_y() - ref.mean_y()) < 1e-6
