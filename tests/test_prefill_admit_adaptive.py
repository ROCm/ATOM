"""--prefill-admit-policy adaptive (rent-or-buy / EOQ prefill admission).

Single-rank (dp_size=1, no group) with an injected clock; a small plant model
drives the delayer so the fitted step costs and the resulting cycle length can
be checked against the closed-form EOQ optimum. No GPU / distributed group.
"""

import math
import random

import pytest

from atom.model_engine.prefill_delayer import (
    _FIXED_SLOTS,
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
    # huge A: EOQ would hold ~100+ ticks; cap = 600 ms / ~26 ms ~ 23 ticks
    d, cyc = _run(400000, 20000, 20.0, 0.3)
    t0, _ = d._adapt_local_fits()
    cap = d._adapt_max_ticks(t0, True)
    assert 18 <= cap <= 28, cap
    # + the vacuous ticks with an empty queue right after a FIRE
    assert cyc <= cap + 2, (cyc, cap)
    assert d._stat_fire_adaptcap > 0


def test_min_ticks():
    # tiny A, big queue: EOQ wants to fire every tick -> held to MIN_TICKS
    _d, cyc = _run(1000, 20000, 20.0, 8.0)
    assert cyc >= 3.5, cyc


def test_slower_prefill_not_shorter_and_bounded():
    _, fast = _run(30000, 20000, 20.0, 2.0)
    _, slow = _run(90000, 20000, 20.0, 2.0)
    assert fast <= slow <= 26, (fast, slow)


@pytest.mark.parametrize("a_us,lam", [(15000, 2.0), (25000, 2.0)])
def test_cycle_matches_eoq_with_fitted_costs(a_us, lam):
    # between the MIN_TICKS floor and the TTFT cap the cycle follows EOQ
    # with the (bounded) fitted T0 and A
    d, cyc = _run(a_us, 20000, c_us=20.0, lam=lam)
    t0, a = d._adapt_local_fits()
    k_eoq = math.sqrt(2 * a * 300 / (lam * t0))
    assert 4 < k_eoq < d._adapt_max_ticks(t0, True)
    assert 0.7 * k_eoq < cyc < 1.4 * k_eoq, (cyc, k_eoq)
