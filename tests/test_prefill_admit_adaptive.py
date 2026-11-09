"""ATOM_PREFILL_ADMIT=adaptive (rent-or-buy / EOQ prefill admission).

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
    admit_policy_from_env,
)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _run(a_us, t0_us, c_us, lam, ticks=6000, seed=0, b0=300, version=1, params=None):
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
        admit_version=version,
        admit_params=params,
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


def test_env_parsing(monkeypatch):
    monkeypatch.delenv("ATOM_PREFILL_ADMIT", raising=False)
    assert admit_policy_from_env() == ""
    for v, want in (("adaptive", "adaptive"), ("eoq", "adaptive"), ("off", "")):
        monkeypatch.setenv("ATOM_PREFILL_ADMIT", v)
        assert admit_policy_from_env() == want
    monkeypatch.setenv("ATOM_PREFILL_ADMIT", "bogus")
    with pytest.raises(ValueError):
        admit_policy_from_env()


def test_default_off_buffer_unchanged():
    d = PrefillDelayer(dp_size=1, cpu_group=None, max_num_batched_tokens=16384)
    assert d.admit_policy == ""
    assert d._reduce_buf.numel() == _FIXED_SLOTS


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


@pytest.mark.parametrize("a_us,t0_us,lam", [(30000, 20000, 2.0), (60000, 20000, 2.0)])
def test_cycle_matches_eoq(a_us, t0_us, lam):
    d, cyc = _run(a_us, t0_us, c_us=20.0, lam=lam)
    t0_fit, a_fit = d._adapt_local_fits()
    assert abs(t0_fit - t0_us) / t0_us < 0.15
    assert abs(a_fit - a_us) / a_us < 0.25
    b = 300
    k_eoq = math.sqrt(2 * a_us * b / (lam * t0_us))
    assert 0.7 * k_eoq < cyc < 1.4 * k_eoq, (cyc, k_eoq)


def test_slower_prefill_lengthens_cycle():
    _, fast = _run(30000, 20000, 20.0, 2.0)
    _, slow = _run(90000, 20000, 20.0, 2.0)
    # EOQ: cycle ~ sqrt(A) -> x sqrt(3)
    assert 1.4 < slow / fast < 2.1, (fast, slow)


# ---------------- v2 (ATOM_PREFILL_ADMIT_VERSION=2, default) ----------------


def test_v2_is_default_and_env(monkeypatch):
    for k in ("VERSION", "T0_FRAC", "MAX_HOLD_MS", "MIN_TICKS"):
        monkeypatch.delenv(f"ATOM_PREFILL_ADMIT_{k}", raising=False)
    d = PrefillDelayer(
        dp_size=1, cpu_group=None, max_num_batched_tokens=16384, admit_policy="adaptive"
    )
    assert d.admit_version == 2 and d._v2["T0_FRAC"] == 0.75
    monkeypatch.setenv("ATOM_PREFILL_ADMIT_VERSION", "1")
    monkeypatch.setenv("ATOM_PREFILL_ADMIT_MAX_HOLD_MS", "300")
    d = PrefillDelayer(
        dp_size=1, cpu_group=None, max_num_batched_tokens=16384, admit_policy="adaptive"
    )
    assert d.admit_version == 1 and d._v2["MAX_HOLD_MS"] == 300.0
    monkeypatch.setenv("ATOM_PREFILL_ADMIT_VERSION", "3")
    with pytest.raises(ValueError):
        PrefillDelayer(
            dp_size=1, cpu_group=None, max_num_batched_tokens=16384,
            admit_policy="adaptive",
        )


def test_v2_t0_is_fraction_of_mean():
    # decode step 25 ms at b~300; v1's intercept would be ~19 ms, v2 = 0.75*mean
    d, _ = _run(60000, 19000, 20.0, 2.0, version=2)
    t0, a = d._adapt_local_fits()
    mean_dec = d._adapt_dec_fit.mean_y()
    assert abs(t0 - 0.75 * mean_dec) < 1e-6
    my = d._adapt_pf_fit.mean_y()
    assert 0.3 * my - 1e-6 <= a <= 1.5 * my + 1e-6


def test_v2_a_clamped_low():
    # A tiny vs the per-token part: intercept below 0.3*mean -> clamped
    d, _ = _run(1000, 20000, 20.0, 2.0, version=2)
    _, a = d._adapt_local_fits()
    assert abs(a - 0.3 * d._adapt_pf_fit.mean_y()) < 1.0


def test_v2_cycle_capped_by_budget():
    # huge A: EOQ would hold ~100+ ticks; cap = 600 ms / ~26 ms ~ 23 ticks
    d, cyc = _run(400000, 20000, 20.0, 0.3, version=2)
    t0, _ = d._adapt_local_fits()
    cap = d._adapt_max_ticks(t0, True)
    assert 18 <= cap <= 28, cap
    # + the vacuous ticks with an empty queue right after a FIRE
    assert cyc <= cap + 2, (cyc, cap)
    assert d._stat_fire_adaptcap > 0


def test_v2_min_ticks():
    # tiny A, big queue: EOQ wants to fire every tick -> held to MIN_TICKS
    d, cyc = _run(1000, 20000, 20.0, 8.0, version=2, params={"MIN_TICKS": 4})
    assert cyc >= 3.5, cyc


def test_v2_max_hold_ms_scales_cap():
    _, c600 = _run(400000, 20000, 20.0, 0.3, version=2)
    _, c300 = _run(400000, 20000, 20.0, 0.3, version=2, params={"MAX_HOLD_MS": 300})
    assert 0.4 < c300 / c600 < 0.65, (c300, c600)


def test_v2_slower_prefill_not_shorter_and_bounded():
    _, fast = _run(30000, 20000, 20.0, 2.0, version=2)
    _, slow = _run(90000, 20000, 20.0, 2.0, version=2)
    assert fast <= slow <= 26, (fast, slow)
