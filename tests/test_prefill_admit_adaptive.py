"""--prefill-admit-policy adaptive (rent-or-buy / EOQ prefill admission).

Single-rank (dp_size=1, no group) with an injected clock; a small plant model
drives the delayer so the fitted step costs and the resulting cycle length can
be checked against the closed-form EOQ optimum. The initial-ramp tests at the
end also run a two-rank group through a mocked all_reduce. No GPU /
distributed group.
"""

import math
import random
from unittest import mock

import pytest

from atom.model_engine.prefill_delayer import (
    _FIXED_SLOTS,
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
            prefill_budget=16384,
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
    d.should_allow_prefill(
        prefillable=True,
        pending_tokens=1,
        running_decode_batch=10,
        prefill_budget=16384,
    )
    for _ in range(50):
        dec = d.should_allow_prefill(
            prefillable=True,
            pending_tokens=1000,
            running_decode_batch=100,
            waiting_requests=5,
            admittable_requests=0,
            prefill_budget=16384,
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


# ---- initial ramp (ATOM_PREFILL_INITIAL_RAMP) under the adaptive policy ----
# reduce-buffer slots: the ranks whose free seq slots fill target_fill of a chunk,
# and (no cost slots) the first adaptive one, the requests a FIRE could admit
_PREFILL_BUDGET = 12
_FULL_CHUNK = 13
_ADMITTABLE = _FIXED_SLOTS


class _Rank0Group:
    def rank(self):
        return 0


def _adaptive_delayer(dp_size=1, prefill_ramp=True, clock=None):
    return PrefillDelayer(
        dp_size=dp_size,
        cpu_group=_Rank0Group() if dp_size > 1 else None,
        max_num_batched_tokens=16384,
        prefill_ramp=prefill_ramp,
        clock=clock or _Clock(),
        admit_policy="adaptive",
    )


def _sibling(full=True, running=10):
    """The other rank of a two-rank group: a whole chunk pending, 5 admittable
    requests, `running` decode seqs; `full`: its free slots fill a chunk."""
    slots = {0: 1, 1: 16384, 2: running, _ADMITTABLE: 5, _PREFILL_BUDGET: 16384}
    if full:
        slots[_FULL_CHUNK] = 1
    return slots


def _tick(d, full=True, running=10, kv_usage=0.5, sibling=None):
    """One decision of a rank with a whole chunk pending and 5 admittable
    requests; `sibling` adds the other rank's reduce slots."""

    def all_reduce(tensor, op=None, group=None):
        for slot, value in (sibling or {}).items():
            tensor[slot] += value

    with mock.patch("torch.distributed.all_reduce", all_reduce):
        return d.should_allow_prefill(
            prefillable=True,
            pending_tokens=16384,
            running_decode_batch=running,
            kv_usage=kv_usage,
            waiting_requests=5,
            admittable_requests=5,
            full_chunk=full,
            prefill_budget=16384,
        ).allow


def _rent_or_buy_cycle(d, **kw):
    # Cold start: the regret reaches A*B within two ticks, so the FIRE waits
    # for the _MIN_TICKS floor.
    got = [_tick(d, **kw) for _ in range(_MIN_TICKS)]
    assert got == [False] * (_MIN_TICKS - 1) + [True]


def test_initial_ramp_fires_before_min_ticks():
    clk = _Clock()
    d = _adaptive_delayer(clock=clk)
    assert _tick(d)  # first call: warmup seed
    # Every rank fills target_fill of a chunk: release at once, every tick.
    for _ in range(_MIN_TICKS - 1):
        d.notify_prefill_executed(8000)
        clk.t += 0.05
        assert _tick(d)
    assert d.ramp_active
    assert d._stat_fire_ramp == _MIN_TICKS - 1
    assert d._stat_fire_adaptive == d._stat_fire_adaptcap == 0
    assert d._stat_adapt_hold_ticks == 0
    # the fits still learn from the ramp's prefill steps
    assert d._adapt_pf_fit.n == _MIN_TICKS - 1


def test_initial_ramp_ends_when_a_rank_cannot_fill():
    d = _adaptive_delayer(dp_size=2)
    full = _sibling()
    assert _tick(d, sibling=full)  # warmup seed
    assert _tick(d, sibling=full)
    assert d._stat_fire_ramp == 1
    # The sibling's free slots no longer fill a chunk: the ramp ends for the
    # group and rent-or-buy decides, from this tick on.
    assert not _tick(d, sibling=_sibling(full=False))
    assert not d.ramp_active
    got = [_tick(d, sibling=full) for _ in range(_MIN_TICKS - 1)]
    assert got == [False] * (_MIN_TICKS - 2) + [True]
    # Every rank full again, or one rank without decode, does not reopen it.
    _rent_or_buy_cycle(d, sibling=full)
    _rent_or_buy_cycle(d, running=0, sibling=full)
    assert d._stat_fire_ramp == 1 and d._stat_fire_adaptive == 3
    # The whole group runs out of decode: the ramp re-arms for the next burst
    # and releases at once, ahead of the no-decode bound.
    assert _tick(d, running=0, sibling=_sibling(running=0))
    assert d._stat_fire_ramp == 2 and d.ramp_active
    assert d._stat_fire_nodecode == 0
    assert _tick(d, sibling=full)
    assert d._stat_fire_ramp == 3


def test_initial_ramp_ends_on_kv_high():
    d = _adaptive_delayer()
    assert _tick(d)  # warmup seed
    assert _tick(d)
    assert d._stat_fire_ramp == 1
    # KV runs high: the KV bound releases and the ramp ends.
    assert _tick(d, kv_usage=0.95)
    assert d._stat_fire_kv == 1 and not d.ramp_active
    _rent_or_buy_cycle(d)
    assert d._stat_fire_ramp == 1 and d._stat_fire_adaptive == 1


def test_ramp_off_ignores_full_chunk():
    # Without the ramp full_chunk changes nothing: rent-or-buy as before.
    with_full = _adaptive_delayer(prefill_ramp=False)
    without = _adaptive_delayer(prefill_ramp=False)
    running = [10] * 9 + [0] + [10] * 5
    got = [_tick(with_full, running=r) for r in running]
    assert got == [_tick(without, full=False, running=r) for r in running]
    cycle = [False] * (_MIN_TICKS - 1) + [True]
    # warmup seed, two cycles, no decode, a cycle, one held tick
    assert got == [True] + cycle * 2 + [True] + cycle + [False]
    stats = [s for s in PrefillDelayer.__slots__ if s.startswith("_stat_")]
    assert [getattr(with_full, s) for s in stats] == [
        getattr(without, s) for s in stats
    ]
    assert with_full._stat_fire_ramp == 0 and not with_full.ramp_active
    assert with_full._stat_fire_adaptive == 3
