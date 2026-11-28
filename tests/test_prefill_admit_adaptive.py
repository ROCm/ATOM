"""--prefill-admit-policy adaptive (rent-or-buy / EOQ prefill admission).

Single-rank (dp_size=1, no group) with an injected clock; a small plant model
drives the delayer so the measured step costs and the resulting cycle length
can be checked against the closed-form EOQ optimum. No GPU / distributed group.
"""

import itertools
import math
import random

import pytest

import atom.model_engine.prefill_delayer as pd_mod
from atom.model_engine.prefill_delayer import (
    _ADAPT_FIXED,
    _FIXED_SLOTS,
    PrefillDelayer,
    _AdaptiveMean,
)

MNBT = 16384

# the shipped floor (fewest decisions per cycle), read before the fixture below
_DEFAULT_FLOOR = pd_mod._FLOOR_TICKS


@pytest.fixture(autouse=True)
def _no_floor(monkeypatch):
    """Most tests check the rule itself, so they run without the floor; the
    floor tests set it explicitly."""
    monkeypatch.setattr(pd_mod, "_FLOOR_TICKS", 1)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _run(
    a_us,
    t0_us,
    c_us,
    lam,
    ticks=6000,
    seed=0,
    b0=300,
    x_us=0.0,
    w0=0,
    fires=None,
    fin_us=0.0,
):
    """Plant: B decode seqs, arrivals ~lam/tick, each FIRE admits all waiting
    (up to a full pass of 400-token prompts). A prefill step takes
    a_us + 2 us/token; a decode step t0_us + c_us per seq, plus fin_us for
    every request that finishes in it; the decode step right after a prefill
    takes x_us more. w0 requests wait at the start (a flood). `fires` collects
    (reason, n) of every FIRE. Returns (delayer, mean decode ticks between
    fires)."""
    rng = random.Random(seed)
    clk = _Clock()
    d = PrefillDelayer(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=MNBT,
        prefill_decode_interval=32,
        clock=clk,
        admit_policy="adaptive",
    )
    running, waiting, n_fires, decode_ticks = b0, w0, 0, 0
    after_prefill = False
    for _ in range(ticks):
        k = int(lam) + (1 if rng.random() < lam - int(lam) else 0)
        waiting += k
        running -= k  # closed loop: as many finish as arrive
        dec = d.should_allow_prefill(
            prefillable=waiting > 0,
            pending_tokens=min(MNBT, 400 * waiting),
            running_decode_batch=running,
            waiting_requests=waiting,
            admittable_requests=waiting,
            arrived_requests=k,
            prefill_budget=MNBT,
        )
        if dec.allow and waiting > 0:
            if fires is not None and d._adapt_last_fire is not None:
                fires.append(d._adapt_last_fire[:2])
            admit = min(waiting, MNBT // 400)
            toks = 400 * admit
            running += admit
            waiting -= admit
            d.notify_prefill_executed(toks)
            clk.t += (a_us + 2.0 * toks + rng.gauss(0, 500)) / 1e6
            n_fires += 1
            after_prefill = True
        else:
            tail = x_us if after_prefill else 0.0
            clk.t += (
                t0_us + c_us * running + fin_us * k + tail + rng.gauss(0, 300)
            ) / 1e6
            decode_ticks += 1
            after_prefill = False
    return d, decode_ticks / max(n_fires, 1)


def _adaptive():
    return PrefillDelayer(
        dp_size=1, cpu_group=None, max_num_batched_tokens=MNBT, admit_policy="adaptive"
    )


def _zeros(d):
    return [0] * d._reduce_buf.numel()


def _decode_steps(d, n, step_s, running=100):
    zeros = _zeros(d)
    for _ in range(n):
        d._adapt_observe(zeros, step_s, False, running)


def _prefill_step(d, toks, step_s, running=100):
    red = _zeros(d)
    red[d._adapt_at + _ADAPT_FIXED + d._rank] = toks
    d._adapt_observe(red, step_s, True, running)


def test_default_off_buffer_unchanged():
    d = PrefillDelayer(dp_size=1, cpu_group=None, max_num_batched_tokens=MNBT)
    assert d.admit_policy == "interval"
    assert d._reduce_buf.numel() == _FIXED_SLOTS


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        PrefillDelayer(
            dp_size=1, cpu_group=None, max_num_batched_tokens=MNBT, admit_policy=""
        )


def test_adaptive_ignores_interval():
    d = PrefillDelayer(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=MNBT,
        prefill_decode_interval=24,
        admit_policy="adaptive",
    )
    assert d.prefill_decode_interval == 0


def test_no_admittable_holds_without_regret():
    d = _adaptive()
    d.should_allow_prefill(
        prefillable=True, pending_tokens=1, running_decode_batch=10, prefill_budget=MNBT
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


def test_no_decode_fires():
    # nothing decoding: holding wastes the GPU, a must-fire bound before EOQ
    d = _adaptive()
    d.should_allow_prefill(
        prefillable=True, pending_tokens=1, running_decode_batch=10, prefill_budget=MNBT
    )
    dec = d.should_allow_prefill(
        prefillable=True,
        pending_tokens=1000,
        running_decode_batch=0,
        waiting_requests=1,
        admittable_requests=1,
        prefill_budget=MNBT,
    )
    assert dec.allow
    assert d._stat_fire_nodecode == 1


# ---- cold start: none of its own ----


def test_unmeasured_holds_until_ttft_bound():
    # no estimates yet (frozen clock: no step is ever timed): the rule holds
    # and the existing TTFT bound releases -- also when more requests wait
    # than decode (W >= B); there is no cold cadence to fall back on
    for waiting in (1, 600, 5000):
        d = PrefillDelayer(
            dp_size=1,
            cpu_group=None,
            max_num_batched_tokens=MNBT,
            clock=_Clock(),
            admit_policy="adaptive",
        )
        d.should_allow_prefill(
            prefillable=True,
            pending_tokens=1,
            running_decode_batch=10,
            prefill_budget=MNBT,
        )
        ns, n = [], 0
        while len(ns) < 3:
            n += 1
            dec = d.should_allow_prefill(
                prefillable=True,
                pending_tokens=min(MNBT, 400 * waiting),
                running_decode_batch=500,
                waiting_requests=waiting,
                admittable_requests=waiting,
                prefill_budget=MNBT,
            )
            if dec.allow:
                ns.append(n)
                n = 0
        assert ns == [d.ttft_max_ticks + 1] * 3, ns
        assert d._stat_fire_ttft == 3 and d._stat_fire_adaptive == 0
        assert d._adapt_local_fits() == (0.0, 0.0)


def test_measured_within_a_few_decisions_in_a_flood():
    # 3000 requests wait at start (warm-up flood, W >> B): the seed pass times
    # C, the few decisions held after it time T0 (the pass's aftermath step,
    # then plain decode steps), and EOQ decides from then on, a few decisions
    # after T0 and C are measured. A flood is a backlog -- every pass leaves
    # requests behind -- so from the second pass on the throughput rule
    # (sum W*T_d >= C*B) decides; it fires no later than the product
    # rule would.
    fires = []
    d, _ = _run(40000, 20000, 20.0, 0.5, ticks=3000, w0=3000, fires=fires)
    assert fires and {r for r, _ in fires} <= {"adaptive", "backlog"}, {
        r for r, _ in fires
    }
    assert "backlog" in {r for r, _ in fires}
    assert fires[0][1] <= 12, fires[:3]
    t0, c = d._adapt_local_fits()
    assert t0 > 0 and c > 0


def test_prefill_only_warmup_does_not_lock_t0():
    # A warm-up of one-output-token requests (the agentic bench's warm-up
    # phase): every pass finishes its requests, nothing ever decodes, so the
    # no-decode bound fires each decision and no T0 sample can exist. That is
    # no lock: the first decode steps of the real load time T0 and EOQ takes
    # over within a few decisions, with no further no-decode fire.
    clk = _Clock()
    d = PrefillDelayer(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=MNBT,
        clock=clk,
        admit_policy="adaptive",
    )
    for _ in range(1500):
        dec = d.should_allow_prefill(
            prefillable=True,
            pending_tokens=8000,
            running_decode_batch=0,
            waiting_requests=20,
            admittable_requests=20,
            prefill_budget=MNBT,
        )
        assert dec.allow
        d.notify_prefill_executed(8000)
        clk.t += 0.25
    assert d._adapt_dec.n == 0 and d._stat_fire_nodecode == 1499
    assert d._adapt_pf.n > 0
    fires = []
    rng = random.Random(1)
    running, waiting = 200, 0
    first_eoq_at = None
    for tick in range(3000):
        arr = 1 if rng.random() < 0.5 else 0
        waiting += arr
        dec = d.should_allow_prefill(
            prefillable=waiting > 0,
            pending_tokens=min(MNBT, 400 * waiting),
            running_decode_batch=running,
            waiting_requests=waiting,
            admittable_requests=waiting,
            arrived_requests=arr,
            prefill_budget=MNBT,
        )
        if dec.allow and waiting > 0:
            fires.append(d._adapt_last_fire[0])
            if first_eoq_at is None and fires[-1] == "adaptive":
                first_eoq_at = tick
            d.notify_prefill_executed(400 * waiting)
            clk.t += 0.15
            waiting = 0
        else:
            clk.t += 0.020
    assert set(fires) == {"adaptive"}, {r: fires.count(r) for r in set(fires)}
    assert first_eoq_at is not None and first_eoq_at < d.ttft_max_ticks
    assert abs(d._adapt_local_fits()[0] - 20000.0) < 1000.0


# ---- pass cost C ----

_N = 8  # passes per test (any count: a mean is used from its first sample)


def _warm(d):
    _decode_steps(d, 66, 0.020)


def test_pass_cost_includes_next_step_excess():
    d = _adaptive()
    _warm(d)
    for _ in range(_N):
        _prefill_step(d, 8000, 0.050)
        _decode_steps(d, 1, 0.061)  # 41 ms over the decode mean
        _decode_steps(d, 5, 0.020)
    t0, c = d._adapt_local_fits()
    assert abs(t0 - 20000.0) < 1e-6
    assert abs(c - 91000.0) < 1.0, c
    # the steps after a pass stay out of the decode mean
    assert abs(d._adapt_dec.mean() - 20000.0) < 1e-6


def test_pass_cost_includes_decaying_aftermath():
    # the step after the pass (GPU spill) and the one after it (ranks late to
    # the DP sync) are both part of the pass; the first step back at the
    # decode mean closes it
    d = _adaptive()
    _warm(d)
    for _ in range(_N):
        _prefill_step(d, 8000, 0.050)
        _decode_steps(d, 1, 0.061)
        _decode_steps(d, 1, 0.028)
        _decode_steps(d, 5, 0.020)
    assert abs(d._adapt_local_fits()[1] - 99000.0) < 1.0
    assert abs(d._adapt_dec.mean() - 20000.0) < 1e-6


def test_aftermath_ends_at_a_step_no_shorter():
    # a step above the mean but no shorter than the one before (the batch got
    # wider) is plain decoding again: it closes the pass and, picked by its
    # time, is no T0 sample; the step after it is
    d = _adaptive()
    _warm(d)
    n0 = d._adapt_dec.n
    _prefill_step(d, 8000, 0.050)
    _decode_steps(d, 1, 0.061)
    _decode_steps(d, 1, 0.025)
    _decode_steps(d, 1, 0.026)  # closes
    assert d._adapt_pf_open is None and d._adapt_pf.n == 1
    assert abs(d._adapt_pf.mean() - (50000.0 + 41000.0 + 5000.0)) < 1.0
    assert d._adapt_dec.n == n0
    _decode_steps(d, 1, 0.026)
    assert d._adapt_dec.n == n0 + 1


def test_pass_cost_faster_next_step_adds_nothing():
    d = _adaptive()
    _warm(d)
    for _ in range(_N):
        _prefill_step(d, 8000, 0.050)
        _decode_steps(d, 1, 0.015)
    assert abs(d._adapt_local_fits()[1] - 50000.0) < 1.0


@pytest.mark.parametrize("case", ["no_decode", "no_decode_mean", "prefill_next"])
def test_pass_cost_without_excess(case):
    d = _adaptive()
    if case != "no_decode_mean":
        _warm(d)
    running = 0 if case == "no_decode" else 100
    for _ in range(_N):
        _prefill_step(d, 8000, 0.050, running=running)
        if case == "prefill_next":
            _prefill_step(d, 8000, 0.050)
        else:
            _decode_steps(d, 1, 0.061, running=running)
    pf = d._adapt_pf
    assert pf.n >= _N - 1
    assert abs(pf.mean() - 50000.0) < 1.0


def test_aftermath_steps_not_in_t0():
    d = _adaptive()
    _warm(d)
    n = d._adapt_dec.n
    _prefill_step(d, 8000, 0.090)
    _decode_steps(d, 1, 0.060)  # aftermath
    assert d._adapt_dec.n == n
    _decode_steps(d, 1, 0.019)  # back below the mean: closes the pass, no sample
    assert d._adapt_pf_open is None
    assert abs(d._adapt_pf.mean() - (90000.0 + 40000.0)) < 1.0
    assert d._adapt_dec.n == n
    _decode_steps(d, 1, 0.019)  # plain decoding again
    assert d._adapt_dec.n == n + 1


# ---- pass cost weighted by the decisions of its cycle ----


def _cycle(d, decisions, pf_s, tail_s=0.0):
    """decisions-1 plain decode steps, then a pass (and its aftermath)."""
    _decode_steps(d, decisions - 1, 0.020)
    _prefill_step(d, 8000, pf_s)
    if tail_s:
        _decode_steps(d, 1, 0.020 + tail_s)


def test_pass_cost_weighted_by_cycle_decisions():
    d = _adaptive()
    _warm(d)
    _cycle(d, 1, 0.050)  # first pass: its cycle is not known yet (weight 1)
    ref = _AdaptiveMean()
    prev, prev_n = 0.050, 1.0
    for n, p in [(40, 0.090), (10, 0.050), (25, 0.120), (40, 0.060)]:
        _cycle(d, n, p)
        ref.add(prev * 1e6, prev_n)
        prev, prev_n = p, float(n)
    _decode_steps(d, 2, 0.020)  # closes the last pass
    ref.add(prev * 1e6, prev_n)
    assert abs(d._adapt_pf.mean() - ref.mean()) < 1e-6


def test_weighted_mean_counts_decisions():
    # same memory: a 100-decision cycle's pass outweighs ten 1-decision ones
    m = _AdaptiveMean()
    m.memory = 1e9
    for _ in range(10):
        m.add(250000.0, 1.0, 1.0)
        m.memory = 1e9
    m.add(90000.0, 100.0, 1.0)
    want = (10 * 250000.0 + 100 * 90000.0) / 110
    assert abs(m.mean() - want) < 1.0, (m.mean(), want)


def test_burst_does_not_outweigh_following_cycles():
    # 300 back-to-back passes of 250 ms (a warm-up flood) then 20 cycles of
    # 50 decisions with 90 ms passes: C is back near 90 ms
    d = _adaptive()
    _warm(d)
    for _ in range(300):
        _prefill_step(d, MNBT, 0.250)
    for _ in range(20):
        _cycle(d, 50, 0.090)
    _decode_steps(d, 2, 0.020)
    c = d._adapt_local_fits()[1]
    assert c < 110000.0, c


# ---- T0: the decode step's part that does not grow with the batch ----


def test_t0_is_intercept_of_decode_step():
    # decode step ~25 ms at b~300 (19 ms + 20 us a seq): a held request costs
    # the 19 ms the step does not owe to the batch, not the whole step
    d, _ = _run(60000, 19000, 20.0, 2.0)
    t0, c = d._adapt_local_fits()
    dec, k = d._adapt_dec, d._adapt_slope.slope()
    assert abs(t0 - (dec.mean() - k * dec.mx)) < 1e-6
    assert abs(c - d._adapt_pf.mean()) < 1e-6
    assert 22000 < dec.mean() < 28000, dec.mean()
    assert abs(t0 / 19000.0 - 1) < 0.15, t0
    assert abs(k / 20.0 - 1) < 0.25, k


@pytest.mark.parametrize("t0_us", [500.0, 12000.0])
def test_t0_follows_the_physical_intercept(t0_us):
    # a step that is nearly all per-seq cost has a small intercept, and T0 is
    # that (not clamped to a share of the step); with a fixed part it is that
    d, _ = _run(60000, t0_us, 60.0, 0.5)
    t0 = d._adapt_local_fits()[0]
    step = d._adapt_dec.mean()
    assert 0.0 <= t0 < step
    assert abs(t0 - t0_us) < 0.1 * step, (t0, t0_us, step)


def test_completion_steps_do_not_pull_the_intercept_to_zero():
    # every request that finishes adds 0.8 ms of completion work to its step;
    # within a cycle the batch moves only by completions. The intercept stays
    # the step's fixed part (12 ms), it does not absorb the completions
    d, _ = _run(60000, 12000, 35.0, 0.5, fin_us=800.0)
    t0 = d._adapt_local_fits()[0]
    assert abs(t0 / 12000.0 - 1) < 0.15, t0
    assert t0 > 0.5 * 12000.0, t0


def test_constant_batch_is_the_mean_step():
    # until the batch has varied there is no slope: T0 is the mean step
    d = _adaptive()
    _decode_steps(d, 200, 0.020, running=150)
    assert d._adapt_slope.slope() == 0.0
    assert abs(d._adapt_local_fits()[0] - 20000.0) < 1.0


def test_line_recovers_slope_and_intercept():
    # y = 10 ms + 40 us * x (x 100..200), CV 5%: the weighted fit finds both
    rng = random.Random(7)
    m = _AdaptiveMean()
    for i in range(20000):
        x = 150.0 + 50.0 * math.sin(i / 300.0)
        y = (10000.0 + 40.0 * x) * (1 + 0.05 * rng.gauss(0, 1))
        m.add(y, 1.0, 40.0, x=x, fit=True)
    assert abs(m.slope() / 40.0 - 1) < 0.1, m.slope()
    assert abs(m.intercept() / 10000.0 - 1) < 0.05, m.intercept()


def test_slope_kept_in_the_physical_range():
    # a falling step (noise, a phase change) gives no negative slope, and the
    # intercept is never negative
    m = _AdaptiveMean()
    for i in range(400):
        x = 100.0 + (i % 50)
        m.add(30000.0 - 100.0 * x, 1.0, 20.0, x=x, fit=True)
    assert m.slope() == 0.0 and m.intercept() == m.mean()
    m = _AdaptiveMean()
    for i in range(400):
        x = 100.0 + (i % 50)
        m.add(-20000.0 + 400.0 * x, 1.0, 20.0, x=x, fit=True)
    assert m.intercept() >= -1e-6, m.intercept()


@pytest.mark.parametrize("cycle", [20.0, 40.0])
def test_memory_resolves_the_slope_share(cycle):
    # T0 removes k * mean x from the mean step; the variance of that share is
    # mean x^2 / var x times the mean's, and the slope's memory resolves it to
    # one decision of the cycle
    rng = random.Random(8)
    m = _AdaptiveMean()
    mems = []
    for i in range(30000):
        x = 100.0 + (20.0 if (i // 7) % 2 else -20.0)
        y = 10000.0 + 50.0 * x + 1000.0 * rng.gauss(0, 1)
        m.add(y, 1.0, cycle, x=x, fit=True)
        mems.append(m.memory)
    cv2 = 1000.0**2 / 10000.0**2
    vif = 100.0**2 / 20.0**2
    want = (cv2 * vif * cycle * cycle + 1) / 2
    tail = sorted(mems[15000:])
    mem = tail[len(tail) // 2]
    assert 0.4 * want < mem < 1.3 * want, (mem, want)


# ---- memory sized by the samples ----


def _feed(m, ys, cycle):
    mems, means = [], []
    for y in ys:
        m.add(y, 1.0, cycle)
        mems.append(m.memory)
        means.append(m.mean())
    return mems, means


@pytest.mark.parametrize("cycle", [20.0, 40.0])
def test_memory_resolves_one_decision_of_the_cycle(cycle):
    # stationary samples, CV 0.3: the memory is the fewest samples whose
    # standard error is about 1/cycle of the mean (n_eff = 2M-1 = CV^2 N^2)
    rng = random.Random(1)
    ys = [10000.0 * (1 + 0.3 * rng.gauss(0, 1)) for _ in range(20000)]
    mems, means = _feed(_AdaptiveMean(), ys, cycle)
    want = (0.09 * cycle * cycle + 1) / 2
    tail_m = sorted(mems[5000:])
    mem = tail_m[len(tail_m) // 2]
    assert 0.5 * want < mem < 1.2 * want, (mem, want)
    tail = means[5000:]
    mu = sum(tail) / len(tail)
    sd = math.sqrt(sum((x - mu) ** 2 for x in tail) / len(tail))
    assert sd / mu < 1.5 / cycle, (sd / mu, 1 / cycle)


def test_effective_samples_of_the_mean():
    # an exponentially weighted mean of memory M rests on 2M-1 equal samples;
    # the innovations are standardized by that n_eff
    m = _AdaptiveMean()
    for i in range(500):
        m.memory = 10.0  # hold the memory fixed for the identity
        m.add(10000.0 + (i % 2), 1.0, 1.0)
    assert abs(m.sw * m.sw / m.sww - 19.0) < 1e-6, m.sw * m.sw / m.sww


def test_memory_shrinks_on_a_level_shift():
    # samples at 10 ms (CV 0.1) with a 100-decision cycle keep ~50 of them;
    # a jump to 15 ms makes the innovations exceed the noise, the memory
    # collapses and the mean follows within a few samples (a fixed memory of
    # 50 would take ~110 samples to get within 5%)
    rng = random.Random(2)
    m = _AdaptiveMean()
    _feed(m, [10000.0 * (1 + 0.1 * rng.gauss(0, 1)) for _ in range(3000)], 100.0)
    before = m.memory
    assert 25 < before < 80, before
    shifted = [15000.0 * (1 + 0.1 * rng.gauss(0, 1)) for _ in range(400)]
    mems, means = _feed(m, shifted, 100.0)
    settle = next(i for i, x in enumerate(means) if abs(x / 15000.0 - 1) < 0.05)
    assert settle <= 20, settle
    assert min(mems[: settle + 1]) < before / 4, (min(mems[: settle + 1]), before)
    # once the new level is the norm, the memory grows back
    assert mems[-1] > 4 * min(mems), (mems[-1], min(mems))


def test_one_outlier_does_not_freeze_the_memory():
    # pass costs ~130 ms (CV 0.07, a short memory); one 10x pass (a stall)
    # inflates the successive-difference noise. The memory regrows one sample
    # a sample instead of jumping to thousands, so a +30% load change 20
    # passes later is still followed within a few dozen passes.
    rng = random.Random(6)
    m = _AdaptiveMean()
    for _ in range(2000):
        m.add(130000.0 * (1 + 0.07 * rng.gauss(0, 1)), 40.0)
    m.add(1300000.0, 40.0)
    mems = []
    for _ in range(20):
        m.add(130000.0 * (1 + 0.07 * rng.gauss(0, 1)), 40.0)
        mems.append(m.memory)
    assert max(mems) <= 22.0, max(mems)
    settle = None
    for i in range(400):
        m.add(169000.0 * (1 + 0.07 * rng.gauss(0, 1)), 40.0)
        if settle is None and abs(m.mean() / 169000.0 - 1) < 0.05:
            settle = i
    assert settle is not None and settle <= 40, settle


def test_t0_follows_a_decode_step_jump():
    # delayer level: the plain decode step jumps from 20 to 30 ms (a load
    # change); T0 is within 10% of the new step well within one cycle
    d = _adaptive()
    rng = random.Random(3)
    for _ in range(40):
        for _ in range(49):
            _decode_steps(d, 1, 0.020 * (1 + 0.05 * rng.gauss(0, 1)))
        _prefill_step(d, 8000, 0.060)
    lag = None
    for i in range(200):
        _decode_steps(d, 1, 0.030 * (1 + 0.05 * rng.gauss(0, 1)))
        if lag is None and abs(d._adapt_local_fits()[0] / 30000.0 - 1) < 0.10:
            lag = i
    assert lag is not None and lag < 25, lag


# ---- closed loop ----


def test_pass_cost_tracks_plant_tail():
    # the decode step after a pass is x_us slower: C ~ prefill step + x_us
    d0, _ = _run(40000, 20000, 20.0, 0.5)
    d1, _ = _run(40000, 20000, 20.0, 0.5, x_us=40000.0)
    c0 = d0._adapt_local_fits()[1]
    c1 = d1._adapt_local_fits()[1]
    # the longer cycle with the tail admits larger passes (2 us/token)
    assert 35000 < c1 - c0 < 55000, (c0, c1)


@pytest.mark.parametrize("a_us,lam,x_us", [(20000, 0.5, 0.0), (20000, 0.5, 15000.0)])
def test_cycle_matches_eoq_with_measured_costs(a_us, lam, x_us):
    d, cyc = _run(a_us, 20000, c_us=20.0, lam=lam, x_us=x_us)
    t0, c = d._adapt_local_fits()
    k_eoq = math.sqrt(2 * c * 300 / (lam * t0))
    assert 0.7 * k_eoq < cyc < 1.4 * k_eoq, (cyc, k_eoq)
    reason, n, eoq_n, *_ = d._adapt_last_fire
    assert reason == "adaptive" and eoq_n == n


def test_longest_hold_is_ttft_bound():
    # a very costly pass among few arrivals: EOQ alone would hold for longer
    # than ttft_max_ticks; the existing TTFT bound releases on its tick
    fires = []
    d, _cyc = _run(800000, 20000, 20.0, 0.1, fires=fires)
    late = [n for r, n in fires[20:]]
    assert max(late) == d.ttft_max_ticks + 1, max(late)
    assert any(r == "ttft" for r, _ in fires[20:])
    assert all(r in ("adaptive", "ttft") for r, _ in fires[20:])


def test_slower_prefill_not_shorter_and_bounded():
    d, fast = _run(30000, 20000, 20.0, 0.5)
    _, slow = _run(90000, 20000, 20.0, 0.5)
    assert fast <= slow <= d.ttft_max_ticks + 1, (fast, slow)


def test_overload_paced_by_eoq_without_cap():
    # Arrivals jump past what the current cadence can take (512-token prompts,
    # 32 fill a pass; 2 per decision need a pass every <= 16 decisions). No
    # capacity cap: the leftover keeps W high from the first decision of the
    # next cycle and fewer requests decode, so EOQ shortens the cycle until
    # the passes keep up -- the queue stops growing.
    clk = _Clock()
    d = PrefillDelayer(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=MNBT,
        clock=clk,
        admit_policy="adaptive",
    )
    rng = random.Random(5)
    running, waiting = 300, 0
    gaps, gap, queue = [], 0, []
    for t in range(24000):
        lam = 0.3 if t < 8000 else 2.0
        k = int(lam) + (1 if rng.random() < lam - int(lam) else 0)
        waiting += k
        running -= k
        dec = d.should_allow_prefill(
            prefillable=waiting > 0,
            pending_tokens=min(MNBT, 512 * waiting),
            running_decode_batch=running,
            waiting_requests=waiting,
            admittable_requests=waiting,
            arrived_requests=k,
            prefill_budget=MNBT,
        )
        gap += 1
        if dec.allow and waiting > 0:
            admit = min(waiting, MNBT // 512)
            running += admit
            waiting -= admit
            d.notify_prefill_executed(512 * admit)
            clk.t += (40000 + 2.0 * 512 * admit) / 1e6
            gaps.append((t, gap))
            gap = 0
        else:
            clk.t += (20000 + 20.0 * running) / 1e6
        if t >= 8000:
            queue.append(waiting)
    before = [g for t, g in gaps if 4000 <= t < 8000]
    after = [g for t, g in gaps if t >= 16000]
    # EOQ settles at the capacity edge (one pass per 16 decisions) ...
    assert sum(after) / len(after) <= 16 * 1.02, sum(after) / len(after)
    assert sum(after) / len(after) < 0.5 * sum(before) / len(before)
    # ... with a bounded queue: that of the last 8000 decisions is no deeper
    # than that of the 8000 before them
    assert max(queue[-8000:]) <= max(queue[-16000:-8000]) + 32, (
        max(queue[-8000:]),
        max(queue[-16000:-8000]),
    )


# ---- backlog (closed loop, passes at capacity) ----


def _backlog(t0_rule=None, fixed=None, ticks=16000, seed=0, fin_us=0.0):
    """A closed-loop backlog calibrated to the measured c512 load peak: 300
    requests in the system (waiting + decoding), a pass admits at most 32 and
    stalls the batch ~282 ms, a decoding request finishes w.p. 1/190 a step and
    rejoins the queue, the decode step is 12.4 ms + 35 us a seq (fin_us more
    per completion). fixed: release every `fixed` decisions instead of the
    adaptive rule. t0_rule="mean_step": charge the whole mean step as T0.
    Returns (mean decisions between passes, mean decode batch), second half."""
    rng = random.Random(seed)
    clk = _Clock()
    d = PrefillDelayer(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=MNBT,
        clock=clk,
        admit_policy="adaptive",
    )
    if t0_rule == "mean_step":
        # the rule charging the whole mean step: a slope that never
        # sees a sample stays 0
        class _NoSlope(_AdaptiveMean):
            __slots__ = ()

            def add(self, *args, **kwargs):
                pass

        d._adapt_slope = _NoSlope()
    b, w, gap, arrived = 150, 150, 0, 0
    gaps, bs = [], []
    for _ in range(ticks):
        if fixed is not None:
            allow = gap + 1 >= fixed
        else:
            allow = d.should_allow_prefill(
                prefillable=w > 0,
                pending_tokens=MNBT,
                running_decode_batch=b,
                waiting_requests=w,
                admittable_requests=w,
                arrived_requests=arrived,
                prefill_budget=MNBT,
            ).allow
        gap += 1
        if allow and w > 0:
            admit = min(w, 32)
            w -= admit
            b += admit
            d.notify_prefill_executed(MNBT)
            clk.t += 0.282 * (1 + 0.05 * rng.gauss(0, 1))
            gaps.append(gap)
            gap = 0
        else:
            fin = sum(1 for _ in range(b) if rng.random() < 1.0 / 190)
            clk.t += (12400.0 + 35.0 * b + fin_us * fin + 300 * rng.gauss(0, 1)) / 1e6
            b -= fin
            w += fin
            arrived = fin
        bs.append(b)
    g, bb = gaps[len(gaps) // 2 :], bs[len(bs) // 2 :]
    return sum(g) / len(g), sum(bb) / len(bb)


def test_backlog_admits_at_least_as_often_as_the_fixed_interval():
    # the product rule C*(B + W) alone held the c512 backlog past the fixed
    # rule's 40 decisions (44-55 a pass, throughput -5.3%). In a sustained
    # backlog the pass is weighed by the throughput rule sum W*T_d >= C*B: it admits
    # at least as often as the fixed rule, so the queue drains no slower.
    gap7, _ = _backlog()
    gapf, _ = _backlog(fixed=40)
    assert gap7 <= gapf, (gap7, gapf)


def test_backlog_completions_leave_the_cycle_alone():
    # completion work (0.8 ms a finished request) does not move T0 to 0 and so
    # does not stretch the backlog cycle
    gap0, _ = _backlog()
    gap1, _ = _backlog(fin_us=800.0)
    assert abs(gap1 / gap0 - 1) < 0.1, (gap0, gap1)


def test_floor_backlog_runs_the_fixed_cadence(monkeypatch):
    # with the fixed rule's cadence as the floor (the shipped 40), the
    # backlog cycle is the fixed rule's: never more often (decode keeps the
    # fixed rule's interactivity), and the throughput rule fires on the
    # floor, so never less often either
    monkeypatch.setattr(pd_mod, "_FLOOR_TICKS", _DEFAULT_FLOOR)
    gap7, b7 = _backlog()
    gapf, bf = _backlog(fixed=40)
    assert gapf <= gap7 <= 1.02 * gapf, (gap7, gapf)
    assert abs(b7 / bf - 1) < 0.03, (b7, bf)


def test_floor_default_is_the_fixed_cadence():
    # a built-in default (the fixed interval rule's 40), not an env knob
    assert _DEFAULT_FLOOR == 40
    import inspect

    assert "MIN_TICKS" not in inspect.getsource(pd_mod)
    assert "MAX_HOLD_MS" not in inspect.getsource(pd_mod)


def test_floor_holds_even_when_the_rule_would_fire(monkeypatch):
    monkeypatch.setattr(pd_mod, "_FLOOR_TICKS", _DEFAULT_FLOOR)
    d = _adaptive()
    red = _zeros(d)
    at = d._adapt_at
    # W 1000 against B 10: either rule would fire on the first decision
    red[at : at + 6] = [1000, 1000, 100000, 1, 1, 2000]
    d._adapt_backlog_prev = True
    n = 0
    while not d._adapt_decide(list(red), 10, 0):
        n += 1
    assert n + 1 == 40, n


def test_sustained_backlog_uses_the_throughput_rule():
    # this pass and the one before left work behind: sum W*T_d >= C*B
    d = _adaptive()
    red = _zeros(d)
    at = d._adapt_at
    # W 10, T0 1 ms, C 100 ms, 1 rank, backlog on 1 rank, T_d 2 ms
    red[at : at + 6] = [10, 1000, 100000, 1, 1, 2000]
    d._adapt_backlog_prev = True
    n = 0
    while True:
        n += 1
        if d._adapt_decide(list(red), 90, 0):
            break
    # 10 * 2 ms a decision against 100 ms * 90
    assert n == 450, n
    assert d._adapt_last_fire[0] == "backlog"


@pytest.mark.parametrize(
    "backlog_now,backlog_prev", [(1, False), (0, True), (0, False)]
)
def test_one_pass_leftover_keeps_the_product_rule(backlog_now, backlog_prev):
    # a single pass that left work (one long burst, a chunked prompt) is no
    # backlog yet, nor is a backlog the last pass cleared: the product rule
    # sum W*T0 >= C*(B + W) still decides
    d = _adaptive()
    red = _zeros(d)
    at = d._adapt_at
    red[at : at + 6] = [10, 1000, 100000, 1, backlog_now, 2000]
    d._adapt_backlog_prev = backlog_prev
    n = 0
    while True:
        n += 1
        if d._adapt_decide(list(red), 90, 0):
            break
    assert n == 1000, n


def test_sustained_backlog_respects_the_floor(monkeypatch):
    # the throughput rule would fire on decision 5 (W 1000 * 2 ms against
    # C 100 ms * B 10 at decision 1), the floor holds it to the 40th
    monkeypatch.setattr(pd_mod, "_FLOOR_TICKS", _DEFAULT_FLOOR)
    d = _adaptive()
    red = _zeros(d)
    at = d._adapt_at
    red[at : at + 6] = [1000, 1, 100000, 1, 1, 2000]
    d._adapt_backlog_prev = True
    n = 0
    while True:
        n += 1
        if d._adapt_decide(list(red), 10, 0):
            break
    assert n == _DEFAULT_FLOOR, n
    assert d._adapt_last_fire[0] == "backlog"


def test_fire_restarts_arrivals_and_keeps_the_reduced_backlog():
    # on a FIRE every rank restarts its arrival count and remembers the
    # group's (reduced) backlog flag, so the next cycle's rule choice is the
    # same on every rank; a HOLD changes neither
    d = _adaptive()
    at = d._adapt_at
    red = _zeros(d)
    red[at + 4] = 3  # three ranks reported a backlog
    d._adapt_arr_since_fire = 7
    d._adapt_backlog_prev = False
    d._hold_ticks = 0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PrefillDelayer, "_decide_inner", lambda self, r, s: False)
        assert not d._decide(list(red), 0.0)
        assert d._adapt_arr_since_fire == 7 and not d._adapt_backlog_prev
        mp.setattr(PrefillDelayer, "_decide_inner", lambda self, r, s: True)
        assert d._decide(list(red), 0.0)
        assert d._adapt_arr_since_fire == 0 and d._adapt_backlog_prev
        red[at + 4] = 0
        assert d._decide(list(red), 0.0)
        assert not d._adapt_backlog_prev


def test_backlog_flag_counts_requests_older_than_the_fire():
    d = _adaptive()
    at = d._adapt_at

    def tick(waiting, arrived, partial=False):
        d.should_allow_prefill(
            prefillable=waiting > 0,
            pending_tokens=1000 * waiting,
            running_decode_batch=50,
            has_partial=partial,
            waiting_requests=waiting,
            admittable_requests=waiting,
            arrived_requests=arrived,
            prefill_budget=MNBT,
        )
        return int(d._reduce_buf[at + 4])

    d._decide([0] * d._reduce_buf.numel(), 0.0)  # first call fires
    d._adapt_arr_since_fire = 0
    assert tick(2, 2) == 0  # both arrived since the FIRE
    assert tick(3, 1) == 0
    d._adapt_arr_since_fire = 0  # a FIRE: the pass took 2 of 3
    assert tick(1, 0) == 1  # one request older than the FIRE still waits
    d._adapt_arr_since_fire = 0
    assert tick(1, 0, partial=True) == 0  # only a chunked prompt's remainder


def test_threshold_counts_waiting_and_decoding():
    # the pass is weighed over the B decoding and W waiting requests: with the
    # same T0 and C, a decision fires when sum W*T0 reaches C*(B + W)
    d = _adaptive()
    red = _zeros(d)
    at = d._adapt_at
    red[at : at + 4] = [10, 1000, 100000, 1]  # W 10, T0 1 ms, C 100 ms, 1 rank
    n = 0
    while True:
        n += 1
        if d._adapt_decide(list(red), 90, 0):
            break
    # 10 * 1 ms a decision against 100 ms * (90 + 10)
    assert n == 1000, n


# ---- decode mean T0 ----


def test_decode_spike_clipped_at_twice_mean():
    clipped, ref = _adaptive(), _AdaptiveMean()
    _decode_steps(clipped, 66, 0.020)
    for _ in range(64):  # the first two steps after start time no sample
        ref.add(20000.0, 1.0, 1.0)
    _decode_steps(clipped, 1, 2.0)  # a stall
    ref.add(40000.0, 1.0, 1.0)
    assert abs(clipped._adapt_dec.mean() - ref.mean()) < 1e-6


# ---- DP lockstep ----


def _lockstep(arrive_p, cap=None, ticks=1500):
    """DP ranks (one per entry of arrive_p), each timing its own steps (rank r
    is 15% slower per r) and seeing its own arrivals (w.p. arrive_p[r] a
    tick); a pass admits at most `cap` requests per rank (None: all). Returns
    (delayers, per-rank allow lists)."""
    import threading
    from unittest import mock

    import torch

    R = len(arrive_p)
    shared = [None] * R
    barrier = threading.Barrier(R)

    class _Group:
        def __init__(self, r):
            self.r = r

        def rank(self):
            return self.r

    def fake_all_reduce(tensor, op=None, group=None):
        shared[group.r] = tensor.clone()
        barrier.wait()
        total = torch.stack(shared).sum(0)
        barrier.wait()
        tensor.copy_(total)

    allows = [[] for _ in range(R)]
    delayers = [None] * R

    def rank_loop(r):
        rng = random.Random(10 + r)
        clk = _Clock()
        d = PrefillDelayer(
            dp_size=R,
            cpu_group=_Group(r),
            max_num_batched_tokens=MNBT,
            exchange_prefill_costs=True,
            clock=clk,
            admit_policy="adaptive",
        )
        delayers[r] = d
        slow = 1.0 + 0.15 * r
        running, waiting = 60, 0
        for _ in range(ticks):
            k = 1 if rng.random() < arrive_p[r] else 0
            waiting += k
            running -= min(k, running)
            dec = d.should_allow_prefill(
                prefillable=waiting > 0,
                pending_tokens=min(MNBT, 400 * waiting),
                running_decode_batch=running,
                waiting_requests=waiting,
                admittable_requests=waiting,
                arrived_requests=k,
                prefill_budget=MNBT,
            )
            allows[r].append(dec.allow)
            if dec.allow and waiting > 0:
                admit = waiting if cap is None else min(waiting, cap)
                d.notify_prefill_executed(400 * admit)
                clk.t += slow * (0.05 + 2e-6 * 400 * admit)
                running += admit
                waiting -= admit
            else:
                clk.t += slow * (0.010 + 2e-5 * running) * (1 + 0.05 * rng.gauss(0, 1))

    with mock.patch("torch.distributed.all_reduce", fake_all_reduce):
        threads = [threading.Thread(target=rank_loop, args=(r,)) for r in range(R)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
    assert all(len(a) == ticks for a in allows)
    return delayers, allows


def test_ranks_decide_alike_on_their_own_clocks():
    # their T0/C differ, but the rule reads only the SUM-reduced values, so
    # every rank FIREs on the same decisions
    delayers, allows = _lockstep([0.3] * 4)
    for r in range(1, 4):
        assert allows[r] == allows[0], r
    assert delayers[0]._stat_fire_adaptive > 10, delayers[0]._stat_fire_adaptive
    for r in range(1, 4):
        assert delayers[r]._stat_fire_adaptive == delayers[0]._stat_fire_adaptive
        assert delayers[r]._adapt_regret == delayers[0]._adapt_regret
    t0s = [d._adapt_local_fits()[0] for d in delayers]
    assert max(t0s) > 1.3 * min(t0s), t0s  # the local estimates do differ


def test_ranks_decide_alike_when_one_rank_has_a_backlog(monkeypatch):
    # only rank 0 is overloaded (arrivals above what a capped pass takes), so
    # only it reports a backlog; the flag rides the SUM all_reduce and every
    # rank switches to the throughput rule (and back) on the same decisions,
    # with the shipped floor
    monkeypatch.setattr(pd_mod, "_FLOOR_TICKS", _DEFAULT_FLOOR)
    delayers, allows = _lockstep([0.9, 0.2, 0.2, 0.2], cap=8, ticks=3000)
    for r in range(1, 4):
        assert allows[r] == allows[0], r
    d0 = delayers[0]
    assert d0._stat_fire_backlog > 5, d0._stat_fire_backlog
    for d in delayers[1:]:
        assert d._stat_fire_backlog == d0._stat_fire_backlog
        assert d._stat_fire_adaptive == d0._stat_fire_adaptive
        assert d._adapt_backlog_prev == d0._adapt_backlog_prev
        assert d._adapt_regret_td == d0._adapt_regret_td
    # no cycle shorter than the floor
    fires = [i for i, a in enumerate(allows[0]) if a]
    gaps = [b - a for a, b in itertools.pairwise(fires[1:])]
    assert gaps and min(gaps) >= _DEFAULT_FLOOR, min(gaps)


# ---- wiring of the slope into the delayer ----


def _line_steps(d, n, a_us, k_us, xs, seed=0):
    """n plain decode steps whose time is a_us + k_us * (the batch reported at
    the decision before the step, i.e. the batch the step ran), 300 us noise."""
    rng = random.Random(seed)
    zeros = _zeros(d)
    prev = xs(0)
    d._adapt_observe(zeros, 0.0, False, prev)
    for i in range(1, n + 1):
        x = xs(i)
        y = a_us + k_us * prev + rng.gauss(0, 300)
        d._adapt_observe(zeros, y / 1e6, False, x)
        prev = x


def test_slope_is_fit_on_the_batch_the_step_ran():
    # the step timed at a decision ran the batch reported at the decision
    # before it; with the batch alternating every step the fit must pair them
    # so, or it sees no slope (T0 = T_d) or a falling one
    d = _adaptive()
    _line_steps(d, 4000, 10000.0, 50.0, lambda i: 100 if i % 2 else 200)
    t0 = d._adapt_local_fits()[0]
    assert abs(d._adapt_slope.slope() / 50.0 - 1) < 0.15, d._adapt_slope.slope()
    assert abs(t0 / 10000.0 - 1) < 0.1, t0


def test_delayer_t0_kept_within_the_step():
    # the delayer's T0 uses the clipped slope: a step falling with the batch
    # gives T0 = T_d (not above it), a steeper-than-proportional one T0 >= 0
    d = _adaptive()
    _line_steps(d, 3000, 30000.0, -100.0, lambda i: 100 + i % 50)
    t0 = d._adapt_local_fits()[0]
    assert abs(t0 - d._adapt_dec.mean()) < 1e-6, (t0, d._adapt_dec.mean())
    d = _adaptive()
    _line_steps(d, 3000, -20000.0, 400.0, lambda i: 100 + i % 50)
    assert 0.0 <= d._adapt_local_fits()[0] < 0.05 * d._adapt_dec.mean()


def test_slope_memory_is_sized_by_its_own_rule():
    # the delayer's slope estimate is an _AdaptiveMean fed the same samples
    # with the measured cycle: its memory is not overridden or shared with the
    # fast decode mean (a fixed or borrowed memory is a tuned constant)
    rng = random.Random(11)
    d = _adaptive()
    _warm(d)
    for _ in range(12):
        _prefill_step(d, 8000, 0.050)
        _decode_steps(d, 39, 0.020)
    ys, xs = [], []
    zeros = _zeros(d)
    prev = 100
    for i in range(6000):
        x = 100 + (40 if (i // 9) % 2 else 0)
        y = (12000.0 + 60.0 * prev) * (1 + 0.04 * rng.gauss(0, 1))
        d._adapt_observe(zeros, y / 1e6, False, x)
        ys.append(y)
        xs.append(prev)
        prev = x
    s = d._adapt_slope
    ref = _AdaptiveMean()
    # the decode samples before the line ran at 20 ms and batch 100
    cycle = max(d._adapt_pf.mean_weight(), 1.0)
    for _ in range(s.n - len(ys)):
        ref.add(20000.0, 1.0, cycle, x=100.0, fit=True)
    for y, x in zip(ys, xs):
        ref.add(min(y, 2.0 * d._adapt_dec.mean()), 1.0, cycle, x=float(x), fit=True)
    assert s.memory != d._adapt_dec.memory
    assert abs(s.memory / ref.memory - 1) < 0.05, (s.memory, ref.memory)
    assert s.memory > 3 * d._adapt_dec.memory, (s.memory, d._adapt_dec.memory)


def test_measured_zero_intercept_counts_as_measured():
    # a rank whose step is all per-seq cost measures T0 = 0: it still counts
    # among the measured ranks (with 0), or the group mean T0 would skip it
    # and charge holds as if its T0 were the others'
    d = _adaptive()
    _warm(d)
    _prefill_step(d, 8000, 0.050)
    _decode_steps(d, 2, 0.020)
    # the slope at (or past) its bound T_d / B: T0 = 0

    class _Steep(_AdaptiveMean):
        __slots__ = ()

        def slope(self):
            return 1e9

    d._adapt_slope = _Steep()
    t0, c = d._adapt_local_fits()
    assert int(t0) == 0 and c > 0 and d._adapt_dec.mean() > 0, (t0, c)
    d.should_allow_prefill(
        prefillable=True,
        pending_tokens=MNBT,
        running_decode_batch=120,
        waiting_requests=5,
        admittable_requests=5,
        prefill_budget=MNBT,
    )
    at = d._adapt_at
    assert d._reduce_buf[at + 3] == 1 and d._reduce_buf[at + 1] == 0
