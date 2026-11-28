# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from atom.model_engine.prefill_duty import PrefillDuty


def _run(duty, steps, state=None):
    """steps: (kind, device seconds) in schedule order; returns the hold
    decisions taken before each. As under deferred output, the wall time
    between schedule calls k and k+1 is the device time of step k-1."""
    state = state if state is not None else {"t": 0.0, "prev": 0.0}
    holds = []
    for kind, dt in steps:
        holds.append(duty.hold(state["t"], decoding=True))
        duty.observe(state["t"], kind)
        state["t"] += state["prev"]
        state["prev"] = dt
    return holds


def test_off_below_target():
    d = PrefillDuty(0.5, tau_s=1.0, max_hold_s=2.0)
    holds = _run(d, [("d", 0.01)] * 50)
    assert not any(holds)
    assert d.share() == 0.0


def test_holds_after_prefill_burst_and_recovers():
    d = PrefillDuty(0.5, tau_s=1.0, max_hold_s=5.0)
    st = {"t": 0.0, "prev": 0.0}
    _run(d, [("d", 0.01)] * 20 + [("p", 0.2)] * 5, st)
    # Two observations later the burst is charged: share is well above target.
    _run(d, [("d", 0.01)] * 2, st)
    assert d.share() > 0.5
    holds = _run(d, [("d", 0.01)] * 400, st)
    assert holds[0]
    # Decode time accumulates until the share falls back under the target.
    assert not holds[-1]
    assert d.share() <= 0.5


def test_no_hold_without_decode():
    d = PrefillDuty(0.1, tau_s=1.0, max_hold_s=5.0)
    _run(d, [("p", 0.2)] * 10)
    assert d.share() > 0.9
    assert not d.hold(10.0, decoding=False)


def test_max_wait_overrides_hold():
    d = PrefillDuty(0.01, tau_s=100.0, max_hold_s=1000.0, max_wait_s=3.0)
    _run(d, [("p", 1.0)] * 3 + [("d", 0.01)] * 2)
    assert d.hold(100.0, decoding=True, oldest_wait_s=lambda: 1.0)
    assert not d.hold(100.1, decoding=True, oldest_wait_s=lambda: 3.5)
    # Without the callable the bound cannot apply.
    assert d.hold(100.2, decoding=True)


def test_max_hold_releases():
    d = PrefillDuty(0.01, tau_s=100.0, max_hold_s=0.5)
    _run(d, [("p", 1.0)] * 3 + [("d", 0.01)] * 2)
    t0 = 100.0
    assert d.hold(t0, decoding=True)
    assert d.hold(t0 + 0.4, decoding=True)
    assert not d.hold(t0 + 0.6, decoding=True)


def test_zero_max_hold_never_holds():
    d = PrefillDuty(target=0.1, tau_s=10.0, max_hold_s=0.0)
    for i in range(6):
        d.observe(float(i), "p")
    assert d.share() > d.target
    assert not d.hold(10.0, True)
    assert not d.hold(10.1, True)
