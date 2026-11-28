# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Flag ATOM_SCHED_ITL_RISK_MS: hold prefill while its stall would push a
# running request across the target per-request ITL
# (atom/model_engine/prefill_itl_risk.py).
import time
from collections import deque
from types import SimpleNamespace
from unittest import mock

from conftest import MockConfig

from atom.model_engine.prefill_itl_risk import ItlRiskHold
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus, SequenceType
from atom.sampling_params import SamplingParams


def _seq(i, max_tokens, done, first_age_s, now=1000.0, partial=False):
    return SimpleNamespace(
        id=i,
        max_tokens=max_tokens,
        num_completion_tokens=done,
        is_partial_prefill=partial,
        first_token_time=(now - first_age_s) if first_age_s is not None else 0.0,
    )


def _gate(theta_ms=5.0, fixed=31.6, per_k=18.65):
    g = ItlRiskHold(theta_ms, 0.3, fixed, per_k)
    g.decode_step_s = 0.012  # 12 ms decode step
    g.tokens_per_step = 4.0
    return g


def test_due_only_for_requests_crossing_theta():
    g = _gate()
    now = 1000.0
    # 101 tokens, 50 done in 0.15 s: 13 steps left -> base (0.15 + 0.156)/100 =
    # 3.06 ms/token; a 1K-token stall (31.6 + 18.65 ms) adds 0.50 ms -> 3.56 < 5.
    assert not g.due(now, [_seq(1, 101, 50, 0.15)], 1000)
    # Same request, older: base (0.30 + 0.156)/100 = 4.56 ms; +0.50 -> 5.06 > 5.
    assert g.due(now, [_seq(1, 101, 50, 0.30)], 1000)
    # Already above theta: lost either way, not held for.
    assert not g.due(now, [_seq(1, 101, 50, 0.60)], 1000)
    # A long request absorbs the stall: base 4.6 ms, +50 ms/999 = 0.05 ms.
    assert not g.due(now, [_seq(1, 1000, 500, 2.0)], 1000)
    # A bigger pending prefill costs more: base (0.20 + 0.156)/100 = 3.56 ms;
    # 1K tokens (+0.50) stays below theta, 8K tokens (184 ms, +1.84) crosses.
    assert not g.due(now, [_seq(1, 101, 50, 0.20)], 1000)
    assert g.due(now, [_seq(1, 101, 50, 0.20)], 8192)
    # Partial prefills, requests with no first token yet and 1-token requests
    # are not decodes at risk.
    assert not g.due(now, [_seq(1, 101, 50, 0.30, partial=True)], 1000)
    assert not g.due(now, [_seq(1, 101, 0, None)], 1000)
    assert not g.due(now, [_seq(1, 1, 0, 0.30)], 1000)
    assert g.due(now, [_seq(2, 1000, 500, 2.0), _seq(1, 101, 50, 0.30)], 1000)
    assert not g.due(now, [], 1000)


def test_not_due_before_rates_are_measured():
    g = ItlRiskHold(5.0, 0.3, 31.6, 18.65)
    assert not g.due(1000.0, [_seq(1, 101, 50, 0.30)], 8192)


def test_observe_learns_rates_from_consecutive_decode_ticks():
    g = ItlRiskHold(5.0, 0.3, 31.6, 18.65)
    a, b = _seq(1, 400, 10, 1.0), _seq(2, 400, 20, 1.0)
    t = 10.0
    for _ in range(3):  # warm-up: needs three decode ticks in a row
        g.observe(t, "d", [a, b])
        t += 0.012
        a.num_completion_tokens += 4
        b.num_completion_tokens += 4
    assert abs(g.decode_step_s - 0.012) < 1e-9
    assert abs(g.tokens_per_step - 4.0) < 1e-9
    # A prefill tick and the two decode ticks after it are not samples (the
    # interval after a prefill tick carries the prefill step).
    g.observe(t, "p", [a, b])
    t += 0.150
    g.observe(t, "d", [a, b])
    t += 0.150
    g.observe(t, "d", [a, b])
    assert abs(g.decode_step_s - 0.012) < 1e-9
    # Clean again: samples move the averages.
    t += 0.022
    a.num_completion_tokens += 6
    b.num_completion_tokens += 6
    g.observe(t, "d", [a, b])
    assert 0.012 < g.decode_step_s < 0.022
    assert 4.0 < g.tokens_per_step < 6.0


def test_hold_is_bounded_and_resets():
    g = _gate()
    assert g.hold(10.0, True)
    assert g.hold(10.2, True)
    assert not g.hold(10.3, True)  # bound (0.3 s) reached: one tick released
    assert g.releases == 1
    assert g.hold(10.4, True)
    assert not g.hold(10.5, False)
    assert g.hold(20.0, True)


def _sched(**env):
    with mock.patch.dict("os.environ", {k: str(v) for k, v in env.items()}):
        return Scheduler(
            MockConfig(
                max_num_seqs=4,
                max_num_batched_tokens=512,
                max_model_len=4096,
                num_kvcache_blocks=400,
                enable_chunked_prefill=True,
            )
        )


def _decoding(seq_factory, max_tokens, done, first_age_s):
    seq = seq_factory(
        [1, 2, 3, 4], sampling_params=SamplingParams(max_tokens=max_tokens)
    )
    seq.status = SequenceStatus.RUNNING
    seq.type = SequenceType.DECODE
    for _ in range(done):
        seq.append_token(7)
    seq.first_token_time = time.time() - first_age_s
    return seq


def test_off_by_default():
    assert _sched()._itl_risk is None


def test_holds_prefill_for_request_at_risk(seq_factory):
    sched = _sched(ATOM_SCHED_ITL_RISK_MS=5.0, ATOM_SCHED_ITL_RISK_MAX_MS=300)
    assert sched._itl_risk is not None
    sched._itl_risk.decode_step_s = 0.012
    sched._itl_risk.tokens_per_step = 4.0
    # 101-token request, 50 done, first token 0.33 s ago: base 4.86 ms/token,
    # a 64-token prefill (32.8 ms) adds 0.33 -> crosses 5 ms: at risk.
    sched.running = deque([_decoding(seq_factory, 101, 50, 0.33)])
    sched.add(seq_factory(list(range(64))))
    batch, _ = sched.schedule()
    assert batch.total_seqs_num_prefill == 0
    assert batch.total_seqs_num_decode == 1
    assert len(sched.waiting) == 1
    # Past the bound the prefill goes.
    with mock.patch(
        "atom.model_engine.scheduler.time.perf_counter",
        return_value=sched._itl_risk._since + 1.0,
    ):
        batch, _ = sched.schedule()
    assert batch.total_seqs_num_prefill == 1


def test_no_hold_when_nobody_crosses(seq_factory):
    for done, age in ((50, 0.15), (50, 0.60)):  # far below / already above
        sched = _sched(ATOM_SCHED_ITL_RISK_MS=5.0)
        sched._itl_risk.decode_step_s = 0.012
        sched._itl_risk.tokens_per_step = 4.0
        sched.running = deque([_decoding(seq_factory, 101, done, age)])
        sched.add(seq_factory(list(range(64))))
        batch, _ = sched.schedule()
        assert batch.total_seqs_num_prefill == 1


def test_no_hold_without_decode(seq_factory):
    sched = _sched(ATOM_SCHED_ITL_RISK_MS=5.0)
    sched._itl_risk.decode_step_s = 0.012
    sched._itl_risk.tokens_per_step = 4.0
    sched.add(seq_factory(list(range(64))))
    batch, _ = sched.schedule()
    assert batch.total_seqs_num_prefill == 1


def test_due_is_inclusive_at_theta():
    # base == theta exactly (1.0 s over 100 gaps, 10 ms): still at risk, since
    # any stall pushes it above. A `<` instead of `<=` would miss it.
    g = ItlRiskHold(10.0, 0.3, 31.6, 18.65)
    g.decode_step_s = 0.012
    g.tokens_per_step = 4.0
    assert g.due(1000.0, [_seq(1, 101, 101, 1.0)], 1000)


def test_zero_max_hold_never_holds():
    g = ItlRiskHold(5.0, 0.0, 31.6, 18.65)
    assert not g.hold(10.0, True)
    assert not g.hold(10.1, True)
    assert g.holds == 0
