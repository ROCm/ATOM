# SPDX-License-Identifier: MIT
"""Online STS: censored calibration fit and verdict pairing (CPU only)."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from atom.spec_decode.online_sts import OnlineSTS, censored_ece, fit_temperatures

TRUE = np.array([2.0, 1.5, 0.7, 1.0, 0.85])


def simulate(n, temperatures, seed=0, censor=True):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.05, 0.95, size=(n, 5))
    logit = np.log(c) - np.log1p(-c)
    p = 1 / (1 + np.exp(-logit / temperatures))
    ok = rng.uniform(size=(n, 5)) < p
    accepted = np.where(ok.all(1), 5, ok.argmin(1))
    # The scheduler verified a prefix; drafts past it are censored.
    verified = rng.integers(1, 6, size=n) if censor else np.full(n, 5)
    return c, verified, np.minimum(accepted, verified)


def test_fit_recovers_temperatures_under_censoring():
    c, v, a = simulate(40000, TRUE)
    fitted, positions = fit_temperatures(c, v, a, np.ones(5))
    assert positions == [0, 1, 2, 3, 4]
    # Early positions are well identified; later ones see fewer survivors, so
    # the check there is the objective itself: as calibrated as the truth.
    assert fitted[:2].tolist() == TRUE[:2].tolist()
    ece = censored_ece(c, v, a, fitted)
    assert ece <= censored_ece(c, v, a, TRUE) + 0.002
    assert ece < censored_ece(c, v, a, np.ones(5)) / 5


def test_censored_positions_are_not_negatives():
    """Truncating at the verified length must not read as rejection: with the
    same draws, censoring leaves the fit where full observation puts it."""
    c, v, a = simulate(40000, np.ones(5))
    fitted = fit_temperatures(c, v, a, np.ones(5))[0]
    assert fitted[:2].tolist() == [1.0, 1.0]
    assert censored_ece(c, v, a, fitted) <= censored_ece(c, v, a, np.ones(5))


def test_positions_without_enough_observations_keep_their_start():
    c, v, a = simulate(2000, TRUE)
    v[:] = 1
    a = np.minimum(a, 1)
    fitted, positions = fit_temperatures(c, v, a, np.ones(5), min_rows=200)
    assert positions == [0]
    assert fitted[1:].tolist() == [1.0] * 4


class Event:
    def query(self):
        return True

    def record(self, stream=None):
        pass


def test_verdicts_pair_with_the_drafts_two_forwards_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: NS(wait_stream=None))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(
        torch.cuda, "stream", lambda s: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(
        torch.Tensor, "pin_memory", lambda self, *a, **k: self, raising=False
    )
    real_empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *a, pin_memory=False, **k: real_empty(*a, **k)
    )
    stream = NS(wait_stream=lambda other: None)
    scheduler = NS(
        runner=NS(tokenID_processor=NS(async_copy_stream=stream), device="cpu"),
        sts_temperatures=None,
    )
    sts = OnlineSTS(
        scheduler, 5, group=object(), sample_interval=1, refit_interval=1000
    )
    conf = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
    sts.observe()  # forward 1 drafts for requests 7, 8
    sts.capture(conf)
    sts.record([7, 8])
    sts.observe([9], [0], [1])  # forward 2: verdict of forward 1's (other) drafts
    assert not sts.rows
    sts.capture(conf * 0)
    sts.record([7, 8])
    # Forward 3 reports forward 2's verification of forward 1's drafts: the
    # verified lengths are the ragged ones, not the full reservation (5).
    sts.observe([8, 7], np.array([0, 3]), np.array([2, 4]))
    rows = {round(float(r[0][0]), 1): r[1:] for r in sts.rows}
    assert rows == {0.9: (4, 3), 0.1: (2, 0)}


def test_free_buffers_are_reused_by_position_not_by_comparison(monkeypatch):
    """Two pooled buffers of different heights: picking the second must not
    compare tensors elementwise (that raised a size mismatch in serving)."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: NS(wait_stream=None))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(
        torch.cuda, "stream", lambda s: __import__("contextlib").nullcontext()
    )
    real_empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *a, pin_memory=False, **k: real_empty(*a, **k)
    )
    scheduler = NS(
        runner=NS(
            tokenID_processor=NS(async_copy_stream=NS(wait_stream=lambda o: None)),
            device="cpu",
        ),
        sts_temperatures=None,
    )
    sts = OnlineSTS(scheduler, 5, group=object(), sample_interval=1)
    sts.free = [torch.zeros(8, 5), torch.zeros(16, 5)]
    sts.capture(torch.rand(10, 5))
    sts.record(list(range(10)))
    assert len(sts.free) == 1 and sts.free[0].shape[0] == 8
    assert sts.pending[0][2].shape[0] == 16


def test_temperatures_publish_as_the_evidence_weighted_mean(monkeypatch):
    """No DP peers here: the published curve is the local fit, positions
    below the evidence floor stay at 1."""
    scheduler = NS(runner=NS(device="cpu"), sts_temperatures=None)
    sts = OnlineSTS(scheduler, 5, group=object(), min_rows=10)
    sts.rows.extend((np.full(5, 0.5), 2, 1) for _ in range(20))
    sts._fitted = np.array([2.0, 1.5, 0.7, 1.0, 0.85])
    sts._publish()
    assert sts.temperatures.tolist() == [2.0, 1.5, 1.0, 1.0, 1.0]
    assert scheduler.sts_temperatures.tolist() == pytest.approx([2, 1.5, 1, 1, 1])


def test_an_unrecorded_capture_is_dropped_at_the_next_forward(monkeypatch):
    """A forward that captured confidence but never recorded it (warmup, a
    skipped draft) must not lend it to the next forward's request ids."""
    scheduler = NS(runner=NS(device="cpu"), sts_temperatures=None)
    sts = OnlineSTS(scheduler, 5, group=object(), sample_interval=1)
    sts.capture(torch.rand(3, 5))
    sts.observe()
    called = []
    monkeypatch.setattr(torch.cuda, "Event", lambda: called.append(1))
    sts.record([1, 2, 3])
    assert not sts.pending and not called
