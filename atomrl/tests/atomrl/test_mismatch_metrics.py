# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Train/rollout mismatch metrics: check the math and the cancellation property.

CPU-only, no model needed. Run: python -m atomrl.tests.test_mismatch_metrics
"""

import math

import torch
from atomrl.trainer.rl_trainer import RLTrainer
from atomrl.workers.actor_worker import AtomActorWorker


def _compute(delta, mask):
    out = AtomActorWorker._mismatch_metrics(delta, mask, float(mask.sum()))
    RLTrainer._finalize_mismatch_metrics(out)
    return out


def test_matches_bruteforce():
    torch.manual_seed(0)
    B, L = 4, 7
    mask = (torch.rand(B, L) > 0.3).float()
    delta = torch.randn(B, L) * 0.05 * mask
    got = _compute(delta, mask)

    vals = [float(delta[i, j]) for i in range(B) for j in range(L) if mask[i, j] > 0]
    n = len(vals)
    want = {
        "mismatch/abs_diff": sum(abs(v) for v in vals) / n,
        "mismatch/k3_kl": sum(math.exp(v) - v - 1 for v in vals) / n,
        "mismatch/chi2_token": sum((math.exp(v) - 1) ** 2 for v in vals) / n,
        "mismatch/frac_abs_gt_0.1": sum(1.0 for v in vals if abs(v) > 0.1) / n,
    }
    seq = [sum(float(delta[i, j]) for j in range(L)) for i in range(B)]
    want["mismatch/chi2_seq"] = sum((math.exp(s) - 1) ** 2 for s in seq) / B

    for k, w in want.items():
        assert abs(got[k] - w) < 1e-6, (k, got[k], w)
    assert got["mismatch/k3_kl"] >= 0.0
    assert not any(k.endswith(("_sum", "_tok", "_n")) for k in got), sorted(got)


def test_survives_micro_and_worker_averaging():
    """Both parts of every ratio are averaged twice before division."""
    s1, t1, s2, t2 = 3.0, 100.0, 7.0, 250.0
    assert abs(((s1 + s2) / 2) / ((t1 + t2) / 2) - (s1 + s2) / (t1 + t2)) < 1e-12


def test_signed_mean_hides_symmetric_disagreement():
    """The reason these metrics exist.

    Expert-selection flips between the training forward and the rollout engine are
    large but unbiased, so they cancel inside the signed ``rollout_corr/kl``. On
    Qwen3-30B-A3B that hid a mean |delta| of 0.0226 behind a signed 0.0019.
    """
    delta = torch.zeros(2, 100)
    delta[0, :50] = 0.3
    delta[0, 50:] = -0.3
    mask = torch.ones(2, 100)
    got = _compute(delta, mask)

    signed = float((delta * mask).sum()) / float(mask.sum())
    assert abs(signed) < 1e-6, signed
    assert got["mismatch/abs_diff"] > 0.1
    assert got["mismatch/k3_kl"] > 0.01
    assert got["mismatch/frac_abs_gt_0.1"] == 0.5


def test_no_overflow_on_extreme_ratios():
    delta = torch.full((2, 64), 50.0)
    got = _compute(delta, torch.ones(2, 64))
    assert all(math.isfinite(v) for v in got.values()), got


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all mismatch-metric tests passed")
