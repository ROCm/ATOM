# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.algorithms.loss_functions import (
    agg_loss,
    asymmetric_clip_loss,
    entropy_bonus,
    kl_penalty,
    policy_gradient_loss,
)


def test_agg_loss_token_mean_matches_masked_reference() -> None:
    loss_mat = torch.tensor([[1.0, 2.0, 9.0], [3.0, 7.0, 8.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    actual = agg_loss(loss_mat, mask, "token-mean")

    assert torch.allclose(actual, torch.tensor(2.0))


def test_policy_gradient_loss_gradient_flows() -> None:
    b, t = 2, 4
    logp = torch.randn(b, t, requires_grad=True)
    old_logp = torch.randn(b, t).detach()
    adv = torch.randn(b, t)
    loss = policy_gradient_loss(logp, old_logp, adv, clip_ratio=0.2)
    loss.backward()
    assert logp.grad is not None
    assert torch.isfinite(logp.grad).all()


def test_asymmetric_clip_loss() -> None:
    logp = torch.tensor([[0.5, 0.5]])
    old_logp = torch.zeros(1, 2)
    adv = torch.ones(1, 2)
    loss = asymmetric_clip_loss(logp, old_logp, adv, clip_low=0.2, clip_high=0.28)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_asymmetric_clip_loss_uses_miles_sequence_mean_with_dp_compensation() -> None:
    logp = torch.zeros(2, 3, requires_grad=True)
    old_logp = torch.zeros_like(logp)
    advantages = torch.tensor([[1.0, 3.0, 100.0], [2.0, 4.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    loss = asymmetric_clip_loss(
        logp,
        old_logp,
        advantages,
        clip_low=0.2,
        clip_high=0.28,
        mask=mask,
        loss_agg_mode="seq-mean-token-mean",
        global_batch_size=4,
        dp_size=2,
    )

    # Local sequence means are -2 and -4. Dividing their sum by global B=4
    # and compensating DP=2 gives the global-average contribution -3.
    torch.testing.assert_close(loss, torch.tensor(-3.0))
    loss.backward()
    torch.testing.assert_close(
        logp.grad,
        torch.tensor([[-0.25, -0.75, 0.0], [-1.0 / 3.0, -2.0 / 3.0, -1.0]]),
    )


def test_asymmetric_clip_loss_uses_miles_clip_bounds() -> None:
    ratios = torch.tensor([[2.0, 0.5]])
    logp = ratios.log().requires_grad_(True)
    old_logp = torch.zeros_like(logp)
    advantages = torch.tensor([[1.0, -1.0]])

    loss = asymmetric_clip_loss(
        logp,
        old_logp,
        advantages,
        clip_low=0.2,
        clip_high=0.28,
    )

    # Positive advantage clips at 1.28; negative advantage clips at 0.8.
    torch.testing.assert_close(loss, torch.tensor((-1.28 + 0.8) / 2))
    loss.backward()
    torch.testing.assert_close(logp.grad, torch.zeros_like(logp))


def test_kl_penalty_nonneg() -> None:
    logp = torch.tensor([[0.0, 0.0]])
    ref = torch.tensor([[0.5, 0.5]])
    kl = kl_penalty(logp, ref)
    assert float(kl) >= 0.0


def test_entropy_bonus() -> None:
    logp = torch.tensor([[-1.0, -0.5]])
    ent = entropy_bonus(logp)
    assert ent.ndim == 0
    assert float(ent) >= 0.0
