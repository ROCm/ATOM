# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import pytest
import torch
from atomrl.algorithms.loss_functions import (
    agg_loss,
    asymmetric_clip_loss,
    entropy_bonus,
    kl_penalty,
    policy_gradient_loss,
    reference_kl_loss,
    reference_kl_tokens,
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


def _nemo_rl_kl(logprobs, ref, kl_type, input_clamp=20.0, output_clamp=10.0):
    """The KL term of NeMo-RL v0.8's GRPO loss written out from its definition
    (clamped log-ratio, estimator, on-policy weight exp(logp - logp.detach())
    detached where the clamp binds, clamped product): what the loss is held to."""
    weights = torch.exp(logprobs - logprobs.detach())
    logr = ref - logprobs
    logr_clamped = logr.clamp(min=-input_clamp, max=input_clamp)
    weights = torch.where(logr == logr_clamped, weights, weights.detach())
    logr = logr_clamped
    if kl_type == "k1":
        kl = -logr
    elif kl_type == "k2":
        kl = torch.square(logr) / 2
    else:
        kl = torch.exp(logr) - 1 - logr
    return (weights * kl).clamp(min=-output_clamp, max=output_clamp)


@pytest.mark.parametrize(
    ("kl_penalty_name", "nemo_type"),
    [("kl", "k1"), ("mse", "k2"), ("k3", "k3"), ("low_var_kl", "k3")],
)
def test_reference_kl_matches_nemo_rl_in_value_and_gradient(
    kl_penalty_name, nemo_type
) -> None:
    torch.manual_seed(0)
    base = torch.randn(3, 7) * 3
    ref = torch.randn(3, 7) * 3
    ref[0, 0] = base[0, 0] + 25.0  # past the input clamp
    ref[1, 1] = base[1, 1] - 25.0
    ours_lp = base.clone().requires_grad_(True)
    theirs_lp = base.clone().requires_grad_(True)

    ours = reference_kl_tokens(ours_lp, ref, kl_penalty_name)
    theirs = _nemo_rl_kl(theirs_lp, ref, nemo_type)
    ours.sum().backward()
    theirs.sum().backward()

    torch.testing.assert_close(ours, theirs)
    torch.testing.assert_close(ours_lp.grad, theirs_lp.grad)


def test_reference_kl_carries_the_score_function_gradient() -> None:
    """d/dlogp of w * k3 with w = exp(logp - logp.detach()) is k3 + dk3/dlogp;
    without the weight only the second term would remain."""
    logp = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    ref = torch.tensor([[-1.5, -1.0]])
    reference_kl_tokens(logp, ref, "k3").sum().backward()
    logr = ref - logp.detach()
    k3 = torch.exp(logr) - 1 - logr
    dk3 = 1 - torch.exp(logr)
    torch.testing.assert_close(logp.grad, k3 + dk3)


def test_reference_kl_loss_is_normalized_like_the_policy_loss() -> None:
    logp = torch.tensor([[-1.0, -2.0, -0.5]])
    ref = torch.tensor([[-1.5, -1.0, -0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss = reference_kl_loss(
        logp, ref, mask=mask, kl_type="k3", dp_size=2, batch_num_tokens=8
    )
    per_token = reference_kl_tokens(logp, ref, "k3")
    torch.testing.assert_close(loss, (per_token * mask).sum() / 8 * 2)


def test_reference_kl_refuses_an_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="abs"):
        reference_kl_tokens(torch.zeros(1, 1), torch.zeros(1, 1), "abs")


def test_entropy_bonus() -> None:
    logp = torch.tensor([[-1.0, -0.5]])
    ent = entropy_bonus(logp)
    assert ent.ndim == 0
    assert float(ent) >= 0.0
