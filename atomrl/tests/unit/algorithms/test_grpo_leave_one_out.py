# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""``adv_estimator: grpo_leave_one_out`` against NeMo-RL's GRPO advantage."""

from __future__ import annotations

import pytest
import torch
from atomrl.algorithms.advantage_estimators import ADV_ESTIMATOR_REGISTRY
from atomrl.core.config import AtomRLConfig
from atomrl.core.protocol import DataProto


def _nemo_rl_advantages(rewards: torch.Tensor, g: int) -> torch.Tensor:
    """Sample by sample, from the definition: the others' mean as baseline,
    their sample std as scale, applied only when the others vary."""
    out = torch.zeros_like(rewards)
    for start in range(0, rewards.numel(), g):
        group = rewards[start : start + g]
        for i in range(g):
            others = torch.cat([group[:i], group[i + 1 :]])
            if others.numel() == 0:
                continue
            adv = group[i] - others.mean()
            std = others.std(unbiased=True) if others.numel() > 1 else torch.tensor(0.0)
            if std > 0 and others.min() != others.max():
                adv = adv / (std + 1e-6)
            out[start + i] = adv
    return out


def _config(g: int) -> AtomRLConfig:
    cfg = AtomRLConfig()
    cfg.algorithm.grpo.num_generations = g
    return cfg


def _advantages(
    rewards: torch.Tensor, g: int, name="grpo_leave_one_out"
) -> torch.Tensor:
    batch = DataProto(tensors={"rewards": rewards.clone()})
    return ADV_ESTIMATOR_REGISTRY[name](batch, _config(g)).tensors["advantages"]


@pytest.mark.parametrize("g", [2, 3, 32])
def test_matches_nemo_rl_on_binary_and_continuous_rewards(g) -> None:
    torch.manual_seed(g)
    binary = (torch.rand(8 * g) > 0.7).float()
    binary[:g] = 1.0  # a group that is all correct
    binary[g : 2 * g] = 0.0
    binary[g] = 1.0  # a group with one correct answer
    continuous = torch.randn(8 * g)
    torch.testing.assert_close(
        _advantages(binary, g), _nemo_rl_advantages(binary, g), rtol=1e-5, atol=1e-6
    )
    # Like NeMo-RL, the variance is E[x^2] - E[x]^2 in float32; where two
    # others nearly coincide that differs from a two-pass std by ~1e-3.
    torch.testing.assert_close(
        _advantages(continuous, g),
        _nemo_rl_advantages(continuous, g),
        rtol=2e-3,
        atol=1e-5,
    )


def test_the_single_correct_answer_of_a_group_stays_unscaled() -> None:
    """Its 31 siblings all score 0, so NeMo-RL does not divide by their (zero)
    std: the advantage is 1, where the group mean/std estimator gives ~5.57. A
    wrong answer is measured against 30 zeros and the one: -1/sqrt(31)."""
    rewards = torch.zeros(32)
    rewards[0] = 1.0
    adv = _advantages(rewards, 32)
    assert adv[0].item() == pytest.approx(1.0)
    assert adv[1].item() == pytest.approx(-(31**-0.5), rel=1e-4)
    assert _advantages(rewards, 32, "grpo")[0].item() == pytest.approx(5.568, rel=1e-3)


def test_a_group_of_one_has_no_advantage() -> None:
    assert _advantages(torch.tensor([1.0, 0.0, 1.0]), 1).tolist() == [0.0, 0.0, 0.0]
