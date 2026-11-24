# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.algorithms.dapo import DAPOAlgorithm
from atomrl.algorithms.loss_functions import asymmetric_clip_loss
from atomrl.core.config import AtomRLConfig
from atomrl.core.protocol import DataProto


def test_compute_advantages_with_dynamic_sampling() -> None:
    cfg = AtomRLConfig()
    cfg.algorithm.dapo.num_generations = 2
    cfg.algorithm.dapo.dynamic_sampling = True
    cfg.algorithm.dapo.overlong_reward_shaping = False
    algo = DAPOAlgorithm(cfg)
    rewards = torch.tensor([1.0, 1.0, 1.0, 2.0])
    batch = DataProto(tensors={"rewards": rewards})
    out = algo.compute_advantages(batch)
    mask = out.tensors["dapo_sample_mask"]
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert torch.allclose(mask, expected)


def test_dapo_uses_dapo_estimator_even_when_recipe_says_grpo() -> None:
    cfg = AtomRLConfig()
    cfg.algorithm.name = "dapo"
    cfg.algorithm.adv_estimator = "grpo"
    cfg.algorithm.dapo.num_generations = 2
    cfg.algorithm.dapo.dynamic_sampling = False
    cfg.algorithm.dapo.overlong_buffer.enable = True
    cfg.algorithm.dapo.overlong_buffer.len = 20
    cfg.algorithm.dapo.overlong_buffer.penalty_factor = 1.0
    cfg.algorithm.dapo.max_resp_len = 100

    batch = DataProto(
        tensors={"rewards": torch.zeros(2)},
        meta={"response_lengths": [80, 100]},
    )
    out = DAPOAlgorithm(cfg).compute_advantages(batch)

    assert "dapo_sample_mask" in out.tensors
    assert torch.all(out.tensors["dapo_sample_mask"] == 1)
    assert out.tensors["advantages"][0] > 0
    assert out.tensors["advantages"][1] < 0


def test_asymmetric_clip() -> None:
    cfg = AtomRLConfig()
    algo = DAPOAlgorithm(cfg)
    dapo = cfg.algorithm.dapo
    b, t = 2, 3
    logp = torch.randn(b, t, requires_grad=True)
    old_logp = torch.zeros(b, t)
    adv = torch.ones(b, t)
    batch = DataProto(
        tensors={
            "log_probs": logp,
            "old_log_probs": old_logp,
            "advantages": adv,
            "dapo_sample_mask": torch.ones(b, t),
        }
    )
    sm = torch.ones_like(logp)
    expected = asymmetric_clip_loss(
        logp,
        old_logp,
        adv,
        float(dapo.clip_ratio_low),
        float(dapo.clip_ratio_high),
        mask=sm,
    )
    loss, _ = algo.compute_loss(batch)
    assert torch.allclose(loss, expected)
