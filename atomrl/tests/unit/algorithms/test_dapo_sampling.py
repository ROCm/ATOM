# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for verl-faithful DAPO sampling and rollout correction helpers."""

from __future__ import annotations

import torch
from atomrl.algorithms.dapo import DAPOAlgorithm
from atomrl.algorithms.dapo_sampling import (
    filter_groups_keep_mask,
    overlong_buffer_penalty,
)
from atomrl.algorithms.loss_functions import asymmetric_clip_loss
from atomrl.core.config import AtomRLConfig
from atomrl.core.protocol import DataProto


# --------------------------------------------------------------------------- #
# Overlong soft buffer (verl reward_manager/dapo)
# --------------------------------------------------------------------------- #
def test_overlong_buffer_penalty_zones() -> None:
    # max_resp_len=1000, buffer=200 -> expected_len=800.
    lens = [700, 800, 900, 1000, 1100]
    pen = overlong_buffer_penalty(
        lens, max_resp_len=1000, buffer_len=200, penalty_factor=1.0
    )
    # <=800: no penalty; 900: -(100/200)=-0.5; 1000: -1.0; 1100: -1.5
    expected = torch.tensor([0.0, 0.0, -0.5, -1.0, -1.5])
    assert torch.allclose(pen, expected, atol=1e-6)


def test_overlong_buffer_disabled() -> None:
    lens = [5000]
    assert torch.allclose(overlong_buffer_penalty(lens, 1000, 0, 1.0), torch.zeros(1))
    assert torch.allclose(overlong_buffer_penalty(lens, 1000, 200, 0.0), torch.zeros(1))


# --------------------------------------------------------------------------- #
# filter_groups dynamic sampling (verl recipe/dapo)
# --------------------------------------------------------------------------- #
def test_filter_groups_drops_zero_std() -> None:
    # 3 prompts × 2 samples. p0 all wrong, p1 mixed, p2 all correct.
    uids = ["p0", "p0", "p1", "p1", "p2", "p2"]
    acc = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    keep, kept_uids = filter_groups_keep_mask(acc, uids)
    assert kept_uids == ["p1"]
    assert keep.tolist() == [False, False, True, True, False, False]


def test_filter_groups_keeps_singletons() -> None:
    uids = ["a", "b"]
    acc = [1.0, 0.0]
    keep, kept_uids = filter_groups_keep_mask(acc, uids)
    assert keep.tolist() == [True, True]
    assert kept_uids == ["a", "b"]


# --------------------------------------------------------------------------- #
# TIS weights consumed in the DAPO loss
# --------------------------------------------------------------------------- #
def test_dapo_loss_applies_rollout_is_weights() -> None:
    cfg = AtomRLConfig()
    algo = DAPOAlgorithm(cfg)
    dapo = cfg.algorithm.dapo
    b, t = 2, 3
    logp = torch.randn(b, t)
    old_logp = torch.zeros(b, t)
    adv = torch.ones(b, t)
    weights = torch.full((b, t), 0.5)
    batch = DataProto(
        tensors={
            "log_probs": logp,
            "old_log_probs": old_logp,
            "advantages": adv,
            "dapo_sample_mask": torch.ones(b, t),
            "rollout_is_weights": weights,
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
        clip_ratio_c=float(dapo.clip_ratio_c),
        rollout_is_weights=weights,
    )
    loss, _ = algo.compute_loss(batch)
    assert torch.allclose(loss, expected)

    # And the IS-weighted loss must differ from the unweighted one.
    batch_no = DataProto(
        tensors={
            "log_probs": logp,
            "old_log_probs": old_logp,
            "advantages": adv,
            "dapo_sample_mask": torch.ones(b, t),
        }
    )
    loss_no, _ = algo.compute_loss(batch_no)
    assert not torch.allclose(loss, loss_no)
