# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""GRPO dense BF16 end-to-end smoke (multi-GPU, slow)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from atomrl.core.config import AtomRLConfig
from atomrl.trainer.rl_trainer import RLTrainer

pytestmark = [pytest.mark.multigpu, pytest.mark.slow]


_RISING_REWARD_HISTORY: list[float] = []


def _compute_rising_rewards(
    self,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    gts_expanded: list[str],
) -> tuple[torch.Tensor, list[str], list[float]]:
    """Return deterministic rising rewards through the current trainer API."""
    del self, attention_mask, prompt_lengths, gts_expanded
    batch_size = int(sequences.shape[0])
    step = len(_RISING_REWARD_HISTORY) + 1
    rewards = torch.linspace(
        0.1,
        0.2,
        batch_size,
        device=sequences.device,
        dtype=torch.float32,
    ) + 0.05 * float(step)
    _RISING_REWARD_HISTORY.append(float(rewards.mean().item()))
    return rewards, [""] * batch_size, [0.0] * batch_size


def test_grpo_dense_bf16_convergence(
    e2e_config_dir: Path,
    tmp_checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")

    yaml_path = e2e_config_dir / "grpo_dense_bf16.yaml"
    assert yaml_path.is_file()

    _RISING_REWARD_HISTORY.clear()
    monkeypatch.setattr(RLTrainer, "_compute_rewards_full", _compute_rising_rewards)

    cfg = AtomRLConfig.from_yaml(
        yaml_path,
        overrides=[
            "num_training_steps=12",
            "cluster.num_nodes=1",
            "cluster.gpus_per_node=2",
            f"checkpointing.checkpoint_dir={tmp_checkpoint_dir!s}",
        ],
    )
    assert cfg.algorithm.name.lower() == "grpo"

    if os.environ.get("ATOMRL_E2E_SMOKE", "") != "1":
        pytest.skip("Set ATOMRL_E2E_SMOKE=1 to run multi-GPU Ray trainer smoke.")

    trainer = RLTrainer(cfg)
    try:
        trainer.setup()
        trainer.train()
    finally:
        if trainer._cluster is not None:
            trainer._cluster.shutdown()

    h = _RISING_REWARD_HISTORY
    assert len(h) >= 8, "expected multiple reward-worker invocations"
    k = min(4, len(h) // 2)
    assert sum(h[-k:]) / k > sum(h[:k]) / k + 1e-6
    assert all(not math.isnan(v) for v in trainer.last_metrics.values())
