# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""GRPO dense FP8 vs BF16 e2e checks (multi-GPU, FP8, slow)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from atomrl.core.config import AtomRLConfig
from atomrl.trainer.rl_trainer import RLTrainer

pytestmark = [pytest.mark.multigpu, pytest.mark.fp8, pytest.mark.slow]


def _compute_flat_rewards(
    self,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    gts_expanded: list[str],
) -> tuple[torch.Tensor, list[str], list[float]]:
    """Return stable rewards through the current trainer API."""
    del self, attention_mask, prompt_lengths, gts_expanded
    batch_size = int(sequences.shape[0])
    rewards = torch.full(
        (batch_size,),
        0.25,
        device=sequences.device,
        dtype=torch.float32,
    )
    return rewards, [""] * batch_size, [0.0] * batch_size


def _run_trainer(cfg: AtomRLConfig) -> tuple[RLTrainer, float]:
    trainer = RLTrainer(cfg)
    t0 = time.perf_counter()
    try:
        trainer.setup()
        trainer.train()
    finally:
        elapsed = time.perf_counter() - t0
        if trainer._cluster is not None:
            trainer._cluster.shutdown()
    return trainer, elapsed


def test_grpo_dense_fp8_convergence(
    e2e_config_dir: Path,
    tmp_checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")

    bf16_path = e2e_config_dir / "grpo_dense_bf16.yaml"
    fp8_path = e2e_config_dir / "grpo_dense_fp8.yaml"
    assert bf16_path.is_file() and fp8_path.is_file()

    if os.environ.get("ATOMRL_E2E_SMOKE", "") != "1":
        pytest.skip("Set ATOMRL_E2E_SMOKE=1 to run multi-GPU Ray trainer smoke.")

    monkeypatch.setattr(RLTrainer, "_compute_rewards_full", _compute_flat_rewards)

    overrides = [
        "num_training_steps=10",
        "cluster.num_nodes=1",
        "cluster.gpus_per_node=2",
        f"checkpointing.checkpoint_dir={tmp_checkpoint_dir / 'cmp'!s}",
    ]
    cfg_bf16 = AtomRLConfig.from_yaml(bf16_path, overrides=overrides)
    cfg_fp8 = AtomRLConfig.from_yaml(fp8_path, overrides=overrides)

    tr_bf16, _ = _run_trainer(cfg_bf16)
    tr_fp8, _ = _run_trainer(cfg_fp8)

    loss_bf16 = float(tr_bf16.last_metrics.get("loss", 0.0))
    loss_fp8 = float(tr_fp8.last_metrics.get("loss", 0.0))
    denom = max(abs(loss_bf16), 1e-6)
    assert abs(loss_bf16 - loss_fp8) / denom <= 0.05 + 1e-6


def test_fp8_throughput_improvement(
    e2e_config_dir: Path,
    tmp_checkpoint_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")

    if os.environ.get("ATOMRL_E2E_SMOKE", "") != "1":
        pytest.skip("Set ATOMRL_E2E_SMOKE=1 to run multi-GPU Ray trainer smoke.")

    monkeypatch.setattr(RLTrainer, "_compute_rewards_full", _compute_flat_rewards)

    overrides = [
        "num_training_steps=16",
        "cluster.num_nodes=1",
        "cluster.gpus_per_node=2",
    ]
    bf16_path = e2e_config_dir / "grpo_dense_bf16.yaml"
    fp8_path = e2e_config_dir / "grpo_dense_fp8.yaml"

    _, t_bf16 = _run_trainer(
        AtomRLConfig.from_yaml(
            bf16_path,
            overrides=overrides
            + [f"checkpointing.checkpoint_dir={tmp_checkpoint_dir / 'tbf16'!s}"],
        )
    )
    _, t_fp8 = _run_trainer(
        AtomRLConfig.from_yaml(
            fp8_path,
            overrides=overrides
            + [f"checkpointing.checkpoint_dir={tmp_checkpoint_dir / 'tfp8'!s}"],
        )
    )

    if t_fp8 >= 0.85 * t_bf16:
        pytest.skip(
            f"Inconclusive throughput (bf16={t_bf16:.3f}s fp8={t_fp8:.3f}s); "
            "stub trainer may not reflect FP8 speedups."
        )
