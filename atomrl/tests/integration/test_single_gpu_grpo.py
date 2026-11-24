# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Single-GPU GRPO integration against RLTrainer + Ray stub workers."""

from __future__ import annotations

import pytest
import ray
from atomrl.core.config import AtomRLConfig
from atomrl.trainer.rl_trainer import RLTrainer

pytestmark = pytest.mark.gpu


@pytest.fixture
def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def test_single_gpu_grpo_loop(grpo_config: AtomRLConfig, _require_cuda: None) -> None:
    """Run a short GRPO loop and validate metrics + reward tensors from stubs."""
    trainer = RLTrainer(grpo_config)
    try:
        trainer.setup()
        trainer.train()
    finally:
        if trainer._cluster is not None:
            trainer._cluster.shutdown()
        elif ray.is_initialized():
            ray.shutdown()

    metrics = trainer.last_metrics
    assert "loss" in metrics
    assert "loss_pg" in metrics
    assert "loss_total" in metrics

    batch = trainer._build_rollout_batch()
    batch = trainer.rollout_wg.dispatch_and_call("rollout", batch)  # type: ignore[union-attr]
    batch = trainer.reward_wg.dispatch_and_call("compute_rewards", batch)  # type: ignore[union-attr]
    assert "rewards" in batch.tensors
    assert batch.tensors["rewards"].shape[0] == batch.batch_size
