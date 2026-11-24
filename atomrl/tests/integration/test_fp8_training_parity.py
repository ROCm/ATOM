# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Training loop BF16 vs configured FP8 training path (Lumen hooks when installed)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from atomrl.core.protocol import DataProto
from atomrl.workers.actor_worker import AtomActorWorker

pytestmark = pytest.mark.fp8


@pytest.fixture
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _actor_config(fp8: str | None) -> dict[str, Any]:
    return {
        "policy": {
            "model_name": "fp8-parity-stub",
            "training_backend": "fsdp2",
            "lr": 3e-3,
            "training": {
                "tiny_lm": {"vocab_size": 512, "dim": 128, "n_layers": 1},
                "fsdp_cfg": {"enabled": False},
            },
        },
        "quantization": {
            "training": {
                "fp8": fp8,
                "fp8_recipe": "blockwise",
                "fp8_weight_cache": False,
            }
        },
    }


def _run_steps(cfg: dict[str, Any], steps: int, seed: int) -> list[float]:
    torch.manual_seed(seed)
    worker = AtomActorWorker(rank=0, world_size=1, config=cfg)
    worker.init_model()
    b, t = 2, 16
    ids = torch.randint(0, 512, (b, t), dtype=torch.long)
    mask = torch.ones(b, t, dtype=torch.long)
    # old_log_probs has to come from the policy itself. Left out (or zeroed) the
    # PPO ratio starts at exp(-log V) ~ 2e-3, the surrogate is pinned to the clip
    # constant, and both arms then agree on a number the compute dtype never
    # touched -- the test would pass without exercising anything.
    old_log_probs = worker.compute_log_probs(
        DataProto(tensors={"input_ids": ids, "attention_mask": mask})
    )["log_probs"]
    batch = DataProto(
        tensors={
            "input_ids": ids,
            "attention_mask": mask,
            "response_mask": mask,
            "old_log_probs": old_log_probs,
            "advantages": torch.ones(b, t, dtype=torch.float32),
        },
        meta={
            "algorithm": "dapo",
            "dp_size": 1,
            "batch_num_tokens": int(mask[:, 1:].sum()),
        },
    )
    losses: list[float] = []
    for _ in range(steps):
        m = worker.update_policy(batch)
        losses.append(m["loss"])
    worker.cleanup()
    return losses


def test_fp8_bf16_loss_tracking(_require_cuda: None) -> None:
    """N training steps: BF16 vs FP8-labeled config should stay within a loose band."""
    steps = 8
    seed = 123
    losses_bf16 = _run_steps(_actor_config(fp8=None), steps, seed)
    losses_fp8 = _run_steps(_actor_config(fp8="e4m3"), steps, seed)

    assert len(losses_bf16) == len(losses_fp8) == steps
    for a, b in zip(losses_bf16, losses_fp8, strict=True):
        denom = max(abs(a), 1e-6)
        assert abs(a - b) / denom <= 0.25
