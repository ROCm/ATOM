# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared fixtures and markers for integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from atomrl.core.config import AtomRLConfig
from atomrl.core.protocol import DataProto

REPO_ROOT = Path(__file__).resolve().parents[2]
GRPO_YAML = REPO_ROOT / "configs" / "grpo_dense_bf16.yaml"


def pytest_configure(config: pytest.Config) -> None:
    """Register markers (also declared in pyproject.toml for discovery)."""
    for name, desc in (
        ("gpu", "requires 1 GPU"),
        ("multigpu", "requires 2+ GPUs"),
        ("moe", "requires MoE model or MoE-related stack"),
        ("fp8", "requires FP8-capable path or numerics"),
        ("slow", "takes > 60 seconds"),
    ):
        config.addinivalue_line("markers", f"{name}: {desc}")


@pytest.fixture
def small_dense_model() -> dict:
    """Minimal dense policy config for FSDP2 TinyLM."""
    return {
        "policy": {
            "model_name": "integration-dense-stub",
            "training_backend": "fsdp2",
            "lr": 1e-3,
            "training": {
                "tiny_lm": {"vocab_size": 512, "dim": 64, "n_layers": 1},
                "fsdp_cfg": {"enabled": False},
            },
        },
        "quantization": {"training": {}},
    }


@pytest.fixture
def small_moe_model() -> dict:
    """MoE-oriented worker config (metadata + TinyLM body for local tests)."""
    return {
        "policy": {
            "model_name": "integration-moe-stub",
            "training_backend": "fsdp2",
            "lr": 1e-3,
            "training": {
                "tiny_lm": {"vocab_size": 512, "dim": 64, "n_layers": 2},
                "megatron_cfg": {
                    "tensor_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "num_experts": 8,
                },
                "fsdp_cfg": {"enabled": False},
            },
        },
        "quantization": {"training": {}},
        "moe": {"r3": {"enabled": True, "replay_mode": "distribution"}},
    }


@pytest.fixture
def sample_batch() -> DataProto:
    """CPU DataProto matching RLTrainer / GRPO tensor expectations."""
    b, t = 4, 16
    return DataProto(
        tensors={
            "input_ids": torch.randint(0, 256, (b, t), dtype=torch.long),
            "attention_mask": torch.ones(b, t, dtype=torch.long),
            "old_log_probs": torch.randn(b, t, dtype=torch.float32) * 0.1,
        },
        meta={"num_generations": 2},
    )


@pytest.fixture
def grpo_config() -> AtomRLConfig:
    """GRPO config from repo YAML; path resolved relative to this package."""
    assert GRPO_YAML.is_file(), f"Expected config at {GRPO_YAML}"
    return AtomRLConfig.from_yaml(
        GRPO_YAML,
        overrides=[
            "num_training_steps=3",
            "cluster.gpus_per_node=1",
            "cluster.num_nodes=1",
        ],
    )


@pytest.fixture(autouse=True)
def _clear_cuda_cache() -> None:
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
