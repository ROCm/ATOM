# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import pytest
from atomrl.architecture.assembly.policy_validator import validate_backend_policy
from atomrl.architecture.config.assembly_config import RuntimeAssemblyConfig


def test_rejects_non_atom_inference_backend() -> None:
    cfg = RuntimeAssemblyConfig(
        training_backend="fsdp",
        inference_backend="vllm",
    )
    with pytest.raises(ValueError, match="Inference backend"):
        validate_backend_policy(cfg)


def test_rejects_training_backend_outside_allowlist() -> None:
    cfg = RuntimeAssemblyConfig(
        training_backend="ddp",
        inference_backend="atom",
    )
    with pytest.raises(ValueError, match="Training backend"):
        validate_backend_policy(cfg)


def test_rejects_removed_legacy_megatron_backend() -> None:
    cfg = RuntimeAssemblyConfig(
        training_backend="megatron",
        inference_backend="atom",
    )
    with pytest.raises(ValueError, match="Training backend"):
        validate_backend_policy(cfg)
