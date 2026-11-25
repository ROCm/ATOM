# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Validation for backend policy constraints."""

from __future__ import annotations

from atomrl.architecture.config.assembly_config import RuntimeAssemblyConfig

ALLOWED_TRAINING = {"fsdp", "fsdp2"}
ALLOWED_INFERENCE = {"atom"}


def validate_backend_policy(cfg: RuntimeAssemblyConfig) -> None:
    """Validate configured backends against project policy."""
    if cfg.training_backend not in ALLOWED_TRAINING:
        raise ValueError(f"Training backend must be one of {sorted(ALLOWED_TRAINING)}")
    if cfg.inference_backend not in ALLOWED_INFERENCE:
        raise ValueError(
            f"Inference backend must be one of {sorted(ALLOWED_INFERENCE)}"
        )
