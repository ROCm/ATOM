# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The superseded async_training trainer says what to use instead."""

from __future__ import annotations

from atomrl.core.config import AtomRLConfig
from atomrl.trainer import async_trainer
from atomrl.trainer.async_trainer import AsyncRLTrainer


def test_async_training_points_to_the_ray_async_modes(tmp_path, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        async_trainer.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args),
    )
    config = AtomRLConfig()
    config.async_training.enabled = True
    config.async_training.weight_sync_dir = str(tmp_path)

    AsyncRLTrainer(config)

    [warning] = warnings
    assert "never loads updated weights" in warning
    assert "controller.ray.fully_async" in warning
    assert "controller.ray.one_step_off_policy" in warning
