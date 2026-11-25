# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

from atomrl.core.config import QuantizationConfig, TrainingQuantConfig
from atomrl.quantization.fp8_config import FP8Config


def test_default_values() -> None:
    cfg = FP8Config()
    assert cfg.precision == "bf16"
    assert cfg.recipe == "blockwise"
    assert cfg.use_deep_gemm is True
    assert cfg.use_weight_pow2_scale is False
    assert cfg.is_fp8_enabled() is False


def test_from_config() -> None:
    q = QuantizationConfig()
    q.rollout.precision = "bf16"
    q.training = TrainingQuantConfig(fp8="e4m3", fp8_recipe="tensorwise")
    cfg = FP8Config.from_config(q)
    assert cfg.precision == "e4m3"
    assert cfg.recipe == "tensorwise"
    assert cfg.is_fp8_enabled() is True


def test_is_fp8_enabled() -> None:
    assert FP8Config(precision="fp8").is_fp8_enabled() is True
    assert FP8Config(precision="hybrid_fp8").is_fp8_enabled() is True
    assert FP8Config(precision="bf16").is_fp8_enabled() is False
