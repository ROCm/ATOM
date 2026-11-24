# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import math

import pytest
import torch
from atomrl.core.config import RolloutCorrectionConfig
from atomrl.core.protocol import DataProto
from atomrl.quantization.rollout_correction import (
    apply_rollout_correction,
    token_level_mis,
    token_level_tis,
)


def test_tis_clipping() -> None:
    bf16 = torch.zeros(2, 4)
    fp8 = torch.full((2, 4), -50.0)
    adv = torch.ones_like(bf16)
    out = token_level_tis(bf16, fp8, adv, clip=1.5)
    ratio = torch.exp(torch.clamp(bf16 - fp8, min=-20.0, max=20.0))
    clipped = torch.clamp(ratio, 1.0 / 1.5, 1.5)
    assert torch.allclose(out, clipped * adv)


def test_mis_normalization() -> None:
    bsz, tok = 3, 5
    bf16 = torch.randn(bsz, tok)
    fp8 = bf16.clone()
    adv = torch.ones(bsz, tok)
    out = token_level_mis(bf16, fp8, adv)
    assert torch.allclose(out, adv)
    assert out.sum().item() == pytest.approx(float(bsz * tok))


def test_mis_self_normalizing() -> None:
    """MIS ratio weights should average to ~1 after mean normalization."""
    bsz, tok = 4, 10
    bf16 = torch.randn(bsz, tok)
    fp8 = bf16 + 0.5 * torch.randn(bsz, tok)
    adv = torch.ones(bsz, tok)
    out = token_level_mis(bf16, fp8, adv)
    ratio = torch.exp(torch.clamp(bf16 - fp8, min=-20.0, max=20.0))
    normalized = ratio / ratio.mean().clamp(min=1e-8)
    assert torch.allclose(out, normalized * adv)
    assert normalized.mean().item() == pytest.approx(1.0, abs=1e-5)


def test_tis_with_known_values() -> None:
    bf16 = torch.tensor([[math.log(10.0)]])
    fp8 = torch.zeros(1, 1)
    adv = torch.tensor([[1.0]])
    clip = 2.0
    out = token_level_tis(bf16, fp8, adv, clip=clip)
    log_ratio = torch.clamp(bf16 - fp8, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 / clip, clip)
    expected = clipped * adv
    assert torch.allclose(out, expected)
    assert float(out.item()) == pytest.approx(2.0)


def test_mis_with_known_values() -> None:
    bf16 = torch.tensor([[0.0, math.log(2.0)]])
    fp8 = torch.tensor([[0.0, 0.0]])
    adv = torch.tensor([[3.0, 5.0]])
    out = token_level_mis(bf16, fp8, adv)
    r = torch.exp(torch.clamp(bf16 - fp8, min=-20.0, max=20.0))
    normalized = r / r.mean().clamp(min=1e-8)
    assert torch.allclose(out, normalized * adv)
    mean_ratio = (1.0 + 2.0) / 2.0  # raw ratios: 1.0, 2.0
    assert float(out[0, 0].item()) == pytest.approx(3.0 * (1.0 / mean_ratio))
    assert float(out[0, 1].item()) == pytest.approx(5.0 * (2.0 / mean_ratio))


def test_apply_rollout_correction_on_dataproto() -> None:
    batch = DataProto(
        tensors={
            "bf16_logprobs": torch.tensor([[0.0, 1.0]]),
            "fp8_logprobs": torch.tensor([[0.0, 0.0]]),
            "advantages": torch.tensor([[2.0, 3.0]]),
        }
    )
    rcfg = RolloutCorrectionConfig(enabled=True, method="tis", clip=2.0)
    out = apply_rollout_correction(batch, rcfg)
    expected_adv = token_level_tis(
        batch["bf16_logprobs"],
        batch["fp8_logprobs"],
        batch["advantages"],
        clip=2.0,
    )
    assert torch.allclose(out["advantages"], expected_adv)
    assert out.meta.get("rollout_correction", {}).get("method") == "tis"

    disabled = RolloutCorrectionConfig(enabled=False)
    passthrough = apply_rollout_correction(batch, disabled)
    assert passthrough is batch
