# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Rollout-side FP8 vs BF16 parity and TIS-style correction behavior."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from atomrl.core.config import AtomRLConfig, QuantizationConfig, RolloutCorrectionConfig
from atomrl.core.protocol import DataProto
from atomrl.engine.training.fsdp_backend import FSDP2Backend
from atomrl.quantization.fp8_config import FP8Config
from atomrl.quantization.rollout_correction import apply_rollout_correction
from atomrl.quantization.weight_quantizer import WeightQuantizer

pytestmark = pytest.mark.fp8


@pytest.fixture
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _tiny_lm() -> torch.nn.Module:
    return FSDP2Backend.build_model(
        "",
        {"tiny_lm": {"vocab_size": 512, "dim": 256, "n_layers": 1}},
    )


def test_fp8_bf16_logprob_distribution(_require_cuda: None) -> None:
    """BF16 vs FP8-packed weights (via WeightQuantizer) yield close full-vocab distributions."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _tiny_lm().to(device=device, dtype=torch.bfloat16)
    b, t = 2, 24
    input_ids = torch.randint(0, 512, (b, t), device=device, dtype=torch.long)

    wq = WeightQuantizer(FP8Config())
    dq_sd = wq.dequantize_state_dict(wq.quantize_state_dict(model.state_dict()))
    twin = _tiny_lm().to(device=device, dtype=torch.bfloat16)
    twin.load_state_dict(dq_sd, strict=True)

    model.eval()
    twin.eval()
    with torch.no_grad():
        log_p = F.log_softmax(model(input_ids)[:, :-1], dim=-1)
        q = F.softmax(twin(input_ids)[:, :-1], dim=-1)
        kl = F.kl_div(log_p, q, reduction="batchmean", log_target=False)
    assert kl.isfinite()
    assert float(kl) < 0.25


def test_tis_reduces_gap() -> None:
    """TIS-style clamping caps importance weights vs an uncapped exponential."""
    rc = RolloutCorrectionConfig(enabled=True, method="tis", clip=1.5)
    quant = QuantizationConfig(rollout_correction=rc)
    cfg = AtomRLConfig(quantization=quant)

    b, t = 2, 8
    old_lp = torch.zeros(b, t)
    fp8_lp = torch.full((b, t), -3.0)
    adv = torch.ones(b, t)

    batch = DataProto(
        tensors={
            "old_log_probs": old_lp,
            "fp8_log_probs": fp8_lp,
            "advantages": adv.clone(),
        },
    )
    uncapped_ratio = old_lp - fp8_lp
    uncapped_weights = torch.exp(uncapped_ratio)
    adv_uncapped = adv * uncapped_weights

    corrected = apply_rollout_correction(batch, cfg)
    adv_tis = corrected.tensors["advantages"]

    assert adv_tis.abs().max() <= adv_uncapped.abs().max() + 1e-6
    assert float(adv_tis.abs().max()) <= float(
        adv.abs().max() * math.exp(math.log(rc.clip)) + 1e-5
    )
