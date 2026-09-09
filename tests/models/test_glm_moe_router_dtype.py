# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The GLM MoE router must resolve to fp32, and the reason it must.

These are CPU-only: they pin the dtype decision and the numerical fact behind
it, without needing a GPU or a 408 GB checkpoint.
"""

import pytest
import torch
from transformers import PretrainedConfig

# `deepseek_v2` imports AITER at module scope, so this module cannot even be
# collected on the CPU-only Pre Checkin runner without the guard.
pytest.importorskip("aiter")

from atom.models.deepseek_v2 import _moe_router_dtype


def _config(**kw) -> PretrainedConfig:
    cfg = PretrainedConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_glm_moe_dsa_forces_fp32_without_the_config_key():
    # GLM-5 / 5.1 / 5.2 predate `moe_router_dtype` and cannot ask for fp32,
    # but their bias needs it just as much as GLM-5.3's. Keying on model_type
    # is what makes those generations correct too.
    assert _moe_router_dtype(_config(model_type="glm_moe_dsa")) is torch.float32


def test_explicit_config_key_is_honoured():
    assert (
        _moe_router_dtype(_config(model_type="something", moe_router_dtype="float32"))
        is torch.float32
    )


def test_other_models_keep_the_model_dtype():
    # None, not torch.bfloat16: DeepSeek/Kimi must go on creating the parameter
    # and calling the gate exactly as before.
    assert _moe_router_dtype(_config(model_type="deepseek_v3")) is None
    assert _moe_router_dtype(PretrainedConfig()) is None


def test_none_means_default_dtype_at_parameter_creation():
    # The call site passes the result straight to torch.empty(dtype=...), so
    # None has to keep meaning "default dtype" for every other model.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        assert torch.empty(4, dtype=_moe_router_dtype(PretrainedConfig())).dtype is (
            torch.bfloat16
        )
    finally:
        torch.set_default_dtype(previous)


def test_bf16_destroys_a_glm_shaped_correction_bias():
    """Why fp32 is required, as a number rather than an assertion.

    Modelled on the real tensor, not on its extremes: within ONE layer of
    GLM-5.3-MXFP4 the 256 values are a narrow band riding a large offset
    (layer 10 measured min 6.817, max 7.063, std 0.040, 238 distinct), and it
    is the narrowness that does the damage. bf16's ULP at ~7 is 1/64, so a
    0.25-wide band has only ~16 representable levels in it. Spreading the same
    256 values over the [6.02, 8.11] range seen ACROSS layers would leave 65
    levels and understate the loss by 4x.
    """
    torch.manual_seed(0)
    bias = 6.94 + torch.randn(256) * 0.040

    distinct_fp32 = len(torch.unique(bias))
    distinct_bf16 = len(torch.unique(bias.to(torch.bfloat16).float()))

    assert distinct_fp32 > 200
    # Not "slightly fewer" -- an order of magnitude fewer. If a future change
    # lands the bias in bf16 again, this is the damage it does.
    assert distinct_bf16 < 20, (
        f"bf16 collapsed {distinct_fp32} distinct bias values onto "
        f"{distinct_bf16}; the router loses its selection signal."
    )
