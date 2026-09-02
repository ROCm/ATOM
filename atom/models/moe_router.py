# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which dtype the MoE router runs in, decided from the HF config alone.

Kept out of `deepseek_v2`, which imports AITER at module scope: this is a
policy decision with a numerical justification worth pinning on a plain CI
runner, and it needs nothing but the config and a torch dtype.
"""

import torch
from transformers import PretrainedConfig


def moe_router_dtype(config: PretrainedConfig) -> torch.dtype | None:
    """Dtype the MoE router must run in, or None to keep the model dtype.

    None preserves the historical behaviour and is what every model that does
    not ask for something else gets.

    GLM's `noaux_tc` correction bias is ~256 values packed into [6.02, 8.11]
    with a median spacing of 6e-6 between neighbours. bf16 carries 8 mantissa
    bits, so its ULP up there is 1/64 -- four orders of magnitude coarser than
    the spacing. Storing that tensor at the model dtype collapses 238 distinct
    values onto 8 and discards almost all of the selection signal it exists to
    carry. Every `glm_moe_dsa` checkpoint on disk ships it as fp32, and vLLM
    forces fp32 routing for this model_type unconditionally -- including
    GLM-5/5.1/5.2, whose configs predate `moe_router_dtype` and therefore
    cannot ask for it. Match that, keyed on the same signal, so a model does
    not silently depend on whether its config generation happened to carry
    the key.

    The gate output dtype is not separately useful -- rounding the logits to
    bf16 barely moves the top-k -- but it is not independent either: aiter's
    `biased_grouped_topk` dispatches on `gating_output.dtype()` and then
    reinterpret_casts `correction_bias` to that same `scalar_t`. The two must
    agree or the kernel reads the bias buffer at the wrong width, so this one
    dtype governs both.
    """
    if getattr(config, "model_type", None) == "glm_moe_dsa":
        return torch.float32
    if getattr(config, "moe_router_dtype", None) == "float32":
        return torch.float32
    return None
