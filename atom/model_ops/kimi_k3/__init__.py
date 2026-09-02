# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused model operations for Kimi-K3."""

from atom.model_ops.kimi_k3.activations import (
    rmsnorm_gated,
    situ_and_mul,
    situ_and_mul_maybe_quant,
    situ_and_mul_quant,
)
from atom.model_ops.kimi_k3.attention_residual import apply_attn_res
from atom.model_ops.kimi_k3.kda_state import gather_kda_initial_state
from atom.model_ops.kimi_k3.quant import strided_per_token_quant

__all__ = [
    "apply_attn_res",
    "gather_kda_initial_state",
    "rmsnorm_gated",
    "situ_and_mul",
    "situ_and_mul_maybe_quant",
    "situ_and_mul_quant",
    "strided_per_token_quant",
]
