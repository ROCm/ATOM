# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused model operations for Kimi-K3."""

from atom.model_ops.kimi_k3.activations import (
    rmsnorm_gated,
    situ_and_mul,
)
from atom.model_ops.kimi_k3.attention_residual import apply_attn_res

__all__ = [
    "apply_attn_res",
    "rmsnorm_gated",
    "situ_and_mul",
]
