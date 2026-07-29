# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused dual RMSNorm operation for Kimi-K3."""

from __future__ import annotations

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.fused_qk_norm_rope_cache_quant import (
    _fused_qk_rmsnorm_kernel as _aiter_fused_qk_rmsnorm_kernel,
)


def _dual_rmsnorm_fake(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch_compile_guard(gen_fake=_dual_rmsnorm_fake, mutates_args=[])
def dual_rmsnorm(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The AITER kernel honors row strides; K3 split views have inner stride 1.
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _aiter_fused_qk_rmsnorm_kernel(
        q,
        q_weight,
        q_eps,
        k,
        k_weight,
        k_eps,
        q_out,
        k_out,
    )
    return q_out, k_out
