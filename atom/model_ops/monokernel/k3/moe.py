# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Kimi-K3-specific launchers for the existing A16W4 MoE kernels."""

from __future__ import annotations

import torch

from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
)

_HIDDEN = 3584
_INTER = 384
_EXPERTS = 896
_TOP_K = 16
_BLOCK_M = 16

# These are the production Kimi-K3 entries selected from AITER's tuned table.
# Keeping the fixed-shape choices here avoids changing the generic MoE package or
# depending on an external CSV at runtime.
_GEMM1_CONFIGS = {
    1: (32, 256, 2, 4),
    2: (32, 128, 4, 0),
    3: (64, 256, 2, 0),
    4: (64, 256, 2, 4),
    5: (64, 256, 2, 4),
    6: (64, 256, 2, 4),
    7: (64, 256, 2, 4),
    8: (128, 256, 1, 1),
}
_GEMM2_XCD = {1: 4, 2: 0, 3: 0, 4: 4, 5: 4, 6: 4, 7: 4, 8: 0}


def _check_samples(samples: int) -> None:
    if samples not in _GEMM1_CONFIGS:
        raise ValueError(f"Kimi-K3 MoE supports 1-8 samples, got {samples}")


def kimi_k3_mxfp4_gemm1(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    samples: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    """Run Kimi-K3 routed expert up/gate plus SiTUv2."""

    _check_samples(samples)
    tile_n, tile_k, k_wave, xcd_swizzle = _GEMM1_CONFIGS[samples]
    return flydsl_a16w4_gemm1(
        a_bf16=activation,
        w1_u8=weight,
        w1_scale_u8=weight_scale,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        m_indices=sorted_token_ids,
        inter_sorted_bf16=output,
        n_tokens=samples,
        NE=_EXPERTS,
        D_HIDDEN=_HIDDEN,
        D_INTER=_INTER,
        topk=_TOP_K,
        tile_m=_BLOCK_M,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=None,
        k_wave=k_wave,
        b_nt=2,
        xcd_swizzle=xcd_swizzle,
        gate_mode="separated",
        act="situv2",
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        w_dtype="fp4",
        w_layout="standard",
        stream=torch.cuda.current_stream(),
    )


def kimi_k3_mxfp4_gemm2(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    output: torch.Tensor,
    *,
    samples: int,
    max_sorted: int,
) -> torch.Tensor:
    """Run Kimi-K3 routed expert down projection and weighted scatter."""

    _check_samples(samples)
    return flydsl_a16w4_gemm2(
        inter_sorted_bf16=activation,
        w2_u8=weight,
        w2_scale_u8=weight_scale,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        flat_out=output,
        M_logical=samples,
        max_sorted=max_sorted,
        NE=_EXPERTS,
        D_HIDDEN=_HIDDEN,
        D_INTER=_INTER,
        topk=_TOP_K,
        tile_m=_BLOCK_M,
        tile_n=128,
        tile_k=128,
        waves_per_eu=None,
        b_nt=2,
        xcd_swizzle=_GEMM2_XCD[samples],
        w_dtype="fp4",
        persist=False,
        epilog="atomic",
        stream=torch.cuda.current_stream(),
    )
