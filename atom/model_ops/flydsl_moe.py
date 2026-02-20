# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE backend for ATOM.

When CK MOE sorting is unavailable (e.g. AITER built with ENABLE_CK=0) and
ATOM_USE_FLYDSL_MOE=1, this module routes FP8 MOE through FlyDSL's MLIR-compiled
2-stage kernels instead of the Triton fallback.

Priority: CK/ASM > FlyDSL > Triton.
"""

import logging
import os
from typing import Optional, Tuple

import torch

from atom.utils import envs

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
_flydsl_moe_available: Optional[bool] = None


def _has_flydsl_moe() -> bool:
    """Check if FlyDSL MOE backend is enabled and importable."""
    global _flydsl_moe_available
    if _flydsl_moe_available is not None:
        return _flydsl_moe_available

    if not envs.ATOM_USE_FLYDSL_MOE:
        _flydsl_moe_available = False
        return False

    try:
        from kernels.moe_gemm_2stage import (
            compile_moe_gemm1,
            compile_moe_gemm2,
        )  # noqa: F401

        _flydsl_moe_available = True
        _logger.info("FlyDSL MOE kernels detected and available")
    except Exception as e:
        _flydsl_moe_available = False
        _logger.warning(
            "ATOM_USE_FLYDSL_MOE=1 but FlyDSL MOE kernels not importable: %s. "
            "Ensure FlyDSL repo is on PYTHONPATH.",
            e,
        )
    return _flydsl_moe_available


# ---------------------------------------------------------------------------
# Torch-native MOE sorting (no CK dependency)
# ---------------------------------------------------------------------------
def moe_sorting_torch_native(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch MOE sorting matching CK/FlyDSL kernel expectations.

    Returns:
        sorted_ids: int32 with packed (topk_slot<<24 | token_id) encoding
        sorted_weights: fp32 aligned with sorted_ids
        sorted_expert_ids: int32, one expert id per M-block
        num_tokens_post_pad: int32 [2], [0]=total padded tokens, [1]=num_tokens
    """
    device = topk_ids.device
    M = topk_ids.shape[0]
    topk = topk_ids.shape[1]

    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    init_val = (topk << 24) | M
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_val, dtype=torch.int32, device=device
    )
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=device
    )
    num_tokens_post_pad = torch.empty((2,), dtype=torch.int32, device=device)

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expert_id in range(num_experts):
        token_id, topk_id = torch.where(topk_ids == expert_id)
        tokens_num = token_id.numel()
        sorted_expert_ids_num = (tokens_num + block_size - 1) // block_size
        tokens_num_pad = sorted_expert_ids_num * block_size

        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = (
            topk_id.to(torch.int32) << 24
        ) | token_id.to(torch.int32)
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[
            token_id, topk_id
        ].to(torch.float32)

        sorted_ids_begin += tokens_num_pad
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = (expert_id - skip_expert_num)
        sorted_expert_ids_begin += sorted_expert_ids_num

    num_tokens_post_pad[0] = sorted_ids_begin
    num_tokens_post_pad[1] = M

    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


# ---------------------------------------------------------------------------
# Per-token FP8 quantization
# ---------------------------------------------------------------------------
def _pertoken_quant_fp8(
    x: torch.Tensor, fp8_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row dynamic FP8 quantization.

    Args:
        x: Input tensor, any shape with last dim as the quant dimension.
        fp8_dtype: Target FP8 dtype.

    Returns:
        x_fp8: Quantized tensor (same shape as x).
        scale_1d: 1D scale tensor [num_rows].
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).float()
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x_2d.abs().amax(dim=-1, keepdim=True)  # [rows, 1]
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_scaled = (x_2d / scale).clamp(-fp8_max, fp8_max)
    x_fp8 = x_scaled.to(fp8_dtype).reshape(orig_shape)
    scale_1d = scale.view(-1).to(torch.float32)
    return x_fp8, scale_1d


# ---------------------------------------------------------------------------
# FlyDSL FP8 MOE dispatch
# ---------------------------------------------------------------------------
def flydsl_fp8_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    top_k: int,
    block_quant: bool,
    quant_type,
) -> torch.Tensor:
    """Execute FP8 MOE using FlyDSL MLIR-compiled 2-stage kernels.

    Drop-in replacement for _triton_fp8_moe(). Expects preshuffled weights
    (same CK layout as ASM/CK path).

    Pipeline:
        1. Sort tokens via torch-native MOE sorting
        2. Quantize activations to FP8
        3. Stage 1: GEMM1 + SiLU gating  (x @ w13^T)
        4. Quantize intermediate to FP8
        5. Stage 2: GEMM2 with atomic accumulation (intermediate @ w2^T)
    """
    from kernels.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2

    if block_quant:
        raise NotImplementedError(
            "FlyDSL MOE does not support block quantization yet. "
            "Set ATOM_USE_FLYDSL_MOE=0 or use Triton fallback."
        )

    M, model_dim = x.shape
    E = w13.shape[0]
    inter_dim_2 = w13.shape[1]  # 2 * inter_dim
    inter_dim = inter_dim_2 // 2
    actual_top_k = topk_ids.numel() // M
    device = x.device

    # Detect FP8 dtype from weight dtype
    fp8_dtype = w13.dtype

    # Tile sizes (configurable via env vars)
    tile_m = int(os.environ.get("ATOM_FLYDSL_MOE_TILE_M", "64"))
    tile_n = int(os.environ.get("ATOM_FLYDSL_MOE_TILE_N", "128"))
    tile_k = int(os.environ.get("ATOM_FLYDSL_MOE_TILE_K", "64"))

    # --- Step 1: Sort tokens ---
    sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad = (
        moe_sorting_torch_native(
            topk_ids.to(torch.int32),
            topk_weights.to(torch.float32),
            num_experts=E,
            block_size=tile_m,
        )
    )
    num_valid_ids = num_tokens_post_pad[:1].contiguous()
    blocks = sorted_expert_ids.numel()

    # --- Step 2: Compile kernels (cached via @lru_cache) ---
    exe1 = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=actual_top_k,
        in_dtype="fp8",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
    )
    exe2 = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=actual_top_k,
        in_dtype="fp8",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=True,
    )

    # --- Step 3: Quantize activations ---
    x_fp8, scale_x = _pertoken_quant_fp8(x, fp8_dtype)
    x_fp8 = x_fp8.contiguous().view(M, model_dim)
    scale_x_1d = scale_x.view(-1).contiguous()

    # --- Step 4: Flatten weights for FlyDSL (expert dim merged into N) ---
    # w13: [E, 2*inter_dim, model_dim] -> [E*2*inter_dim, model_dim]
    w13_flat = w13.contiguous().view(E * inter_dim_2, model_dim)

    # --- Step 5: Flatten weight scales ---
    # Handle different scale shapes:
    #   per-tensor: [E] -> broadcast to [E*2*inter_dim]
    #   per-channel: [E, 2*inter_dim, 1] -> [E*2*inter_dim]
    #   per-tensor (after max reduction): [E] -> broadcast
    if w13_scale.dim() == 1:
        # per-tensor [E]: broadcast each expert's scale to all its rows
        scale_w13_1d = (
            w13_scale.unsqueeze(1).expand(E, inter_dim_2).contiguous().view(-1)
        )
    elif w13_scale.dim() == 3:
        # per-channel [E, 2*inter_dim, 1]
        scale_w13_1d = w13_scale.contiguous().view(-1)
    elif w13_scale.dim() == 2:
        # [E, 2*inter_dim] or [E, 2] -> handle both
        if w13_scale.shape[1] == inter_dim_2:
            scale_w13_1d = w13_scale.contiguous().view(-1)
        else:
            # [E, 2] per-tensor per-shard -> broadcast
            scale_w13_1d = (
                w13_scale.unsqueeze(2).expand(E, 2, inter_dim).contiguous().view(-1)
            )
    else:
        scale_w13_1d = w13_scale.contiguous().view(-1)

    # --- Step 6: Stage 1 (GEMM1 + SiLU) ---
    out1 = torch.empty((M, actual_top_k, inter_dim), device=device, dtype=torch.float16)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe1(
        out1,
        x_fp8,
        w13_flat,
        scale_x_1d,
        scale_w13_1d,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights.view(-1),
        num_valid_ids,
        M,
        inter_dim,
        model_dim,
        blocks,
        stream_ptr,
    )

    # --- Step 7: Quantize intermediate for Stage 2 ---
    out1_fp32 = out1.to(torch.float32)
    a2_fp8, scale_a2 = _pertoken_quant_fp8(out1_fp32, fp8_dtype)
    a2_flat = a2_fp8.contiguous().view(-1)
    scale_a2_1d = scale_a2.view(-1).contiguous()

    # --- Step 8: Flatten w2 weights and scales ---
    # w2: [E, model_dim, inter_dim] -> [E*model_dim, inter_dim] -> flat 1D
    w2_flat = w2.contiguous().view(-1)

    if w2_scale.dim() == 1:
        # per-tensor [E]: broadcast to [E*model_dim]
        scale_w2_1d = w2_scale.unsqueeze(1).expand(E, w2.shape[1]).contiguous().view(-1)
    elif w2_scale.dim() == 3:
        # per-channel [E, model_dim, 1]
        scale_w2_1d = w2_scale.contiguous().view(-1)
    elif w2_scale.dim() == 2:
        scale_w2_1d = w2_scale.contiguous().view(-1)
    else:
        scale_w2_1d = w2_scale.contiguous().view(-1)

    # --- Step 9: Stage 2 (GEMM2 with atomic accumulation) ---
    out2 = torch.zeros((M, model_dim), device=device, dtype=torch.float16)

    exe2(
        out2.view(-1),
        a2_flat,
        w2_flat,
        scale_a2_1d,
        scale_w2_1d,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights.view(-1),
        num_valid_ids,
        M,
        model_dim,
        inter_dim,
        blocks,
        stream_ptr,
    )

    return out2.to(x.dtype)
