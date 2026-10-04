# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache

import torch
import triton
import triton.language as tl

_K3_EXPERTS = 896
_K3_TOPK = 16


@triton.jit
def _k3_sigmoid_biased_top16(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    stride_logits,
    stride_weights,
    stride_ids,
    n_rows,
    routed_scaling,
    N_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_RENORM: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = (row < n_rows) & (cols < N_EXPERTS)
    logits = tl.load(logits_ptr + row * stride_logits + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scores = 1.0 / (1.0 + tl.exp(-logits))
    choice = tl.where(mask, scores + bias, float("-inf"))
    # AITER ignores NaNs during its strict-greater local maximum scan.
    choice = tl.where(choice == choice, choice, float("-inf"))  # noqa: PLR0124

    slots = tl.arange(0, TOPK)
    selected_ids = tl.zeros((TOPK,), dtype=tl.int32)
    selected_weights = tl.zeros((TOPK,), dtype=tl.float32)

    # Preserve AITER's gfx1250 wave32 tie order. Its float4 LDS kernel assigns
    # vectors round-robin to lanes and reduces with the legacy DPP/permlane
    # priority below; local positions keep source order.
    lane = (cols // 4) % 32
    local_pos = (cols // 128) * 4 + cols % 4
    low2 = (lane & 3) ^ tl.where((lane & 4) != 0, 3, 0)
    bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2
    lane_rank = ((lane ^ 0x38) & 0x38) | bit2 | low2
    tie_rank = lane_rank * 32 + local_pos

    for k in tl.static_range(TOPK):
        max_choice = tl.max(choice, axis=0)
        rank = tl.where(choice == max_choice, tie_rank, 0x7FFFFFFF)
        idx = tl.argmin(rank, axis=0, tie_break_left=True).to(tl.int32)
        value = tl.sum(tl.where(cols == idx, scores, 0.0), axis=0)
        selected_ids = tl.where(slots == k, idx, selected_ids)
        selected_weights = tl.where(slots == k, value, selected_weights)
        choice = tl.where(cols == idx, float("-inf"), choice)

    if NEED_RENORM:
        selected_weights *= routed_scaling / tl.sum(selected_weights, axis=0)
    else:
        selected_weights *= routed_scaling

    row_mask = row < n_rows
    tl.store(
        ids_ptr + row * stride_ids + slots,
        selected_ids,
        mask=row_mask,
    )
    tl.store(
        weights_ptr + row * stride_weights + slots,
        selected_weights,
        mask=row_mask,
    )


@cache
def _is_gfx1250(device_index: int) -> bool:
    arch = getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx1250"


def can_use_k3_biased_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int,
) -> bool:
    return (
        gating_output.is_cuda
        and gating_output.shape[0] > 0
        and gating_output.ndim == 2
        and gating_output.shape[1] == _K3_EXPERTS
        and gating_output.dtype == torch.bfloat16
        and gating_output.stride(1) == 1
        and correction_bias.shape == (_K3_EXPERTS,)
        and correction_bias.dtype == torch.bfloat16
        and correction_bias.device == gating_output.device
        and correction_bias.is_contiguous()
        and topk == _K3_TOPK
        and topk_weights.shape == (gating_output.shape[0], topk)
        and topk_ids.shape == (gating_output.shape[0], topk)
        and topk_weights.dtype == torch.float32
        and topk_ids.dtype == torch.int32
        and topk_weights.device == gating_output.device
        and topk_ids.device == gating_output.device
        and topk_weights.stride(1) == 1
        and topk_ids.stride(1) == 1
        and num_expert_group == 1
        and topk_group == 1
        and num_fused_shared_experts == 0
        and _is_gfx1250(gating_output.device.index or 0)
    )


def k3_biased_grouped_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    need_renorm: bool,
    routed_scaling_factor: float,
) -> None:
    _k3_sigmoid_biased_top16[(gating_output.shape[0],)](
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        gating_output.shape[0],
        routed_scaling_factor,
        N_EXPERTS=_K3_EXPERTS,
        TOPK=_K3_TOPK,
        BLOCK_N=1024,
        NEED_RENORM=need_renorm,
        num_warps=4,
    )
