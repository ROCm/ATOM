# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py
# Copyright 2023 The vLLM team.
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from math import prod
from aiter import ActivationType
from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
from aiter.ops.triton.utils._triton.arch_info import get_arch
from atom.utils import envs

if envs.ATOM_USE_TRITON_GEMM or envs.ATOM_USE_TRITON_MOE or envs.ATOM_USE_TRITON_MOE_DECODE:
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        moe_gemm_a8w4,
        swizzle_scales as swizzle_scales_a8w4,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
        moe_gemm_a16w4,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
        moe_gemm_a4w4,
        mxfp4_quant,
        swizzle_scales as swizzle_scales_cdna4,
    )
    from aiter.ops.triton.moe.quant_moe import downcast_to_static_fp8
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

from atom.model_ops.moe import MoEActivationQuant


def _swizzle_scales_for_kernel(scale, act_quant: MoEActivationQuant):
    """Swizzle MoE weight scales for the kernel that act_quant selects.

    FP8 (a8w4): arch-agnostic swizzle (CDNA4 on gfx950, GFX1250 on gfx1250).
    BF16/FP4 (a16w4/a4w4): CDNA4 swizzle on gfx942/gfx950, no swizzle elsewhere.
    """
    if act_quant == MoEActivationQuant.FP8:
        return swizzle_scales_a8w4(scale)
    # TODO: move arch dispatch into aiter's a4w4/a16w4 swizzle_scales (like a8w4)
    if get_arch() in ("gfx942", "gfx950"):
        return swizzle_scales_cdna4(scale), "CDNA4_SCALE"
    return scale, None


def _swizzle_mxfp4(
    w1,
    w1_scale,
    w2,
    w2_scale,
    w_dtype,
    N_1,
    K_1,
    N_2,
    K_2,
    TP=1,
    act_quant: MoEActivationQuant = MoEActivationQuant.BF16,
):
    """Weight swizzle for mxfp4 moe, used for aiter triton mxfp4 moe kernels."""
    assert envs.ATOM_USE_TRITON_GEMM or envs.ATOM_USE_TRITON_MOE

    # Transposing for expected layout of aiter triton kernels
    w1_triton_layout = w1.transpose(-2, -1)
    w1_scale_triton_layout = w1_scale.transpose(-2, -1)
    w2_triton_layout = w2.transpose(-2, -1)
    w2_scale_triton_layout = w2_scale.transpose(-2, -1)

    if N_1 % 32 == 0 and K_1 % (32 * 8) == 0:
        w1_scale_triton_layout, w1_swizzle_layout = _swizzle_scales_for_kernel(
            w1_scale_triton_layout, act_quant
        )
    else:
        w1_swizzle_layout = None

    if N_2 % 32 == 0 and K_2 % (32 * 8) == 0:
        w2_scale_triton_layout, w2_swizzle_layout = _swizzle_scales_for_kernel(
            w2_scale_triton_layout, act_quant
        )
    else:
        w2_swizzle_layout = None

    return (
        w1_triton_layout,
        w1_scale_triton_layout,
        w1_swizzle_layout,
        w2_triton_layout,
        w2_scale_triton_layout,
        w2_swizzle_layout,
    )


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert (
        prod(v) <= x.numel()
    ), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: ActivationType = ActivationType.Silu,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w13_swizzle_layout: torch.Tensor | None = None,
    w2_swizzle_layout: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    act_quant: MoEActivationQuant = MoEActivationQuant.BF16,
) -> torch.Tensor:
    routing_data, gather_idx, scatter_idx = routing(
        gating_output, topk, sm_first=not renormalize
    )

    output = torch.empty_like(hidden_states)

    return triton_kernel_fused_experts(
        output,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        topk=topk,
        activation=activation,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        a13_scale=a13_scale,
        a2_scale=a2_scale,
        w13_swizzle_layout=w13_swizzle_layout,
        w2_swizzle_layout=w2_swizzle_layout,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        swiglu_limit=swiglu_limit,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        act_quant=act_quant,
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx -> tensor
    scatter_indx,  # ScatterIndx -> tensor
    topk: int,
    activation: ActivationType = ActivationType.Silu,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w13_swizzle_layout: torch.Tensor | None = None,
    w2_swizzle_layout: torch.Tensor | None = None,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    act_quant: MoEActivationQuant = MoEActivationQuant.BF16,
) -> torch.Tensor:
    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert w1_bias is None or w1_bias.dtype == torch.float32
    assert w2_bias is None or w2_bias.dtype == torch.float32

    # Shape check
    # Changes to weight handling before this function, therefore shape check change
    assert hidden_states.ndim == 2

    # aiter kernels expect 2d inputs/outputs
    M, K = hidden_states.shape[-2:]
    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    half_N = N // 2

    if intermediate_cache is None:
        intermediate_cache = torch.empty(
            (M * topk, half_N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    # Add batch_dim to output buffer because matmul_ogs expects 3D output
    intermediate_cache = _resize_cache(intermediate_cache, (M * topk, half_N))

    output_tensor = _resize_cache(output_tensor, (M, K))

    gammas = routing_data.gate_scal if routing_data else None

    if activation == ActivationType.Swiglu:
        # SwiGLU (GPT OSS): fused activation with interleaved [gate, up] layout
        if act_quant == MoEActivationQuant.FP8:
            assert a13_scale is not None
            assert a2_scale is not None

            quant_dtype = torch.float8_e4m3fn
            if get_arch() == "gfx942":
                quant_dtype = torch.float8_e4m3fnuz

            hidden_states = downcast_to_static_fp8(hidden_states, a13_scale)
            interm_cache = moe_gemm_a8w4(
                hidden_states,
                w1,
                None,
                w13_scale,
                a13_scale,
                a2_scale,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale=w13_swizzle_layout,
                out_dtype=quant_dtype,
                apply_swiglu=True,
                alpha=swiglu_alpha,
                limit=swiglu_limit,
                swiglu_add_residual=True,
            )
            output_tensor = moe_gemm_a8w4(
                interm_cache,
                w2,
                None,
                w2_scale,
                a2_scale,
                None,
                w2_bias,
                routing_data,
                scatter_indx=scatter_indx,
                gammas=None if apply_router_weight_on_input else gammas,
                swizzle_mx_scale=w2_swizzle_layout,
            )
        else:
            interm_cache = moe_gemm_a16w4(
                hidden_states,
                w1,
                None,
                w13_scale,
                None,
                None,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale=w13_swizzle_layout,
                apply_swiglu=True,
                alpha=swiglu_alpha,
                limit=swiglu_limit,
                swiglu_add_residual=True,  # gpt-oss `(up + 1)`
            )
            output_tensor = moe_gemm_a16w4(
                interm_cache,
                w2,
                None,
                w2_scale,
                None,
                None,
                w2_bias,
                routing_data,
                scatter_indx=scatter_indx,
                gammas=None if apply_router_weight_on_input else gammas,
                swizzle_mx_scale=w2_swizzle_layout,
            )
    else:
        # SiLU (DeepSeek): concatenated [gate | up] layout, manual activation.
        # The activation precision selects the routed GEMM: MXFP4 activations
        # (a4w4) when act_quant is FP4, otherwise bf16 activations (a16w4).
        if act_quant == MoEActivationQuant.FP8:
            raise NotImplementedError(
                "SiLU activation with FP8 act_quant is not implemented in the "
                "triton MoE kernel. Only the SwiGLU branch supports FP8 "
                "activations (moe_gemm_a8w4)."
            )
        if act_quant == MoEActivationQuant.FP4:
            hidden_states_fp4, hidden_states_mx_scale = mxfp4_quant(hidden_states)
            raw_intermediate = moe_gemm_a4w4(
                hidden_states_fp4,
                w1,
                hidden_states_mx_scale,
                w13_scale,
                None,
                None,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale=w13_swizzle_layout,
                apply_swiglu=False,
            )
        else:
            raw_intermediate = moe_gemm_a16w4(
                hidden_states,
                w1,
                None,
                w13_scale,
                None,
                None,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale=w13_swizzle_layout,
                apply_swiglu=False,
            )

        raw_2d = raw_intermediate.view(M * topk, N)
        intermediate_cache = intermediate_cache.view(M * topk, half_N)
        fused_clamp_act_mul(
            raw_2d,
            out=intermediate_cache,
            swiglu_limit=swiglu_limit,
            activation="silu",
            dtype_quant=None,
        )

        if act_quant == MoEActivationQuant.FP4:
            intermediate_fp4, intermediate_mx_scale = mxfp4_quant(intermediate_cache)
            output_tensor = moe_gemm_a4w4(
                intermediate_fp4,
                w2,
                intermediate_mx_scale,
                w2_scale,
                None,
                None,
                w2_bias,
                routing_data,
                scatter_indx=scatter_indx,
                gammas=None if apply_router_weight_on_input else gammas,
                swizzle_mx_scale=w2_swizzle_layout,
            )
        else:
            output_tensor = moe_gemm_a16w4(
                intermediate_cache,
                w2,
                None,
                w2_scale,
                None,
                None,
                w2_bias,
                routing_data,
                scatter_indx=scatter_indx,
                gammas=None if apply_router_weight_on_input else gammas,
                swizzle_mx_scale=w2_swizzle_layout,
            )

        return output_tensor

    output_tensor = output_tensor.view(M, K)
    return output_tensor


def triton_kernel_fused_experts_a8w4_silu(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    topk: int,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_swizzle_layout,
    w2_swizzle_layout,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_limit: float = 10.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode-only A8W4 MoE for SiLU models (DSv4).

    Pipeline: MXFP8 quant -> GEMM1(a8w4) -> SiLU -> MXFP8 quant -> GEMM2(a8w4).
    Weights must be in preshuffled layout (shuffle_weight_gfx1250).
    Scales must be GFX1250-swizzled.
    """
    assert hidden_states.ndim == 2
    assert hidden_states.dtype == torch.bfloat16

    M, K = hidden_states.shape
    N = w1.shape[-1] * 16
    half_N = N // 2

    gammas = routing_data.gate_scal if routing_data else None

    x_fp8, x_scale = downcast_to_mxfp(
        hidden_states, torch.float8_e4m3fn, axis=-1
    )

    raw_intermediate = moe_gemm_a8w4(
        x_fp8,
        w1,
        x_scale,
        w13_scale,
        a13_scale,
        None,
        w1_bias,
        routing_data,
        gather_indx=gather_indx,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale=w13_swizzle_layout,
        apply_swiglu=False,
        preshuffled=True,
    )

    raw_2d = raw_intermediate.view(M * topk, N)
    intermediate = torch.empty(
        (M * topk, half_N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    fused_clamp_act_mul(
        raw_2d,
        out=intermediate,
        swiglu_limit=swiglu_limit,
        activation="silu",
        dtype_quant=None,
    )

    interm_fp8, interm_scale = downcast_to_mxfp(
        intermediate, torch.float8_e4m3fn, axis=-1
    )

    output_tensor = moe_gemm_a8w4(
        interm_fp8,
        w2,
        interm_scale,
        w2_scale,
        a2_scale,
        None,
        w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=w2_swizzle_layout,
        preshuffled=True,
    )

    return output_tensor
