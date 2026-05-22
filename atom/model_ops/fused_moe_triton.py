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
from typing import Any
import logging
from math import prod
from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.fusions.fused_routing_from_topk import (
    fused_routing_from_topk as _aiter_fused_routing_from_topk,
)
from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
from atom.model_ops.moe_utils import (
    check_and_swizzle_scales,
)
from atom.utils import envs

from triton import next_power_of_2


logger = logging.getLogger("atom")

if envs.ATOM_USE_TRITON_GEMM:
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        moe_gemm_a8w4,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
        moe_gemm_a16w4,
        get_kernel_config,
    )
    
    
    from aiter.ops.triton.moe import moe_op_gemm_a16w4 as _aiter_a16w4
    import logging
    logging.getLogger("atom").warning(get_gfx())
    if get_gfx() == "gfx950":
        _orig_get_kernel_config = _aiter_a16w4.get_kernel_config
        _MAX_BLOCK_N = envs.ATOM_TRITON_MOE_MAX_BLOCK_N
        def _amd_smem_safe_tile(m, n, k, routing_data):
            cfg = _orig_get_kernel_config(m, n, k, routing_data)
            # Cap block_n on CDNA4: BLOCK_M=128 * BLOCK_N=512 * 4B FP32 acc = 256 KiB,
            # which exceeds MI355X's 160 KiB LDS when triton spills the accumulator.
            # Pinning block_n <= 256 keeps acc <= 128 KiB.
            # previous safe_mem function to prevent LDS overflow
            if cfg["block_m"] >= 64 and cfg["block_n"] > _MAX_BLOCK_N:
                cfg["block_n"] = _MAX_BLOCK_N
            return cfg
        _aiter_a16w4.get_kernel_config = _amd_smem_safe_tile


        logging.getLogger("atom").warning(
            "[atom] aiter moe_op_gemm_a16w4.get_kernel_config patched -> %s (max_block_n=%d)",
            _aiter_a16w4.get_kernel_config.__name__, _MAX_BLOCK_N,
        )
    
    from aiter.ops.triton.moe.quant_moe import downcast_to_static_fp8


# amd_smem_safe_tile is no longer needed as atom now uses aiter kernels rather than triton_kernels matmul_ogs
# @contextmanager
# def _amd_smem_safe_tile():
#     """Cap matmul_ogs tile size on AMD CDNA4 to fit MI355X's 160 KiB LDS.

#     triton_kernels' AMD opt_flags has a special-case
#     `if cdna4 and block_m == 128: block_n = 512`, which makes BLOCK_M*BLOCK_N
#     = 64K FP32 entries — large enough that triton 3.6+/3.7+ spills the
#     accumulator into LDS and overflows the 160 KiB budget (observed 269 KiB
#     on V4-Pro FP8 MoE). triton 3.5 happened to keep more of the acc in
#     registers and slipped under the limit, hence the version-dependent OOM.

#     Pin block_n ≤ ATOM_TRITON_MOE_MAX_BLOCK_N (default 256) so BLOCK_M*BLOCK_N
#     stays at 32K. Default block_n in compute_block_nk is already capped at
#     256 except for that single cdna4 branch, so this only sidesteps the bad
#     path on gfx950.
#     """
#     if get_gfx() != "gfx950":
#         yield
#         return
#     try:
#         from triton_kernels.matmul_ogs_details.opt_flags import (
#             update_opt_flags_constraints,
#             reset_opt_flags_constraints,
#         )
#     except ImportError:
#         yield
#         return
#     # Defaults chosen so BLOCK_M*BLOCK_N stays ≤ 16384 entries (64 KiB FP32
#     # acc), comfortably fitting MI355X's register file. Override via env if
#     # a future compiler/kernel update relaxes the budget.
#     block_m = int(os.getenv("ATOM_TRITON_MOE_BLOCK_M", "32"))
#     block_n = int(os.getenv("ATOM_TRITON_MOE_BLOCK_N", "256"))
#     update_opt_flags_constraints({"block_m": block_m, "block_n": block_n})
#     try:
#         yield
#     finally:
#         reset_opt_flags_constraints()


def _swizzle_mxfp4(w1, w1_scale, w2, w2_scale, w_dtype, N_1, K_1, N_2, K_2, TP=1):
    """weight swizzle for mxfp4 moe, used for aiter triton weight mxfp4 moe method/kernels"""

    # assert environment variable is active for use with triton gemm
    assert envs.ATOM_USE_TRITON_GEMM

    # transposing for expected layout of aiter triton kernels
    w1_triton_layout = w1.transpose(-2, -1)
    w1_scale_triton_layout = w1_scale.transpose(-2, -1)
    w2_triton_layout = w2.transpose(-2, -1)
    w2_scale_triton_layout = w2_scale.transpose(-2, -1)

    w1_scale_triton_layout, w1_swizzle_layout = check_and_swizzle_scales(w1_scale_triton_layout, N_1, K_1)
    w2_scale_triton_layout, w2_swizzle_layout = check_and_swizzle_scales(
        w2_scale_triton_layout, N_2, K_2
    )
    
    return w1_triton_layout, w1_scale_triton_layout, w1_swizzle_layout, w2_triton_layout, w2_scale_triton_layout, w2_swizzle_layout


def fused_routing_from_topk_triton(topk_weights, topk_ids, n_expts_tot):
    """Build matmul_ogs routing data via the AITER fused-routing kernel.

    Thin bridge over ``aiter.ops.triton.fused_routing_from_topk``: invokes
    the single-CTA counting-sort kernel for small NK and packages the
    resulting indices into the ``RoutingData`` / ``GatherIndx`` /
    ``ScatterIndx`` structures consumed by
    ``triton_kernels.matmul_ogs``. For ``NK = n_tokens * n_expts_act``
    above the kernel's single-CTA budget (prefill-shaped inputs), falls
    back to the multi-kernel ``routing_from_topk`` reference defined
    below — that path does the per-row sort + global stable argsort in
    plain torch and is correctness-stable at any NK.

    Equivalence vs reference: the fused kernel skips the per-row sort,
    so ``topk_indx`` / ``gate_indx`` differ at intra-expert ordering.
    ``hist`` and the per-(token, expert, weight) bucket assignments
    match exactly; ``matmul_ogs`` is commutative over per-expert slices
    so the MoE output is unchanged (up to FP non-associativity).
    """
    if not has_triton_kernels():
        return routing_from_topk(topk_weights, topk_ids, n_expts_tot)

    n_tokens, n_expts_act = topk_weights.shape
    n_gates_pad = n_tokens * n_expts_act

    if n_gates_pad > 4096:
        # Single-CTA design exceeded; fall back rather than degrading
        # silently. Typically only hit during prefill.
        return routing_from_topk(topk_weights, topk_ids, n_expts_tot)

    hist, topk_indx, gate_indx, gate_scal = _aiter_fused_routing_from_topk(
        topk_weights, topk_ids, n_expts_tot
    )

    from aiter.ops.triton.moe.moe_routing.routing import (
        routing,
        RoutingData,
    )

    from moe_utils import compute_expt_data

    m = n_tokens * n_expts_act
    tokens_per_expt = max(1, m // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    expt_data = compute_expt_data(hist, n_expts_tot, n_gates_pad, block_m)

    routing_data = RoutingData(block_m, gate_scal, hist, n_expts_tot, n_expts_act, expt_data)
    return routing_data, topk_indx, gate_indx #topk_indx = gather_indx, gate_indx = scatter_indx


def routing_from_topk(topk_weights, topk_ids, n_expts_tot):
    """Convert FusedMoE.select_experts output to triton routing data structures.

    This bridges the gap between ATOM's grouped topk / sigmoid routing
    (which triton_kernels routing() does not support) and the triton
    matmul_ogs compute kernels.

    Args:
        topk_weights: (n_tokens, n_expts_act) routing weights from select_experts
        topk_ids: (n_tokens, n_expts_act) expert indices from select_experts
        n_expts_tot: total number of experts (global, before EP)

    Returns:
        (RoutingData, GatherIndx, ScatterIndx) compatible with triton_kernel_fused_experts
    """
    from aiter.ops.triton.moe.moe_routing.routing import (
        routing,
        RoutingData,
    )

    from atom.model_ops.moe_utils import compute_expt_data

    n_tokens, n_expts_act = topk_weights.shape
    n_gates_pad = n_tokens * n_expts_act

    # Sort each token's selected experts by expert_id (required by triton kernels)
    expt_indx_sorted, sort_indices = torch.sort(topk_ids.int(), dim=1)
    expt_scal_sorted = torch.gather(topk_weights, 1, sort_indices.long())

    # Flatten to 1D
    expt_scal = expt_scal_sorted.reshape(-1).to(topk_weights.dtype)
    expt_indx = expt_indx_sorted.reshape(-1).to(torch.int32)

    # Sort by expert_id globally so experts are contiguous for the matmul
    topk_indx = torch.argsort(expt_indx, stable=True).int()
    gate_indx = torch.argsort(topk_indx, stable=True).int()
    gate_scal = expt_scal[topk_indx.long()].to(torch.bfloat16)

    # Histogram of tokens over experts
    hist = torch.histc(expt_indx.float(), bins=n_expts_tot, max=n_expts_tot - 1).int()

    # Build routing data structures using triton-accelerated compute_expt_data
    m = n_tokens * n_expts_act
    tokens_per_expt = max(1, m // n_expts_tot)
    block_m = max(16, min(next_power_of_2(tokens_per_expt), 128))
    expt_data = compute_expt_data(hist, n_expts_tot, n_gates_pad, block_m)

    routing_data = RoutingData(block_m, gate_scal, hist, n_expts_tot, n_expts_act, expt_data)
    return routing_data, topk_indx, gate_indx #topk_indx = gather_indx, gate_indx = scatter_indx


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
    activation: str = "silu",
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w13_swizzle_layout: torch.Tensor | None = None,
    w2_swizzle_layout: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    x_q_dtype: str | None = None,
    static_scale: torch.Tensor | None = None,
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
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        x_q_dtype=x_q_dtype,
        static_scale=static_scale
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
    activation: str = "silu",
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
    a1q_scale: torch.Tensor | None = None,
    x_q_dtype: str | None = None,
    static_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert w1_bias is None or w1_bias.dtype == torch.float32
    assert w2_bias is None or w2_bias.dtype == torch.float32

    # Shape check
    # Changes to weight handling before this function, therefore shape check change
    assert hidden_states.ndim == 2
    # assert hidden_states.shape[-1] == w1.shape[-2] * 2 
    # assert w2.shape[-1] == w1.shape[-2]  * 2

    # unecessary since aiter kernels expect 2d inputs/outputs
    # batch_dim = 1
    M, K = hidden_states.shape[-2:] # 
    E, _, N = w1.shape 

    if global_num_experts == -1:
        global_num_experts = E

    half_N = N // 2

    # if intermediate_cache is None:
    #     intermediate_cache = torch.empty(
    #         (M * topk, half_N),
    #         device=hidden_states.device,
    #         dtype=hidden_states.dtype,
    #     )
    
    # # Add batch_dim to output buffer because matmul_ogs expects 3D output
    # intermediate_cache = _resize_cache(
    #     intermediate_cache, (M * topk, half_N)
    # )

    output_tensor = _resize_cache(output_tensor, (M, K))

    gammas = routing_data.gate_scal if routing_data else None

    # raw_intermediate = torch.empty(
    #     (M * topk, N),
    #     device=hidden_states.device,
    #     dtype=hidden_states.dtype,
    # )

    if activation == ActivationType.Swiglu:
        # SwiGLU (GPT OSS): fused activation with interleaved [gate, up] layout
        x_q_dtype_base = x_q_dtype.split("_")[0] if x_q_dtype else None
        if (x_q_dtype_base == "fp8"):
            from aiter.ops.triton.utils._triton.arch_info import get_arch
            # If input type is fp8, input scales must be available
            assert a13_scale is not None
            assert a2_scale is not None

            # vllm-like processing
            a13_scale = a13_scale.max().to(torch.float32)
            a2_scale = a2_scale.max().to(torch.float32)

            x_dtype = torch.float8_e4m3fn # model is fp8_e4m3 type

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
                out_dtype=torch.float8_e4m3fn,
                apply_swiglu=True,
                alpha=1.702,          # gpt-oss
                limit=7.0,            # gpt-oss
                add_residual=True,    # gpt-oss `(up + 1)`
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
                alpha=1.702,          # gpt-oss
                limit=7.0,            # gpt-oss
                add_residual=True,    # gpt-oss `(up + 1)`
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
        # SiLU (DeepSeek): concatenated [gate | up] layout, manual activation
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
        
        raw_2d = raw_intermediate #.view(M * topk, N)
        # intermediate_cache = intermediate_cache.view(M * topk, half_N)
        intermediate_cache = fused_clamp_act_mul(
            raw_2d,
            swiglu_limit=swiglu_limit,
            activation="silu",
            dtype_quant=None,
        )

        # logger.warning("intermediate cache")
        # logger.warning(intermediate_cache.view(M * topk, half_N))
        # logger.warning(intermediate_cache.shape)

        # logger.warning("swizzle layouts")
        # logger.warning(w13_swizzle_layout)
        # logger.warning(w2_swizzle_layout)

        # logger.warning("numerics")
        # logger.warning(M * topk)
        # logger.warning(half_N)
        # logger.warning(N)

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
        logger.warning("out")
        logger.warning("output")
        return output_tensor

    output_tensor = output_tensor.view(M, K) 
    return output_tensor
