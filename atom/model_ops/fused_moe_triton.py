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

import os
import torch
from contextlib import contextmanager
from typing import Any
import logging
from math import prod
from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.fusions.fused_routing_from_topk import (
    fused_routing_from_topk as _aiter_fused_routing_from_topk,
)
from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
from atom.model_ops.utils import has_triton_kernels

logger = logging.getLogger("atom")


if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import (
            FnSpecs,
            FusedActivation,
            PrecisionConfig,
            matmul_ogs,
        )
        from triton_kernels.routing import routing
    except (AttributeError, ImportError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


@contextmanager
def _amd_smem_safe_tile():
    """Cap matmul_ogs tile size on AMD CDNA4 to fit MI355X's 160 KiB LDS.

    triton_kernels' AMD opt_flags has a special-case
    `if cdna4 and block_m == 128: block_n = 512`, which makes BLOCK_M*BLOCK_N
    = 64K FP32 entries — large enough that triton 3.6+/3.7+ spills the
    accumulator into LDS and overflows the 160 KiB budget (observed 269 KiB
    on V4-Pro FP8 MoE). triton 3.5 happened to keep more of the acc in
    registers and slipped under the limit, hence the version-dependent OOM.

    Pin block_n ≤ ATOM_TRITON_MOE_MAX_BLOCK_N (default 256) so BLOCK_M*BLOCK_N
    stays at 32K. Default block_n in compute_block_nk is already capped at
    256 except for that single cdna4 branch, so this only sidesteps the bad
    path on gfx950.
    """
    if get_gfx() != "gfx950" or not has_triton_kernels():
        yield
        return
    try:
        from triton_kernels.matmul_ogs_details.opt_flags import (
            update_opt_flags_constraints,
            reset_opt_flags_constraints,
        )
    except ImportError:
        yield
        return
    # Defaults chosen so BLOCK_M*BLOCK_N stays ≤ 16384 entries (64 KiB FP32
    # acc), comfortably fitting MI355X's register file. Override via env if
    # a future compiler/kernel update relaxes the budget.
    block_m = int(os.getenv("ATOM_TRITON_MOE_BLOCK_M", "32"))
    block_n = int(os.getenv("ATOM_TRITON_MOE_BLOCK_N", "256"))
    update_opt_flags_constraints({"block_m": block_m, "block_n": block_n})
    try:
        yield
    finally:
        reset_opt_flags_constraints()


def _swizzle_mxfp4(quant_tensor, scale):
    """weight swizzle for mxfp4 moe, used for OAI mxfp4 kernel"""
    assert has_triton_kernels()
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details.layout import StridedLayout

    value_layout_opts: dict[str, Any] = {}
    scale_layout_opts: dict[str, Any] = {}
    value_layout = StridedLayout
    if get_gfx() == "gfx950":
        from triton_kernels.tensor_details.layout import CDNA4MXScaleLayout

        scale_layout = CDNA4MXScaleLayout
    else:
        scale_layout = StridedLayout

    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    return quant_tensor, InFlexData(), scale


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

    # Package as the matmul_ogs routing data structures.
    from triton_kernels.routing import (
        RoutingData,
        GatherIndx,
        ScatterIndx,
        compute_expt_data,
    )

    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    expt_data = compute_expt_data(hist, n_expts_tot, n_gates_pad)

    routing_data = RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data)
    return routing_data, gather_indx, scatter_indx


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
    from triton_kernels.routing import (
        RoutingData,
        GatherIndx,
        ScatterIndx,
        compute_expt_data,
    )

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
    gate_scal = expt_scal[topk_indx.long()]

    # Histogram of tokens over experts
    hist = torch.histc(expt_indx.float(), bins=n_expts_tot, max=n_expts_tot - 1).int()

    # Build routing data structures using triton-accelerated compute_expt_data
    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    expt_data = compute_expt_data(hist, n_expts_tot, n_gates_pad)

    routing_data = RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data)
    return routing_data, gather_indx, scatter_indx


def bridge_tk_to_aiter_routing(tk_rd, block_m: int):
    """Bridge triton_kernels RoutingData -> aiter RoutingData (for moe_gemm_a8w4).

    triton_kernels carries per-block-m specializations of token_offs_pad /
    block_pid_map as dicts keyed by block_m; the aiter variant is specialized
    for a single block_m and stores those fields as plain tensors.
    """
    from aiter.ops.triton.moe.moe_routing.routing import RoutingData, ExptData

    tk_ed = tk_rd.expt_data
    if block_m not in tk_ed.token_offs_pad:
        raise ValueError(
            f"bridge_tk_to_aiter_routing: block_m={block_m} not present in "
            f"tk_rd.expt_data.token_offs_pad. Keys: {sorted(tk_ed.token_offs_pad.keys())}"
        )
    aiter_expt_data = ExptData(
        hist=tk_ed.hist,
        token_offs_raw=tk_ed.token_offs_raw,
        token_offs_pad=tk_ed.token_offs_pad[block_m],
        block_pid_map=tk_ed.block_pid_map[block_m],
    )
    return RoutingData(
        block_m=block_m,
        gate_scal=tk_rd.gate_scal,
        expt_hist=tk_rd.expt_hist,
        n_expts_tot=tk_rd.n_expts_tot,
        n_expts_act=tk_rd.n_expts_act,
        expt_data=aiter_expt_data,
    )


def _a8w4_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,            # a8w4 layout: uint8 [E, K/2, 2N], stride(-2)==1
    w2: torch.Tensor,            # a8w4 layout: uint8 [E, N/2, K], stride(-2)==1
    w1_scale: torch.Tensor,      # swizzled ue8m0 uint8
    w2_scale: torch.Tensor,      # swizzled ue8m0 uint8
    routing_data,
    gather_indx,
    scatter_indx,
    topk: int,
    swiglu_limit: float,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    apply_router_weight_on_input: bool,
    pre_built_aiter_routing=None,  # if set, skip bridge and use this aiter routing directly
    # Step 9: shared-expert output to fold into GEMM2 reduce_grouped writeback.
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Step 1 minimum a8w4 fused experts.

    Pipeline: downcast_to_mxfp -> moe_gemm_a8w4 (G1, no swiglu) ->
              fused_clamp_act_mul (silu_mul, no quant) -> downcast_to_mxfp ->
              moe_gemm_a8w4 (G2). No apply_swiglu fold, no out_mx_quant fold,
              no residual fold — those are later steps.
    """
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4, recommend_block_m
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

    M, K = hidden_states.shape[-2:]
    BLOCK_M = recommend_block_m(M)
    # Step 3: apply_swiglu fold into GEMM1. Requires interleaved W13 (done at weight load).
    _swiglu_fold = os.environ.get("ATOM_A8W4_SWIGLU_FOLD", "0") == "1"

    # Pre-MoE quant: bf16 -> fp8 e4m3 + ue8m0 per-1x32.
    x_fp8, x_scale = downcast_to_mxfp(
        hidden_states, torch.float8_e4m3fn, axis=-1
    )

    # Step 4: skip bridge when pre_built_aiter_routing is provided.
    if pre_built_aiter_routing is not None:
        aiter_routing = pre_built_aiter_routing
        gammas = aiter_routing.gate_scal
        gather_src_indx = gather_indx  # already raw int32 tensor
        scatter_src_indx = scatter_indx
    else:
        aiter_routing = bridge_tk_to_aiter_routing(routing_data, BLOCK_M)
        gammas = routing_data.gate_scal if routing_data else None
        gather_src_indx = gather_indx.src_indx
        scatter_src_indx = scatter_indx.src_indx

    if w1_scale.dtype != torch.uint8:
        w1_scale = w1_scale.view(torch.uint8)
    if w2_scale.dtype != torch.uint8:
        w2_scale = w2_scale.view(torch.uint8)

    if _swiglu_fold:
        # Step 5: optional fold of MXFP8 (fp8 e4m3 + ue8m0 per-1×32) emit into
        # GEMM1's apply_swiglu writeback. Saves one `downcast_to_mxfp` launch.
        _mx_emit = os.environ.get("ATOM_A8W4_GEMM1_MX_EMIT", "0") == "1"
        if _mx_emit:
            inter_fp8, inter_scale = moe_gemm_a8w4(
                x=x_fp8,
                w=w1,
                x_scales=x_scale,
                w_scales=w1_scale,
                bias=w1_bias,
                routing_data=aiter_routing,
                gather_indx=gather_src_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale="CDNA4_SCALE",
                apply_swiglu=True,
                alpha=1.0,
                limit=swiglu_limit,
                add_residual=False,
                out_dtype=torch.bfloat16,  # ignored when out_mx_quant=True
                out_mx_quant=True,
            )
        else:
            # Step 3: GEMM1 with apply_swiglu=True. Reads interleaved W13. Returns bf16 [M*topk, N].
            # add_residual=False = V4-Flash semantics (silu(gate) * up). True would compute s*(up+1).
            inter_bf16 = moe_gemm_a8w4(
                x=x_fp8,
                w=w1,
                x_scales=x_scale,
                w_scales=w1_scale,
                bias=w1_bias,
                routing_data=aiter_routing,
                gather_indx=gather_src_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale="CDNA4_SCALE",
                apply_swiglu=True,
                alpha=1.0,
                limit=swiglu_limit,
                add_residual=False,
                out_dtype=torch.bfloat16,
            )
            inter_bf16 = inter_bf16.view(inter_bf16.shape[-2], inter_bf16.shape[-1])
            inter_fp8, inter_scale = downcast_to_mxfp(
                inter_bf16, torch.float8_e4m3fn, axis=-1
            )
    else:
        # GEMM1: fp8 act x fp4 weight, bf16 output [M*topk, 2N].
        raw_intermediate = moe_gemm_a8w4(
            x=x_fp8,
            w=w1,
            x_scales=x_scale,
            w_scales=w1_scale,
            bias=w1_bias,
            routing_data=aiter_routing,
            gather_indx=gather_src_indx,
            gammas=gammas if apply_router_weight_on_input else None,
            swizzle_mx_scale="CDNA4_SCALE",
            apply_swiglu=False,
            out_dtype=torch.bfloat16,
        )
        raw_2d = raw_intermediate.view(
            raw_intermediate.shape[-2], raw_intermediate.shape[-1]
        )
        two_N = raw_2d.shape[-1]
        inter_bf16 = torch.empty(
            (raw_2d.shape[0], two_N // 2),
            device=raw_2d.device,
            dtype=torch.bfloat16,
        )
        fused_clamp_act_mul(
            raw_2d,
            out=inter_bf16,
            swiglu_limit=swiglu_limit,
            activation="silu",
            dtype_quant=None,
        )
        inter_fp8, inter_scale = downcast_to_mxfp(
            inter_bf16, torch.float8_e4m3fn, axis=-1
        )

    # GEMM2: fp8 act x fp4 weight, scatter+combine.
    # Step 9: pass `residual` so reduce_grouped folds the routed+shared add
    # into the writeback (saves the standalone elementwise add launch).
    y = moe_gemm_a8w4(
        x=inter_fp8,
        w=w2,
        x_scales=inter_scale,
        w_scales=w2_scale,
        bias=w2_bias,
        routing_data=aiter_routing,
        scatter_indx=scatter_src_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale="CDNA4_SCALE",
        apply_swiglu=False,
        out_dtype=torch.bfloat16,
        residual=residual,
    )
    return y


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
    w13_precision_config: PrecisionConfig | None = None,
    w2_precision_config: PrecisionConfig | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
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
        w13_precision_config=w13_precision_config,
        w2_precision_config=w2_precision_config,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    topk: int,
    activation: str = "silu",
    w13_precision_config: PrecisionConfig | None = None,
    w2_precision_config: PrecisionConfig | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
    w13_weight_scale_a8w4: torch.Tensor | None = None,
    w2_weight_scale_a8w4: torch.Tensor | None = None,
    swiglu_limit_a8w4: float = 7.0,
) -> torch.Tensor:
    # type check
    assert hidden_states.dtype == torch.bfloat16
    assert w1_bias is None or w1_bias.dtype == torch.float32
    assert w2_bias is None or w2_bias.dtype == torch.float32

    # a8w4 path: dispatched when env is set AND a8w4 scale tensors were plumbed
    # through. matmul_ogs path below stays untouched.
    _use_a8w4 = (
        os.environ.get("ATOM_MOE_BACKEND", "matmul_ogs") == "a8w4"
        and w13_weight_scale_a8w4 is not None
        and w2_weight_scale_a8w4 is not None
    )
    if _use_a8w4:
        return _a8w4_fused_experts(
            output_tensor,
            hidden_states,
            w1,
            w2,
            w13_weight_scale_a8w4,
            w2_weight_scale_a8w4,
            routing_data,
            gather_indx,
            scatter_indx,
            topk,
            swiglu_limit=swiglu_limit_a8w4,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

    # Shape check for matmul_ogs path (a8w4 has different layout)
    assert hidden_states.ndim == 2
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    batch_dim = 1
    M, K = hidden_states.shape[-2:]
    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    half_N = N // 2

    if intermediate_cache is None:
        intermediate_cache = torch.empty(
            (batch_dim, M * topk, half_N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    # Add batch_dim to output buffer because matmul_ogs expects 3D output
    intermediate_cache = _resize_cache(
        intermediate_cache, (batch_dim, M * topk, half_N)
    )
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    gammas = routing_data.gate_scal if routing_data else None

    # NOTE: We intentionally do NOT use the triton fused SwiGLU activation
    # because it expects interleaved [gate0, up0, gate1, up1, ...] layout
    # while our w13 weights produce concatenated [gate | up] output.
    # It also uses a non-standard formula: s*sigmoid(alpha*s)*(linear+1)
    # with alpha=1.702, which differs from the standard SiLU activation
    # (x*sigmoid(x)*up) used by most MoE models.
    # Instead, we compute the matmul without fused activation and apply
    # standard silu(gate) * up manually.
    raw_intermediate = torch.empty(
        (batch_dim, M * topk, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    with _amd_smem_safe_tile():
        if activation == ActivationType.Swiglu:
            # SwiGLU (GPT OSS): fused activation with interleaved [gate, up] layout
            act = FusedActivation(
                FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
                (swiglu_alpha, swiglu_limit),
                2,
            )
            matmul_ogs(
                hidden_states,
                w1,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                precision_config=w13_precision_config,
                gammas=gammas if apply_router_weight_on_input else None,
                fused_activation=act,
                y=intermediate_cache,
            )
        else:
            # SiLU (DeepSeek): concatenated [gate | up] layout, manual activation
            raw_intermediate = matmul_ogs(
                hidden_states,
                w1,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                precision_config=w13_precision_config,
                gammas=gammas if apply_router_weight_on_input else None,
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
            intermediate_cache = intermediate_cache.view(batch_dim, M * topk, half_N)

        matmul_ogs(
            intermediate_cache.view(M * topk, half_N),
            w2,
            w2_bias,
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=w2_precision_config,
            gammas=None if apply_router_weight_on_input else gammas,
            y=output_tensor,
        )

    output_tensor = output_tensor.view(M, K)
    return output_tensor
