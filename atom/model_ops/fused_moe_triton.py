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

from __future__ import annotations

import os
import torch
from contextlib import contextmanager
from typing import Any
import logging
from math import ceil, prod
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


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert (
        prod(v) <= x.numel()
    ), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def _round_up(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def _fused_moe_lora_config(prefix: str) -> dict[str, int]:
    block_m = int(os.getenv(f"ATOM_FUSED_MOE_LORA_{prefix}_BLOCK_M", "64"))
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": int(os.getenv(f"ATOM_FUSED_MOE_LORA_{prefix}_BLOCK_N", "64")),
        "BLOCK_SIZE_K": int(os.getenv(f"ATOM_FUSED_MOE_LORA_{prefix}_BLOCK_K", "32")),
        "GROUP_SIZE_M": int(os.getenv(f"ATOM_FUSED_MOE_LORA_{prefix}_GROUP_M", "8")),
        "NUM_WARPS": int(os.getenv(f"ATOM_FUSED_MOE_LORA_{prefix}_WARPS", "4")),
        "NUM_STAGES": int(os.getenv(f"ATOM_FUSED_MOE_LORA_{prefix}_STAGES", "3")),
        "SPLIT_K": int(os.getenv(f"ATOM_FUSED_MOE_LORA_{prefix}_SPLIT_K", "1")),
    }


def _call_vllm_fused_moe_lora(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a_stacked: tuple[torch.Tensor, ...],
    lora_b_stacked: tuple[torch.Tensor, ...],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    max_lora_rank: int,
    topk: int,
    lora_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_config: dict[str, int],
    expand_config: dict[str, int],
    mul_routed_weight: bool = False,
):
    from vllm.lora.ops.triton_ops import fused_moe_lora

    common_args = (
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )
    kernel_configs = (
        adapter_enabled,
        shrink_config["BLOCK_SIZE_M"],
        shrink_config["BLOCK_SIZE_N"],
        shrink_config["BLOCK_SIZE_K"],
        shrink_config["GROUP_SIZE_M"],
        shrink_config["NUM_WARPS"],
        shrink_config["NUM_STAGES"],
        shrink_config["SPLIT_K"],
        expand_config["BLOCK_SIZE_M"],
        expand_config["BLOCK_SIZE_N"],
        expand_config["BLOCK_SIZE_K"],
        expand_config["GROUP_SIZE_M"],
        expand_config["NUM_WARPS"],
        expand_config["NUM_STAGES"],
        expand_config["SPLIT_K"],
        mul_routed_weight,
        False,
        0,
    )
    num_active_loras = torch.tensor([lora_ids.numel()], dtype=torch.int32, device="cpu")
    try:
        fused_moe_lora(
            *common_args,
            token_lora_mapping,
            max_lora_rank,
            topk,
            lora_ids,
            num_active_loras,
            *kernel_configs,
        )
    except TypeError as new_api_error:
        try:
            fused_moe_lora(
                *common_args,
                max_lora_rank,
                topk,
                lora_ids,
                *kernel_configs,
            )
        except TypeError:
            raise new_api_error


def _build_static_lora_routing(
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
    expert_map: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    from vllm import _custom_ops as ops

    max_loras = 1
    max_num_tokens_padded = _round_up(
        topk_ids.numel() + num_experts * (block_size - 1), block_size
    )
    max_num_m_blocks = ceil(max_num_tokens_padded / block_size)
    sorted_token_ids = torch.empty(
        (max_loras * max_num_tokens_padded,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (max_loras * max_num_m_blocks,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_padded = torch.empty(
        (max_loras,), dtype=torch.int32, device=topk_ids.device
    )
    token_lora_mapping = torch.zeros(
        topk_ids.shape[0], dtype=torch.int32, device=topk_ids.device
    )
    lora_ids = torch.zeros((max_loras,), dtype=torch.int32, device=topk_ids.device)
    adapter_enabled = torch.ones(
        (max_loras + 1,), dtype=torch.int32, device=topk_ids.device
    )

    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        adapter_enabled,
        lora_ids,
    )
    if expert_map is not None:
        mapped_expert_ids = torch.full_like(expert_ids, -1)
        valid_blocks = int(num_tokens_post_padded[0].item()) // block_size
        if valid_blocks > 0:
            valid_expert_ids = expert_ids[:valid_blocks].long()
            valid_mask = (valid_expert_ids >= 0) & (
                valid_expert_ids < expert_map.numel()
            )
            if bool(valid_mask.any().item()):
                mapped_expert_ids[:valid_blocks][valid_mask] = expert_map[
                    valid_expert_ids[valid_mask]
                ].to(mapped_expert_ids.dtype)
        expert_ids = mapped_expert_ids

    return (
        sorted_token_ids.view(max_loras, -1),
        expert_ids.view(max_loras, -1),
        num_tokens_post_padded,
        lora_ids,
        token_lora_mapping,
        max_loras,
    )


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
    static_lora: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert w1_bias is None or w1_bias.dtype == torch.float32
    assert w2_bias is None or w2_bias.dtype == torch.float32

    # Shape check, only check non-mxfp4
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
    lora_routing = None
    if static_lora is not None:
        w13_lora_a = static_lora["w13_a"]
        w13_lora_b = static_lora["w13_b"]
        w2_lora_a = static_lora["w2_a"]
        w2_lora_b = static_lora["w2_b"]
        lora_topk_ids = static_lora["topk_ids"].to(torch.int32)
        lora_topk_weights = static_lora["topk_weights"]
        w13_lora_a_tuple = (w13_lora_a[0].unsqueeze(0), w13_lora_a[1].unsqueeze(0))
        w13_lora_b_tuple = (w13_lora_b[0].unsqueeze(0), w13_lora_b[1].unsqueeze(0))
        w2_lora_a_tuple = (w2_lora_a[0].unsqueeze(0),)
        w2_lora_b_tuple = (w2_lora_b[0].unsqueeze(0),)
        w13_shrink_config = _fused_moe_lora_config("W13_SHRINK")
        w13_expand_config = _fused_moe_lora_config("W13_EXPAND")
        w2_shrink_config = _fused_moe_lora_config("W2_SHRINK")
        w2_expand_config = _fused_moe_lora_config("W2_EXPAND")
        lora_routing = _build_static_lora_routing(
            lora_topk_ids,
            num_experts=global_num_experts,
            block_size=w13_shrink_config["BLOCK_SIZE_M"],
            expert_map=expert_map,
        )
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            lora_ids,
            token_lora_mapping,
            _,
        ) = lora_routing
        adapter_enabled = torch.ones(
            (2,), dtype=torch.int32, device=lora_topk_ids.device
        )

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
            if static_lora is not None:
                raise NotImplementedError(
                    "Static routed LoRA is wired for the SiLU MoE path only."
                )
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
            if static_lora is None:
                raw_2d = raw_intermediate.view(M * topk, N)
            else:
                raw_2d = raw_intermediate.view(-1, N)[gather_indx.dst_indx]
                _call_vllm_fused_moe_lora(
                    raw_2d.view(M, topk, N),
                    hidden_states,
                    w13_lora_a_tuple,
                    w13_lora_b_tuple,
                    lora_topk_weights,
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                    int(w13_lora_a.shape[-2]),
                    topk,
                    lora_ids,
                    token_lora_mapping,
                    adapter_enabled,
                    w13_shrink_config,
                    w13_expand_config,
                    mul_routed_weight=False,
                )
            intermediate_cache = intermediate_cache.view(M * topk, half_N)
            fused_clamp_act_mul(
                raw_2d,
                out=intermediate_cache,
                swiglu_limit=swiglu_limit,
                activation="silu",
                dtype_quant=None,
            )
            intermediate_cache = intermediate_cache.view(batch_dim, M * topk, half_N)

        if static_lora is None:
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
        else:
            w2_output = torch.empty(
                (batch_dim, M * topk, K),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            routing_data.n_expts_act = 1
            matmul_ogs(
                intermediate_cache.view(M * topk, half_N)[gather_indx.src_indx],
                w2,
                w2_bias,
                routing_data,
                scatter_indx=scatter_indx,
                precision_config=w2_precision_config,
                gammas=None if apply_router_weight_on_input else gammas,
                y=w2_output,
            )
            _call_vllm_fused_moe_lora(
                w2_output.view(M, topk, K),
                intermediate_cache.view(M * topk, half_N),
                w2_lora_a_tuple,
                w2_lora_b_tuple,
                lora_topk_weights,
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                int(w2_lora_a.shape[-2]),
                topk,
                lora_ids,
                token_lora_mapping,
                adapter_enabled,
                w2_shrink_config,
                w2_expand_config,
                mul_routed_weight=True,
            )
            output_tensor.view(M, K).copy_(w2_output.view(M, topk, K).sum(dim=1))

    output_tensor = output_tensor.view(M, K)
    return output_tensor
