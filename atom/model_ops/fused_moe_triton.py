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

from math import prod

import torch
from aiter import ActivationType
from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
from aiter.ops.triton.utils._triton.arch_info import get_arch

from atom.utils import envs

if (
    envs.ATOM_USE_TRITON_GEMM
    or envs.ATOM_USE_TRITON_MOE
    or envs.ATOM_USE_TRITON_MOE_DECODE
    # The EP path reaches these same helpers from inside the modular kernel, so
    # it must pull the imports in too -- otherwise they are NameErrors at call
    # time when only the EP flag is set.
    or envs.ATOM_USE_TRITON_MOE_EP
):
    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
        moe_gemm_a4w4,
        mxfp4_quant,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        moe_gemm_a8w4,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
        moe_gemm_a16w4,
    )
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, downcast_to_static_fp8
    from aiter.ops.triton.utils.shuffle import shuffle_scale_moe

from atom.model_ops.moe import MoEActivationQuant


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
):
    """Weight swizzle for mxfp4 moe, used for aiter triton mxfp4 moe kernels.

    The arch -> SWIZZLE_MX_SCALE label decision lives in aiter
    (``shuffle_scale_moe(..., return_layout=True)``), so this stays arch-agnostic.
    """
    assert envs.ATOM_USE_TRITON_GEMM or envs.ATOM_USE_TRITON_MOE

    # Transposing for expected layout of aiter triton kernels
    w1_triton_layout = w1.transpose(-2, -1)
    w1_scale_triton_layout = w1_scale.transpose(-2, -1)
    w2_triton_layout = w2.transpose(-2, -1)
    w2_scale_triton_layout = w2_scale.transpose(-2, -1)

    if N_1 % 32 == 0 and K_1 % (32 * 8) == 0:
        w1_scale_triton_layout, w1_swizzle_layout = shuffle_scale_moe(
            w1_scale_triton_layout, return_layout=True
        )
    else:
        w1_swizzle_layout = None

    if N_2 % 32 == 0 and K_2 % (32 * 8) == 0:
        w2_scale_triton_layout, w2_swizzle_layout = shuffle_scale_moe(
            w2_scale_triton_layout, return_layout=True
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


def _compute_expt_data_triton(hist, n_expts_tot, n_gates, block_m):
    """Cudagraph-safe ExptData, replacing routing.compute_expt_data_torch.

    The torch reference builds `block_pid_map` with a Python loop whose trip
    count comes off a device tensor (`for b in range(n_tiles[e])`), so it syncs
    per expert -- illegal during capture ("operation not permitted when stream
    is capturing"). aiter already ships the same computation as a kernel:
    `_expt_data_only_kernel` does stage1+stage2 from a precomputed histogram,
    which is exactly our situation (we have `hist`, we just need the offsets and
    tile map). Grid is (n_expts_tot,).

    The kernel deliberately does no memset, so `block_pid_map` is pre-filled
    with -1 here: the matmul reads it as "this pid has no work" (`if expt_data ==
    -1: return`), and leaving it uninitialized would let a tile process garbage.

    One benign divergence from the torch reference: its terminal
    `token_offs_raw[n_expts_tot]` is `sum(hist)` while the kernel writes
    `n_gates` (identical in non-EP where the two are equal, far apart under EP).
    That entry is never read -- the matmul indexes `ExptOffs[expt_id]` only for
    `expt_id < n_expts_tot`, and `ExptOffsSum` is fed from `token_offs_pad[-1]`
    (moe_op_gemm_a8w4.py:468), which matches exactly. Verified equal for
    `token_offs_pad` and `block_pid_map` across (NE, n_gates, block_m) shapes.
    """
    from aiter.ops.triton._triton_kernels.moe.moe_routing.expt_data import (
        _expt_data_only_kernel,
    )
    from aiter.ops.triton.moe.moe_routing.routing import (
        ExptData,
        _compute_expt_data_internal,
    )

    (
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
        blocks1,
        BLOCK,
        block_m_log2,
    ) = _compute_expt_data_internal(n_expts_tot, n_gates, block_m, hist.device)

    block_pid_map.fill_(-1)
    _expt_data_only_kernel[(blocks1,)](
        hist,
        n_expts_tot,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
        block_pid_map.shape[0],
        n_gates,
        block_m_log2,
        BLOCK,
        EQUAL_BLOCK=(n_expts_tot == BLOCK),
    )
    return ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)


def routing_from_dispatched(
    dispatch_weights: torch.Tensor,
    dispatch_ids: torch.Tensor,
    expert_map: torch.Tensor,
    num_local_experts: int,
    num_local_tokens: torch.Tensor | None = None,
):
    """Build triton RoutingData / gather / scatter from mori-dispatched rows.

    The EP path cannot use aiter's ``routing()``: that starts from router logits,
    but after the all-to-all the top-k choice is already made and the rows have
    been permuted across ranks. This is ``routing_torch``'s second half --
    everything from ``(expt_scal, expt_indx)`` onward -- adapted for three facts
    about the post-dispatch buffer:

    1. Rows are per-token: mori sends one copy per (token, destination rank), so
       a row carries the full top-k tuple with only *some* entries owned here.
       Non-local entries go to a sentinel bin that is sliced off the histogram,
       so the matmul schedules no block for them.
    2. The flat gate index must stay ``row * topk + slot``, because the matmul
       recovers the activation row as ``gather_idx // N_EXPTS_ACT``. Non-local
       entries are therefore **masked, never compacted** -- compacting would
       break that arithmetic and silently read the wrong rows.
    3. ``num_local_tokens`` is a device tensor and rows past it hold garbage from
       the over-allocated receive buffer. Masking them the same way keeps this
       function sync-free and its shapes static (so it stays cudagraph-safe).

    Returns ``(routing_data, gather_indx, scatter_indx, gate_valid)`` -- the first
    three match ``routing()``; ``gate_valid`` is the extra piece EP needs, since
    ``routing()`` never produces dead gates.
    """
    import triton
    from aiter.ops.triton.moe.moe_routing.routing import RoutingData

    M, topk = dispatch_ids.shape
    device = dispatch_ids.device
    # One extra bin collects everything this rank must not compute.
    sentinel = num_local_experts

    # Garbage rows can hold out-of-range ids; clamp before the gather. `.long()`
    # already copies, so clamping in place cannot touch the caller's tensor.
    ids = dispatch_ids.long().clamp_(0, expert_map.numel() - 1)
    local_ids = expert_map[ids]
    if num_local_tokens is not None:
        rows = torch.arange(M, device=device, dtype=torch.int32).unsqueeze(1)
        local_ids = torch.where(
            rows < num_local_tokens.reshape(1, 1).to(rows.dtype), local_ids, -1
        )

    # Per-gate validity, in flat gate order (row * topk + slot) -- the same
    # layout scatter_indx uses, so reduce_grouped's .view(-1, n_expts_act) lines
    # up slot-for-slot. A dead slot's sorted position is never written by the
    # GEMM (the sentinel keeps the matmul off it), so the reduce must be told to
    # skip it rather than sum uninitialized memory.
    gate_valid = (local_ids >= 0).reshape(-1).to(torch.int32)

    expt_indx = torch.where(local_ids < 0, sentinel, local_ids).reshape(-1).int()
    expt_scal = dispatch_weights.reshape(-1).float()

    # Sort by expert so each expert's rows are contiguous for the matmul. The
    # sentinel is the largest id, so masked entries land in the unused tail.
    topk_indx = torch.argsort(expt_indx, stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)
    gate_scal = expt_scal[topk_indx]

    # Histogram via index_add_ into a fixed-size buffer. NOT torch.bincount: it
    # sizes its output from the data's max, which forces a device->host sync and
    # is rejected outright during cudagraph capture ("operation not permitted
    # when stream is capturing"). Dropping the sentinel bin leaves
    # sum(hist) == the number of (token, local expert) pairs actually computed.
    hist_full = torch.zeros(sentinel + 1, dtype=torch.int32, device=device)
    hist_full.index_add_(
        0, expt_indx.long(), torch.ones_like(expt_indx, dtype=torch.int32)
    )
    hist = hist_full[:num_local_experts]

    n_gates = M * topk
    # Same derivation as routing_torch. Note n_gates counts every gate slot while
    # only ~1/topk are live under EP, so this overstates real per-expert
    # occupancy and picks larger tiles than the work needs. That is a
    # perf/tiling concern, not correctness: the matmul wraps its gather with
    # `offs_x_m % hist[e]` and masks stores with `offs_m < hist[e]`, so a
    # mostly-empty tile recomputes a live row rather than reading garbage.
    tokens_per_expt = max(1, n_gates // max(num_local_experts, 1))
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    expt_data = _compute_expt_data_triton(hist, num_local_experts, n_gates, block_m)
    routing_data = RoutingData(
        block_m, gate_scal, hist, num_local_experts, topk, expt_data
    )
    return routing_data, topk_indx.int(), gate_indx.int(), gate_valid


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


def triton_kernel_fused_experts_a8w4_silu_gguu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
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
) -> torch.Tensor:
    """Decode-only A8W4 MoE for SiLU models, GGUU (separated ``[gate|up]``).

    GGUU keeps gate and up as contiguous halves, so the per-block SiLU cannot be
    fused into GEMM1's write-back (a tile spans only gate *or* only up). The
    activation and quant therefore run as a separate step:

        MXFP8 quant -> GEMM1(a8w4, no swiglu, bf16 [gate|up]) ->
        fused_clamp_act_mul(SiLU(gate)*up on the halves) ->
        MXFP8 quant -> GEMM2(a8w4).

    The intermediate is re-quantized with ``downcast_to_mxfp`` (same op as the x
    path) so GEMM2 sees the identical activation-scale format. Weights are in the
    preshuffled a8w4 layout with w13 gate/up separated.
    """
    assert hidden_states.ndim == 2
    assert hidden_states.dtype == torch.bfloat16

    gammas = routing_data.gate_scal if routing_data else None

    x_fp8, x_scale = downcast_to_mxfp(hidden_states, torch.float8_e4m3fn, axis=-1)

    # GEMM1: raw bf16 [gate|up] output; no fused activation for the separated layout.
    interm = moe_gemm_a8w4(
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
        out_dtype=torch.bfloat16,
        preshuffled=True,
    )

    # Standalone SiLU(gate)*up over the contiguous halves, then MXFP8 quant.
    interm_act = fused_clamp_act_mul(
        interm, swiglu_limit=swiglu_limit, activation="silu"
    )
    interm_fp8, interm_scale = downcast_to_mxfp(
        interm_act, torch.float8_e4m3fn, axis=-1
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


def triton_kernel_fused_experts_a8w4_silu_gugu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
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
    gate_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """A8W4 MoE for SiLU models, GUGU (interleaved ``[gate, up]``).

    Interleaved is the a8w4 kernel's native layout: ``_swiglu`` splits
    ``reshape(M, N // 2, 2)`` on the trailing axis, i.e. adjacent gate/up pairs,
    so a BLOCK_N tile carries both halves and the activation fuses into GEMM1's
    write-back. ``out_mx_quant=True`` folds the MXFP8 requant in with it, so the
    whole layer is two launches:

        MXFP8 quant -> GEMM1(a8w4, fused SiLU + MX requant) -> GEMM2(a8w4)

    versus four on the GGUU path (GEMM1 -> fused_clamp_act_mul ->
    downcast_to_mxfp -> GEMM2), which needs the separate steps precisely because
    a tile there spans only gate *or* only up.

    ``alpha=1.0`` with ``swiglu_add_residual=False`` is plain SiLU (``s * linear``).
    GPT-OSS uses ``swiglu_add_residual=True`` for its ``s * (linear + 1)`` variant,
    which would be wrong for DeepSeek-V4.
    """
    assert hidden_states.ndim == 2
    assert hidden_states.dtype == torch.bfloat16

    gammas = routing_data.gate_scal if routing_data else None

    # Only gfx1250's gluon kernel consumes the WMMA-preshuffled weight; the
    # CDNA triton kernel takes a plain (E, K, N) weight.
    _preshuffled = get_arch() == "gfx1250"

    x_fp8, x_scale = downcast_to_mxfp(hidden_states, torch.float8_e4m3fn, axis=-1)

    # GEMM1: SiLU(gate)*up fused into write-back, emitting (fp8 e4m3, ue8m0)
    # directly. out_mx_quant requires split_k == 1 and no scatter_indx, both of
    # which hold for a GEMM1-style call.
    interm_fp8, interm_scale = moe_gemm_a8w4(
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
        apply_swiglu=True,
        alpha=1.0,
        limit=swiglu_limit,
        swiglu_add_residual=False,
        out_mx_quant=True,
        out_dtype=torch.float8_e4m3fn,
        preshuffled=_preshuffled,
    )

    return moe_gemm_a8w4(
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
        preshuffled=_preshuffled,
        # Only GEMM2 feeds reduce_grouped, so the mask belongs here. GEMM1's
        # dead slots are already skipped by the sentinel histogram.
        gate_valid=gate_valid,
    )
