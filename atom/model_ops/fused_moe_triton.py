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
import triton
from aiter import ActivationType
from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
from aiter.ops.triton.utils._triton.arch_info import get_arch

from atom.utils import envs

if envs.ATOM_USE_TRITON_GEMM or envs.ATOM_USE_TRITON_MOE:
    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
        moe_gemm_a4w4,
        mxfp4_quant,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        get_kernel_config_gluon,
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


def routing_from_dispatched(
    dispatch_weights: torch.Tensor,
    dispatch_ids: torch.Tensor,
    expert_map: torch.Tensor,
    num_local_experts: int,
    num_local_tokens: torch.Tensor,
    ep_scatter_geometry=None,
):
    """Build triton RoutingData / gather / scatter from mori-dispatched rows.
    Thin wrapper over aiter's ``ep_sort_routing``, which owns the sort itself
    (gating, histogram, scatter) because the EP path cannot use ``routing()``:
    that starts from router logits, but after the all-to-all the top-k choice is
    already made and the rows have been permuted across ranks. What stays here is
    the tile geometry, the ExptData allocation and the packaging -- shaped by
    three facts about the post-dispatch buffer:
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
       It is REQUIRED, not optional: the mori buffer always has M > R, so
       skipping the row mask would fold garbage rows into the histogram as live
       gates -- silently wrong rather than an error.
    Returns ``(routing_data, gather_indx, scatter_indx, gate_valid, dst_row)`` --
    the first three match ``routing()``; ``gate_valid`` is the extra piece EP
    needs, since ``routing()`` never produces dead gates. ``dst_row`` is None
    unless ``ep_scatter_geometry`` asked for the combine-scatter map.
    """
    from aiter.ops.triton.moe.moe_routing.routing import (
        ExptData,
        RoutingData,
        _compute_expt_data_internal,
        ep_sort_routing,
    )

    M, topk = dispatch_ids.shape
    device = dispatch_ids.device
    n_gates = M * topk

    # gate_valid is in flat gate order (row * topk + slot) -- the same layout
    # scatter_indx uses, so reduce_grouped's .view(-1, n_expts_act) lines up
    # slot-for-slot. A dead slot's sorted position is never written by the GEMM
    # (the sentinel keeps the matmul off it), so the reduce must be told to skip
    # it rather than sum uninitialized memory.

    # Same derivation as routing_torch. Note n_gates counts every gate slot while
    # only ~1/topk are live under EP, so this overstates real per-expert
    # occupancy and picks larger tiles than the work needs. That is a
    # perf/tiling concern, not correctness: the matmul wraps its gather with
    # `offs_x_m % hist[e]` and masks stores with `offs_m < hist[e]`, so a
    # mostly-empty tile recomputes a live row rather than reading garbage.
    #
    # Allocation-only (no kernel), and hoisted above the sort because
    # ep_sort_routing fills these buffers in the same launches as the histogram
    # and the scatter.
    global_num_experts = max(1, expert_map.numel() - 1)
    tokens_per_expt = max(1, n_gates // global_num_experts)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    expt_data_bufs = _compute_expt_data_internal(
        num_local_experts, n_gates, block_m, device
    )
    token_offs_raw, token_offs_pad, block_pid_map, _blocks1, _BLOCK, _block_m_log2 = (
        expt_data_bufs
    )

    hist_full, topk_indx, gate_indx, gate_scal, gate_valid, dst_row = ep_sort_routing(
        dispatch_weights,
        dispatch_ids,
        expert_map,
        num_local_experts,
        num_local_tokens,
        M,
        topk,
        n_gates,
        expt_data_bufs,
        ep_scatter_geometry=ep_scatter_geometry,
    )
    # The tail bin holds the sentinel (non-local) count, which the matmul must
    # schedule no tile for.
    hist = hist_full[:num_local_experts]
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)

    routing_data = RoutingData(
        block_m, gate_scal, hist, num_local_experts, topk, expt_data
    )
    return routing_data, topk_indx, gate_indx, gate_valid, dst_row


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert (
        prod(v) <= x.numel()
    ), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def _gluon_fused_quant_supported(m, n, k, routing_data) -> bool:
    """Will the gfx1250 kernel that moe_gemm_a8w4 picks support out_mx_quant?

    moe_gemm_a8w4 chooses between three gluon kernels, in this order, and only
    the middle one implements the fused MXFP8 requant epilogue:

        persistent_iters > 1      -> _moe_gemm_a8w4_decode_persistent   no
        block_m == 16             -> _moe_gemm_a8w4_decode              YES
        (otherwise)               -> _moe_gemm_a8w4_prefill             no

    This must be exact, because getting it wrong is silently WRONG rather than
    an error: moe_gemm_a8w4 allocates and returns the y_scale buffer whenever
    out_mx_quant is set, but only the kernels that take HAS_MX_OUT ever write
    it -- an unsupported pick returns uninitialised scales, and there is no
    assert on the aiter side to catch it.

    block_m comes from routing_data (tokens-per-expert, see
    routing_from_dispatched), NOT from a prefill/decode flag: a decode step at
    high enough concurrency raises block_m above 16 and lands on the prefill
    kernel. So this has to be recomputed per call, not cached per layer.

    Defers the persistent decision to aiter's own selector instead of restating
    its thresholds, so this stays correct if that heuristic moves. The selector
    reports it as `persistent_iters` -- the count of N-tiles one program walks --
    and the persistent kernel is the one launched when that exceeds 1, matching
    moe_gemm_a8w4's own `config["persistent_iters"] > 1` test.
    """
    if routing_data is None or getattr(routing_data, "block_m", None) is None:
        return False
    return get_kernel_config_gluon(m, n, k, routing_data)["persistent_iters"] <= 1


def _fused_experts_silu_gugu(
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
    preshuffled: bool | None = None,
    act_quant: MoEActivationQuant = MoEActivationQuant.BF16,
    y_out: torch.Tensor | None = None,
    ep_scatter=None,
) -> torch.Tensor:
    """Fused-SiLU MoE experts over GUGU (interleaved ``[gate, up]``) weights.

    Interleaved is the w4 kernels' native layout: ``_swiglu`` splits
    ``reshape(M, N // 2, 2)`` on the trailing axis, i.e. adjacent gate/up pairs,
    so a BLOCK_N tile carries both halves and the activation fuses into GEMM1's
    write-back. On a8w4 ``out_mx_quant=True`` folds the MXFP8 requant in with it,
    so the whole layer is two launches:

        MXFP8 quant -> GEMM1(a8w4, fused SiLU + MX requant) -> GEMM2(a8w4)

    against four on the GGUU path (GEMM1 -> fused_clamp_act_mul ->
    downcast_to_mxfp -> GEMM2), which needs the separate steps precisely because
    a tile there spans only gate *or* only up.

    ``alpha=1.0`` with ``swiglu_add_residual=False`` is plain SiLU
    (``s * linear``); GPT-OSS's ``s * (linear + 1)`` variant is NOT served here.

    a8w4 is the default. a4w4 -- selected by ``ATOM_USE_TRITON_MOE_A4W4`` or an
    MXFP4-activation checkpoint -- takes the SAME weights (both are w4), so it
    needs no extra weight prep or memory; it just costs one launch more, since
    ``moe_gemm_a4w4`` has no ``out_mx_quant`` and must re-quantise the
    intermediate separately. Measured on gfx950 that was 1.22-1.28x overall (see
    _test/ep_moe_bench_report.md): the a4w4 GEMMs are genuinely faster (halved
    weight traffic) but the doubled quant costs more than the GEMM saving.

    ``y_out``, when given, is where GEMM2's grouped reduction writes -- typically
    the leading rows of a taller caller-owned buffer, so the EP path can hand
    mori's combine a full-M tensor without copying this result into it. It is
    GEMM2's output only; GEMM1's intermediate is always kernel-allocated.

    ``ep_scatter`` replaces that reduction entirely: GEMM2 delivers its un-reduced
    rows into a peer combine-staging window and the sum happens in the EP combine,
    once every rank has delivered. Mutually exclusive with ``y_out`` -- one names a
    reduced output, the other says there is no local reduction to produce one.
    """
    assert hidden_states.ndim == 2
    assert hidden_states.dtype == torch.bfloat16
    assert y_out is None or ep_scatter is None, (
        "y_out and ep_scatter are alternatives: the first names GEMM2's reduced "
        "output, the second says the rows leave unreduced"
    )

    gammas = routing_data.gate_scal if routing_data else None

    # Only gfx1250's gluon kernel consumes the WMMA-preshuffled weight; the
    # CDNA triton kernel takes a plain (E, K, N) weight.
    _preshuffled = (
        (get_arch() == "gfx1250") if preshuffled is None else bool(preshuffled)
    )

    if envs.ATOM_USE_TRITON_MOE_A4W4 or act_quant == MoEActivationQuant.FP4:
        assert ep_scatter is None, (
            "the scatter-fused EP combine is wired through moe_gemm_a8w4 only; "
            "moe_gemm_a4w4 has no ep_scatter, so a4w4 must use the gather combine"
        )
        # ``moe_gemm_a4w4`` has no ``preshuffled`` parameter, so it can only
        # consume the plain (E, K, N) weight -- i.e. a4w4 is a CDNA-only variant.
        # Handed a gfx1250 WMMA-preshuffled weight it reads N as N // 16 and
        # dies inside mxfp4_quant on a block-size assert with no hint as to why,
        # so say it here instead.
        assert not _preshuffled, (
            "ATOM_USE_TRITON_MOE_A4W4 needs the plain (E, K, N) weight layout, "
            "but this layer was prepared WMMA-preshuffled (gfx1250). "
            "moe_gemm_a4w4 has no preshuffled variant -- a4w4 is CDNA-only."
        )
        x_fp4, x_scale = mxfp4_quant(hidden_states)
        interm = moe_gemm_a4w4(
            x_fp4,
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
        )
        # The launch a8w4 avoids via out_mx_quant=True.
        interm_fp4, interm_scale = mxfp4_quant(interm)
        return moe_gemm_a4w4(
            interm_fp4,
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
            # Only GEMM2 feeds reduce_grouped, so the mask belongs here. GEMM1's
            # dead slots are already skipped by the sentinel histogram.
            gate_valid=gate_valid,
            y_out=y_out,
        )

    x_fp8, x_scale = downcast_to_mxfp(hidden_states, torch.float8_e4m3fn, axis=-1)

    # GEMM1: SiLU(gate)*up fused into write-back.
    #
    # out_mx_quant folds the MXFP8 requant into GEMM1's epilogue, saving a whole
    # downcast_to_mxfp launch over the (M, intermediate) intermediate. The CDNA
    # triton kernel has always supported it. On gfx1250 only ONE of the three
    # gluon kernels does, so we must predict which one moe_gemm_a8w4 will pick
    # -- see _gluon_fused_quant_supported for why guessing is unsafe.
    #
    # Derive the shapes exactly as moe_gemm_a8w4 does: M from the gather index,
    # K from the (already fp8) activation, N from the weight -- x16 under the
    # preshuffled layout, whose last dim is N // 16. We pass no unpadded_N /
    # unpadded_K, so it applies no further adjustment.
    if get_arch() != "gfx1250":
        _fuse_requant = True
    else:
        _m = (
            gather_indx.shape[0]
            if gather_indx is not None
            else hidden_states.shape[-2]
        )
        _k = x_fp8.shape[-1]
        _n = w1.shape[-1] * 16 if _preshuffled else w1.shape[-1]
        _fuse_requant = _gluon_fused_quant_supported(_m, _n, _k, routing_data)
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
        apply_swiglu=True,
        alpha=1.0,
        limit=swiglu_limit,
        swiglu_add_residual=False,
        out_mx_quant=_fuse_requant,
        out_dtype=torch.float8_e4m3fn if _fuse_requant else torch.bfloat16,
        preshuffled=_preshuffled,
    )
    if _fuse_requant:
        interm_fp8, interm_scale = interm
    else:
        interm_fp8, interm_scale = downcast_to_mxfp(
            interm, torch.float8_e4m3fn, axis=-1
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
        y_out=y_out,
        ep_scatter=ep_scatter,
    )




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
    # Select the fused-SiLU a8w4/a4w4 experts over GUGU (interleaved
    # [g0,u0,g1,u1,...]) w13, instead of the general path's GGUU ([gate|up]).
    # The caller is the authority: the layout was decided by
    # process_weights_after_loading, and nothing in the tensors themselves
    # distinguishes the two. Named for the TP condition that gates it in
    # Mxfp4MoEMethod, which is the only caller that passes it conditionally --
    # the EP callers pass True unconditionally and do run on CDNA, where
    # `preshuffled` resolves to False.
    use_triton_gfx1250_silu: bool = False,
    # EP only: dead-gate mask for rows the all-to-all did not route here.
    # routing() never produces dead gates, so the TP callers leave it None.
    gate_valid: torch.Tensor | None = None,
    # GUGU only. None = pick by arch (pre-shuffled only on gfx1250, where the
    # gluon kernel supports it). Overridable so the two weight layouts can be
    # A/B'd from one process: pre-shuffle is a pure layout change, so the same
    # source weights through either path must agree numerically, which validates
    # process_weights_after_loading's is_gfx1250 branch without needing a
    # reference implementation.
    preshuffled: bool | None = None,
    # GUGU only. Buffer GEMM2's grouped reduction writes into, in place of a
    # kernel-allocated one; may be a row-slice of a taller tensor. Distinct from
    # `output_tensor`, which the GGUU path resizes to (M, K) and which the GUGU
    # path ignores. The kernel asserts shape/dtype/device/last-dim-contiguity.
    y_out: torch.Tensor | None = None,
    # GUGU only. Deliver GEMM2's un-reduced rows to an EP combine-staging window
    # instead of reducing them locally (aiter's EpCombineScatter). Alternative to
    # y_out, not a companion.
    ep_scatter=None,
) -> torch.Tensor:
    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert w1_bias is None or w1_bias.dtype == torch.float32
    assert w2_bias is None or w2_bias.dtype == torch.float32

    # Shape check
    # Changes to weight handling before this function, therefore shape check change
    assert hidden_states.ndim == 2

    if use_triton_gfx1250_silu:
        # Sits ABOVE the shared preamble rather than inside the SiLU branch
        # below, because that preamble reads N out of `w1.shape[-1]` and sizes an
        # intermediate cache from it. Under GUGU the weight is the pre-shuffled
        # (E, K*16, N//16) view, so N would come out 16x too small -- and the
        # fused path allocates no intermediate cache and writes no caller-owned
        # output buffer anyway, so none of that setup applies.
        assert activation != ActivationType.Swiglu, (
            "GUGU fused experts implement plain SiLU (alpha=1, no residual). "
            "GPT-OSS-style SwiGLU needs alpha=1.702 / swiglu_add_residual=True "
            "and static per-tensor FP8 activations -- use the GGUU path."
        )
        return _fused_experts_silu_gugu(
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_indx,
            scatter_indx,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_swizzle_layout=w13_swizzle_layout,
            w2_swizzle_layout=w2_swizzle_layout,
            a13_scale=a13_scale,
            a2_scale=a2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=swiglu_limit,
            apply_router_weight_on_input=apply_router_weight_on_input,
            gate_valid=gate_valid,
            preshuffled=preshuffled,
            act_quant=act_quant,
            y_out=y_out,
            ep_scatter=ep_scatter,
        )

    assert y_out is None and ep_scatter is None, (
        "y_out / ep_scatter are only wired through the GUGU fused-SiLU path; the "
        "GGUU path writes its result into `output_tensor`."
    )

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
