# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Grouped GEMM (MoE) operations with multi-backend fallback.

All backends are AITER implementations — no torch fallbacks.

Supports multiple quantization modes for grouped matrix multiplication
used in Mixture-of-Experts architectures.

Backends:
    - **Triton GMM**: ``gmm`` / ``ptgmm`` / ``nptgmm`` (BF16/FP16 only)
    - **Triton MOE GEMM**: ``moe_gemm_a8w8``, ``moe_gemm_a8w8_blockscale``
    - **Sequential AITER GEMM**: per-token and MXFP8 grouped execution
    - **CKTile DeepGEMM**: ``deepgemm`` (BF16/FP16/FP8 grouped flat MM)

Quantization modes:
    - ``none``       — BF16 GMM/ptgmm via AITER Triton
    - ``delayed``    — per-tensor FP8 MOE GEMM via AITER Triton
    - ``dynamic``    — per-tensor FP8 MOE GEMM via AITER Triton
    - ``per_token``  — per-token FP8 via sequential AITER GEMM
    - ``blockwise``  — per-block FP8 MOE GEMM via AITER Triton
    - ``blockwise2d`` — same as blockwise (shared kernel)
    - ``mxfp8``      — MXFP8 via sequential AITER GEMM
"""

import logging

import torch

from atomrl.lumen.ops.dispatch import Backend, try_backends

logger = logging.getLogger(__name__)


def _build_routing_data_from_group_sizes(group_sizes, total_tokens):
    """Build aiter RoutingData from Lumen's simple group_sizes tensor.

    Lumen's grouped GEMM API uses expert-sorted tokens with a per-expert
    count vector.  The aiter fused kernels need a full RoutingData with
    block-to-expert mapping.  This function bridges the two.
    """
    import triton
    from aiter.ops.triton.moe.moe_routing.routing import (
        RoutingData,
        compute_expt_data_torch,
    )

    n_expts_tot = group_sizes.shape[0]
    tokens_per_expert = max(1, total_tokens // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expert), 128))
    expt_data = compute_expt_data_torch(group_sizes, n_expts_tot, total_tokens, block_m)
    # gate_scal is unused for expert-sorted tokens (n_expts_act=1, no scatter)
    return RoutingData(
        block_m=block_m,
        gate_scal=None,
        expt_hist=group_sizes,
        n_expts_tot=n_expts_tot,
        n_expts_act=1,
        expt_data=expt_data,
    )


def _quant_blockwise_2d(w, fp8_dtype, block_size=128):
    """Quantize a 3D weight tensor [E, K, N] with 2D block scales.

    Each expert's [K, N] matrix is divided into [block_size, block_size]
    blocks.  Per-block amax determines the FP8 scale.

    Returns (w_fp8 [E, K, N], w_scales [E, ceil(K/bs), ceil(N/bs)]).
    """
    import triton

    from atomrl.lumen.ops.quantize.padding import pad_to_block

    E, K, N = w.shape
    bs = block_size
    bk = triton.cdiv(K, bs)
    bn = triton.cdiv(N, bs)
    fp8_max = torch.finfo(fp8_dtype).max

    w_fp8 = torch.empty(E, K, N, dtype=fp8_dtype, device=w.device)
    w_scales = torch.empty(E, bk, bn, dtype=torch.float32, device=w.device)

    for e in range(E):
        we = w[e]
        we, orig_k = pad_to_block(we, bs, dim=0)
        we, orig_n = pad_to_block(we, bs, dim=1)

        blocks = we.reshape(bk, bs, bn, bs).permute(0, 2, 1, 3)
        amax = blocks.float().abs().amax(dim=(-2, -1)).clamp(min=1e-12)
        scales = amax / fp8_max
        w_scales[e] = scales

        scale_expanded = scales.unsqueeze(-1).unsqueeze(-1)
        q_blocks = (blocks.float() / scale_expanded).clamp(-fp8_max, fp8_max)
        q_full = q_blocks.permute(0, 2, 1, 3).reshape(we.size(0), we.size(1))
        w_fp8[e] = q_full[:orig_k, :orig_n].to(fp8_dtype)

    return w_fp8, w_scales


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _get_gmm():
    from aiter.ops.triton.gemm.grouped.gmm import gmm

    return gmm


def _get_ptgmm():
    from aiter.ops.triton.gemm.grouped.gmm import ptgmm

    return ptgmm


def _get_nptgmm():
    from aiter.ops.triton.gemm.grouped.gmm import nptgmm

    return nptgmm


def _get_deepgemm():
    from aiter.ops.deepgemm import deepgemm

    return deepgemm


def _get_moe_gemm_a8w8():
    from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8

    return moe_gemm_a8w8


def _get_moe_gemm_a8w8_blockscale():
    from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import (
        moe_gemm_a8w8_blockscale,
    )

    return moe_gemm_a8w8_blockscale


# ---------------------------------------------------------------------------
# BF16 Grouped GEMM (no quantization) — all via AITER
# ---------------------------------------------------------------------------


def _gmm_triton(lhs, rhs, group_sizes, bias=None):
    """BF16 grouped GEMM via AITER Triton gmm."""
    fn = _get_gmm()
    return fn(lhs, rhs, group_sizes, bias=bias)


def grouped_gemm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    scaling_type: str = "none",
    x_scale: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    output_features: int | None = None,
) -> torch.Tensor:
    """Grouped GEMM dispatch with multi-backend fallback (all AITER).

    For BF16 (scaling_type="none"):
        out[g] = lhs[group_start:group_end] @ rhs[g].T + bias[g]

    For FP8 modes:
        Uses AITER MOE GEMM kernels where available.

    Args:
        lhs: Activation tensor ``[total_tokens, K]``.
        rhs: Weight tensor ``[num_experts, N, K]`` (or ``[num_experts, K, N]``
             depending on backend).
        group_sizes: Expert token counts ``[num_experts]``.
        scaling_type: Quantization mode.
        x_scale: Activation scale(s).
        w_scale: Weight scale(s).
        bias: Per-expert bias ``[num_experts, N]`` or ``None``.
        block_size: Block size for blockwise mode.
        fp8_dtype: Target FP8 dtype.
        output_features: Original unpadded output width for pre-quantized MXFP8
            weights. Required when their stored row count includes padding.

    Returns:
        Output tensor ``[total_tokens, N]``.
    """
    if scaling_type == "none":
        backends = [(Backend.TRITON, lambda: _gmm_triton(lhs, rhs, group_sizes, bias))]
        return try_backends(backends, op_name="grouped_gemm_bf16")

    if scaling_type in ("delayed", "dynamic"):

        def _moe_a8w8():
            from atomrl.lumen.ops.quantize.linear import quantize_input

            fn = _get_moe_gemm_a8w8()
            x_data, x_static_scale = lhs, x_scale
            w_data, w_expt_scale = rhs, w_scale
            if lhs.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                x_desc = quantize_input(lhs, "dynamic", fp8_dtype)
                x_data, x_static_scale = x_desc.data, x_desc.scale
            if rhs.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                descriptors = [
                    quantize_input(weight.T.contiguous(), "dynamic", fp8_dtype)
                    for weight in rhs
                ]
                w_data = torch.stack([descriptor.data.T for descriptor in descriptors])
                w_expt_scale = torch.stack(
                    [descriptor.scale.reshape(()) for descriptor in descriptors]
                )

            routing_data = _build_routing_data_from_group_sizes(
                group_sizes, int(group_sizes.sum().item())
            )
            static_weight_scale = None
            expert_weight_scale = None
            if w_expt_scale is not None:
                if w_expt_scale.numel() == 1:
                    static_weight_scale = w_expt_scale.reshape(1)
                elif w_expt_scale.numel() == len(group_sizes):
                    expert_weight_scale = w_expt_scale.reshape(-1).float()
                else:
                    raise ValueError(
                        "per-tensor grouped weight scales must contain one value "
                        f"per expert, got {tuple(w_expt_scale.shape)}"
                    )
            return fn(
                x_data,
                w_data,
                None,
                None,
                x_static_scale=x_static_scale,
                w_static_scale=static_weight_scale,
                bias=bias,
                routing_data=routing_data,
                out_dtype=torch.bfloat16,
                w_expt_scale=expert_weight_scale,
            )

        return _moe_a8w8()

    if scaling_type in ("blockwise", "blockwise2d"):
        backends = []

        def _moe_blockscale():
            total_tokens = int(group_sizes.sum().item())
            routing_data = _build_routing_data_from_group_sizes(
                group_sizes, total_tokens
            )

            if lhs.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                if x_scale is None or w_scale is None:
                    raise ValueError(
                        "Pre-quantized FP8 inputs require explicit x_scale and w_scale"
                    )
                lhs_fp8, lhs_scales = lhs, x_scale
                rhs_fp8, rhs_scales = rhs, w_scale
            else:
                from atomrl.lumen.ops.quantize.ops import quant_fp8_blockwise_impl

                lhs_2d = lhs.reshape(-1, lhs.shape[-1]).contiguous()
                lhs_fp8, lhs_scales = quant_fp8_blockwise_impl(
                    lhs_2d, fp8_dtype, axis=1, block_size=block_size
                )
                rhs_fp8, rhs_scales = _quant_blockwise_2d(rhs, fp8_dtype, block_size)

            fn = _get_moe_gemm_a8w8_blockscale()
            return fn(
                x=lhs_fp8,
                w=rhs_fp8,
                x_block_scales=lhs_scales,
                w_block_scales=rhs_scales,
                bias=bias,
                routing_data=routing_data,
                out_dtype=torch.bfloat16,
                per_row_x_scale=True,
            )

        backends.append((Backend.TRITON, _moe_blockscale))

        def _sequential_fallback():
            return _grouped_gemm_fp8_sequential(
                lhs,
                rhs,
                group_sizes,
                x_scale,
                w_scale,
                scaling_type,
                bias,
                block_size=block_size,
                fp8_dtype=fp8_dtype,
            )

        backends.append((Backend.TRITON, _sequential_fallback))
        return try_backends(backends, op_name="grouped_gemm_blockscale")

    if scaling_type == "per_token":
        return _grouped_gemm_fp8_sequential(
            lhs,
            rhs,
            group_sizes,
            x_scale,
            w_scale,
            scaling_type,
            bias,
            output_features=output_features,
        )

    if scaling_type == "mxfp8":
        return _grouped_gemm_fp8_sequential(
            lhs,
            rhs,
            group_sizes,
            x_scale,
            w_scale,
            scaling_type,
            bias,
            output_features=output_features,
        )

    raise ValueError(f"Unknown scaling_type={scaling_type!r}")


def _grouped_gemm_fp8_sequential(
    lhs,
    rhs,
    group_sizes,
    x_scale,
    w_scale,
    scaling_type,
    bias=None,
    block_size=128,
    fp8_dtype=torch.float8_e4m3fn,
    output_features=None,
):
    """Sequential per-expert FP8 GEMM via AITER Triton GEMM backends."""
    from atomrl.lumen.ops.quantize.linear import dispatch_gemm, quantize_input

    outputs = []
    offset = 0
    total_tokens = int(group_sizes.sum().item())
    for g, size in enumerate(group_sizes):
        size = int(size)
        if size == 0:
            continue
        x_g = lhs[offset : offset + size]
        # Grouped AITER kernels consume [E, K, N], while dispatch_gemm follows
        # torch.nn.Linear and consumes each expert weight as [N, K].
        w_g = rhs[g].transpose(0, 1).contiguous()
        if x_scale is None or x_scale.ndim == 0 or x_scale.numel() == 1:
            xs = x_scale
        elif x_scale.shape[0] >= total_tokens:
            xs = x_scale[offset : offset + size]
        else:
            raise ValueError(
                "activation scale must be scalar or token-aligned: "
                f"got {tuple(x_scale.shape)} for {total_tokens} tokens"
            )

        if w_scale is None or w_scale.ndim == 0 or w_scale.numel() == 1:
            ws = w_scale
        elif w_scale.shape[0] == len(group_sizes):
            ws = w_scale[g]
        else:
            raise ValueError(
                "weight scale must be scalar or expert-aligned: "
                f"got {tuple(w_scale.shape)} for {len(group_sizes)} experts"
            )
        b_g = bias[g] if bias is not None else None

        if xs is None and scaling_type != "none":
            desc = quantize_input(x_g, scaling_type, fp8_dtype, block_size)
            x_g, xs = desc.data, desc.scale
        if ws is None and scaling_type != "none":
            desc = quantize_input(w_g, scaling_type, fp8_dtype, block_size)
            w_g, ws = desc.data, desc.scale

        out_g = dispatch_gemm(x_g, w_g, xs, ws, scaling_type, b_g)
        width = output_features if output_features is not None else rhs.shape[-1]
        outputs.append(out_g[:size, :width])
        offset += size
    if not outputs:
        N = (
            output_features
            if output_features is not None
            else rhs.shape[-1] if rhs.dim() == 3 else rhs.shape[1]
        )
        return torch.empty(0, N, device=lhs.device, dtype=torch.bfloat16)
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Grouped GEMM backward (wgrad) — all via AITER
# ---------------------------------------------------------------------------


def grouped_gemm_wgrad(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    group_sizes: torch.Tensor,
    scaling_type: str = "none",
) -> torch.Tensor:
    """Grouped GEMM weight gradient: out[g] = grad[:, g_start:g_end].T @ input[g_start:g_end].

    For BF16 uses AITER's ptgmm (Persistent Transposed GMM).
    For FP8 modes uses sequential per-expert wgrad via AITER Triton GEMM.

    Args:
        grad_output: ``[total_tokens, N]``.
        input_tensor: ``[total_tokens, K]``.
        group_sizes: Expert token counts ``[num_experts]``.
        scaling_type: Quantization mode.

    Returns:
        Weight gradient ``[num_experts, N, K]``.
    """
    if scaling_type == "none":

        def _ptgmm():
            fn = _get_ptgmm()
            return fn(grad_output.t(), input_tensor, group_sizes)

        backends = [(Backend.TRITON, _ptgmm)]
        return try_backends(backends, op_name="grouped_gemm_wgrad")

    # FP8 wgrad: sequential per-expert via AITER GEMM (in BF16 for numerical stability)
    from atomrl.lumen.ops.quantize.linear import dispatch_gemm

    num_experts = len(group_sizes)
    N = grad_output.shape[-1]
    K = input_tensor.shape[-1]
    wgrad = torch.zeros(
        num_experts, N, K, device=grad_output.device, dtype=torch.bfloat16
    )
    offset = 0
    for g, size in enumerate(group_sizes):
        size = int(size)
        if size == 0:
            continue
        g_bf16 = grad_output[offset : offset + size].to(torch.bfloat16)
        x_bf16 = input_tensor[offset : offset + size].to(torch.bfloat16)
        # wgrad[g] = g^T @ x  →  dispatch(g^T, x^T) since kernel does A @ W^T → (g^T) @ (x^T)^T = g^T @ x
        wgrad[g] = dispatch_gemm(
            g_bf16.t().contiguous(),
            x_bf16.t().contiguous(),
            None,
            None,
            "none",
        )
        offset += size
    return wgrad
