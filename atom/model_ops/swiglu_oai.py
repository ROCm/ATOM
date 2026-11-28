# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""SwiGLU-OAI activation used by MiniMax-M3.

MiniMax-M3 stores dense and expert gate/up activations in split layout:
``[gate | up]``.  The activation is:

    gate * sigmoid(alpha * gate) * (up + beta)

with optional clamping.  ATOM only supports this path on AMD GPU, so the
implementation is Triton-only.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE, fp8_groups


@triton.jit
def _swiglu_oai_kernel(
    gate_up_ptr,
    out_ptr,
    n_inter: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    alpha,
    beta,
    limit,
    has_limit: tl.constexpr,
    block_i: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * block_i + tl.arange(0, block_i)
    mask = cols < n_inter

    gate = tl.load(
        gate_up_ptr + row * stride_gm + cols * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gate_up_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if has_limit:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)

    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    tl.store(
        out_ptr + row * stride_om + cols * stride_on,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def swiglu_oai_split(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply MiniMax-M3 SwiGLU-OAI to a split-layout ``[..., 2I]`` tensor."""
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"SwiGLU-OAI expects an even last dimension, got {gate_up.shape[-1]}."
        )
    if not gate_up.is_cuda:
        raise RuntimeError("SwiGLU-OAI is only supported on AMD GPU tensors.")

    orig_shape = gate_up.shape
    two_i = orig_shape[-1]
    n_inter = two_i // 2
    x2 = gate_up.reshape(-1, two_i)
    out = torch.empty(
        (x2.shape[0], n_inter),
        dtype=out_dtype or gate_up.dtype,
        device=gate_up.device,
    )

    block_i = 512 if n_inter >= 2048 else 256
    grid = (x2.shape[0], triton.cdiv(n_inter, block_i))
    _swiglu_oai_kernel[grid](
        x2,
        out,
        n_inter,
        x2.stride(0),
        x2.stride(1),
        out.stride(0),
        out.stride(1),
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        has_limit=limit is not None,
        block_i=block_i,
        num_warps=4,
    )
    return out.reshape(*orig_shape[:-1], n_inter)


@triton.jit
def _swiglu_oai_mxfp8_kernel(
    gate_up_ptr,
    q_ptr,
    scale_ptr,
    n_inter: tl.constexpr,
    stride_gm,
    alpha,
    beta,
    limit,
    has_limit: tl.constexpr,
    block_i: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * block_i + tl.arange(0, block_i)
    mask = cols < n_inter
    gate = tl.load(gate_up_ptr + row * stride_gm + cols, mask=mask, other=0.0)
    up = tl.load(gate_up_ptr + row * stride_gm + n_inter + cols, mask=mask, other=0.0)
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    if has_limit:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    # Round to the activation dtype first: the separate path stores this row
    # in it and quantizes what it reads back, and the bytes must agree.
    out = out.to(gate_up_ptr.dtype.element_ty).to(tl.float32)
    q, code, _ = fp8_groups(tl.reshape(out, (block_i // 32, 32)))
    tl.store(
        q_ptr + row * n_inter + tl.reshape(cols, (block_i // 32, 32)),
        q,
        mask=tl.reshape(mask, (block_i // 32, 32)),
    )
    groups = tile * (block_i // 32) + tl.arange(0, block_i // 32)
    tl.store(
        scale_ptr + row * (n_inter // 32) + groups,
        code.to(tl.uint8),
        mask=groups < n_inter // 32,
    )


def swiglu_oai_split_mxfp8(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``swiglu_oai_split`` followed by ``quantize_fp8`` in one launch.

    Returns the E4M3 values ``[..., I]`` and their E8M0 group32 scales
    ``[..., I // 32]``, byte-identical to quantizing ``swiglu_oai_split``'s
    output (rounded to ``gate_up.dtype``) with ``quantize_fp8``. ``I`` must be
    a multiple of 32.
    """
    two_i = gate_up.shape[-1]
    n_inter = two_i // 2
    if two_i % 64:
        raise ValueError(
            f"SwiGLU-OAI MXFP8 needs the intermediate to be a multiple of 32, "
            f"got {n_inter}."
        )
    if not gate_up.is_cuda:
        raise RuntimeError("SwiGLU-OAI is only supported on AMD GPU tensors.")
    x2 = gate_up.reshape(-1, two_i)
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    q = torch.empty((m, n_inter), dtype=FP8_DTYPE, device=gate_up.device)
    scale = torch.empty(
        (m, n_inter // 32), dtype=torch.float8_e8m0fnu, device=gate_up.device
    )
    if m:
        block_i = 512 if n_inter >= 2048 else 256
        _swiglu_oai_mxfp8_kernel[(m, triton.cdiv(n_inter, block_i))](
            x2,
            q,
            scale.view(torch.uint8),
            n_inter,
            x2.stride(0),
            float(alpha),
            float(beta),
            0.0 if limit is None else float(limit),
            has_limit=limit is not None,
            block_i=block_i,
            num_warps=4,
        )
    lead = gate_up.shape[:-1]
    return q.reshape(*lead, n_inter), scale.reshape(*lead, n_inter // 32)
