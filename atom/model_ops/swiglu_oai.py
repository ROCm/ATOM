# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SwiGLU-OAI activation used by MiniMax-M3.

MiniMax-M3 stores dense and expert gate/up activations in split layout:
``[gate | up]``.  The activation is:

    gate * sigmoid(alpha * gate) * (up + beta)

with optional clamping.  ATOM only supports this path on AMD GPU, so the
implementation is Triton-only.
"""

import functools

import torch
import triton
import triton.language as tl


@functools.cache
def _device_arch(device_index: int) -> str:
    return torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]


def _launch_config(num_rows: int, n_inter: int, arch: str) -> tuple[int, int]:
    # gfx1250 benefits from fewer, wider programs at MiniMax-M3's I=3072.
    # One or two warps also improve occupancy for the bandwidth-bound prefill
    # shapes; decode retains the higher-warp launch needed by its few programs.
    if arch == "gfx1250" and n_inter >= 2048:
        if num_rows < 64:
            return 256, 4
        if num_rows < 256:
            return 512, 2
        if num_rows < 512:
            # Avoid a heavily masked third 1024-wide tile (for example I=2560).
            return (1024, 4) if n_inter % 1024 == 0 else (512, 4)
        if num_rows < 16384:
            return 1024, 2
        return 512, 1
    return (512 if n_inter >= 2048 else 256), 4


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

    device_index = gate_up.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    block_i, num_warps = _launch_config(
        x2.shape[0], n_inter, _device_arch(device_index)
    )
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
        num_warps=num_warps,
    )
    return out.reshape(*orig_shape[:-1], n_inter)
