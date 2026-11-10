# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The Wan VAE's ``silu(WanRMSNorm(x))`` as one Triton pass.

Unfused that is seven elementwise kernels and a reduction over the activation
(cast to fp32, norm, divide, cast back, two scalings, SiLU), followed by the
copy into the conv's staging buffer. Here each element is read once and its
result written once, straight into a view of the staging buffer.

The kernel rounds to the activation dtype at every point the unfused ops do,
and uses IEEE division and square root, so it agrees with them bit for bit up
to the order of the fp32 sum of squares. GPU only: imported on the first GPU
call, so the VAE module imports on a CPU runner.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# One program covers this many elements: BLOCK_P pixels times all channels.
_TILE = 8192


@triton.jit
def _rms_norm_silu_kernel(
    x_ptr,
    gamma_ptr,
    out_ptr,
    PIXELS,
    C,
    T,
    H,
    W,
    sx_b,
    sx_c,
    sx_t,
    sx_h,
    sx_w,
    so_b,
    so_c,
    so_t,
    so_h,
    so_w,
    scale,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pixel = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    channel = tl.arange(0, BLOCK_C)
    mask = (pixel < PIXELS)[:, None] & (channel < C)[None, :]

    w = pixel % W
    rest = pixel // W
    h = rest % H
    rest = rest // H
    t = rest % T
    b = rest // T
    x_at = b * sx_b + t * sx_t + h * sx_h + w * sx_w
    out_at = b * so_b + t * so_t + h * so_h + w * so_w

    dtype = out_ptr.dtype.element_ty
    x = tl.load(
        x_ptr + x_at[:, None] + channel[None, :] * sx_c, mask=mask, other=0.0
    ).to(tl.float32)
    # F.normalize: x / max(||x||, eps), in fp32, then .to(x.dtype).
    norm = tl.maximum(tl.sqrt_rn(tl.sum(x * x, axis=1)), 1e-12)
    y = tl.div_rn(x, norm[:, None]).to(dtype).to(tl.float32)
    y = (y * scale).to(dtype).to(tl.float32)
    gamma = tl.load(gamma_ptr + channel, mask=channel < C, other=0.0)
    y = (y * gamma.to(tl.float32)[None, :]).to(dtype).to(tl.float32)
    # torch's SiLU: x / (1 + exp(-x)) in fp32.
    y = tl.div_rn(y, 1.0 + libdevice.exp(-y))
    tl.store(out_ptr + out_at[:, None] + channel[None, :] * so_c, y.to(dtype), mask)


def rms_norm_silu(
    x: torch.Tensor, gamma: torch.Tensor, scale: float, out: torch.Tensor
) -> None:
    """``out.copy_(F.silu(F.normalize(x.float(), dim=1).to(x.dtype) * scale *
    gamma))`` for logical ``[B, C, T, H, W]`` tensors of any strides; ``gamma``
    holds C contiguous values."""
    if x.dim() != 5 or out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError(
            f"need matching 5-D x and out, got {list(x.shape)} {x.dtype} / "
            f"{list(out.shape)} {out.dtype}"
        )
    b, c, t, h, w = x.shape
    if gamma.numel() != c or not gamma.is_contiguous():
        raise ValueError(f"gamma must hold {c} contiguous values")
    pixels = b * t * h * w
    if not pixels:
        return
    block_c = triton.next_power_of_2(c)
    block_p = max(1, _TILE // block_c)
    _rms_norm_silu_kernel[(triton.cdiv(pixels, block_p),)](
        x,
        gamma,
        out,
        pixels,
        c,
        t,
        h,
        w,
        *x.stride(),
        *out.stride(),
        float(scale),
        BLOCK_P=block_p,
        BLOCK_C=block_c,
    )
