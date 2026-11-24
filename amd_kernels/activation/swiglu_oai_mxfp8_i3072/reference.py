# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of ``swiglu_oai_mxfp8``.

``bf16_chain`` (the DSpark draft's ``SiluAndMulWithClamp.forward_native`` on a bf16 tensor; every aten op
computes in fp32 and rounds to bf16):

    g = clamp(gate, max=limit); u = clamp(up, -limit, limit)
    act = g * sigmoid(alpha * g) * (u + beta)          (bf16 after every op)

then the Triton MXFP8 quantizer of the bf16 ``act`` (1 x 32 blocks, E8M0 scale
``clamp(ceil(log2(amax / 448)) + 127, 0, 254)``, e4m3 codes of ``act * 2^(127 - scale)``), in the layout the
down GEMM reads (row-major or (16, 16) tiles; scales in AITER's shuffled layout).

``fp32`` (the target's dense layers 0-2, ``swiglu_oai_split``): ``swiglu_fp32`` (fp32 math, one bf16
rounding), then the same quantizer.

Tolerance: the kernel's codes and scales equal this reference bit for bit (the same bf16 ``act``, the same
quantizer) on finite inputs; the tests check that, so a change of the SwiGLU's order or rounding shows up."""

import torch


def swiglu_bf16_chain(gate_up: torch.Tensor, alpha: float, beta: float, limit: float) -> torch.Tensor:
    d = gate_up.shape[-1] // 2
    gate = torch.clamp(gate_up[..., :d], max=limit)
    up = torch.clamp(gate_up[..., d:], min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


def swiglu_fp32(gate_up: torch.Tensor, alpha: float, beta: float, limit: float) -> torch.Tensor:
    """``swiglu_oai_split``'s form: fp32 math, one bf16 rounding."""
    d = gate_up.shape[-1] // 2
    gate = torch.clamp(gate_up[..., :d].float(), max=limit)
    up = torch.clamp(gate_up[..., d:].float(), min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).bfloat16()


def quantize_mxfp8(act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-major e4m3 codes [m, K] and e8m0 scale bytes [m, K / 32] of a bf16 [m, K] (the Triton
    quantizer's arithmetic)."""
    m, k = act.shape
    x = act.float().view(m, k // 32, 32)
    amax = x.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    sb = torch.clamp(torch.ceil(torch.log2(amax / 448.0)) + 127.0, 0.0, 254.0)
    q = (x * torch.exp2(127.0 - sb)[..., None]).view(m, k).to(torch.float8_e4m3fn)
    return q, sb.to(torch.uint8)


def shuffle_scales(sb: torch.Tensor) -> torch.Tensor:
    """e8m0 bytes [m, K / 32] -> AITER's shuffled layout [ceil(m / 32) * 32, K / 32] (padding zero)."""
    from aiter.ops.shuffle import shuffle_scale_a16w4

    m, g = sb.shape
    rows = (m + 31) // 32 * 32
    pad = torch.zeros((rows, g), dtype=torch.uint8, device=sb.device)
    pad[:m] = sb
    return shuffle_scale_a16w4(pad, 1, False).reshape(rows, g)
