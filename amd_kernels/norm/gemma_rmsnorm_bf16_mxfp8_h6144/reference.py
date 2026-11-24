# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gemma RMSNorm of bf16 rows, bf16 rounding before 1x32 E8M0 MXFP8.

The norm uses fp32 sums, eps and products; weight is bf16 and multiplied as
(1 + weight). The torch norm is the numerical contract (per-row cosine >=
0.999999). The kernel preserves the served Triton plain-norm reduction tree
as an additional fusion check. Quantization always consumes the bf16 norm.
"""
import torch

def gemma_rmsnorm_reference(x, weight, eps):
    xf=x.float()
    return ((xf * torch.rsqrt(xf.square().mean(-1,keepdim=True) + eps)) *
            (1.0 + weight.float())).bfloat16()


def quantize_mxfp8_reference(norm_bf16):
    """BF16 norm -> row-major E4M3 codes and E8M0 scale bytes, one group per 32 columns."""
    m,h=norm_bf16.shape
    blocks=norm_bf16.float().reshape(m,h//32,32)
    amax=blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    scales=(amax/448.0).log2().ceil().add(127).clamp(0,254)
    codes=(blocks*torch.exp2(127-scales).unsqueeze(-1)).reshape(m,h)
    return codes.to(torch.float8_e4m3fn),scales.to(torch.uint8)


def gemma_rmsnorm_mxfp8_reference(x,weight,eps):
    """Norm + BF16 rounding + quantization; row-major reference layouts."""
    return quantize_mxfp8_reference(gemma_rmsnorm_reference(x,weight,eps))
