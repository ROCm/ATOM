# SPDX-License-Identifier: MIT
"""RMS normalization at the V4.1 activation quantization boundary."""

import torch
from aiter import rmsnorm2d_fwd
from torch import nn


class RMSNorm(nn.Module):
    """Evaluate the statistic and affine product in FP32, then round once.

    Keep the eager reduction order of the published model. A single BF16
    rounding difference can cross the next A8 quantization boundary; fused
    replacements need the full-model numerical gate as well as an operator test.
    """

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, hidden):
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * value).to(hidden.dtype)


class FusedRMSNorm(RMSNorm):
    """Reuse V4's kernel at independently validated normalization boundaries."""

    def forward(self, hidden):
        if not hidden.is_cuda or hidden.numel() == 0:
            return super().forward(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        return rmsnorm2d_fwd(flat, self.weight, self.eps).view_as(hidden)
