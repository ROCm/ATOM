# SPDX-License-Identifier: MIT
"""Grouped output LoRA and reference mHC coefficient projections.

Small output projections reuse V4's BF16 batched GEMM. Larger batches use its
native einsum path. The reference-only mHC helper retains its row policy;
production delayed mHC is implemented separately in mhc_pre_delayed.
"""

import torch
import torch.nn.functional as F
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16


def grouped_output_projection(hidden, weight):
    """BF16 [batch, tokens, groups, channels] x [groups, rank, channels]."""
    rows = hidden.shape[0] * hidden.shape[1]
    # Single-row native GEMM is already as fast on the TP4 checkpoint shape.
    if hidden.is_cuda and 1 < rows <= 32:
        flat = hidden.flatten(0, 1)
        # Token-major output lets wo_b flatten groups without a transpose copy.
        output = torch.empty(
            (rows, weight.shape[0], weight.shape[1]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        batched_gemm_bf16(flat.transpose(0, 1), weight, YQ=output.transpose(0, 1))
        return output.unflatten(0, hidden.shape[:2])
    return torch.einsum("bsgd,grd->bsgr", hidden, weight)


def hc_projection(hidden, weight):
    """FP32 coefficient projection; normalization and Sinkhorn belong to mHC."""
    rows = hidden.numel() // hidden.shape[-1]
    if hidden.is_cuda and 1 < rows <= 64:
        flat = hidden.reshape(rows, hidden.shape[-1])
        padded = F.pad(flat, (0, 0, 0, 128 - rows))
        output = F.linear(padded, weight)
        return output[:rows].view(*hidden.shape[:-1], weight.shape[0])
    return F.linear(hidden, weight)
