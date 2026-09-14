# SPDX-License-Identifier: MIT
"""Native wo_a and mHC projections with bounded small-row padding.

The native GEMM reduction order depends on the row count. With V4.1's
downstream quantization, small prefill chunks can amplify that difference.
Use the validated 128-row arithmetic for 2..64 rows, including batched decode.
This is an operator policy: it never changes attention spans or cache positions.
Single-row decode, larger projections and CPU execution retain native dispatch.
"""

import torch
import torch.nn.functional as F


def grouped_output_projection(hidden, weight):
    """BF16 [batch, tokens, groups, channels] x [groups, rank, channels]."""
    rows = hidden.shape[0] * hidden.shape[1]
    if hidden.is_cuda and 1 < rows <= 64:
        padded = F.pad(hidden.flatten(0, 1), (0, 0, 0, 0, 0, 128 - rows))
        output = torch.einsum("sgd,grd->sgr", padded, weight)
        return output[:rows].unflatten(0, hidden.shape[:2])
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
