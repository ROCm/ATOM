# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Five bf16 [M,6144] taps and bf16 [1536,30720] weight to bf16 [M,1536].

This is an N-shard of the input projection. Each output has its entire K sum
on one rank. There is no bf16 partial sum across ranks and no quantization.
The kernel may split K into fp32 slabs, then reduce those slabs in fixed order
and round once to bf16. Against the fp64 product rounded once to bf16, require
per-row and overall cosine >=0.99999, relative L2 <=1e-3 and bitwise repeatability.
"""
import torch


def concat_linear_reference(taps,weight):
    return (torch.cat(taps,dim=1).double()@weight.double().T).bfloat16()

