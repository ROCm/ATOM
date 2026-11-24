# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Owner embedding lookup, AITER FTZ/positive-zero cleanup, plain Gemma norm,
BF16 materialization, then the BF16-row MXFP8 rule. The raw BF16 embedding is
also returned for the residual stream. Exact fusion on finite weights; norm
accuracy uses the existing plain-norm FP64 contract. Padding remains unwritten.
"""
import torch
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144.reference import gemma_rmsnorm_mxfp8_reference

def lookup_reference(ids,weights):
    shard=weights[0].shape[0]
    safe=ids.clamp(0,4*shard-1).long()
    out=torch.cat(weights)[safe].clone()
    bits=out.view(torch.int16)
    bits[(bits&0x7f80)==0]=0
    out[(ids<0)|(ids>=4*shard)]=0
    return out

def reference(ids,weights,norm_weight,eps=1e-6):
    raw=lookup_reference(ids,weights)
    q,s=gemma_rmsnorm_mxfp8_reference(raw,norm_weight,eps)
    return q,s,raw
