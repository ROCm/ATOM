# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 GEMM with packed activation rows, original bf16 weight, fp32 accumulate.

Activation layout: [m16, k32, k8, row16, feature8], flattened and viewed as
[padded_rows,256]; rows are padded to 16/32/64. Weight remains [200064,256],
row major, shared with every existing reader. Output is row-major bf16.
Reference uses fp64 accumulation, rounded once to bf16. Gate: row cosine
>=0.99999 and relative L2 <=1e-3, deterministic. Accumulation order can move
last bf16 bits against hipBLASLt; sampling gates are required in the model.
"""
import torch
N,K=200064,256


def pack_rows_reference(x,indices):
    n=indices.numel();pad=16 if n<=16 else (32 if n<=32 else 64)
    selected=torch.zeros(pad,K,dtype=torch.bfloat16,device=x.device)
    selected[:n]=x.index_select(0,indices)
    return selected.view(-1,16,8,4,8).permute(0,2,3,1,4).contiguous().view(pad,K)


def unpack_rows_reference(packed,n):
    return packed.view(-1,8,4,16,8).permute(0,3,1,2,4).contiguous().view(-1,K)[:n]


def bf16_gemm_reference(packed,weight,n):
    return (unpack_rows_reference(packed,n).double() @ weight.double().t()).bfloat16()
