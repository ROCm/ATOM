# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract for four-rank bf16 all-reduce, residual add, Gemma norm and optional MXFP8.

The reduction uses owner c=min(row//ceil(rows/4),3), rank order c,c+1,c+2,c+3
modulo four. Each fp32 operand and partial sum flushes denormals to signed zero.
The sum rounds once to bf16. Residual addition is fp32 and writes a bf16 residual.
The norm uses the unrounded fp32 addition, weight offset one and epsilon 1e-6.
Its bf16 output must have per-row cosine >=0.99999 against the fp64 norm below.
MXFP8 always quantizes the bf16-rounded norm output, in blocks of 32, using the
family reference's scale rule. Codes are row-major or (16,16) tiled; scales use
AITER's shuffled layout. Padding rows are unwritten. Determinism is bitwise.

The reduction order equals AITER's two-stage order when rows are divisible by
four. Other counts use whole rows per owner, and small calls use two stages
rather than AITER's one stage: these are explicit summation-order changes.
"""
import torch



def reduce_reference(parts):
    assert len(parts)==4
    m=parts[0].shape[0]
    q=(m+3)//4
    out=torch.empty_like(parts[0])
    def ftz(v):
        return torch.where(v.abs()<2.0**-126,v*0.0,v)
    for c in range(4):
        sl=slice(c*q,min((c+1)*q,m))
        acc=ftz(parts[c][sl].float())
        for r in range(1,4):
            acc=ftz(acc+ftz(parts[(c+r)%4][sl].float()))
        out[sl]=acc.bfloat16()
    return out


def allreduce_norm_reference(parts, residual, weight, eps=1e-6):
    """Return the fp64 normalized value and exact bf16 residual of the reduction."""
    return add_rmsnorm_reference(reduce_reference(parts),residual,weight,eps)


def add_rmsnorm_reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(y in fp64, the residual out bf16) of ``x + residual`` (module docstring,
    steps 2 and 3); ``bf16(y)`` is the bf16 output."""
    s = x.float() + residual.float()
    s64 = s.double()
    eps32 = torch.tensor(eps, dtype=torch.float32).item()
    ss = (s64 * s64).sum(-1, keepdim=True)
    ss = torch.where(ss > torch.finfo(torch.float32).max, torch.full_like(ss, float("inf")), ss)
    y = s64 * torch.rsqrt(ss / 6144 + eps32) * (weight.double() + 1.0)
    return y, s.bfloat16()
