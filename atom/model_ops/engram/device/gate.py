# SPDX-License-Identifier: MIT
"""Engram post-projection gate and residual update.

One program handles one token/HC branch. Reductions and gate arithmetic stay
FP32; only the final residual is cast. The precomputed q*k weights retain the
original operation order. Disabling FP fusion avoids introducing an FMA at the
residual addition. Reduction trees/rsqrt can still differ from PyTorch.
"""

import torch
import triton
import triton.language as tl

from atom.utils import envs


@triton.jit
def _engram_post_wkv_kernel(
    hidden,
    kv,
    weight,
    token_mask,
    output,
    eps,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token, branch = row // HC, row % HC
    col = tl.arange(0, BLOCK)
    live = col < DIM
    residual = tl.load(hidden + row * DIM + col, live, 0).to(tl.float32)
    key = tl.load(kv + token * (HC + 1) * DIM + branch * DIM + col, live, 0).to(
        tl.float32
    )
    w = tl.load(weight + branch * DIM + col, live, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(residual * residual, 0) / DIM + eps)
    rstd *= tl.rsqrt(tl.sum(key * key, 0) / DIM + eps)
    dot = tl.sum((residual * w) * key, 0) * rstd * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    # copysign, including negative zero (which a dot < 0 test would lose).
    signed = (magnitude.to(tl.uint32, bitcast=True) & 0x7FFFFFFF) | (
        dot.to(tl.uint32, bitcast=True) & 0x80000000
    )
    gate = tl.sigmoid(signed.to(tl.float32, bitcast=True))
    if HAS_MASK:
        gate = tl.where(tl.load(token_mask + token), gate, 0.0)
    value = tl.load(kv + token * (HC + 1) * DIM + HC * DIM + col, live, 0).to(
        tl.float32
    )
    tl.store(output + row * DIM + col, residual + gate * value, live)


def engram_post_wkv_reference(hidden, kv, weight, token_mask=None, eps=1e-20):
    """Original torch operation sequence, also used for CPU/strided inputs."""
    hc, dim = hidden.shape[-2:]
    key = kv[..., : hc * dim].reshape(*hidden.shape).float()
    value = kv[..., hc * dim :]
    residual = hidden.float()
    rstd = torch.rsqrt(residual.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (residual * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (residual + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
        hidden.dtype
    )


def engram_post_wkv(hidden, kv, weight, token_mask=None, eps=1e-20):
    """Fused inference path for contiguous [..., HC, DIM] residuals."""
    hc, dim = hidden.shape[-2:]
    if kv.shape != (*hidden.shape[:-2], (hc + 1) * dim):
        raise ValueError("Engram KV projection does not match residual shape")
    if weight.shape != (hc, dim):
        raise ValueError("Engram gate weight does not match residual branches")
    if token_mask is not None and token_mask.shape != hidden.shape[:-2]:
        raise ValueError("Engram token mask must match residual tokens")
    supported = (
        envs.ATOM_ENGRAM_FUSED_GATE
        and hidden.is_cuda
        and hidden.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and kv.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and hidden.is_contiguous()
        and kv.is_contiguous()
        and weight.is_contiguous()
        and (token_mask is None or token_mask.is_contiguous())
        and not (
            torch.is_grad_enabled()
            and any(t.requires_grad for t in (hidden, kv, weight))
        )
    )
    if not supported:
        return engram_post_wkv_reference(hidden, kv, weight, token_mask, eps)
    output = torch.empty_like(hidden)
    if hidden.numel():
        block = triton.next_power_of_2(dim)
        _engram_post_wkv_kernel[(hidden.numel() // dim,)](
            hidden,
            kv,
            weight,
            token_mask,
            output,
            eps,
            DIM=dim,
            HC=hc,
            BLOCK=block,
            HAS_MASK=token_mask is not None,
            num_warps=8 if block >= 2048 else 4,
            enable_fp_fusion=False,
        )
    return output
