"""Fused Triton kernels for Kimi-K3 small ops (gfx1250 perf).

Drop-in replacements for the elementwise/reduction glue that otherwise runs as
several separate torch ops (each a kernel launch + full HBM round-trip):

- ``situ_and_mul``      : SiTU(gate) * linear(up)  (dense/shared-expert MLP act)
- ``rmsnorm_gated``     : rmsnorm(x) * weight * sigmoid(gate)  (KDA o_norm)
- ``apply_attn_res``    : block-residual soft-attention mix (attn_res boundaries)

All gated behind ``ATOM_K3_FUSED`` in kimi_k3.py; the torch paths remain as the
reference and are used when triton is unavailable. Correctness is checked in
my_script/optest_fused.py against the torch reference.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _situ_and_mul_kernel(
        x_ptr, y_ptr,
        M, D,
        stride_xm, stride_ym,
        beta, inv_beta, linear_beta, inv_linear_beta,
        HAS_LINEAR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        g = tl.load(x_ptr + row * stride_xm + col, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * stride_xm + D + col, mask=mask, other=0.0).to(tl.float32)
        # SiTUv2 gate: beta * tanh(gate/beta) * sigmoid(gate); tanh via sigmoid
        # identity (tanh(z) = 2*sigmoid(2z) - 1) for portability across triton.
        out = beta * (2.0 * tl.sigmoid(2.0 * g * inv_beta) - 1.0) * tl.sigmoid(g)
        if HAS_LINEAR:
            u = linear_beta * (2.0 * tl.sigmoid(2.0 * u * inv_linear_beta) - 1.0)
        y = out * u
        tl.store(y_ptr + row * stride_ym + col, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_gated_kernel(
        x_ptr, w_ptr, g_ptr, y_ptr,
        H, eps,
        stride_m,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < H
        x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / H
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(g_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd * w) * tl.sigmoid(gate)
        tl.store(y_ptr + row * stride_m + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


if _HAS_TRITON:

    @triton.jit
    def _attn_res_scores_kernel(
        v_ptr, sw_ptr, scores_ptr,
        Bp, H, eps,
        stride_t, stride_b, stride_st,
        BLOCK_H: tl.constexpr,
    ):
        # one program per (t, b): reduce over H to get var and dot(v, score_weight)
        t = tl.program_id(0)
        b = tl.program_id(1)
        base = t * stride_t + b * stride_b
        acc_sq = 0.0
        acc_dot = 0.0
        for h0 in range(0, H, BLOCK_H):
            cols = h0 + tl.arange(0, BLOCK_H)
            mask = cols < H
            v = tl.load(v_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
            sw = tl.load(sw_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            acc_sq += tl.sum(v * v, axis=0)
            acc_dot += tl.sum(v * sw, axis=0)
        rstd = 1.0 / tl.sqrt(acc_sq / H + eps)
        tl.store(scores_ptr + t * stride_st + b, rstd * acc_dot)


def apply_attn_res(
    prefix_sum: torch.Tensor,        # [T, H]
    block_residual: torch.Tensor,    # [T, B, H]
    proj_weight: torch.Tensor,       # [H] (proj.weight.squeeze(0))
    norm_weight: torch.Tensor,       # [H]
    eps: float,
) -> torch.Tensor:
    """Block-residual soft-attention mix: rmsnorm each of the B+1 candidates,
    score = <normed, norm_weight*proj_weight>, softmax over B+1, weighted sum."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)  # [T,Bp,H]
    T, Bp, H = values.shape
    if not _HAS_TRITON or T == 0:
        return _apply_attn_res_torch(prefix_sum, block_residual, proj_weight, norm_weight, eps)
    score_weight = (norm_weight.float() * proj_weight.float()).contiguous()
    values_c = values.contiguous()
    scores = torch.empty((T, Bp), device=values.device, dtype=torch.float32)
    BLOCK_H = 1024
    _attn_res_scores_kernel[(T, Bp)](
        values_c, score_weight, scores,
        Bp, H, float(eps),
        values_c.stride(0), values_c.stride(1), scores.stride(0),
        BLOCK_H=BLOCK_H,
    )
    probs = scores.softmax(-1).unsqueeze(1)  # [T,1,Bp]
    return torch.matmul(probs, values_c.float()).squeeze(1).to(prefix_sum.dtype)


def _apply_attn_res_torch(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_f = values.float()
    var = values_f.pow(2).mean(-1, keepdim=True)
    normed = values_f * torch.rsqrt(var + eps)
    score_weight = norm_weight.float() * proj_weight.float()
    scores = (normed * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, values_f).squeeze(1).to(prefix_sum.dtype)


def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """SiTUv2 gated activation over the last dim (x[..., :D] gate, x[..., D:] up)."""
    *lead, two_d = x.shape
    assert two_d % 2 == 0
    d = two_d // 2
    x2 = x.reshape(-1, two_d)
    m = x2.shape[0]
    y = torch.empty((m, d), dtype=x.dtype, device=x.device)
    if not _HAS_TRITON or m == 0:
        return _situ_and_mul_torch(x, beta, linear_beta)
    BLOCK = 1024
    grid = (m, triton.cdiv(d, BLOCK))
    has_linear = linear_beta is not None
    _situ_and_mul_kernel[grid](
        x2, y, m, d,
        x2.stride(0), y.stride(0),
        float(beta), 1.0 / float(beta),
        float(linear_beta) if has_linear else 0.0,
        (1.0 / float(linear_beta)) if has_linear else 0.0,
        HAS_LINEAR=has_linear, BLOCK=BLOCK,
    )
    return y.reshape(*lead, d)


def rmsnorm_gated(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """rmsnorm(x) over last dim * weight * sigmoid(gate)."""
    h = x.shape[-1]
    x2 = x.reshape(-1, h)
    g2 = gate.reshape(-1, h)
    m = x2.shape[0]
    if not _HAS_TRITON or m == 0 or h > 8192:
        return _rmsnorm_gated_torch(x, weight, gate, eps)
    x2 = x2.contiguous()
    g2 = g2.contiguous()
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(h)
    _rmsnorm_gated_kernel[(m,)](
        x2, weight, g2, y, h, float(eps), x2.stride(0), BLOCK=BLOCK,
    )
    return y.reshape_as(x)


# --------------------------------------------------------------------------- #
# torch references (also the fallback when triton is unavailable)
# --------------------------------------------------------------------------- #
def _situ_and_mul_torch(x: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate_f = gate.float()
    up_f = up.float()
    out = beta * torch.tanh(gate_f / beta) * torch.sigmoid(gate_f)
    if linear_beta is not None:
        up_f = linear_beta * torch.tanh(up_f / linear_beta)
    return (out * up_f).to(x.dtype)


def _rmsnorm_gated_torch(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    xn = x_f * torch.rsqrt(var + eps)
    return (xn.to(dtype) * weight.to(dtype)) * torch.sigmoid(gate)
