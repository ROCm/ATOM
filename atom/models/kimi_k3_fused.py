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
    def _attn_res_fused_kernel(
        br_ptr, ps_ptr, nw_ptr, pw_ptr, y_ptr,
        B, Bp, H, eps,
        stride_br_t, stride_br_b, stride_ps_t, stride_yt,
        BP: tl.constexpr,       # Bp padded to a power of 2 (vectorized candidate axis)
        BLOCK_H: tl.constexpr,
    ):
        # One program per row t: rmsnorm each of the Bp = B+1 candidates, score =
        # <normed, score_weight>, softmax over Bp, then weighted sum -> y[t].
        # Candidates 0..B-1 are block_residual rows; candidate B is prefix_sum.
        # Read both source tensors directly (no torch.cat materialization); the
        # Bp axis is vectorized, so scores/probs stay in registers and softmax +
        # weighted-sum never touch HBM.
        t = tl.program_id(0)
        b_idx = tl.arange(0, BP)
        b_mask = b_idx < Bp
        is_last = b_idx == B                              # prefix_sum candidate
        br_base = t * stride_br_t + b_idx * stride_br_b   # [BP]
        ps_base = t * stride_ps_t

        acc_sq = tl.zeros((BP,), dtype=tl.float32)
        acc_dot = tl.zeros((BP,), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            cols = h0 + tl.arange(0, BLOCK_H)
            h_mask = cols < H
            br = tl.load(br_ptr + br_base[:, None] + cols[None, :],
                         mask=(b_idx < B)[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            ps = tl.load(ps_ptr + ps_base + cols, mask=h_mask, other=0.0).to(tl.float32)  # [BLOCK_H]
            v = tl.where(is_last[:, None], ps[None, :], br)   # [BP, BLOCK_H], ps broadcast in-reg
            # score_weight = norm_weight * proj_weight, folded in per H-chunk
            nw = tl.load(nw_ptr + cols, mask=h_mask, other=0.0).to(tl.float32)
            pw = tl.load(pw_ptr + cols, mask=h_mask, other=0.0).to(tl.float32)
            sw = nw * pw
            acc_sq += tl.sum(v * v, axis=1)               # [BP]
            acc_dot += tl.sum(v * sw[None, :], axis=1)    # [BP]

        rstd = 1.0 / tl.sqrt(acc_sq / H + eps)
        scores = tl.where(b_mask, rstd * acc_dot, float("-inf"))
        scores = scores - tl.max(scores, axis=0)
        probs = tl.exp(scores)
        probs = probs / tl.sum(probs, axis=0)             # [BP], softmax over Bp

        for h0 in range(0, H, BLOCK_H):
            cols = h0 + tl.arange(0, BLOCK_H)
            h_mask = cols < H
            br = tl.load(br_ptr + br_base[:, None] + cols[None, :],
                         mask=(b_idx < B)[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            ps = tl.load(ps_ptr + ps_base + cols, mask=h_mask, other=0.0).to(tl.float32)
            v = tl.where(is_last[:, None], ps[None, :], br)
            out = tl.sum(probs[:, None] * v, axis=0)      # [BLOCK_H]
            tl.store(y_ptr + t * stride_yt + cols, out.to(y_ptr.dtype.element_ty), mask=h_mask)


def apply_attn_res(
    prefix_sum: torch.Tensor,        # [T, H]
    block_residual: torch.Tensor,    # [T, B, H]
    proj_weight: torch.Tensor,       # [H] (proj.weight.squeeze(0))
    norm_weight: torch.Tensor,       # [H]
    eps: float,
) -> torch.Tensor:
    """Block-residual soft-attention mix: rmsnorm each of the B+1 candidates,
    score = <normed, norm_weight*proj_weight>, softmax over B+1, weighted sum."""
    T, B, H = block_residual.shape
    Bp = B + 1
    if not _HAS_TRITON or T == 0:
        return _apply_attn_res_torch(prefix_sum, block_residual, proj_weight, norm_weight, eps)
    br = block_residual.contiguous()
    ps = prefix_sum.contiguous()
    y = torch.empty((T, H), device=block_residual.device, dtype=prefix_sum.dtype)
    _attn_res_fused_kernel[(T,)](
        br, ps, norm_weight.contiguous(), proj_weight.contiguous(), y,
        B, Bp, H, float(eps),
        br.stride(0), br.stride(1), ps.stride(0), y.stride(0),
        BP=triton.next_power_of_2(Bp), BLOCK_H=1024,
    )
    return y


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


# --------------------------------------------------------------------------- #
# KDA initial-state gather (masked)                                            #
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _gather_kda_state_kernel(
        src_ptr,   # ssm_state viewed as [num_lines, S]
        idx_ptr,   # state_indices [num_seqs]
        mask_ptr,  # has_initial_state [num_seqs] int8 (unused when HAS_MASK=False)
        dst_ptr,   # initial viewed as [num_seqs, S]
        S,
        stride_src,
        stride_dst,
        HAS_MASK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        seq = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        col_mask = offs < S
        if HAS_MASK:
            keep = tl.load(mask_ptr + seq) != 0
            load_mask = col_mask & keep
        else:
            load_mask = col_mask
        # int64 offsets: line * S can exceed int32 for large state caches.
        line = tl.load(idx_ptr + seq).to(tl.int64)
        vals = tl.load(src_ptr + line * stride_src + offs, mask=load_mask, other=0.0)
        tl.store(dst_ptr + seq.to(tl.int64) * stride_dst + offs, vals, mask=col_mask)


def gather_kda_initial_state(
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather ``ssm_state[state_indices]`` into a packed ``[num_seqs, ...]``
    initial-state tensor, zeroing sequences whose ``has_initial_state`` is
    False -- in a single kernel.

    Fuses the ``ssm_state[idx].contiguous()`` gather with the
    ``initial[~has_initial_state] = 0`` masking so fresh sequences are written
    as zeros in the same pass instead of a gather followed by a separate
    zero-write pass. Falls back to the torch path when triton is unavailable or
    there are no sequences.
    """
    num_seqs = int(state_indices.shape[0])
    if not _HAS_TRITON or num_seqs == 0:
        initial = ssm_state[state_indices].contiguous()
        if has_initial_state is not None:
            initial[~has_initial_state] = 0
        return initial
    tail = ssm_state.shape[1:]
    src = ssm_state.reshape(ssm_state.shape[0], -1)
    S = src.shape[1]
    initial = torch.empty(
        (num_seqs, *tail), dtype=ssm_state.dtype, device=ssm_state.device
    )
    dst = initial.reshape(num_seqs, -1)
    has_mask = has_initial_state is not None
    mask_arg = has_initial_state.to(torch.int8) if has_mask else src
    BLOCK = 1024
    grid = (num_seqs, triton.cdiv(S, BLOCK))
    _gather_kda_state_kernel[grid](
        src, state_indices, mask_arg, dst, S,
        src.stride(0), dst.stride(0),
        HAS_MASK=has_mask, BLOCK=BLOCK,
    )
    return initial
