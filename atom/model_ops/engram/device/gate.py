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


# `num_tokens` only scales strides that are multiples of KS, so specializing on
# its value buys nothing: one compile per launch config, whatever the rows.
@triton.jit(do_not_specialize=["num_tokens"])
def _engram_post_wkv_shuf_kernel(
    hidden,
    kv,
    weight,
    token_mask,
    output,
    eps,
    num_tokens,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    TB: tl.constexpr,
    CH: tl.constexpr,
    KS: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """The gate on a residual in aiter's shuffled layout
    resS[k // KS][branch][token][k % KS]: one program takes TB consecutive
    tokens of one branch, TB * KS contiguous elements per KS block. Two passes
    over DIM in CH-column chunks: the three reductions, then the update."""
    branch = tl.program_id(1)
    tok = (tl.program_id(0) * TB + tl.arange(0, TB)).to(tl.int64)
    tlive = tok < num_tokens
    c = tl.arange(0, CH)
    row_kv = tok[:, None] * ((HC + 1) * DIM)
    res_base = branch * (num_tokens * KS) + tok[:, None] * KS
    blk = HC * num_tokens * KS
    ss_r = tl.zeros([TB], tl.float32)
    ss_k = tl.zeros([TB], tl.float32)
    dot = tl.zeros([TB], tl.float32)
    for k0 in tl.static_range(0, DIM, CH):
        col = k0 + c
        live = tlive[:, None] & (col < DIM)[None, :]
        r_off = res_base + (col // KS)[None, :] * blk + (col % KS)[None, :]
        r = tl.load(hidden + r_off, live, 0).to(tl.float32)
        k = tl.load(kv + row_kv + branch * DIM + col[None, :], live, 0).to(tl.float32)
        w = tl.load(weight + branch * DIM + col, col < DIM, 0).to(tl.float32)
        ss_r += tl.sum(r * r, 1)
        ss_k += tl.sum(k * k, 1)
        dot += tl.sum((r * w[None, :]) * k, 1)
    rstd = tl.rsqrt(ss_r / DIM + eps) * tl.rsqrt(ss_k / DIM + eps)
    dot = dot * rstd * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    signed = (magnitude.to(tl.uint32, bitcast=True) & 0x7FFFFFFF) | (
        dot.to(tl.uint32, bitcast=True) & 0x80000000
    )
    gate = tl.sigmoid(signed.to(tl.float32, bitcast=True))
    if HAS_MASK:
        gate = tl.where(tl.load(token_mask + tok, tlive, 0) != 0, gate, 0.0)
    for k0 in tl.static_range(0, DIM, CH):
        col = k0 + c
        live = tlive[:, None] & (col < DIM)[None, :]
        r_off = res_base + (col // KS)[None, :] * blk + (col % KS)[None, :]
        r = tl.load(hidden + r_off, live, 0).to(tl.float32)
        v = tl.load(kv + row_kv + HC * DIM + col[None, :], live, 0).to(tl.float32)
        tl.store(output + r_off, r + gate[:, None] * v, live)


@triton.jit(do_not_specialize=["num_tokens"])
def _engram_post_wkv_shuf_onepass_kernel(
    hidden,
    kv,
    weight,
    token_mask,
    output,
    eps,
    num_tokens,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    TB: tl.constexpr,
    BLOCK: tl.constexpr,
    KS: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """The shuffled gate in one pass: the TB x DIM residual tile is read once
    and held on chip between the reductions and the update. At 8 warps its
    reductions match `_engram_post_wkv_kernel`'s bit for bit. The tile is
    TB x BLOCK wide, so it only pays while the grid is small (decode rows)."""
    branch = tl.program_id(1)
    tok = (tl.program_id(0) * TB + tl.arange(0, TB)).to(tl.int64)
    tlive = tok < num_tokens
    col = tl.arange(0, BLOCK)
    live = tlive[:, None] & (col < DIM)[None, :]
    row_kv = tok[:, None] * ((HC + 1) * DIM)
    r_off = (
        branch * (num_tokens * KS)
        + tok[:, None] * KS
        + (col // KS)[None, :] * (HC * num_tokens * KS)
        + (col % KS)[None, :]
    )
    residual = tl.load(hidden + r_off, live, 0).to(tl.float32)
    key = tl.load(kv + row_kv + branch * DIM + col[None, :], live, 0).to(tl.float32)
    w = tl.load(weight + branch * DIM + col, col < DIM, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(residual * residual, 1) / DIM + eps)
    rstd *= tl.rsqrt(tl.sum(key * key, 1) / DIM + eps)
    dot = tl.sum((residual * w[None, :]) * key, 1) * rstd * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    signed = (magnitude.to(tl.uint32, bitcast=True) & 0x7FFFFFFF) | (
        dot.to(tl.uint32, bitcast=True) & 0x80000000
    )
    gate = tl.sigmoid(signed.to(tl.float32, bitcast=True))
    if HAS_MASK:
        gate = tl.where(tl.load(token_mask + tok, tlive, 0) != 0, gate, 0.0)
    value = tl.load(kv + row_kv + HC * DIM + col[None, :], live, 0).to(tl.float32)
    tl.store(output + r_off, residual + gate[:, None] * value, live)


# (tokens upper bound, (kernel, TB, CH, num_warps)), from a gfx1250 H=5120
# sweep. Decode rows take the one-pass kernel (8 warps: bit-identical to the
# plain-layout gate); past 512 rows its tile costs occupancy, and the two-pass
# kernel streams the residual in CH-column chunks instead.
_SHUF_LAUNCH = (
    (48, ("onepass", 1, None, 8)),
    (96, ("onepass", 2, None, 8)),
    (512, ("onepass", 4, None, 8)),
    (None, ("twopass", 4, 1024, 4)),
)


def _shuf_launch(tokens: int):
    for bound, config in _SHUF_LAUNCH:
        if bound is None or tokens <= bound:
            return config


def _launch_shuf(hidden, kv, weight, token_mask, output, eps, tokens, ks, config):
    kernel, tb, ch, warps = config
    hc, dim = hidden.shape[-2:]
    grid = (triton.cdiv(tokens, tb), hc)
    args = (hidden, kv, weight, token_mask, output, eps, tokens)
    shapes = {
        "DIM": dim,
        "HC": hc,
        "TB": tb,
        "KS": ks,
        "HAS_MASK": token_mask is not None,
    }
    if kernel == "onepass":
        _engram_post_wkv_shuf_onepass_kernel[grid](
            *args,
            **shapes,
            BLOCK=triton.next_power_of_2(dim),
            num_warps=warps,
            enable_fp_fusion=False,
        )
    else:
        _engram_post_wkv_shuf_kernel[grid](
            *args, **shapes, CH=ch, num_warps=warps, enable_fp_fusion=False
        )


def precompile_shuffled_gate(hc: int, dim: int, device, ks: int = 32) -> None:
    """Compile every launch config of the shuffled gate, with and without a
    token mask, on one dummy token, so no decode step pays a Triton compile
    (one per config: `num_tokens` is not specialized)."""
    hidden = torch.zeros(1, hc, dim, dtype=torch.bfloat16, device=device)
    kv = torch.zeros(1, (hc + 1) * dim, dtype=torch.bfloat16, device=device)
    weight = torch.zeros(hc, dim, dtype=torch.float32, device=device)
    mask = torch.ones(1, dtype=torch.bool, device=device)
    output = torch.empty_like(hidden)
    for _, config in _SHUF_LAUNCH:
        for token_mask in (None, mask):
            _launch_shuf(hidden, kv, weight, token_mask, output, 1e-20, 1, ks, config)


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


def engram_post_wkv(
    hidden, kv, weight, token_mask=None, eps=1e-20, res_preshuffle=False, ks=32
):
    """Fused inference path for contiguous [..., HC, DIM] residuals.

    res_preshuffle: ``hidden`` and the returned residual are in aiter's
    mhc_res_shuffle layout ``resS[k // ks][branch][token][k % ks]`` (same
    shape); only the fused kernel reads it.
    """
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
    if res_preshuffle:
        if not supported:
            raise ValueError("a shuffled Engram residual needs the fused gate")
        if dim % ks:
            raise ValueError(f"a shuffled residual needs DIM % {ks} == 0")
        output = torch.empty_like(hidden)
        tokens = hidden.numel() // (hc * dim)
        if tokens:
            _launch_shuf(
                hidden,
                kv,
                weight,
                token_mask,
                output,
                eps,
                tokens,
                ks,
                _shuf_launch(tokens),
            )
        return output
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
