# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`ATOM_K3_DRAFT_QKROPE`: the Kimi-K3 DSpark GQA draft's per-head QK-norm, rope and
paged K/V write in ONE launch per layer.

Without the flag, per draft layer, the block pass runs four launches after the
packed QKV GEMM -- the q RMSNorm, the k RMSNorm, the neox rope and
`reshape_and_cache_flash` -- and the context write runs three (k norm, rope,
cache). Here one program per token does all of it in registers, keeping the two
bf16 rounding points of the unfused chain:

    n = bf16(x * rsqrt(mean(x^2) + eps) * w)          (per 64-wide head, fp32)
    o = bf16(n_lo * cos - n_hi * sin | n_hi * cos + n_lo * sin)  (neox, fp32)
    cache[slot] = fp8(float(o_k) * (1 / k_scale)), fp8(float(v) * (1 / v_scale))

q and k land back in the packed buffer (the strided views the block read takes);
v is untouched. Numerics: the same formulas
and rounding points as the aiter kernels; the fp32 reduction order of the 64-wide
sum of squares (and FMA contraction in the rope) is the compiler's, so an output
can differ from the unfused chain by one bf16 ulp. Draft-only: verification
accepts against the q the draft actually drew from.

Supported pools: bf16, and `float8_e4m3fn` (gfx950's platform fp8). The saturation
bound below is e4m3fn's 448; the gfx942 `float8_e4m3fnuz` pool saturates at 240, so
`qkrope_supported` refuses it and the caller keeps the unfused chain.
"""

import torch
import triton
import triton.language as tl

_FP8_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def _norm_rope_heads(row_ptr, w_ptr, c, s, eps, H: tl.constexpr, D: tl.constexpr):
    """[H, D/2] lo and hi halves of H heads at `row_ptr`, normed then roped (bf16)."""
    HALF: tl.constexpr = D // 2
    h = tl.arange(0, H)
    j = tl.arange(0, HALF)
    offs = h[:, None] * D + j[None, :]
    x_lo = tl.load(row_ptr + offs).to(tl.float32)
    x_hi = tl.load(row_ptr + offs + HALF).to(tl.float32)
    ss = tl.sum(x_lo * x_lo, axis=1) + tl.sum(x_hi * x_hi, axis=1)
    r = tl.rsqrt(ss / D + eps)
    w_lo = tl.load(w_ptr + j).to(tl.float32)
    w_hi = tl.load(w_ptr + HALF + j).to(tl.float32)
    n_lo = (x_lo * r[:, None] * w_lo[None, :]).to(tl.bfloat16).to(tl.float32)
    n_hi = (x_hi * r[:, None] * w_hi[None, :]).to(tl.bfloat16).to(tl.float32)
    o_lo = (n_lo * c[None, :] - n_hi * s[None, :]).to(tl.bfloat16)
    o_hi = (n_hi * c[None, :] + n_lo * s[None, :]).to(tl.bfloat16)
    return o_lo, o_hi, offs


@triton.jit
def _draft_qk_norm_rope_cache_kernel(
    qkv_ptr,  # [n, q_size + 2 * kv_size] bf16, row stride `row_stride`
    row_stride,
    q_w_ptr,  # [D] q_norm weight
    k_w_ptr,  # [D] k_norm weight
    eps,
    pos_ptr,  # [n] absolute positions
    cos_ptr,  # [max_pos, D/2] (row stride `cs_stride`)
    sin_ptr,
    cs_stride,
    slot_ptr,  # [n] destination slots, -1 = skip
    kc_ptr,  # FLASH k pool rows: [blocks, block_size, HKV, D]
    vc_ptr,
    kc_block_stride,
    vc_block_stride,
    block_size,
    inv_k_scale,
    inv_v_scale,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    WITH_Q: tl.constexpr,
    WRITE_K: tl.constexpr,
    FP8: tl.constexpr,
):
    t = tl.program_id(0)
    HALF: tl.constexpr = D // 2
    pos = tl.load(pos_ptr + t).to(tl.int64)
    j = tl.arange(0, HALF)
    c = tl.load(cos_ptr + pos * cs_stride + j).to(tl.float32)
    s = tl.load(sin_ptr + pos * cs_stride + j).to(tl.float32)
    row = qkv_ptr + t.to(tl.int64) * row_stride
    if WITH_Q:
        q_lo, q_hi, q_offs = _norm_rope_heads(row, q_w_ptr, c, s, eps, HQ, D)
        tl.store(row + q_offs, q_lo)
        tl.store(row + q_offs + HALF, q_hi)
    k_row = row + HQ * D
    k_lo, k_hi, k_offs = _norm_rope_heads(k_row, k_w_ptr, c, s, eps, HKV, D)
    if WRITE_K:
        tl.store(k_row + k_offs, k_lo)
        tl.store(k_row + k_offs + HALF, k_hi)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot >= 0:
        blk = slot // block_size
        off = slot - blk * block_size
        kc = kc_ptr + blk * kc_block_stride + off * (HKV * D)
        vc = vc_ptr + blk * vc_block_stride + off * (HKV * D)
        v_row = k_row + HKV * D
        v_lo = tl.load(v_row + k_offs).to(tl.float32)
        v_hi = tl.load(v_row + k_offs + HALF).to(tl.float32)
        if FP8:
            lim = _FP8_E4M3_MAX
            k_lo_c = tl.clamp(k_lo.to(tl.float32) * inv_k_scale, -lim, lim)
            k_hi_c = tl.clamp(k_hi.to(tl.float32) * inv_k_scale, -lim, lim)
            v_lo_c = tl.clamp(v_lo * inv_v_scale, -lim, lim)
            v_hi_c = tl.clamp(v_hi * inv_v_scale, -lim, lim)
            tl.store(kc + k_offs, k_lo_c.to(kc_ptr.dtype.element_ty))
            tl.store(kc + k_offs + HALF, k_hi_c.to(kc_ptr.dtype.element_ty))
            tl.store(vc + k_offs, v_lo_c.to(vc_ptr.dtype.element_ty))
            tl.store(vc + k_offs + HALF, v_hi_c.to(vc_ptr.dtype.element_ty))
        else:
            tl.store(kc + k_offs, k_lo)
            tl.store(kc + k_offs + HALF, k_hi)
            tl.store(vc + k_offs, v_lo.to(vc_ptr.dtype.element_ty))
            tl.store(vc + k_offs + HALF, v_hi.to(vc_ptr.dtype.element_ty))


def qkrope_supported(
    qkv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
    eps_q: float,
    eps_k: float,
    rope,
) -> bool:
    """True when `draft_qk_norm_rope_cache` computes exactly what the unfused
    chain does for these operands; anything else keeps the chain."""
    inner = num_kv_heads * head_dim
    return (
        qkv.dtype == torch.bfloat16
        and qkv.is_contiguous()
        and eps_q == eps_k
        and bool(getattr(rope, "is_neox_style", False))
        and getattr(rope, "rotary_dim", None) == head_dim
        and getattr(rope, "cos_cache", None) is not None
        and getattr(rope, "sin_cache", None) is not None
        and k_cache.dtype == v_cache.dtype
        and k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and all(
            c.dim() == 4
            and c.stride(3) == 1
            and c.stride(2) == head_dim
            and c.stride(1) == inner
            for c in (k_cache, v_cache)
        )
    )


def draft_qk_norm_rope_cache(
    qkv: torch.Tensor,  # [n, HQ*D + 2*HKV*D], unit inner stride
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,  # [n]
    cos_cache: torch.Tensor,  # [max_pos, ..., D/2]
    sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [n]
    k_cache: torch.Tensor,  # [blocks, block_size, HKV, D]
    v_cache: torch.Tensor,
    k_scale: float,
    v_scale: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    write_k: bool = True,
) -> None:
    """In place on `qkv`: q (if `q_weight`) and k (if `write_k`) normed + roped;
    k and v written to the pool at `slot_mapping` (fp8 pools take
    `float(x) * (1 / scale)` like `reshape_and_cache_flash`)."""
    n = qkv.shape[0]
    if n == 0:
        return
    assert qkv.stride(1) == 1 and head_dim % 2 == 0
    assert k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
    assert (
        k_cache.dim() == 4 and k_cache.stride(3) == 1 and k_cache.stride(2) == head_dim
    )
    assert k_cache.stride(1) == num_kv_heads * head_dim
    assert v_cache.stride(1) == num_kv_heads * head_dim and v_cache.stride(3) == 1
    cos2 = cos_cache.reshape(cos_cache.shape[0], -1)
    sin2 = sin_cache.reshape(sin_cache.shape[0], -1)
    assert cos2.shape[1] == head_dim // 2 and cos2.stride() == sin2.stride()
    _draft_qk_norm_rope_cache_kernel[(n,)](
        qkv,
        qkv.stride(0),
        q_weight if q_weight is not None else k_weight,
        k_weight,
        eps,
        positions,
        cos2,
        sin2,
        cos2.stride(0),
        slot_mapping,
        k_cache,
        v_cache,
        k_cache.stride(0),
        v_cache.stride(0),
        k_cache.shape[1],
        1.0 / k_scale,
        1.0 / v_scale,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        D=head_dim,
        WITH_Q=q_weight is not None,
        WRITE_K=write_k,
        FP8=k_cache.dtype.itemsize == 1,
        num_warps=1,
    )
