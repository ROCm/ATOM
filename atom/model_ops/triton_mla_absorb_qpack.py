# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""W_K absorb of the MLA query, stored as FP8 straight into the packed ``q_out``.

The two-kernel path runs aiter's ``batched_gemm_bf16`` into a bf16 ``[H, T,
kv_lora_rank]`` intermediate, which ``fused_qk_rope_concat_and_cache_mla`` then
reads back to write ``q_out[..., :kv_lora_rank]`` as ``cast(x * (1/q_scale))``.
This kernel keeps the fp32 accumulator and applies that epilogue itself, so the
intermediate is never written.

Numerics: the two-kernel path rounds the accumulator to bf16 before the FP8
cast, this one does not, and the fp32 reduction order differs. Both only move a
value across an e4m3 rounding boundary, so the stored code differs by at most
one e4m3 step. The clamp to the finite FP8 range is the HIP pack's.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_absorb_q_pack_kernel(
    q_nope_ptr,  # [H, T, P], last dim contiguous
    w_k_ptr,  # [H, L, P], last dim contiguous
    q_out_ptr,  # [T, H, >= L], last dim contiguous
    q_scale_ptr,  # fp32 scalar
    num_tokens,
    stride_q_h,
    stride_q_t,
    stride_w_h,
    stride_w_l,
    stride_out_t,
    stride_out_h,
    P: tl.constexpr,
    L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_p = tl.arange(0, P)
    m_mask = offs_m < num_tokens
    rows = offs_m.to(tl.int64)[:, None]

    a = tl.load(
        q_nope_ptr + head * stride_q_h + rows * stride_q_t + offs_p[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    inv_scale = 1.0 / tl.load(q_scale_ptr)
    w_base = w_k_ptr + head * stride_w_h
    out_base = q_out_ptr + head * stride_out_h + rows * stride_out_t
    for n in tl.range(0, L, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        w = tl.load(w_base + offs_n[:, None] * stride_w_l + offs_p[None, :])
        acc = tl.dot(a, tl.trans(w))
        out = tl.clamp(acc * inv_scale, -FP8_MAX, FP8_MAX)
        tl.store(
            out_base + offs_n[None, :],
            out.to(q_out_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )


def mla_absorb_q_pack(
    q_nope: torch.Tensor,
    w_k: torch.Tensor,
    q_out: torch.Tensor,
    q_scale: torch.Tensor,
) -> None:
    """``q_out[t, h, :L] = e4m3(q_nope[h, t] @ w_k[h].T / q_scale)``, in place.

    Args:
        q_nope: ``[H, T, P]`` bf16; the transposed nope slice of ``q_proj``'s
            output, so only the last dim need be contiguous.
        w_k: ``[H, L, P]`` bf16, the BF16 absorb weight (``W_K`` / ``W_K_qrep``).
        q_out: ``[T, H, >= L]`` FP8; columns ``L:`` are not touched.
        q_scale: fp32 per-tensor scale (``MLAAttention._q_scale``).
    """
    num_heads, num_tokens, p = q_nope.shape
    l = w_k.shape[1]
    assert w_k.shape == (num_heads, l, p) and q_out.shape[1] == num_heads
    assert q_nope.stride(-1) == 1 and w_k.stride(-1) == 1 and q_out.stride(-1) == 1
    if num_tokens == 0:
        return
    block_m, block_n = 256, 64
    _mla_absorb_q_pack_kernel[(triton.cdiv(num_tokens, block_m), num_heads)](
        q_nope,
        w_k,
        q_out,
        q_scale,
        num_tokens,
        q_nope.stride(0),
        q_nope.stride(1),
        w_k.stride(0),
        w_k.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        P=p,
        L=l,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        FP8_MAX=torch.finfo(q_out.dtype).max,
        num_warps=4,
        num_stages=2,
    )
