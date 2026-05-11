# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton fused kernel: GemmaRMSNorm + RoPE + KV cache write for Qwen3-Next.

Accepts separate q, k, v tensors (after torch.split from QKVGParallelLinear):
  - Q: GemmaRMSNorm + neox RoPE → contiguous q_out
  - K: GemmaRMSNorm + neox RoPE → contiguous k_out + FP8 quant SHUFFLE cache write
  - V: FP8 quant + SHUFFLE cache write (no norm, no RoPE)

Returns freshly allocated contiguous (q_out, k_out).
"""

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def _fused_qkv_norm_rope_cache_kernel(
    # Separate Q, K, V input pointers (may be strided views from torch.split)
    q_ptr,
    q_stride_t,
    k_ptr,
    k_stride_t,
    v_ptr,
    v_stride_t,
    # Contiguous output pointers
    q_out_ptr,
    k_out_ptr,
    # Norm weights
    qw_ptr,
    kw_ptr,
    # RoPE cos/sin caches: [max_pos, rotary_dim//2]
    cos_cache_ptr,
    sin_cache_ptr,
    cos_sin_stride_pos,
    # Positions
    pos_ptr,
    # KV cache pointers (SHUFFLE layout)
    k_cache_ptr,
    v_cache_ptr,
    # KV cache strides (5D SHUFFLE: [num_blocks, num_kv_heads, D//X, block_size, X])
    kc_stride_block,
    kc_stride_head,
    kc_stride_dx,
    kc_stride_slot,
    kc_stride_x,
    # V cache strides (5D SHUFFLE: [num_blocks, num_kv_heads, block_size//X, head_dim, X])
    vc_stride_block,
    vc_stride_head,
    vc_stride_sc,
    vc_stride_d,
    vc_stride_x,
    # KV scale pointers (per-token scales: [num_blocks, num_kv_heads, block_size])
    k_scale_ptr,
    v_scale_ptr,
    ks_stride_block,
    ks_stride_head,
    vs_stride_block,
    vs_stride_head,
    # Slot mapping
    slot_mapping_ptr,
    # M-RoPE parameters
    pos_stride_row,  # stride between position rows (for 2D positions)
    # Dimensions
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    eps: tl.constexpr,
    # Cache layout
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROTARY_DIM_HALF: tl.constexpr,
    IS_FP8: tl.constexpr,
    # M-RoPE section boundaries (cumulative)
    MROPE_S0: tl.constexpr = 0,
    MROPE_S1: tl.constexpr = 0,
    IS_MROPE: tl.constexpr = False,
):
    # Grid: (num_tokens * (num_heads + num_kv_heads),)
    pid = tl.program_id(0)
    total_heads = num_heads + num_kv_heads
    token_id = pid // total_heads
    head_id = pid % total_heads

    d_offs = tl.arange(0, BLOCK_D)

    if head_id < num_heads:
        # ============ Q head processing: GemmaRMSNorm + RoPE ============
        h = head_id
        # Input offset uses strided layout
        q_in_offset = token_id * q_stride_t + h * BLOCK_D

        # Load q from strided input
        q = tl.load(q_ptr + q_in_offset + d_offs).to(tl.float32)

        # GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight)
        variance = tl.sum(q * q, axis=0) / BLOCK_D
        q_normed = q * tl.math.rsqrt(variance + eps)
        qw = tl.load(qw_ptr + d_offs).to(tl.float32)
        q_normed = q_normed * (1.0 + qw)

        # RoPE (neox-style, partial rotary)
        rot_mask = d_offs < ROTARY_DIM
        first_half_mask = d_offs < ROTARY_DIM_HALF
        d_cos_idx = tl.where(
            first_half_mask,
            d_offs,
            tl.where(
                d_offs < ROTARY_DIM,
                d_offs - ROTARY_DIM_HALF,
                tl.zeros_like(d_offs),
            ),
        )

        if IS_MROPE:
            # M-RoPE: per-dim position selection based on mrope_section
            pos_t = tl.load(pos_ptr + 0 * pos_stride_row + token_id)
            pos_h = tl.load(pos_ptr + 1 * pos_stride_row + token_id)
            pos_w = tl.load(pos_ptr + 2 * pos_stride_row + token_id)
            pos_per_dim = tl.where(
                d_cos_idx < MROPE_S0,
                pos_t,
                tl.where(d_cos_idx < MROPE_S1, pos_h, pos_w),
            )
            cos_base = pos_per_dim * cos_sin_stride_pos
        else:
            pos = tl.load(pos_ptr + token_id)
            cos_base = pos * cos_sin_stride_pos

        cos_vals = tl.load(
            cos_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=1.0
        ).to(tl.float32)
        sin_vals = tl.load(
            sin_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=0.0
        ).to(tl.float32)

        # Neox RoPE rotation in registers (no global memory scratch)
        gather_idx = tl.where(
            first_half_mask,
            d_offs + ROTARY_DIM_HALF,
            tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs),
        )
        q_gathered_raw = tl.load(q_ptr + q_in_offset + gather_idx).to(tl.float32)
        qw_gathered = tl.load(qw_ptr + gather_idx).to(tl.float32)
        q_gathered_normed = (
            q_gathered_raw * tl.math.rsqrt(variance + eps) * (1.0 + qw_gathered)
        )
        q_rot = tl.where(first_half_mask, -q_gathered_normed, q_gathered_normed)
        q_rot = tl.where(rot_mask, q_rot, 0.0)

        q_roped = q_normed * cos_vals + q_rot * sin_vals

        # Write to contiguous q_out: [T, num_heads * BLOCK_D]
        q_out_offset = token_id * (num_heads * BLOCK_D) + h * BLOCK_D
        tl.store(
            q_out_ptr + q_out_offset + d_offs,
            q_roped.to(q_out_ptr.dtype.element_ty),
        )
    else:
        # ============ KV head processing ============
        kv_h = head_id - num_heads

        # --- K: GemmaRMSNorm + RoPE → contiguous k_out + cache write ---
        # Input offset uses strided layout
        k_in_offset = token_id * k_stride_t + kv_h * BLOCK_D
        k = tl.load(k_ptr + k_in_offset + d_offs).to(tl.float32)

        # GemmaRMSNorm on k
        k_variance = tl.sum(k * k, axis=0) / BLOCK_D
        k_normed = k * tl.math.rsqrt(k_variance + eps)
        kw = tl.load(kw_ptr + d_offs).to(tl.float32)
        k_normed = k_normed * (1.0 + kw)

        # RoPE on k (neox-style, partial rotary)
        rot_mask = d_offs < ROTARY_DIM
        first_half_mask = d_offs < ROTARY_DIM_HALF
        d_cos_idx = tl.where(
            first_half_mask,
            d_offs,
            tl.where(
                d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, tl.zeros_like(d_offs)
            ),
        )

        if IS_MROPE:
            pos_t = tl.load(pos_ptr + 0 * pos_stride_row + token_id)
            pos_h = tl.load(pos_ptr + 1 * pos_stride_row + token_id)
            pos_w = tl.load(pos_ptr + 2 * pos_stride_row + token_id)
            pos_per_dim = tl.where(
                d_cos_idx < MROPE_S0,
                pos_t,
                tl.where(d_cos_idx < MROPE_S1, pos_h, pos_w),
            )
            cos_base = pos_per_dim * cos_sin_stride_pos
        else:
            pos = tl.load(pos_ptr + token_id)
            cos_base = pos * cos_sin_stride_pos

        cos_vals = tl.load(
            cos_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=1.0
        ).to(tl.float32)
        sin_vals = tl.load(
            sin_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=0.0
        ).to(tl.float32)

        # Neox RoPE rotation in registers
        gather_idx = tl.where(
            first_half_mask,
            d_offs + ROTARY_DIM_HALF,
            tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs),
        )
        k_gathered_raw = tl.load(k_ptr + k_in_offset + gather_idx).to(tl.float32)
        kw_gathered = tl.load(kw_ptr + gather_idx).to(tl.float32)
        k_gathered_normed = (
            k_gathered_raw * tl.math.rsqrt(k_variance + eps) * (1.0 + kw_gathered)
        )
        k_rot = tl.where(first_half_mask, -k_gathered_normed, k_gathered_normed)
        k_rot = tl.where(rot_mask, k_rot, 0.0)

        k_roped = k_normed * cos_vals + k_rot * sin_vals

        # Write to contiguous k_out: [T, num_kv_heads * BLOCK_D]
        k_out_offset = token_id * (num_kv_heads * BLOCK_D) + kv_h * BLOCK_D
        tl.store(
            k_out_ptr + k_out_offset + d_offs,
            k_roped.to(k_out_ptr.dtype.element_ty),
        )

        # --- V: load for cache write (no norm, no RoPE) ---
        v_in_offset = token_id * v_stride_t + kv_h * BLOCK_D
        v = tl.load(v_ptr + v_in_offset + d_offs)

        # === KV cache write (SHUFFLE layout) ===
        slot = tl.load(slot_mapping_ptr + token_id).to(tl.int64)
        if slot >= 0:
            block_idx = slot // BLOCK_SIZE
            slot_in_block = slot % BLOCK_SIZE

            if IS_FP8:
                # FP8 per-token quantization for k
                k_abs_max = tl.max(tl.abs(k_roped), axis=0)
                k_scale = k_abs_max / 240.0
                k_scale = tl.where(k_scale == 0.0, 1.0, k_scale)
                k_quant = (k_roped / k_scale).to(k_cache_ptr.dtype.element_ty)

                tl.store(
                    k_scale_ptr
                    + block_idx * ks_stride_block
                    + kv_h * ks_stride_head
                    + slot_in_block,
                    k_scale,
                )
            else:
                k_quant = k_roped.to(k_cache_ptr.dtype.element_ty)

            # K cache SHUFFLE write: [num_blocks, num_kv_heads, head_dim//X, block_size, X]
            k_quant_2d = tl.reshape(k_quant, (BLOCK_D // X_SIZE, X_SIZE))
            dx_offs = tl.arange(0, BLOCK_D // X_SIZE).to(tl.int64)
            x_offs = tl.arange(0, X_SIZE).to(tl.int64)
            k_cache_ptrs = (
                k_cache_ptr
                + block_idx * kc_stride_block
                + kv_h * kc_stride_head
                + dx_offs[:, None] * kc_stride_dx
                + slot_in_block * kc_stride_slot
                + x_offs[None, :] * kc_stride_x
            )
            tl.store(k_cache_ptrs, k_quant_2d)

            if IS_FP8:
                # FP8 per-token quantization for v
                v_f32 = v.to(tl.float32)
                v_abs_max = tl.max(tl.abs(v_f32), axis=0)
                v_scale = v_abs_max / 240.0
                v_scale = tl.where(v_scale == 0.0, 1.0, v_scale)
                v_quant = (v_f32 / v_scale).to(v_cache_ptr.dtype.element_ty)

                tl.store(
                    v_scale_ptr
                    + block_idx * vs_stride_block
                    + kv_h * vs_stride_head
                    + slot_in_block,
                    v_scale,
                )
            else:
                v_quant = v.to(v_cache_ptr.dtype.element_ty)

            # V cache SHUFFLE write: [num_blocks, num_kv_heads, block_size//X, head_dim, X]
            slot_chunk = slot_in_block // X_SIZE
            x_off = slot_in_block % X_SIZE
            v_cache_ptrs = (
                v_cache_ptr
                + block_idx * vc_stride_block
                + kv_h * vc_stride_head
                + slot_chunk * vc_stride_sc
                + d_offs.to(tl.int64) * vc_stride_d
                + x_off * vc_stride_x
            )
            tl.store(v_cache_ptrs, v_quant)


def triton_fused_norm_rope_cache(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    positions: Tensor,
    q_norm: torch.nn.Module,
    k_norm: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    k_cache: Tensor,
    v_cache: Tensor,
    k_scale: Tensor | None,
    v_scale: Tensor | None,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
) -> tuple[Tensor, Tensor]:
    """GemmaRMSNorm + RoPE + KV cache write for Qwen3-Next.

    Accepts separate q, k, v tensors (from QKVGParallelLinear split).
    Inputs may be strided views from torch.split.
    - Q: GemmaRMSNorm + RoPE → freshly allocated contiguous q_out
    - K: GemmaRMSNorm + RoPE → freshly allocated contiguous k_out
         + FP8 quant SHUFFLE cache write
    - V: FP8 quant SHUFFLE cache write (no norm, no RoPE)

    Returns contiguous (q_out, k_out).
    """
    T = q.shape[0]
    eps = q_norm.variance_epsilon
    rotary_dim = rotary_emb.rotary_dim

    # Separate cos/sin caches: [max_pos, rotary_dim//2]
    cos_cache = rotary_emb.cos_cache.squeeze(-2).squeeze(-2)
    sin_cache = rotary_emb.sin_cache.squeeze(-2).squeeze(-2)

    is_fp8 = kv_cache_dtype == "fp8"

    block_size = k_cache.shape[3]  # k_cache: [B, H, D//X, block_size, X]
    x_size = k_cache.shape[4]

    # Detect M-RoPE (2D positions: [3, num_tokens])
    is_mrope = positions.ndim == 2
    mrope_section = getattr(rotary_emb, "mrope_section", None)
    if is_mrope:
        assert mrope_section is not None, "M-RoPE requires rotary_emb.mrope_section"
        s0 = mrope_section[0]
        s1 = s0 + mrope_section[1]
        pos_stride_row = positions.stride(0)
    else:
        s0 = 0
        s1 = 0
        pos_stride_row = 0

    # Allocate contiguous output tensors
    q_out = q.new_empty((T, num_heads * head_dim), dtype=q.dtype)
    k_out = k.new_empty((T, num_kv_heads * head_dim), dtype=k.dtype)

    total_heads = num_heads + num_kv_heads
    grid = (T * total_heads,)

    _fused_qkv_norm_rope_cache_kernel[grid](
        q,
        q.stride(0),
        k,
        k.stride(0),
        v,
        v.stride(0),
        q_out,
        k_out,
        q_norm.weight,
        k_norm.weight,
        cos_cache,
        sin_cache,
        cos_cache.stride(0),
        positions,
        k_cache,
        v_cache,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        k_cache.stride(4),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        v_cache.stride(4),
        k_scale if k_scale is not None else q,  # dummy pointer when no scale
        v_scale if v_scale is not None else q,
        k_scale.stride(0) if k_scale is not None and k_scale.dim() >= 1 else 0,
        k_scale.stride(1) if k_scale is not None and k_scale.dim() >= 2 else 0,
        v_scale.stride(0) if v_scale is not None and v_scale.dim() >= 1 else 0,
        v_scale.stride(1) if v_scale is not None and v_scale.dim() >= 2 else 0,
        slot_mapping,
        pos_stride_row,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        eps=eps,
        BLOCK_SIZE=block_size,
        X_SIZE=x_size,
        BLOCK_D=head_dim,
        ROTARY_DIM=rotary_dim,
        ROTARY_DIM_HALF=rotary_dim // 2,
        IS_FP8=is_fp8,
        MROPE_S0=s0,
        MROPE_S1=s1,
        IS_MROPE=is_mrope,
    )

    return q_out, k_out
