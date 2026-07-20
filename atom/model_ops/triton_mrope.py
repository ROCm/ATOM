# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Triton kernels for Qwen3.5 MRoPE.

This module currently specializes the hot MRoPE path used by Qwen3.5:
positions are 3D T/H/W ids, RoPE is Neox-style, head_size=256, and
rotary_dim=64. Unsupported shapes return ``None`` so callers can fall back to
the generic rotary embedding implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _mrope_qk_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    q_stride_t: tl.constexpr,
    k_stride_t: tl.constexpr,
    q_out_stride_t: tl.constexpr,
    k_out_stride_t: tl.constexpr,
    pos_stride_row: tl.constexpr,
    cos_stride_pos: tl.constexpr,
    sin_stride_pos: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    head_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    rotary_half: tl.constexpr,
    section_h: tl.constexpr,
    section_w: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    total_heads = num_q_heads + num_k_heads
    token_id = pid // total_heads
    head_id = pid - token_id * total_heads
    d = tl.arange(0, block_d)
    mask = d < head_size

    is_q = head_id < num_q_heads
    local_head = tl.where(is_q, head_id, head_id - num_q_heads)

    q_base_in = token_id * q_stride_t + local_head * head_size
    k_base_in = token_id * k_stride_t + local_head * head_size
    q_base_out = token_id * q_out_stride_t + local_head * head_size
    k_base_out = token_id * k_out_stride_t + local_head * head_size

    x_q = tl.load(q_ptr + q_base_in + d, mask=mask & is_q, other=0.0).to(tl.float32)
    x_k = tl.load(k_ptr + k_base_in + d, mask=mask & ~is_q, other=0.0).to(tl.float32)
    x = x_q + x_k

    rot_mask = d < rotary_dim
    first_half = d < rotary_half
    freq_idx = tl.where(first_half, d, d - rotary_half)
    pair_d = tl.where(
        first_half,
        d + rotary_half,
        tl.where(d < rotary_dim, d - rotary_half, d),
    )
    pair_q = tl.load(q_ptr + q_base_in + pair_d, mask=mask & is_q, other=0.0).to(
        tl.float32
    )
    pair_k = tl.load(k_ptr + k_base_in + pair_d, mask=mask & ~is_q, other=0.0).to(
        tl.float32
    )
    pair = pair_q + pair_k

    pos_t = tl.load(positions_ptr + token_id)
    pos_h = tl.load(positions_ptr + pos_stride_row + token_id)
    pos_w = tl.load(positions_ptr + 2 * pos_stride_row + token_id)

    use_h = ((freq_idx % 3) == 1) & (freq_idx < section_h * 3)
    use_w = ((freq_idx % 3) == 2) & (freq_idx < section_w * 3)
    pos = tl.where(use_h, pos_h, tl.where(use_w, pos_w, pos_t))

    cos = tl.load(
        cos_ptr + pos * cos_stride_pos + freq_idx,
        mask=rot_mask,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + pos * sin_stride_pos + freq_idx,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)

    rotated = tl.where(first_half, -pair, pair)
    out = tl.where(rot_mask, x * cos + rotated * sin, x)

    tl.store(q_out_ptr + q_base_out + d, out, mask=mask & is_q)
    tl.store(k_out_ptr + k_base_out + d, out, mask=mask & ~is_q)


@triton.jit
def _mrope_qk_tiled_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    q_stride_t: tl.constexpr,
    k_stride_t: tl.constexpr,
    q_out_stride_t: tl.constexpr,
    k_out_stride_t: tl.constexpr,
    pos_stride_row: tl.constexpr,
    cos_stride_pos: tl.constexpr,
    sin_stride_pos: tl.constexpr,
    num_tokens,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    head_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    rotary_half: tl.constexpr,
    section_h: tl.constexpr,
    section_w: tl.constexpr,
    block_t: tl.constexpr,
    block_d: tl.constexpr,
):
    token_block = tl.program_id(0)
    head_id = tl.program_id(1)
    rows = token_block * block_t + tl.arange(0, block_t)
    d = tl.arange(0, block_d)
    row_mask = rows < num_tokens
    d_mask = d < head_size

    is_q = head_id < num_q_heads
    local_head = tl.where(is_q, head_id, head_id - num_q_heads)

    offsets_q = rows[:, None] * q_stride_t + local_head * head_size + d[None, :]
    offsets_k = rows[:, None] * k_stride_t + local_head * head_size + d[None, :]
    mask = row_mask[:, None] & d_mask[None, :]

    x_q = tl.load(q_ptr + offsets_q, mask=mask & is_q, other=0.0).to(tl.float32)
    x_k = tl.load(k_ptr + offsets_k, mask=mask & ~is_q, other=0.0).to(tl.float32)
    x = x_q + x_k

    rot_mask = d < rotary_dim
    first_half = d < rotary_half
    freq_idx = tl.where(first_half, d, d - rotary_half)
    pair_d = tl.where(
        first_half,
        d + rotary_half,
        tl.where(d < rotary_dim, d - rotary_half, d),
    )
    pair_offsets_q = (
        rows[:, None] * q_stride_t + local_head * head_size + pair_d[None, :]
    )
    pair_offsets_k = (
        rows[:, None] * k_stride_t + local_head * head_size + pair_d[None, :]
    )
    pair_q = tl.load(q_ptr + pair_offsets_q, mask=mask & is_q, other=0.0).to(tl.float32)
    pair_k = tl.load(k_ptr + pair_offsets_k, mask=mask & ~is_q, other=0.0).to(
        tl.float32
    )
    pair = pair_q + pair_k

    pos_t = tl.load(positions_ptr + rows, mask=row_mask, other=0)
    pos_h = tl.load(positions_ptr + pos_stride_row + rows, mask=row_mask, other=0)
    pos_w = tl.load(positions_ptr + 2 * pos_stride_row + rows, mask=row_mask, other=0)

    use_h = ((freq_idx % 3) == 1) & (freq_idx < section_h * 3)
    use_w = ((freq_idx % 3) == 2) & (freq_idx < section_w * 3)
    pos = tl.where(
        use_h[None, :],
        pos_h[:, None],
        tl.where(use_w[None, :], pos_w[:, None], pos_t[:, None]),
    )

    cos = tl.load(
        cos_ptr + pos * cos_stride_pos + freq_idx[None, :],
        mask=row_mask[:, None] & rot_mask[None, :],
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + pos * sin_stride_pos + freq_idx[None, :],
        mask=row_mask[:, None] & rot_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    rotated = tl.where(first_half[None, :], -pair, pair)
    out = tl.where(rot_mask[None, :], x * cos + rotated * sin, x)

    out_offsets_q = rows[:, None] * q_out_stride_t + local_head * head_size + d[None, :]
    out_offsets_k = rows[:, None] * k_out_stride_t + local_head * head_size + d[None, :]
    tl.store(q_out_ptr + out_offsets_q, out, mask=mask & is_q)
    tl.store(k_out_ptr + out_offsets_k, out, mask=mask & ~is_q)


def try_mrope_qk_fused(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Try the specialized Qwen3.5 MRoPE Triton path.

    Returns ``None`` for unsupported shapes (or whenever invoked under
    ``torch.compile``) so callers fall back to the generic rotary embedding.

    The Python-level shape branches below (``positions.shape[0] != 3``,
    ``q.shape[1] != num_q_heads * head_size``, ...) would otherwise force
    Dynamo to specialize symbolic dims, conflicting with dims marked dynamic
    by the runner (MMStar fails with ConstraintViolationError on
    ``L['positions'].size()[0]``). We previously tried
    ``@torch.compiler.disable`` here, but the resulting graph break inside
    the compiled Qwen3NextAttention forward re-enters ATOM's VllmBackend and
    trips ``AssertionError: VllmBackend can only be called once``. Skipping
    the fused path under compile (eager fallback to ``self.rotary_emb``) is
    the safe option: SGLang eager mode keeps the perf gain, compile mode
    behaves identically to ``main``.
    """
    if torch.compiler.is_compiling():
        return None

    mrope_section = getattr(rotary_emb, "mrope_section", None)
    if (
        positions.ndim != 2
        or positions.shape[0] != 3
        or positions.stride(1) != 1
        or mrope_section is None
        or len(mrope_section) < 3
        or not getattr(rotary_emb, "mrope_interleaved", False)
        or head_size != 256
        or getattr(rotary_emb, "rotary_dim", None) != 64
        or not getattr(rotary_emb, "is_neox_style", False)
        or q.ndim != 2
        or k.ndim != 2
        or q.shape[0] != positions.shape[1]
        or k.shape[0] != positions.shape[1]
        or q.stride(1) != 1
        or k.stride(1) != 1
        or q.shape[1] != num_q_heads * head_size
        or k.shape[1] != num_k_heads * head_size
    ):
        return None

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    num_tokens = positions.shape[1]
    block_d = triton.next_power_of_2(head_size)
    cos_cache = rotary_emb.cos_cache
    sin_cache = rotary_emb.sin_cache

    if num_tokens >= 128:
        block_t = 16
        _mrope_qk_tiled_kernel[
            (triton.cdiv(num_tokens, block_t), num_q_heads + num_k_heads)
        ](
            q,
            k,
            q_out,
            k_out,
            positions,
            cos_cache,
            sin_cache,
            q.stride(0),
            k.stride(0),
            q_out.stride(0),
            k_out.stride(0),
            positions.stride(0),
            cos_cache.stride(0),
            sin_cache.stride(0),
            num_tokens,
            num_q_heads,
            num_k_heads,
            head_size,
            64,
            32,
            mrope_section[1],
            mrope_section[2],
            block_t,
            block_d,
            num_warps=8,
            num_stages=1,
        )
    else:
        _mrope_qk_kernel[(num_tokens * (num_q_heads + num_k_heads),)](
            q,
            k,
            q_out,
            k_out,
            positions,
            cos_cache,
            sin_cache,
            q.stride(0),
            k.stride(0),
            q_out.stride(0),
            k_out.stride(0),
            positions.stride(0),
            cos_cache.stride(0),
            sin_cache.stride(0),
            num_q_heads,
            num_k_heads,
            head_size,
            64,
            32,
            mrope_section[1],
            mrope_section[2],
            block_d,
            num_warps=8,
            num_stages=1,
        )
    return q_out, k_out
