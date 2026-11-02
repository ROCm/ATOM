# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Triton kernels for Qwen3.5 MRoPE.

This module currently specializes the hot MRoPE path used by Qwen3.5:
positions are 3D T/H/W ids, RoPE is Neox-style, head_size=256, and
rotary_dim=64. Unsupported shapes return ``None`` so callers can fall back to
the generic rotary embedding implementation.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx
from torch import nn

if TYPE_CHECKING:
    from atom.model_ops.layernorm import DualRMSNorm


@triton.jit(do_not_specialize=["pos_stride_row"])
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
    pos_stride_row,
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


@triton.jit(do_not_specialize=["pos_stride_row", "num_tokens"])
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
    pos_stride_row,
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


def _mrope_rotary_supported(rotary_emb: nn.Module, head_size: int) -> bool:
    """Whether the rotary embedding is the interleaved Qwen3.5 MRoPE these kernels take:
    head 256, rotary 64, Neox."""
    mrope_section = getattr(rotary_emb, "mrope_section", None)
    return (
        mrope_section is not None
        and len(mrope_section) >= 3
        and getattr(rotary_emb, "mrope_interleaved", False)
        and head_size == 256
        and getattr(rotary_emb, "rotary_dim", None) == 64
        and getattr(rotary_emb, "is_neox_style", False)
    )


def _mrope_qk_supported(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_size: int,
) -> bool:
    """Whether q, k and the rotary embedding have the Qwen3.5 MRoPE shapes these kernels take."""
    return (
        _mrope_rotary_supported(rotary_emb, head_size)
        and positions.ndim == 2
        and positions.shape[0] == 3
        and positions.stride(1) == 1
        and q.ndim == 2
        and k.ndim == 2
        and q.shape[0] == positions.shape[1]
        and k.shape[0] == positions.shape[1]
        and q.stride(1) == 1
        and k.stride(1) == 1
        and q.shape[1] == num_q_heads * head_size
        and k.shape[1] == num_k_heads * head_size
    )


@functools.cache
def _gfx1250_aiter_op(name: str):
    """aiter's fused Q/K norm + MRoPE op ``name`` on gfx1250, or None (other arch, or an
    aiter without it)."""
    if get_gfx() != "gfx1250":
        return None
    try:
        from aiter.ops.triton.fusions import fused_qk_norm_mrope
    except ImportError:
        return None
    return getattr(fused_qk_norm_mrope, name, None)


def try_fused_qk_norm_mrope(
    qk_norm: DualRMSNorm,
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The Q/K Gemma RMSNorm and the Qwen3.5 MRoPE in one aiter Gluon kernel on gfx1250.

    Returns ``None`` where the kernel does not apply (under ``torch.compile`` as
    ``try_mrope_qk_fused``), so callers keep ``qk_norm`` and ``try_mrope_qk_fused``.
    """
    if torch.compiler.is_compiling():
        return None
    fused_qk_norm_mrope = _gfx1250_aiter_op("fused_qk_norm_mrope")
    if (
        fused_qk_norm_mrope is None
        or not qk_norm.add_unit_offset
        or not _mrope_qk_supported(
            rotary_emb, positions, q, k, num_q_heads, num_k_heads, head_size
        )
    ):
        return None
    half = rotary_emb.rotary_dim // 2
    return fused_qk_norm_mrope(
        q,
        k,
        qk_norm.q_norm.weight,
        qk_norm.k_norm.weight,
        qk_norm._eps,
        positions,
        rotary_emb.cos_cache.view(-1, half),
        rotary_emb.sin_cache.view(-1, half),
        rotary_emb.mrope_section,
    )


@functools.cache
def _gfx1250_fused_qk_norm_mrope_cache_fp8() -> bool:
    """Whether aiter's kernel also writes an fp8 cache (its ``kv_scale`` argument)."""
    op = _gfx1250_aiter_op("fused_qk_norm_mrope_cache")
    return op is not None and "kv_scale" in inspect.signature(op).parameters


def gfx1250_fused_qk_norm_mrope_cache_applies(
    rotary_emb: nn.Module, head_size: int, kv_cache_dtype: str
) -> bool:
    """Whether ``try_fused_qk_norm_mrope_cache`` takes every batch of an attention:
    gfx1250 with the aiter kernel, the interleaved Qwen3.5 MRoPE (head 256, rotary
    64, Neox) and a cache in the model dtype, or an fp8 cache read by unified
    attention when the kernel writes fp8 (one per-tensor scale, Q in fp8)."""
    from atom.utils import envs

    fp8_ok = (
        kv_cache_dtype == "fp8"
        and envs.ATOM_USE_UNIFIED_ATTN
        and _gfx1250_fused_qk_norm_mrope_cache_fp8()
    )
    return (
        _gfx1250_aiter_op("fused_qk_norm_mrope_cache") is not None
        and (not kv_cache_dtype.startswith("fp8") or fp8_ok)
        and _mrope_rotary_supported(rotary_emb, head_size)
    )


def try_fused_qk_norm_mrope_cache(
    q_norm: nn.Module,
    k_norm: nn.Module,
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_size: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The Q/K Gemma RMSNorm, the Qwen3.5 MRoPE and the K/V write into the SHUFFLE
    KV cache in one aiter Gluon kernel on gfx1250.

    For an fp8 cache pass its per-tensor ``kv_scale``: K and V are stored scaled and
    clamped, and Q comes back in fp8 (clamped, unscaled), ready for unified attention.

    Returns ``None`` where the kernel does not apply (also under ``torch.compile``,
    and for an fp8 cache without ``kv_scale`` or with an aiter that lacks it), so
    callers keep their other paths.
    """
    if torch.compiler.is_compiling():
        return None
    fused_qk_norm_mrope_cache = _gfx1250_aiter_op("fused_qk_norm_mrope_cache")
    fp8 = kv_scale is not None
    if fp8 and (
        k_cache.element_size() != 1 or not _gfx1250_fused_qk_norm_mrope_cache_fp8()
    ):
        return None
    if (
        fused_qk_norm_mrope_cache is None
        or (not fp8 and k_cache.dtype != q.dtype)
        or k_cache.dim() != 5
        or not _mrope_qk_supported(
            rotary_emb, positions, q, k, num_q_heads, num_k_heads, head_size
        )
    ):
        return None
    if v_cache.dim() == 4:
        # The SHUFFLE view of a [blocks, heads, head_size, block] V cache.
        n, nh, hd, bs = v_cache.shape
        x = k_cache.shape[4]
        v_cache = v_cache.view(n, nh, bs // x, hd, x)
    half = rotary_emb.rotary_dim // 2
    return fused_qk_norm_mrope_cache(
        q,
        k,
        v,
        q_norm.weight,
        k_norm.weight,
        q_norm.variance_epsilon,
        positions,
        rotary_emb.cos_cache.view(-1, half),
        rotary_emb.sin_cache.view(-1, half),
        rotary_emb.mrope_section,
        k_cache,
        v_cache,
        slot_mapping,
        **({"kv_scale": kv_scale} if fp8 else {}),
    )


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
    if torch.compiler.is_compiling() or not _mrope_qk_supported(
        rotary_emb, positions, q, k, num_q_heads, num_k_heads, head_size
    ):
        return None

    mrope_section = rotary_emb.mrope_section
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
