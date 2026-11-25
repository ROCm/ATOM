# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fused Rotary Positional Embedding (RoPE) — AITER Triton backend only.

Dispatches to AITER's Triton RoPE kernels for SBHD, THD, 2D, and 3D
layouts.  No PyTorch fallback; AITER must be available.

AITER kernel mapping:
  apply_rotary_pos_emb   -> rope_cached_fwd          (SBHD, single tensor)
  fused_rope             -> rope_cached_fwd * 2       (Q + K separately)
  apply_rotary_pos_emb_2d -> rope_fwd_2d             (vision 2D)
  apply_rotary_pos_emb_3d -> rope_fwd_3d             (video 3D)
"""

import functools
import os

import torch

_SKIP_SYNC = os.environ.get("LUMEN_SKIP_BACKEND_SYNC", "0") == "1"

NEOX_STYLE = 0
GPTJ_STYLE = 1


# ── Lazy AITER getters ──────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _get_aiter_rope_cached_fwd():
    from aiter.ops.triton.rope.rope import rope_cached_fwd

    return rope_cached_fwd


@functools.lru_cache(maxsize=1)
def _get_aiter_rope_cached_bwd():
    from aiter.ops.triton.rope.rope import rope_cached_bwd

    return rope_cached_bwd


@functools.lru_cache(maxsize=1)
def _get_aiter_rope_cached_fwd_lowlevel():
    from aiter.ops.triton.rope.rope import _rope_cached_fwd

    return _rope_cached_fwd


@functools.lru_cache(maxsize=1)
def _get_aiter_rope_cached_bwd_lowlevel():
    from aiter.ops.triton.rope.rope import _rope_cached_bwd

    return _rope_cached_bwd


@functools.lru_cache(maxsize=1)
def _get_aiter_rope_fwd_2d():
    from aiter.ops.triton.rope.rope import rope_fwd_2d

    return rope_fwd_2d


@functools.lru_cache(maxsize=1)
def _get_aiter_rope_fwd_3d():
    from aiter.ops.triton.rope.rope import rope_fwd_3d

    return rope_fwd_3d


# ── Layout adapters ─────────────────────────────────────────────────────────


def _bhsd_to_sbhd(x: torch.Tensor) -> torch.Tensor:
    """[B, H, S, D] -> [S, B, H, D]"""
    return x.permute(2, 0, 1, 3).contiguous()


def _sbhd_to_bhsd(x: torch.Tensor) -> torch.Tensor:
    """[S, B, H, D] -> [B, H, S, D]"""
    return x.permute(1, 2, 0, 3).contiguous()


# ── Frequency tensor adapter ────────────────────────────────────────────────


def _ensure_cos_sin_4d(t: torch.Tensor) -> torch.Tensor:
    """Reshape cos/sin to [S, 1, 1, rotary_dim] expected by the AITER kernel.

    Accepts [S, D], [S, 1, D], or [S, 1, 1, D].
    """
    if t.dim() == 2:
        return t.unsqueeze(1).unsqueeze(1)
    if t.dim() == 3:
        return t.unsqueeze(1)
    return t


# ── Core RoPE dispatch ──────────────────────────────────────────────────────


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply RoPE to a single tensor using AITER Triton kernel.

    Args:
        x: Input tensor [B, H, S, D] (BHSD layout).
        cos: Cosine frequencies — [S, rotary_dim], [S, 1, rotary_dim],
             or [S, 1, 1, rotary_dim].
        sin: Sine frequencies — same shapes as cos.
        interleaved: If True, use GPT-J interleaved layout.

    Returns:
        Rotated tensor, same shape as x.
    """
    assert x.dim() == 4, f"Expected 4D BHSD tensor, got {x.dim()}D"
    assert x.is_cuda, "AITER RoPE requires a CUDA tensor"

    rope_cached_fwd = _get_aiter_rope_cached_fwd()
    rotate_style = GPTJ_STYLE if interleaved else NEOX_STYLE

    cos_4d = _ensure_cos_sin_4d(cos)
    sin_4d = _ensure_cos_sin_4d(sin)

    x_sbhd = _bhsd_to_sbhd(x)
    out_sbhd = rope_cached_fwd(
        x_sbhd,
        cos_4d,
        sin_4d,
        rotate_style,
        True,  # reuse_freqs_front_part
        False,  # nope_first
    )
    if not _SKIP_SYNC:
        torch.cuda.synchronize()
    return _sbhd_to_bhsd(out_sbhd)


def fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fused RoPE to query and key tensors using AITER Triton.

    Dispatches to AITER's rope_cached_fwd for each of Q and K.

    Args:
        q: Query tensor [B, H, S, D].
        k: Key tensor [B, H_k, S, D].
        cos: Cosine frequencies [S, rotary_dim].
        sin: Sine frequencies [S, rotary_dim].
        interleaved: Use GPT-J interleaved layout.

    Returns:
        Tuple of (q_rotated, k_rotated).
    """
    q_rot = apply_rotary_pos_emb(q, cos, sin, interleaved)
    k_rot = apply_rotary_pos_emb(k, cos, sin, interleaved)
    return q_rot, k_rot


# ── Autograd-aware RoPE (training) ─────────────────────────────────────────
# The plain apply_rotary_pos_emb above is forward-only (calls rope_cached_fwd).
# Training needs gradients through RoPE, so wrap fwd/bwd kernels in an
# autograd.Function. Verified vs the HF reference (q*cos + rotate_half(q)*sin):
# fwd SNR ~56 dB, dX SNR ~53 dB (bf16). NEOX layout, full cos/sin [S,D].


def _rope_cached_bhsd(x, cos4, sin4, low_level_fn):
    """Apply cached RoPE to a [B,H,S,D] tensor with zero layout copies.

    The AITER rope kernel works on [S,B,H,D] and is stride-aware (it reads x and
    writes out via their .stride()). So instead of permute().contiguous() round
    trips, we feed a strided [S,B,H,D] *view* of the BHSD input and have the
    kernel write through a strided [S,B,H,D] view of a freshly-allocated, already
    contiguous [B,H,S,D] output. No contiguous() on either side; the returned
    tensor is contiguous BHSD as attention expects. Math is identical.
    """
    b, h, s, d = x.shape
    out = torch.empty((b, h, s, d), dtype=x.dtype, device=x.device)
    x_sbhd = x.permute(2, 0, 1, 3)  # [S,B,H,D] view, no copy
    out_sbhd = out.permute(2, 0, 1, 3)  # [S,B,H,D] view of contiguous BHSD out
    low_level_fn(
        x_sbhd,
        out_sbhd,
        cos4,
        sin4,
        None,
        None,
        NEOX_STYLE,
        False,
        False,
        False,
        False,  # rotate_style, reuse, nope, inplace, transpose_output
    )
    return out


class _RoPEAutograd(torch.autograd.Function):
    """RoPE on one [B,H,S,D] tensor; cos4/sin4 are [S,1,1,D] (full, NEOX)."""

    @staticmethod
    def forward(ctx, x, cos4, sin4):
        out = _rope_cached_bhsd(x, cos4, sin4, _get_aiter_rope_cached_fwd_lowlevel())
        ctx.save_for_backward(cos4, sin4)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        cos4, sin4 = ctx.saved_tensors
        # AITER's RoPE backward divides by stride(0). An expanded grad has
        # stride 0 and crashes the kernel.
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        gx = _rope_cached_bhsd(
            grad_out, cos4, sin4, _get_aiter_rope_cached_bwd_lowlevel()
        )
        return gx, None, None


def apply_rotary_qk_autograd(q, k, cos, sin):
    """Autograd-aware fused RoPE for Q and K, HF-compatible signature.

    Args:
        q, k: [B, H, S, D] (BHSD).
        cos, sin: HF-style full cos/sin — [B, S, D] or [S, D] (NEOX, front half
            duplicated). RoPE is batch-independent so batch dim (if present) is
            dropped.

    Returns:
        (q_rot, k_rot), same shapes as q, k, with gradients.
    """
    if cos.dim() == 3:  # [B, S, D] -> [S, D]
        cos, sin = cos[0], sin[0]
    cos4 = cos.unsqueeze(1).unsqueeze(1)  # [S, 1, 1, D]
    sin4 = sin.unsqueeze(1).unsqueeze(1)
    return _RoPEAutograd.apply(q, cos4, sin4), _RoPEAutograd.apply(k, cos4, sin4)


# ── 2D / 3D RoPE variants ──────────────────────────────────────────────────


def apply_rotary_pos_emb_2d(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    img_height: int,
    img_width: int,
) -> torch.Tensor:
    """Apply 2D RoPE for vision models using AITER Triton kernel.

    Args:
        x: Input tensor [B, WH, H, D] (batch, spatial, heads, dim).
        cos_h / sin_h: Height-direction frequencies.
        cos_w / sin_w: Width-direction frequencies.
        img_height: Image height in patches.
        img_width: Image width in patches.

    Returns:
        Rotated tensor, same shape as x.
    """
    assert x.is_cuda, "AITER 2D RoPE requires a CUDA tensor"
    rope_fwd_2d = _get_aiter_rope_fwd_2d()
    out = rope_fwd_2d(
        x,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
        int(img_height),
        int(img_width),
        NEOX_STYLE,  # rotate_style
        True,  # reuse_freqs_front_part
        False,  # nope_first
    )
    if not _SKIP_SYNC:
        torch.cuda.synchronize()
    return out


def apply_rotary_pos_emb_3d(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    sp_size: int = 1,
    sp_rank: int = 0,
) -> torch.Tensor:
    """Apply 3D RoPE for video models using AITER Triton kernel.

    Args:
        x: Input tensor [B, S, num_heads, C].
        grid_sizes: Grid dimensions tensor.
        freqs: Complex-valued frequency tensor.
        sp_size: Sequence parallelism world size.
        sp_rank: Sequence parallelism rank.

    Returns:
        Rotated tensor (float32).
    """
    assert x.is_cuda, "AITER 3D RoPE requires a CUDA tensor"
    rope_fwd_3d = _get_aiter_rope_fwd_3d()
    out = rope_fwd_3d(x, grid_sizes, freqs, sp_size, sp_rank)
    if not _SKIP_SYNC:
        torch.cuda.synchronize()
    return out
