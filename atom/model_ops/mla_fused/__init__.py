# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Fused MLA decode kernels for gfx1250 (Gluon) and the checks that gate them.

* ATOM_MLA_FUSED_KUP_QPE: K-up GEMM + q_pe RoPE + KV-cache write in one kernel
  (fused_kup_qpe.kup_qpe_fused).

The gate below only uses attributes and shapes, so it imports without the GPU
stack; the kernel is imported lazily by the caller and needs a TDM arch
(aiter `arch_info.is_tdm_avail()`, checked once by the caller). GPU byte
comparisons are required when changing the kernel or its supported shapes.
"""

import torch

_KUP_QPE_LDS_BUDGET = 224 * 1024


def _kup_qpe_lds_bytes(latent_rank: int, nope_dim: int, rope_dim: int) -> int:
    """Raw LDS for the fixed 64x256, two-buffer launch, before layout padding.

    224 KiB is ATOM's conservative budget, below gfx1250's 320 KiB capacity.
    Keep it fixed until the supported shape matrix has GPU measurements.
    """
    half_m = 64 // (latent_rank // 256)
    return (
        2 * 256 * nope_dim
        + 4 * 64 * nope_dim
        + 2 * 64 * 256
        + 8 * half_m * rope_dim
        + 2 * rope_dim
    )


def kup_qpe_fusion_reason(
    *,
    use_qrep: bool,
    q_out: torch.Tensor | None,
    w_k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    kv_cache: torch.Tensor | None,
    k_scale,
) -> str | None:
    """None means eligible; otherwise return the specific decline reason.

    The caller owns the flag/TDM, prefill/DCP and slot-count checks. Keep
    use_qrep and the kernel shape/dtype contracts.
    """
    if use_qrep:
        return "query replication"
    if q_out is None or kv_cache is None:
        return "no prestored q_out or shuffled KV cache"
    num_tokens = q_out.shape[0]
    if positions.dim() != 1 or positions.shape[0] != num_tokens:
        return "positions must be one row per decode token"
    if positions.dtype not in (torch.int32, torch.int64):
        return "positions dtype must be int32 or int64"
    if (
        q_out.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        or kv_cache.dtype != q_out.dtype
    ):
        return "query/cache dtype must be matching FP8"
    if w_k.dtype != torch.bfloat16 or w_k.dim() != 3:
        return "W_K must be a rank-3 BF16 tensor"
    num_heads, latent_rank, nope_dim = w_k.shape
    if (
        num_heads <= 0
        or latent_rank not in (256, 512, 1024)
        or nope_dim not in (32, 64, 128)
    ):
        return "unsupported W_K head/latent/nope shape"
    if w_k.stride(2) != 1:
        return "W_K last dimension must be contiguous"
    if rope_dim not in (32, 64, 128, 256):
        return "unsupported RoPE width"
    if _kup_qpe_lds_bytes(latent_rank, nope_dim, rope_dim) > _KUP_QPE_LDS_BUDGET:
        return "shape exceeds ATOM's conservative LDS budget"
    cos, sin = cos_cache, sin_cache
    while cos.dim() > 2 and cos.shape[1] == 1:
        cos = cos.squeeze(1)
    while sin.dim() > 2 and sin.shape[1] == 1:
        sin = sin.squeeze(1)
    if cos.dim() != 2 or cos.shape[0] == 0:
        return "cos cache must have nonempty position/frequency dimensions"
    if (
        cos.shape[-1] * 2 != rope_dim
        or cos.stride(-1) != 1
        or cos.dtype != torch.bfloat16
    ):
        return "cos cache must be contiguous half-width BF16"
    if sin.shape != cos.shape or sin.stride() != cos.stride() or sin.dtype != cos.dtype:
        return "sin/cos cache layout or dtype mismatch"
    if kv_cache.dim() != 4 or kv_cache.shape[1] != 1:
        return "KV cache must be a shuffled single-head block view"
    if (
        kv_cache.shape[2] <= 0
        or kv_cache.shape[2] % 16
        or kv_cache.shape[3] != latent_rank + rope_dim
    ):
        return "KV block size or entry width is unsupported"
    if kv_cache.stride(3) != 1:
        return "KV cache last dimension must be contiguous"
    # The parent allocates q_out. This also catches a mismatched W_K shape
    # before its dimensions reach the launch descriptors.
    if q_out.shape != (num_tokens, num_heads, latent_rank + rope_dim):
        return "prestored q_out and W_K dimensions mismatch"
    if q_out.stride(2) != 1 or num_tokens * q_out.stride(0) >= 2**31:
        return "q_out stride or int32 byte offset is unsupported"
    if any(
        t.device != q_out.device
        for t in (w_k, cos_cache, sin_cache, positions, kv_cache)
    ):
        return "query/weight/positions/cache device mismatch"
    if k_scale is not None and not (
        isinstance(k_scale, torch.Tensor) and k_scale.numel() == 1
    ):
        return "k_scale must be one tensor element or absent"
    if isinstance(k_scale, torch.Tensor) and k_scale.device != q_out.device:
        return "k_scale and query device mismatch"
    return None if num_tokens > 0 else "no decode tokens"


def kup_qpe_fusion_ok(**kwargs) -> bool:
    return kup_qpe_fusion_reason(**kwargs) is None
