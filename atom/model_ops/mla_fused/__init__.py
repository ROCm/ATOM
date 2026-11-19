# SPDX-License-Identifier: MIT
"""Fused MLA decode kernels for gfx1250 (Gluon) and the checks that gate them.

* ATOM_MLA_FUSED_KUP_QPE: K-up GEMM + q_pe RoPE + KV-cache write in one kernel
  (fused_kup_qpe.kup_qpe_fused).

The gate below only uses attributes and shapes, so it imports without the GPU
stack; the kernel is imported lazily by the caller and needs a TDM arch
(aiter `arch_info.is_tdm_avail()`, checked once by the caller). GPU byte
comparisons are required when changing the kernel or its supported shapes.
"""

import torch


def _is_fp8(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def kup_qpe_fusion_ok(
    *,
    enabled: bool,
    is_prefill: bool,
    use_qrep: bool,
    dcp_world_size: int,
    q_out: torch.Tensor | None,
    num_tokens: int,
    slot_mapping: torch.Tensor,
    w_k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    kv_cache: torch.Tensor | None,
    k_scale,
) -> bool:
    """q_out is the prestored q_out of the BF16-absorb + shuffled-KV path (None
    when that path is off); kv_cache is the shuffled (blocks, 1, bs, d) view.
    `enabled` is the flag on a TDM arch, resolved once at layer init."""
    if not enabled:
        return False
    # Pure decode only: no prefill (so no PCP), no DCP / query replication.
    if is_prefill or use_qrep or dcp_world_size > 1:
        return False
    if q_out is None or kv_cache is None:
        return False
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] != num_tokens:
        return False
    if (
        positions.dim() != 1
        or positions.shape[0] != num_tokens
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        return False
    if not (_is_fp8(q_out) and kv_cache.dtype == q_out.dtype):
        return False
    if w_k.dtype != torch.bfloat16 or w_k.dim() != 3:
        return False
    num_heads, kv_lora_rank, nope_dim = w_k.shape
    # arange and the WMMA / q_pe partition require power-of-two tiles. With
    # BLOCK_N=256 this bounds NSPLIT_N to {1, 2, 4} and HALF_M to {64, 32, 16}.
    if (
        num_heads <= 0
        or kv_lora_rank not in (256, 512, 1024)
        or nope_dim not in (32, 64, 128)
    ):
        return False
    if w_k.stride(2) != 1:
        return False
    if rope_dim not in (32, 64, 128, 256):
        return False
    half_m = 64 // (kv_lora_rank // 256)
    lds_bytes = (
        2 * 256 * nope_dim
        + 4 * 64 * nope_dim
        + 2 * 64 * 256
        + 8 * half_m * rope_dim
        + 2 * rope_dim
    )
    # Leave 32 KiB for padded shared layouts. In particular L=256, P=128,
    # R=256 exceeds the 256 KiB limit even before padding.
    if lds_bytes > 224 * 1024:
        return False
    cos = cos_cache
    while cos.dim() > 2 and cos.shape[1] == 1:
        cos = cos.squeeze(1)
    if cos.dim() != 2 or cos.shape[0] == 0:
        return False
    if (
        cos.shape[-1] * 2 != rope_dim
        or cos.stride(-1) != 1
        or cos.dtype != torch.bfloat16
    ):
        return False
    sin = sin_cache
    while sin.dim() > 2 and sin.shape[1] == 1:
        sin = sin.squeeze(1)
    if sin.shape != cos.shape or sin.stride() != cos.stride() or sin.dtype != cos.dtype:
        return False
    if kv_cache.dim() != 4 or kv_cache.shape[1] != 1:
        return False
    if (
        kv_cache.shape[2] <= 0
        or kv_cache.shape[2] % 16
        or kv_cache.shape[3] != kv_lora_rank + rope_dim
    ):
        return False
    if kv_cache.stride(3) != 1:
        return False
    if q_out.shape != (num_tokens, num_heads, kv_lora_rank + rope_dim):
        return False
    if q_out.stride(2) != 1 or num_tokens * q_out.stride(0) >= 2**31:
        return False
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        return False
    if any(
        t.device != q_out.device
        for t in (w_k, cos_cache, sin_cache, positions, kv_cache, slot_mapping)
    ):
        return False
    if k_scale is not None and not (
        isinstance(k_scale, torch.Tensor) and k_scale.numel() == 1
    ):
        return False
    if isinstance(k_scale, torch.Tensor) and k_scale.device != q_out.device:
        return False
    return num_tokens > 0
