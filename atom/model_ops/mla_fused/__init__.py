# SPDX-License-Identifier: MIT
"""Fused MLA decode kernels for gfx1250 (Gluon) and the checks that gate them.

* ATOM_MLA_FUSED_KUP_QPE: K-up GEMM + q_pe RoPE + KV-cache write in one kernel
  (fused_kup_qpe.kup_qpe_fused).

Bit-exact with the path it replaces. The gate below only uses attributes and
shapes, so it imports without the GPU stack; the kernel is imported lazily by
the caller.
"""

import functools

import torch

_GFX1250 = ("gfx1250", "gfx1250-strict")


@functools.cache
def gfx_arch() -> str:
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime()
    except Exception:  # noqa: BLE001 - no GPU stack means no gfx1250
        return ""


def _is_fp8(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def kup_qpe_fusion_ok(
    *,
    enabled: bool,
    arch: str,
    is_prefill: bool,
    use_qrep: bool,
    dcp_world_size: int,
    q_out: torch.Tensor | None,
    num_tokens: int,
    slot_mapping: torch.Tensor,
    w_k: torch.Tensor,
    cos_cache: torch.Tensor,
    rope_dim: int,
    kv_cache: torch.Tensor | None,
    k_scale,
) -> bool:
    """q_out is the prestored q_out of the BF16-absorb + shuffled-KV path (None
    when that path is off); kv_cache is the shuffled (blocks, 1, bs, d) view."""
    if not enabled or arch not in _GFX1250:
        return False
    # Pure decode only: no prefill (so no PCP), no DCP / query replication.
    if is_prefill or use_qrep or dcp_world_size > 1:
        return False
    if q_out is None or kv_cache is None:
        return False
    if slot_mapping.dim() != 1 or slot_mapping.shape[0] != num_tokens:
        return False
    if not (_is_fp8(q_out) and kv_cache.dtype == q_out.dtype):
        return False
    if w_k.dtype != torch.bfloat16 or w_k.dim() != 3:
        return False
    num_heads, kv_lora_rank, nope_dim = w_k.shape
    if kv_lora_rank % 256 or nope_dim % 32 or nope_dim > 128:
        return False
    if rope_dim not in (32, 64, 128, 256):
        return False
    cos = cos_cache
    while cos.dim() > 2 and cos.shape[1] == 1:
        cos = cos.squeeze(1)
    if cos.shape[-1] * 2 != rope_dim or cos.stride(-1) != 1:
        return False
    if kv_cache.dim() != 4 or kv_cache.shape[1] != 1:
        return False
    if kv_cache.shape[2] % 16 or kv_cache.shape[3] != kv_lora_rank + rope_dim:
        return False
    if kv_cache.stride(3) != 1:
        return False
    if q_out.shape != (num_tokens, num_heads, kv_lora_rank + rope_dim):
        return False
    if q_out.stride(2) != 1 or num_tokens * q_out.stride(0) >= 2**31:
        return False
    if k_scale is not None and not (
        isinstance(k_scale, torch.Tensor) and k_scale.numel() == 1
    ):
        return False
    return num_tokens > 0
