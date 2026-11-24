# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the DSpark draft's fused q / k norm, RoPE and
fp8 K / V insert."""

import functools
import struct

import torch

from amd_kernels.warm_up import entry

from ..device import on_gfx950
from . import kernel as _kernel


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(nq: int, nkv: int, block_size: int, insert: bool):
    return _kernel.compile_qkv_norm_rope_insert(nq, nkv, block_size, insert)


def supports_qkv_norm_rope_insert(
    qkv: torch.Tensor, cos_sin_cache: torch.Tensor, kv_cache: torch.Tensor | None
) -> bool:
    if not on_gfx950() or qkv.dtype != torch.bfloat16 or qkv.stride(1) != 1:
        return False
    if cos_sin_cache.dtype != torch.bfloat16 or cos_sin_cache.shape[1] != _kernel.ROT:
        return False
    if kv_cache is None:
        return True
    return (
        kv_cache.dim() == 4
        and kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn)
        and kv_cache.shape[3] == _kernel.TOK_B
        and kv_cache.stride(3) == 1
    )


def _qkv_norm_rope_insert(
    qkv: torch.Tensor,  # [n, (nq + 2 nkv) * 128] bf16, unit column stride
    positions: torch.Tensor,  # [n] int64
    q_norm_weight: torch.Tensor,  # [128] bf16
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] bf16 (cos | sin)
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    q_out: torch.Tensor,  # [n, nq * 128] bf16
    kv_cache: torch.Tensor | None = None,  # [blocks, nkv, block_size, 256] e4m3
    slot_mapping: torch.Tensor | None = None,  # int64, < 0 skips the token
    k_scale: torch.Tensor | None = None,  # fp32 scalars
    v_scale: torch.Tensor | None = None,
) -> None:
    """q_out = RoPE(norm(q)); with ``kv_cache``, RoPE(norm(k)) and v of the first
    len(slot_mapping) tokens go into the pages at ``slot_mapping`` (e4m3 of value /
    scale)."""
    n = qkv.shape[0]
    if n == 0:
        return
    nq, nkv = num_heads, num_kv_heads
    D = _kernel.D
    assert qkv.shape[1] == (nq + 2 * nkv) * D and qkv.stride(1) == 1
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert cos_sin_cache.is_contiguous() and cos_sin_cache.shape[1] == _kernel.ROT
    for w in (q_norm_weight, k_norm_weight):
        assert w.dtype == torch.bfloat16 and w.is_contiguous() and w.numel() == D
    assert q_out.dtype == torch.bfloat16 and q_out.is_contiguous()
    insert = kv_cache is not None and slot_mapping is not None
    if insert:
        assert kv_cache.shape[1] == nkv and kv_cache.shape[3] == _kernel.TOK_B
        assert kv_cache.stride(3) == 1 and slot_mapping.dtype == torch.int64
        assert slot_mapping.is_contiguous()
        n_slots = min(n, slot_mapping.shape[0])
        block_size = kv_cache.shape[2]
        page_b, head_b, tok_b = (
            kv_cache.stride(0) * kv_cache.element_size(),
            kv_cache.stride(1) * kv_cache.element_size(),
            kv_cache.stride(2) * kv_cache.element_size(),
        )
        kv_ptr, sl_ptr = kv_cache.data_ptr(), slot_mapping.data_ptr()
        ks_ptr, vs_ptr = k_scale.data_ptr(), v_scale.data_ptr()
    else:
        block_size, page_b, head_b, tok_b, n_slots = 1, 0, 0, 0, 0
        kv_ptr = sl_ptr = ks_ptr = vs_ptr = 0
    _run_compiled(
        _get_kernel(nq, nkv, block_size, insert),
        qkv.data_ptr(),
        positions.data_ptr(),
        sl_ptr,
        cos_sin_cache.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        q_out.data_ptr(),
        kv_ptr,
        ks_ptr,
        vs_ptr,
        n,
        cos_sin_cache.shape[0],
        page_b,
        head_b,
        tok_b,
        qkv.stride(0),
        struct.unpack("<q", struct.pack("<d", eps))[0],
        n_slots,
        torch.cuda.current_stream(),
    )


def _qkv_mode(qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads, num_kv_heads, q_out,
              kv_cache=None, slot_mapping=None, k_scale=None, v_scale=None):
    """What a call fixes about its kernel (``amd_kernels.warm_up``; the token count does not)."""
    insert = kv_cache is not None and slot_mapping is not None
    return num_heads, num_kv_heads, insert, kv_cache.shape[2] if insert else 1


def _qkv_inputs(mode, n, qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads, num_kv_heads,
                q_out, kv_cache=None, slot_mapping=None, k_scale=None, v_scale=None):
    """One token (zeros; slot -1 writes nothing; the caller's cache, nothing launches)."""
    dev = qkv.device
    x = torch.zeros((n, qkv.shape[1]), dtype=qkv.dtype, device=dev)
    pos = torch.zeros(n, dtype=torch.int64, device=dev)
    q = torch.empty((n, num_heads * _kernel.D), dtype=torch.bfloat16, device=dev)
    slots = torch.full((n,), -1, dtype=torch.int64, device=dev) if mode[2] else None
    return (x, pos, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads, num_kv_heads, q,
            kv_cache if mode[2] else None, slots, k_scale, v_scale), {}


qkv_norm_rope_insert = entry("attention/norm_rope_fp8_insert_d128_rot64:qkv", mode=_qkv_mode,
                             sizes=lambda mode: (1,), inputs=_qkv_inputs)(_qkv_norm_rope_insert)
