# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The attention input of an NVFP4 sparse layer from the QKV projection in one call: q (bf16) and index-q out, K / V
and index-K into their NVFP4 pages.

- ``qkv_norm_rope_nvfp4_insert``: from the bf16 QKV rows (any projection).
- ``mxfp8_gemm_qkv_norm_rope_nvfp4_insert``: from the MXFP8 codes of the QKV GEMM's input (``gemm.mxfp8_common``'s
  pre-quantized layout): the one-launch entry ``mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128`` where it
  applies (N 2560, K 6144, 16 / 1 / 1 heads, rotary 64, bf16, 1-1536 rows in the shuffled-code tile) and
  ``one_launch`` allows it; else the GEMM (where the codes are in the shuffled tile, its deferred split-K partials go
  to ``splitk_reduce_norm_rope_nvfp4_insert``) and the insert. Bitwise either way.

``one_launch=False``: the one-launch entry keeps one cross-workgroup counter set per stream and graph capture (512 per
device); a caller that captures every layer's segment on its own (breakable PIECEWISE graphs) passes False.
"""

import torch

from .host import norm_rope_nvfp4_insert, splitk_reduce_norm_rope_nvfp4_insert

D = 128


def _outputs(rows, num_heads, num_index_heads, index_q_dtype, like):
    q = like.new_empty((rows, num_heads * D), dtype=torch.bfloat16)
    index_q = like.new_empty((rows, num_index_heads * D), dtype=index_q_dtype)
    return q, index_q


def qkv_norm_rope_nvfp4_insert(
    qkv: torch.Tensor,  # [n, (nq + 2 nkv + niq + 1) * 128] bf16
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim] bf16
    positions: torch.Tensor,  # [n] int64
    slot_mapping: torch.Tensor,  # [n] int64, -1 skips the token
    index_slot_mapping: torch.Tensor,  # [n] int64, -1 skips the token
    kv_cache: torch.Tensor,  # [blocks, 2 nkv, 128, 72] uint8
    index_cache: torch.Tensor,  # [blocks, 128, 72] uint8
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
    index_q_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``norm_rope_nvfp4_insert`` with its outputs allocated: returns (q [n, nq * 128]
    bf16, index-q [n, niq * 128] ``index_q_dtype``)."""
    q, index_q = _outputs(qkv.shape[0], num_heads, num_index_heads, index_q_dtype, qkv)
    norm_rope_nvfp4_insert(
        qkv, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions,
        slot_mapping, index_slot_mapping, kv_cache, index_cache, k_scale, v_scale, q, index_q,
        num_heads=num_heads, num_kv_heads=num_kv_heads, num_index_heads=num_index_heads, eps=eps,
    )
    return q, index_q


def _one_launch_entry():
    try:
        from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128 import host
    except ImportError:
        return None
    return host


def mxfp8_gemm_qkv_norm_rope_nvfp4_insert(
    activation: torch.Tensor,  # e4m3 codes of the GEMM input (prequant_code_tile_rows layout, or row-major)
    activation_scale: torch.Tensor,  # their shuffled E8M0 scales
    weight: torch.Tensor,  # preshuffle_mxfp8_weight codes [N, K]
    weight_scale: torch.Tensor,  # its shuffled scales
    m: int,  # tokens
    out_dtype: torch.dtype,  # the projection's output dtype
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    index_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
    index_q_dtype: torch.dtype = torch.float8_e4m3fn,
    one_launch: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The QKV GEMM from its MXFP8 input + norms, RoPE and the NVFP4 inserts (module
    doc); returns (q, index-q). The codes' layout is the GEMM table row's for ``m``
    rows (``gemm.prequant_code_tile_rows``: its tile, or row-major where it is 0)."""
    from amd_kernels.gemm.mxfp8_common import gemm as G

    n, k = weight.shape
    ins = (q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions,
           slot_mapping, index_slot_mapping, kv_cache, index_cache, k_scale, v_scale)
    kw = dict(num_heads=num_heads, num_kv_heads=num_kv_heads, num_index_heads=num_index_heads, eps=eps)
    f = _one_launch_entry() if one_launch else None
    if (
        f is not None
        and out_dtype == torch.bfloat16
        and (n, k) == (f.N, f.K)
        and (num_heads, num_kv_heads, num_index_heads) == (f._kernel.NQ, f._kernel.NKV, f._kernel.NIQ)
        and cos_sin_cache.shape[1] == f._kernel.ROT
        and index_q_dtype == torch.float8_e4m3fn
        and 1 <= m <= f.MAX_ROWS
        and G.prequant_code_tile_rows(m, f.N, f.K, torch.bfloat16)
    ):
        q, index_q = _outputs(m, num_heads, num_index_heads, index_q_dtype, activation)
        f.mxfp8_qkv_norm_rope_nvfp4_insert(activation, activation_scale, weight, weight_scale, m, *ins, q, index_q,
                                           **kw)
        return q, index_q
    if m == 0:
        return qkv_norm_rope_nvfp4_insert(torch.empty((0, n), dtype=out_dtype, device=weight.device), *ins, **kw,
                                          index_q_dtype=index_q_dtype)
    if G.prequant_code_tile_rows(m, n, k, out_dtype):  # codes in the GEMM row's tile
        out = G.mxfp8_gemm_shuffled(activation, activation_scale, weight, weight_scale, out_dtype, m,
                                    defer_reduce=True)
    else:
        out = G.mxfp8_gemm(activation, activation_scale, weight, weight_scale, out_dtype,
                           activation_scale_preshuffled=True)
    if isinstance(out, G.SkPartials):
        q, index_q = _outputs(out.tmp.shape[1], num_heads, num_index_heads, index_q_dtype, out.tmp)
        splitk_reduce_norm_rope_nvfp4_insert(out.tmp, out.split_k, *ins, q, index_q, **kw)
        return q, index_q
    return qkv_norm_rope_nvfp4_insert(out.reshape(m, n), *ins, **kw, index_q_dtype=index_q_dtype)
