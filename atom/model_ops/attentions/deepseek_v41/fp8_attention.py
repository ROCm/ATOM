# SPDX-License-Identifier: MIT
"""Sparse attention over contiguous MXFP8 or NVFP4 CSA and MXFP8 SWA planes.

CSA and SWA have independent row spaces. A prefill's SWA source contains
the current batch's rings followed by the current chunk; decode uses the ring directly.
Legacy packed rows remain accepted by the standalone wrappers.
"""

import torch
from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_v41

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE
from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src_reference
from atom.utils.decorators import mark_trace


def _split(rows, dim, scale_dtype=torch.float8_e8m0fnu):
    """View the data and scale bytes for MXFP8 (D + D/32) or NVFP4
    (D/2 + D/16) rows without copying either source."""
    fp4 = scale_dtype == torch.float8_e4m3fn
    data_dtype = torch.uint8 if fp4 else FP8_DTYPE
    width = dim // 2 if fp4 else dim
    if isinstance(rows, tuple):
        values, scales = rows
        if not values.is_contiguous() or not scales.is_contiguous():
            raise ValueError("KV data and scale planes must each be contiguous")
        return values.view(data_dtype), scales.view(scale_dtype)
    return rows[:, :width].view(data_dtype), rows[:, width:].view(scale_dtype)


@mark_trace
def sparse_attn_v41_paged_2src(
    query,
    pool,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out=None,
    *,
    pool_scale_dtype=torch.float8_e8m0fnu,
    num_kv_splits=None,
):
    """Sparse attention with MXFP8 queries: each token attends to its
    prefix rows of `pool` and its extend rows of `kv`.

    Args:
      query:   `(values [T, H, D] fp8, scales [T, H, D // 32] E8M0)`.
      pool:    CSA planes: MXFP8 `(data [rows, D], scale [rows, D // 32])`,
        or NVFP4 `(data [rows, D // 2], scale [rows, D // 16])`.
      kv:      the SWA pair: ring plus current chunk (prefill), or ring (decode).
        Both planes are contiguous; each source has its own CSR row space.
      kv_indices_* / kv_indptr_*: int32 CSR row lists into each source.
      out:     optional `[T, H, D]` bf16 output.
      pool_scale_dtype: E8M0 for MXFP8 CSA, or E4M3 for NVFP4 CSA.
      num_kv_splits: optional AITER split override; otherwise AITER chooses it.
    """
    values, scales = query
    dim = values.shape[-1]
    return mla_sparse_prefill_v41(
        values,
        scales,
        *_split(pool, dim, pool_scale_dtype),
        kv_indices_prefix,
        kv_indptr_prefix,
        *_split(kv, dim),
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out=out,
        num_kv_splits=num_kv_splits,
    )


def dequantize_rows(values, scales):
    """MXFP8 `(values [..., D], scales [..., D // 32])` as fp32."""
    groups = values.float().unflatten(-1, (-1, 32))
    steps = torch.exp2(scales.view(torch.uint8).float() - 127)
    return (groups * steps[..., None]).flatten(-2)


def sparse_attn_v41_paged_2src_reference(
    query,
    pool,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
):
    """The same attention on the dequantized rows, through V4's reference."""
    values, scales = query
    dim = values.shape[-1]
    return sparse_attn_v4_paged_2src_reference(
        dequantize_rows(values, scales),
        dequantize_rows(*_split(pool, dim)),
        kv_indices_prefix,
        kv_indptr_prefix,
        dequantize_rows(*_split(kv, dim)),
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
    )
