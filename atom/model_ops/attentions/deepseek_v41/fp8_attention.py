# SPDX-License-Identifier: MIT
"""Sparse attention over the FP8 pool's MXFP8 rows.

A row is `head_dim` E4M3 values then `head_dim // 32` E8M0 scales, RoPE dims
included (`V41PoolGeometry.row_bytes`); the query is quantized the same way
(`rope_quant_window`'s `query_fp8`). Only AITER's `mla_sparse_prefill_v41`
reads this record: the V3 (576 dims, a scale a tensor) and V4 (FP8 NoPE + BF16
RoPE) FP8 kernels take the same dtypes and compute garbage on it, so nothing
here dispatches by dtype.
"""

import torch

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE
from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src_reference
from atom.utils.decorators import mark_trace

try:
    from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_v41
except ImportError:
    mla_sparse_prefill_v41 = None


def _split(rows, dim):
    """`[n, dim + dim // 32]` byte rows as their (values, scales) views, at the
    rows' own stride."""
    return rows[:, :dim].view(FP8_DTYPE), rows[:, dim:].view(torch.float8_e8m0fnu)


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
):
    """`sparse_attn_v4_paged_2src` on MXFP8 rows: each token attends to its
    prefix rows of `pool` and its extend rows of `kv`.

    Args:
      query:   `(values [T, H, D] fp8, scales [T, H, D // 32] E8M0)`.
      pool:    `[rows, D + D // 32]` uint8, the prefix source (the cache's pool).
      kv:      `[rows, D + D // 32]` uint8 at `pool`'s row stride, the extend
        source: this forward's rows (prefill) or the layer's ring (decode).
      kv_indices_* / kv_indptr_*: int32 CSR row lists into each source.
      out:     optional `[T, H, D]` bf16 output.
    """
    if mla_sparse_prefill_v41 is None:
        raise RuntimeError(
            "The FP8 V4.1 KV cache needs AITER's mla_sparse_prefill_v41 (aiter#6675)"
        )
    values, scales = query
    dim = values.shape[-1]
    return mla_sparse_prefill_v41(
        values,
        scales,
        *_split(pool, dim),
        kv_indices_prefix,
        kv_indptr_prefix,
        *_split(kv, dim),
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out=out,
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
