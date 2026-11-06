# SPDX-License-Identifier: MIT
"""Sparse attention over separate, contiguous MXFP8 KV data/scale planes.

CSA and SWA have independent row spaces. A prefill's SWA source contains
the current batch's rings followed by the current chunk; decode uses the ring directly.
Legacy packed rows remain accepted by the standalone wrappers.
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
    if isinstance(rows, tuple):
        values, scales = rows
        if not values.is_contiguous() or not scales.is_contiguous():
            raise ValueError("KV data and scale planes must each be contiguous")
        return values.view(FP8_DTYPE), scales.view(torch.float8_e8m0fnu)
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
      pool:    `(data [rows, D], scale [rows, D // 32])`, the CSA source.
      kv:      the SWA pair: ring plus current chunk (prefill), or ring (decode).
        Both planes are contiguous; each source has its own CSR row space.
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
