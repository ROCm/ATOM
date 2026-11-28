# SPDX-License-Identifier: MIT
"""Sparse attention over contiguous MXFP8 or NVFP4 CSA and MXFP8 SWA planes.

CSA and SWA have independent row spaces. A prefill's SWA source contains
the current batch's rings followed by the current chunk; decode uses the ring directly.
AITER's wrapper also accepts legacy packed rows; the BF16-query attention
takes only contiguous `(data, scale)` planes.

A device AITER's MXFP8 kernel does not serve (gfx942 today) attends with a
BF16 query instead, on a Triton kernel that decodes the planes in registers
(`fp8_sparse_attention`). Full-width prefill launches, where that kernel is
slower, decode each row the step names once into bounded BF16 scratch for
V4's BF16 attention, and fall back to the kernel past the bound.
"""

import functools
import logging

import torch
import triton
import triton.language as tl
from aiter.ops.mla_sparse_prefill import (
    get_mla_sparse_prefill_v41_backend,
    mla_sparse_prefill_v41,
)

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE
from atom.model_ops.v4_kernels import (
    sparse_attn_v4_paged_2src,
    sparse_attn_v4_paged_2src_reference,
)
from atom.model_ops.v4_kernels.paged_2src import _BLOCK_H, _csa_num_splits
from atom.utils.decorators import mark_trace

from .fp8_sparse_attention import _planes, sparse_attn_v41_paged_2src_fp8kv
from .packed_rows import decode_rows

logger = logging.getLogger("atom")

# Decoded rows held at once by the BF16 adapter (256 MiB at D=512).
_SCRATCH_ROWS = 1 << 18


@functools.cache
def native_fp8_attention_on(device, heads, dim):
    """Whether AITER's MXFP8 sparse attention serves `heads` x `dim` queries
    on `device`, for the tensors ATOM hands it; asked once per shape."""
    return (
        get_mla_sparse_prefill_v41_backend(
            device,
            (1, heads, dim),
            unified_kv_dtype=FP8_DTYPE,
            softmax_scale=dim**-0.5,
            q_dtype=FP8_DTYPE,
            q_scale_dtype=torch.float8_e8m0fnu,
            unified_kv_scale_dtype=torch.float8_e8m0fnu,
            kv_dtype=FP8_DTYPE,
            kv_scale_dtype=torch.float8_e8m0fnu,
            layout="contiguous",
        )
        is not None
    )


def native_fp8_attention(query):
    """Whether an FP8 KV cache attends `query` [..., H, D] with AITER's
    MXFP8 kernel; the choice is logged once per device and shape."""
    *_, heads, dim = query.shape
    return _announced(query.device, heads, dim)


@functools.cache
def _announced(device, heads, dim):
    native = native_fp8_attention_on(device, heads, dim)
    logger.info(
        "V4.1 FP8 KV attention on %s: %s",
        device,
        (
            "AITER mla_sparse_prefill_v41, MXFP8 query"
            if native
            else "BF16 query over the MXFP8 planes (AITER has no MXFP8 sparse "
            "attention here)"
        ),
    )
    return native


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


@triton.jit(do_not_specialize=["N", "ROWS"])
def _mark_named(indices, used, marks, N, ROWS, BLOCK: tl.constexpr):
    # Only the CSR's `used` entries are written; the plane past them is
    # uninitialized and may hold any value.
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = (j < N) & (j < tl.load(used))
    row = tl.load(indices + j, inside, other=-1)
    named = inside & (row >= 0) & (row < ROWS)
    tl.store(marks + row, tl.full((BLOCK,), 1, tl.int32), named)


@triton.jit(do_not_specialize=["N", "ROWS"])
def _renumber(indices, used, rank, local, N, ROWS, BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = j < N
    live = inside & (j < tl.load(used))
    row = tl.load(indices + j, live, other=-1)
    named = live & (row >= 0) & (row < ROWS)
    r = tl.load(rank + row, named, other=0)
    tl.store(local + j, tl.where(named, r - 1, -1), inside)


@triton.jit(do_not_specialize=["N"])
def _order_marked(rank, order, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = i < N
    rank_i = tl.load(rank + i, inside, other=0)
    rank_prev = tl.load(rank + i - 1, inside & (i > 0), other=0)
    tl.store(order + rank_i - 1, i, inside & (rank_i > rank_prev))


def _decoded(source, dim):
    data = source[0]
    rows = data.shape[0]
    out = torch.empty((rows, dim), device=data.device, dtype=torch.bfloat16)
    order = torch.arange(rows, device=data.device, dtype=torch.int32)
    decode_rows(source, order, out)
    return out


def _unique_rows(source, indices, indptr, dim):
    """Each source row that the CSR (`indices` up to `indptr[-1]`) names,
    decoded once, and `indices` renumbered into it. Entries past the CSR, and
    any -1 or ID past the source, become -1; V4's CSA kernels do not skip -1,
    so this relies on the CSR naming only live rows, as the serving CSR does.
    A layer without a CSA pool attends over a 0-row source. None when the
    rows exceed the scratch bound; the count is the one host read.

    Besides the scratch, `marks` and `rank` are sized by the whole source, not
    by the step: 8 B per pool row (about 1.5% of an MXFP8 pool at D=512), a
    cumsum and an `_order_marked` scan over them for every CSA layer of a
    prefill, freed after each layer."""
    data = source[0]
    rows, entries = data.shape[0], indices.numel()
    if not rows:
        out = torch.empty((1, dim), device=data.device, dtype=torch.bfloat16)
        return out, torch.full_like(indices, -1, dtype=torch.int32)
    used = indptr[-1:]
    marks = torch.zeros(rows, device=data.device, dtype=torch.int32)
    if entries:
        _mark_named[(triton.cdiv(entries, 1024),)](
            indices, used, marks, entries, rows, BLOCK=1024
        )
    rank = torch.cumsum(marks, 0, dtype=torch.int32)
    count = int(rank[-1])
    if count > _SCRATCH_ROWS:
        return None
    out = torch.empty((max(count, 1), dim), device=data.device, dtype=torch.bfloat16)
    if count:
        order = torch.empty(count, device=data.device, dtype=torch.int32)
        _order_marked[(triton.cdiv(rows, 1024),)](rank, order, rows, BLOCK=1024)
        decode_rows(source, order, out[:count])
    local = torch.empty(entries, device=data.device, dtype=torch.int32)
    if entries:
        _renumber[(triton.cdiv(entries, 1024),)](
            indices, used, rank, local, entries, rows, BLOCK=1024
        )
    return out, local


def sparse_attn_v41_paged_2src_gather(
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
    """`sparse_attn_v41_paged_2src` for a BF16 `query` [T, H, D] over the
    same MXFP8 `(data, scale)` CSA and SWA planes, on V4's BF16 attention.

    A prefill decodes its SWA source whole (it is no larger than the
    selection) and each CSA row it names once, and attends in one launch.
    A decode's ring, and CSA rows past the scratch bound, take the in-register
    kernel instead.
    """
    tokens, _, dim = query.shape
    if out is None:
        out = torch.empty_like(query)
    if tokens == 0:
        return out
    pool_planes, kv_planes = _planes(pool), _planes(kv)
    unique = None
    if kv_planes[0].shape[0] <= kv_indices_extend.numel():
        unique = _unique_rows(pool_planes, kv_indices_prefix, kv_indptr_prefix, dim)
    if unique is None:
        return sparse_attn_v41_paged_2src_fp8kv(
            query,
            pool,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            softmax_scale,
            out=out,
        )
    prefix, prefix_indices = unique
    result = sparse_attn_v4_paged_2src(
        query,
        prefix,
        prefix_indices,
        kv_indptr_prefix[: tokens + 1],
        _decoded(kv_planes, dim),
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out=out,
    )
    if result.data_ptr() != out.data_ptr():
        out.copy_(result)
    return out


def _in_kernel(query, kv_indices_prefix, kv_indices_extend):
    # The decode path reads a row count on the host, which capture forbids.
    if torch.cuda.is_current_stream_capturing():
        return True
    tokens, heads, _ = query.shape
    if tokens == 0:
        return True
    keys = (kv_indices_prefix.numel() + kv_indices_extend.numel()) / tokens
    return _csa_num_splits(tokens, triton.cdiv(heads, _BLOCK_H), keys) > 1


@mark_trace
def sparse_attn_v41_paged_2src_bf16(
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
    """`sparse_attn_v41_paged_2src` for a BF16 `query` [T, H, D] over the
    same MXFP8 `(data, scale)` CSA and SWA planes.

    Launches V4 would split (decode, short chunks) and graph capture take the
    in-register kernel. A full-width prefill decodes its rows once instead,
    about 5x faster than the in-register kernel at 8192 tokens on MI308X."""
    attend = (
        sparse_attn_v41_paged_2src_fp8kv
        if _in_kernel(query, kv_indices_prefix, kv_indices_extend)
        else sparse_attn_v41_paged_2src_gather
    )
    return attend(
        query,
        pool,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
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
