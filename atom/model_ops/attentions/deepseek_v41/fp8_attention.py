# SPDX-License-Identifier: MIT
"""Sparse attention over contiguous MXFP8 or NVFP4 CSA and MXFP8 SWA planes.

CSA and SWA have independent row spaces. A prefill's SWA source contains
the current batch's rings followed by the current chunk; decode uses the ring directly.
Legacy packed rows remain accepted by the standalone wrappers.

gfx942, which AITER's MXFP8 kernel does not serve, attends with a BF16 query instead, on a
Triton kernel that decodes the planes in registers (`fp8_sparse_attention`)
for split-KV launches (decode, short chunks) and during graph capture. Full-width
prefill launches, where that kernel is slower, decode the rows each source
contributes into bounded BF16 scratch for V4's BF16 attention, as the packed
FP4 adapter does.
"""

import functools

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_v41

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE
from atom.model_ops.v4_kernels import (
    sparse_attn_v4_paged_2src,
    sparse_attn_v4_paged_2src_reference,
)
from atom.model_ops.v4_kernels.paged_2src import _csa_num_splits
from atom.utils.decorators import mark_trace

from .fp8_sparse_attention import _BLOCK_H, sparse_attn_v41_paged_2src_fp8kv
from .packed_rows import _read_plane

# Decoded rows held at once by the BF16 adapter (256 MiB at D=512).
_SCRATCH_ROWS = 1 << 18


@functools.cache
def native_fp8_attention():
    """Whether AITER's MXFP8 sparse attention serves this device; gfx942 has
    no build of it."""
    return get_gfx() != "gfx942"


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


@triton.jit
def _decode_rows(data, scale, out, N, D: tl.constexpr):
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    valid = i < N
    dims = tl.arange(0, D)
    value = _read_plane(data, scale, i, dims, valid, D, False)
    tl.store(out + i.to(tl.int64)[:, None] * D + dims[None, :], value, valid[:, None])


@triton.jit(do_not_specialize=["first", "end", "CAPACITY"])
def _gather_rows(
    data, scale, rows, ptr, first, end, out, local, CAPACITY, D: tl.constexpr
):
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    start = tl.load(ptr + first)
    count = tl.load(ptr + end) - start
    inside = i < CAPACITY
    valid = (i < count) & inside
    row = tl.load(rows + start + i, valid, other=-1)
    # A -1 sentinel stays one, so V4 skips it rather than scoring a zero key.
    live = valid & (row >= 0)
    dims = tl.arange(0, D)
    value = _read_plane(data, scale, tl.where(live, row, 0), dims, live, D, False)
    tl.store(out + i.to(tl.int64)[:, None] * D + dims[None, :], value, inside[:, None])
    tl.store(local + i, tl.where(live, i, -1), inside)


@triton.jit
def _mark_named(indices, used, marks, N, ROWS, BLOCK: tl.constexpr):
    # Only the CSR's `used` entries are written; the plane past them is
    # uninitialized and may hold any value.
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = (j < N) & (j < tl.load(used))
    row = tl.load(indices + j, inside, other=-1)
    named = inside & (row >= 0) & (row < ROWS)
    tl.store(marks + row, tl.full((BLOCK,), 1, tl.int32), named)


@triton.jit
def _renumber(indices, used, rank, local, N, ROWS, BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = j < N
    live = inside & (j < tl.load(used))
    row = tl.load(indices + j, live, other=-1)
    named = live & (row >= 0) & (row < ROWS)
    r = tl.load(rank + row, named, other=0)
    tl.store(local + j, tl.where(named, r - 1, -1), inside)


@triton.jit
def _order_marked(rank, order, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = i < N
    rank_i = tl.load(rank + i, inside, other=0)
    rank_prev = tl.load(rank + i - 1, inside & (i > 0), other=0)
    tl.store(order + rank_i - 1, i, inside & (rank_i > rank_prev))


@triton.jit
def _decode_at(data, scale, order, out, N, D: tl.constexpr):
    k = tl.program_id(0) * 16 + tl.arange(0, 16)
    inside = k < N
    row = tl.load(order + k, inside, other=0)
    dims = tl.arange(0, D)
    value = _read_plane(data, scale, row, dims, inside, D, False)
    tl.store(out + k.to(tl.int64)[:, None] * D + dims[None, :], value, inside[:, None])


def _planes(source):
    return tuple(plane.view(torch.uint8) for plane in source)


def _decoded(source, dim):
    data, scale = source
    rows = data.shape[0]
    out = torch.empty((rows, dim), device=data.device, dtype=torch.bfloat16)
    if rows:
        _decode_rows[(triton.cdiv(rows, 16),)](data, scale, out, rows, dim)
    return out


def _gathered(source, indices, indptr, first, end, capacity, dim):
    data, scale = source
    # One row at least: an empty selection still needs a source tensor.
    capacity = max(capacity, 1)
    out = torch.empty((capacity, dim), device=data.device, dtype=torch.bfloat16)
    local = torch.empty(capacity, device=data.device, dtype=torch.int32)
    _gather_rows[(triton.cdiv(capacity, 16),)](
        data, scale, indices, indptr, first, end, out, local, capacity, dim
    )
    return out, local


def _unique_rows(source, indices, indptr, dim):
    """Each source row that the CSR (`indices` up to `indptr[-1]`) names,
    decoded once, and `indices` renumbered into it. -1, any ID past the
    source, and every entry past the CSR stay -1: a layer without a CSA pool
    attends over a 0-row source. None when the rows exceed the scratch bound;
    the count is the one host read."""
    data, scale = source
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
        _decode_at[(triton.cdiv(count, 16),)](data, scale, order, out, count, dim)
    local = torch.empty(entries, device=data.device, dtype=torch.int32)
    if entries:
        _renumber[(triton.cdiv(entries, 1024),)](
            indices, used, rank, local, entries, rows, BLOCK=1024
        )
    return out, local


def _tiles(tokens, indptrs, totals):
    """`(begin, end, rows per gathered source)` query ranges within
    `_SCRATCH_ROWS`. The CSR offsets are read on the host only when the
    whole step would not fit."""
    if sum(totals) <= _SCRATCH_ROWS:
        return [(0, tokens, totals)]
    ptrs = [indptr.tolist() for indptr in indptrs]
    tiles, begin = [], 0
    while begin < tokens:
        end = begin + 1
        while end < tokens and (
            sum(p[end + 1] - p[begin] for p in ptrs) <= _SCRATCH_ROWS
        ):
            end += 1
        tiles.append((begin, end, [p[end] - p[begin] for p in ptrs]))
        begin = end
    return tiles


def _rebased(indptr, begin, end):
    return indptr[begin : end + 1] - indptr[begin]


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

    Selected CSA rows are decoded per query range. The SWA source is decoded
    whole when it is no larger than its selection (a prefill's batch-local
    window), and gathered like CSA otherwise (a decode's ring). A prefill
    decodes each CSA row it references once and attends in one launch, or
    one row per CSR entry when those rows exceed the scratch bound.
    """
    tokens, _, dim = query.shape
    if out is None:
        out = torch.empty_like(query)
    if tokens == 0:
        return out
    pool, kv = _planes(pool), _planes(kv)
    whole = kv[0].shape[0] <= kv_indices_extend.numel()
    window = _decoded(kv, dim) if whole else None
    unique = (
        _unique_rows(pool, kv_indices_prefix, kv_indptr_prefix, dim) if whole else None
    )
    if unique is not None:
        prefix, prefix_indices = unique
        result = sparse_attn_v4_paged_2src(
            query,
            prefix,
            prefix_indices,
            _rebased(kv_indptr_prefix, 0, tokens),
            window,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            softmax_scale,
            out=out,
        )
        if result.data_ptr() != out.data_ptr():
            out.copy_(result)
        return out
    indptrs, totals = [kv_indptr_prefix], [kv_indices_prefix.numel()]
    if not whole:
        indptrs.append(kv_indptr_extend)
        totals.append(kv_indices_extend.numel())
    for begin, end, rows in _tiles(tokens, indptrs, totals):
        prefix, prefix_indices = _gathered(
            pool, kv_indices_prefix, kv_indptr_prefix, begin, end, rows[0], dim
        )
        if whole:
            extend, extend_indices = window, kv_indices_extend
            extend_indptr = kv_indptr_extend[begin : end + 1]
        else:
            extend, extend_indices = _gathered(
                kv, kv_indices_extend, kv_indptr_extend, begin, end, rows[1], dim
            )
            extend_indptr = _rebased(kv_indptr_extend, begin, end)
        result = sparse_attn_v4_paged_2src(
            query[begin:end],
            prefix,
            prefix_indices,
            _rebased(kv_indptr_prefix, begin, end),
            extend,
            extend_indices,
            extend_indptr,
            attn_sink,
            softmax_scale,
            out=out[begin:end],
        )
        if result.data_ptr() != out[begin:end].data_ptr():
            out[begin:end].copy_(result)
    return out


def _in_kernel(query, kv_indices_prefix, kv_indices_extend):
    # The scratch path reads a row count on the host, which capture forbids.
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
    same MXFP8 `(data, scale)` CSA and SWA planes."""
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
