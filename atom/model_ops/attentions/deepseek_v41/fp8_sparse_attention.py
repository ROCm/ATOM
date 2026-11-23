# SPDX-License-Identifier: MIT
"""Sparse attention reading MXFP8 CSA and SWA planes directly, BF16 query.

The kernel loads each selected row's FP8 values as 32-bit words and its
E8M0 group-32 codes once per group, decodes them in registers to BF16, and runs the arithmetic of V4's BF16 CSA
kernels: the same tiles, dot products, online softmax, split-KV policy, split
combine and sink. No scratch, no host reads.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import FP8_TL_DTYPE
from atom.model_ops.v4_kernels.paged_2src import (
    _COMBINE_BLOCK_DC,
    _csa_num_splits,
    _sparse_attn_v4_split_combine_kernel,
)
from atom.model_ops.v4_kernels.pool_index import row_offset

from .packed_rows import _e8m0

_BLOCK_H = 16
_BLOCK_K = 32
_SPLIT_BLOCK_K = 16
_NUM_WARPS = 4
_NUM_STAGES = 1


@triton.jit
def _fp8_byte(words, k: tl.constexpr, step):
    raw = ((words >> (8 * k)) & 0xFF).to(tl.uint8)
    return (raw.to(FP8_TL_DTYPE, bitcast=True).to(tl.float32) * step).to(tl.bfloat16)


@triton.jit
def _mxfp8_rows(data, scale, slot, valid, d_offs, D: tl.constexpr):
    # Byte-wide loads compile to one load per value on gfx942; words keep the
    # loads vectorized, and the joins restore the element order exactly.
    W: tl.constexpr = D // 4
    G: tl.constexpr = D // 32
    BK: tl.constexpr = valid.shape[0]
    row = tl.where(valid, slot, 0)
    words = tl.load(
        data.to(tl.pointer_type(tl.uint32))
        + row_offset(row, W)[:, None]
        + tl.arange(0, W)[None, :],
        mask=valid[:, None],
        other=0,
    )
    code = tl.load(
        scale + row_offset(row, G)[:, None] + tl.arange(0, G)[None, :],
        mask=valid[:, None],
        other=127,
    )
    step = tl.reshape(tl.broadcast_to(_e8m0(code)[:, :, None], (BK, G, 8)), (BK, W))
    even = tl.join(_fp8_byte(words, 0, step), _fp8_byte(words, 2, step))
    odd = tl.join(_fp8_byte(words, 1, step), _fp8_byte(words, 3, step))
    return tl.reshape(tl.join(even, odd), (BK, D))


@triton.jit
def _attend_rows(q, kv, valid, h_mask, m_i, l_i, acc, qk_scale):
    neg_large = -3.4028234663852886e38
    scores = tl.dot(q, tl.trans(kv)) * qk_scale
    scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.where(
        h_mask[:, None] & valid[None, :], tl.exp2(scores - m_new[:, None]), 0.0
    )
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
    return m_new, l_i, acc


@triton.jit
def _mxfp8_2src_kernel(
    q_ptr,  # [N, H, D] bf16
    pool_data,  # [rows, D] uint8 CSA values
    pool_scale,  # [rows, D // 32] uint8 E8M0
    kv_indices_prefix_ptr,
    kv_indptr_prefix_ptr,
    kv_data,  # [rows, D] uint8 SWA values
    kv_scale,
    kv_indices_extend_ptr,
    kv_indptr_extend_ptr,
    attn_sink_ptr,
    out_ptr,
    part_acc_ptr,  # [N, S, H, D] fp32 when NUM_SPLITS > 1
    part_ml_ptr,  # [N, S, H, 2] fp32 when NUM_SPLITS > 1
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    H: tl.constexpr,
    softmax_scale: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program takes split `s` of a token's keys, prefix then extend;
    # NUM_SPLITS == 1 is the whole list, finalized here with the sink.
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    s = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, D)
    h_mask = h_offs < H
    q = tl.load(
        q_ptr + t * q_stride_t + h_offs[:, None] * q_stride_h + d_offs[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    qk_scale = softmax_scale * 1.4426950408889634
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    k_offs = tl.arange(0, BLOCK_K)

    p_start = tl.load(kv_indptr_prefix_ptr + t)
    p_len = tl.load(kv_indptr_prefix_ptr + t + 1) - p_start
    e_start = tl.load(kv_indptr_extend_ptr + t)
    e_len = tl.load(kv_indptr_extend_ptr + t + 1) - e_start
    total = p_len + e_len
    chunk = tl.cdiv(tl.cdiv(total, NUM_SPLITS), BLOCK_K) * BLOCK_K
    lo = s * chunk
    hi = tl.minimum(lo + chunk, total)

    p_hi = tl.minimum(hi, p_len)
    for k_start in tl.range(lo, p_hi, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < p_hi
        slot = tl.load(kv_indices_prefix_ptr + p_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0)
        kv = _mxfp8_rows(pool_data, pool_scale, slot, valid, d_offs, D)
        m_i, l_i, acc = _attend_rows(q, kv, valid, h_mask, m_i, l_i, acc, qk_scale)

    e_lo = tl.maximum(lo, p_len) - p_len
    e_hi = hi - p_len
    for e_k_start in tl.range(e_lo, e_hi, BLOCK_K):
        k_pos = e_k_start + k_offs
        in_range = k_pos < e_hi
        slot = tl.load(kv_indices_extend_ptr + e_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0)
        kv = _mxfp8_rows(kv_data, kv_scale, slot, valid, d_offs, D)
        m_i, l_i, acc = _attend_rows(q, kv, valid, h_mask, m_i, l_i, acc, qk_scale)

    if NUM_SPLITS == 1:
        sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(
            tl.float32
        )
        sink = sink * 1.4426950408889634
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp2(m_i - m_final)
        l_final = l_i * alpha + tl.exp2(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0
        )
        tl.store(
            out_ptr
            + t * out_stride_t
            + h_offs[:, None] * out_stride_h
            + d_offs[None, :],
            out,
            mask=h_mask[:, None],
        )
    else:
        row = (t * NUM_SPLITS + s) * H + h_offs
        tl.store(
            part_acc_ptr + row[:, None] * D + d_offs[None, :], acc, mask=h_mask[:, None]
        )
        tl.store(part_ml_ptr + row * 2, m_i, mask=h_mask)
        tl.store(part_ml_ptr + row * 2 + 1, l_i, mask=h_mask)


def sparse_attn_v41_paged_2src_fp8kv(
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
    block_k=None,
    split_block_k=None,
    num_warps=None,
    num_splits=None,
):
    """BF16 `query` [T, H, D] over MXFP8 `(data, scale)` CSA `pool` and SWA
    `kv` planes, each with its own int32 CSR row lists; -1 rows are skipped.
    The keyword overrides exist for tuning only."""
    tokens, heads, dim = query.shape
    if out is None:
        out = torch.empty_like(query)
    if tokens == 0:
        return out
    if query.dtype != torch.bfloat16 or query.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("FP8-KV attention takes a BF16 query with unit dim stride")
    pool_data, pool_scale = (plane.view(torch.uint8) for plane in pool)
    kv_data, kv_scale = (plane.view(torch.uint8) for plane in kv)
    head_blocks = triton.cdiv(heads, _BLOCK_H)
    if num_splits is None:
        num_splits = _csa_num_splits(
            tokens,
            head_blocks,
            (kv_indices_prefix.numel() + kv_indices_extend.numel()) / tokens,
        )
    split = num_splits > 1
    if split:
        part_acc = torch.empty(
            (tokens * num_splits * heads, dim), device=query.device, dtype=torch.float32
        )
        part_ml = torch.empty(
            (tokens * num_splits * heads, 2), device=query.device, dtype=torch.float32
        )
    else:
        part_acc = part_ml = out
    meta = {
        "NUM_SPLITS": num_splits,
        "BLOCK_H": _BLOCK_H,
        "D": dim,
        "BLOCK_K": (split_block_k or _SPLIT_BLOCK_K)
        if split
        else (block_k or _BLOCK_K),
        "num_warps": num_warps or _NUM_WARPS,
        "num_stages": _NUM_STAGES,
    }
    _mxfp8_2src_kernel[(tokens, head_blocks, num_splits)](
        query,
        pool_data,
        pool_scale,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_data,
        kv_scale,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        part_acc,
        part_ml,
        query.stride(0),
        query.stride(1),
        out.stride(0),
        out.stride(1),
        heads,
        float(softmax_scale),
        **meta,
    )
    if split:
        _sparse_attn_v4_split_combine_kernel[
            (tokens, head_blocks, triton.cdiv(dim, _COMBINE_BLOCK_DC))
        ](
            part_acc,
            part_ml,
            attn_sink,
            out,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            heads,
            NUM_SPLITS=num_splits,
            BLOCK_H=_BLOCK_H,
            BLOCK_D=dim,
            BLOCK_DC=_COMBINE_BLOCK_DC,
            num_warps=2,
        )
    return out
