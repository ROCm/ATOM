# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA operators for cache preparation, index selection and sparse paged GQA.

Core scoring and attention kernels are in aiter; this file keeps the
lightweight cache-utility kernels that are specific to ATOM's scheduler.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.qsa import (  # noqa: F401
    FP4_GROUP,
    FP8_DTYPE,
    FP8_MAX,
    QSA_CACHE_FORMATS,
    qsa_cache_row_layout,
    qsa_expand_block_indices,
    qsa_paged_mqa_logits,
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
    qsa_store_quantized_rows,
)


def qsa_cache_formats(atom_config) -> tuple[str, str]:
    """`(K/V format, compressed index-key format)` this engine serves.

    The raw index keys stay BF16 whatever is asked: they are mean-pooled
    before normalization, and a group can straddle a prefill chunk.
    An unset index dtype follows the KV cache's, as `Config` resolves it.
    """
    kv = getattr(atom_config, "kv_cache_dtype", "bf16") or "bf16"
    kv = "bf16" if kv in ("auto", "bfloat16") else kv
    index = getattr(atom_config, "index_cache_dtype", None) or kv
    index = "bf16" if index in ("auto", "bfloat16") else index
    if kv not in ("bf16", "fp8") or index not in QSA_CACHE_FORMATS:
        raise ValueError(
            f"Qwen QSA serves bf16/fp8 KV and bf16/fp8/fp4 index caches, got "
            f"kv_cache_dtype={kv!r} index_cache_dtype={index!r}"
        )
    return kv, index


# ---------------------------------------------------------------------------
# Utility kernels kept local to ATOM
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["N", "REAL"])
def _draft_decode_metadata(
    Lengths,
    Tables,
    Reject,
    Slots,
    Positions,
    Requests,
    Compressed,
    N,
    REAL,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_REJECT: tl.constexpr,
    B: tl.constexpr,
):
    req = tl.program_id(0) * B + tl.arange(0, B)
    live = req < REAL
    length = tl.load(Lengths + req, req < N, 0)
    if HAS_REJECT:
        length -= tl.load(Reject + req, live, 0)
    length = tl.where(live, length, 0)
    pos = length - 1
    valid = live & (pos >= 0)
    page = tl.load(Tables + req * TS + pos // PAGE, valid, 0)
    slot = tl.where(valid, page.to(tl.int64) * PAGE + pos % PAGE, -1)
    tl.store(Lengths + req, length, req < N)
    tl.store(Positions + req, pos, req < N)
    tl.store(Requests + req, tl.where(live, req, -1), req < N)
    tl.store(Slots + req, slot, req < N)
    tl.store(
        Compressed + req,
        tl.where(valid & (length % RATIO == 0), slot // RATIO, -1),
        req < N,
    )


def qsa_draft_decode_metadata(
    lengths: torch.Tensor,
    tables: torch.Tensor,
    rejected: torch.Tensor | None,
    slots: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    compressed: torch.Tensor,
    real_requests: int,
    block_size: int,
    compress_ratio: int,
) -> None:
    """Update draft lengths and physical write slots after verification."""
    _draft_decode_metadata[(triton.cdiv(lengths.numel(), 128),)](
        lengths,
        tables,
        rejected,
        slots,
        positions,
        requests,
        compressed,
        lengths.numel(),
        real_requests,
        tables.stride(0),
        block_size,
        compress_ratio,
        rejected is not None,
        128,
    )


@triton.jit
def _qsa_compressed_slots_kernel(
    slots_ptr,
    positions_ptr,
    out_ptr,
    num_tokens,
    stride_slots,
    stride_positions,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = tokens < num_tokens
    slot = tl.load(slots_ptr + tokens * stride_slots, mask, -1).to(tl.int64)
    position = tl.load(positions_ptr + tokens * stride_positions, mask, -1)
    closes_group = (position >= 0) & ((position + 1) % RATIO == 0) & (slot >= 0)
    tl.store(out_ptr + tokens, tl.where(closes_group, slot // RATIO, -1), mask)


def qsa_compressed_slots(
    slot_mapping: torch.Tensor,
    logical_positions: torch.Tensor,
    compress_ratio: int,
    out: torch.Tensor,
) -> None:
    """Map complete groups into the existing persistent compressed-slot buffer."""
    tokens = slot_mapping.numel()
    if (
        compress_ratio <= 0
        or logical_positions.numel() != tokens
        or out.numel() != tokens
    ):
        raise ValueError(
            "QSA compressed slots require equal token lengths and a positive ratio"
        )
    if out.ndim != 1 or out.stride(0) != 1:
        raise ValueError("QSA compressed slots output must be contiguous")
    if tokens:
        _qsa_compressed_slots_kernel[(triton.cdiv(tokens, 256),)](
            slot_mapping,
            logical_positions,
            out,
            tokens,
            slot_mapping.stride(0),
            logical_positions.stride(0),
            RATIO=compress_ratio,
            BLOCK=256,
            num_warps=4,
        )


@triton.jit
def _qsa_mrope_kernel(
    q_ptr,
    k_ptr,
    out_q_ptr,
    out_k_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    pos_axis_stride,
    pos_token_stride,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    SECTION_T: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // (Q_HEADS + K_HEADS)
    head = row % (Q_HEADS + K_HEADS)
    dims = tl.arange(0, BLOCK_D)
    if head < Q_HEADS:
        src = q_ptr + token * q_token_stride + head * q_head_stride
        dst = out_q_ptr + (token * Q_HEADS + head) * HEAD_DIM
    else:
        src = k_ptr + token * k_token_stride + (head - Q_HEADS) * k_head_stride
        dst = out_k_ptr + (token * K_HEADS + head - Q_HEADS) * HEAD_DIM
    x = tl.load(src + dims, dims < HEAD_DIM, other=0)
    half = ROTARY_DIM // 2
    freq = dims % half
    if INTERLEAVED:
        axis = tl.where((freq % 3 == 1) & (freq < 3 * SECTION_H), 1, 0)
        axis = tl.where((freq % 3 == 2) & (freq < 3 * SECTION_W), 2, axis)
    else:
        axis = tl.where(
            freq < SECTION_T, 0, tl.where(freq < SECTION_T + SECTION_H, 1, 2)
        )
    position = tl.load(pos_ptr + axis * pos_axis_stride + token * pos_token_stride)
    cos = tl.load(cos_ptr + position * half + freq).to(x.dtype).to(tl.float32)
    sin = tl.load(sin_ptr + position * half + freq).to(x.dtype).to(tl.float32)
    paired = tl.where(dims < half, dims + half, dims - half)
    y = tl.load(src + paired, dims < ROTARY_DIM, other=0).to(tl.float32)
    y = tl.where(dims < half, -y, y)
    xc = (x.to(tl.float32) * cos).to(x.dtype).to(tl.float32)
    ys = (y * sin).to(x.dtype).to(tl.float32)
    result = tl.where(dims < ROTARY_DIM, (xc + ys).to(x.dtype), x)
    tl.store(dst + dims, result, dims < HEAD_DIM)


def qsa_apply_mrope(
    rotary_emb,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Partial NeoX mRoPE for [tokens, heads, dim], without dummy indexer K."""
    tokens, heads, dim = q.shape
    if q.stride(-1) != 1 or not rotary_emb.is_neox_style:
        raise ValueError(
            "QSA mRoPE requires contiguous head dimensions and NeoX layout"
        )
    if k is not None and (
        k.shape[0] != tokens or k.shape[2] != dim or k.stride(-1) != 1
    ):
        raise ValueError("QSA mRoPE Q/K shapes or strides disagree")
    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    out_k = (
        torch.empty(k.shape, dtype=k.dtype, device=k.device) if k is not None else None
    )
    if tokens == 0:
        return out_q, out_k
    rotary_dim = rotary_emb.rotary_dim
    sections = getattr(rotary_emb, "mrope_section", None) or [rotary_dim // 2, 0, 0]
    if positions.ndim not in (1, 2) or (
        positions.ndim == 2 and positions.shape[0] not in (1, 3)
    ):
        raise ValueError("QSA positions must be [tokens], [1, tokens] or [3, tokens]")
    if positions.shape[-1] < tokens:
        raise ValueError("QSA positions are shorter than Q/K")
    axis_stride = (
        positions.stride(0) if positions.ndim == 2 and positions.shape[0] == 3 else 0
    )
    kv_heads = k.shape[1] if k is not None else 0
    _qsa_mrope_kernel[(tokens * (heads + kv_heads),)](
        q,
        k if k is not None else q,
        out_q,
        out_k if k is not None else out_q,
        rotary_emb.cos_cache,
        rotary_emb.sin_cache,
        positions,
        q.stride(0),
        q.stride(1),
        k.stride(0) if k is not None else 0,
        k.stride(1) if k is not None else 0,
        axis_stride,
        positions.stride(-1),
        Q_HEADS=heads,
        K_HEADS=kv_heads,
        HEAD_DIM=dim,
        ROTARY_DIM=rotary_dim,
        SECTION_T=sections[0],
        SECTION_H=sections[1],
        SECTION_W=sections[2],
        INTERLEAVED=getattr(rotary_emb, "mrope_interleaved", False),
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out_q, out_k


@triton.jit
def _qsa_store_rows_kernel(
    cache_ptr,
    slots_ptr,
    values_ptr,
    stride_cache_slot,
    stride_cache_head,
    stride_cache_dim,
    stride_value_row,
    stride_value_head,
    stride_value_dim,
    num_rows,
    num_slots,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)

    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_slots)
    if not valid:
        return

    values = tl.load(
        values_ptr
        + row * stride_value_row
        + head * stride_value_head
        + dims * stride_value_dim,
        mask=dims < HEAD_DIM,
        other=0,
    )
    tl.store(
        cache_ptr
        + slot.to(tl.int64) * stride_cache_slot
        + head * stride_cache_head
        + dims * stride_cache_dim,
        values,
        mask=dims < HEAD_DIM,
    )


def qsa_store_rows(
    cache: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Scatter `values [rows, heads, dim]` into `cache` at `slots`, skipping -1."""
    if cache.ndim != 4:
        raise ValueError("cache must be [pages, page_size, heads, dim]")
    if values.ndim != 3:
        raise ValueError("values must be [rows, heads, dim]")
    if values.shape[1] != cache.shape[2] or values.shape[2] != cache.shape[3]:
        raise ValueError("values and cache disagree on heads/dim")
    rows = values.shape[0]
    if rows == 0:
        return
    if slots.shape[0] < rows:
        raise ValueError("slot mapping is shorter than the value rows")

    flat = cache.view(cache.shape[0] * cache.shape[1], cache.shape[2], cache.shape[3])
    head_dim = values.shape[2]
    _qsa_store_rows_kernel[(rows, values.shape[1])](
        flat,
        slots,
        values,
        flat.stride(0),
        flat.stride(1),
        flat.stride(2),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        rows,
        flat.shape[0],
        NUM_HEADS=values.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


def _table_width_bucket(width: int) -> int:
    """Page-table width as seen by the kernels, rounded up to a power of two."""
    return 1 << (max(int(width), 1) - 1).bit_length()


@triton.jit
def _qsa_compress_groups_kernel(
    raw_cache_ptr,
    position_cache_ptr,
    page_table_ptr,
    token_to_request_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_page,
    stride_raw_token,
    stride_raw_dim,
    stride_position_page,
    stride_position_token,
    stride_position_axis,
    stride_table_request,
    stride_table_page,
    stride_pooled_row,
    stride_pooled_dim,
    stride_first_row,
    stride_first_axis,
    num_rows,
    num_pages,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_POSITIONS: tl.constexpr,
    ROPE_POSITION_OFFSET: tl.constexpr,
) -> None:
    """Mean-pool the group that ends at each token, reading the paged raw cache."""
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)

    request = tl.load(token_to_request_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if valid_row:
        for offset in tl.range(0, COMPRESS_RATIO):
            position = end_position - (COMPRESS_RATIO - 1 - offset)
            logical_page = position // PAGE_SIZE
            page_offset = position % PAGE_SIZE
            valid = logical_page < PAGE_TABLE_WIDTH
            physical_page = tl.load(
                page_table_ptr
                + safe_request * stride_table_request
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_pages)
            accumulator += tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_page
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    axes = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_POSITIONS:
        logical_page = first_position // PAGE_SIZE
        page_offset = first_position % PAGE_SIZE
        valid = valid_row & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_request
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_pages)
        values = tl.load(
            position_cache_ptr
            + tl.maximum(physical_page, 0).to(tl.int64) * stride_position_page
            + page_offset * stride_position_token
            + axes * stride_position_axis,
            mask=valid & (axes < 3),
            other=0,
        )
        tl.store(
            first_positions_ptr + row * stride_first_row + axes * stride_first_axis,
            values,
            mask=(row < num_rows) & (axes < 3),
        )
    else:
        first_position = tl.where(valid_row, first_position + ROPE_POSITION_OFFSET, 0)
        tl.store(
            first_positions_ptr + row * stride_first_row + axes * stride_first_axis,
            first_position,
            mask=(row < num_rows) & (axes < 3),
        )


def qsa_compress_groups(
    raw_key_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    position_cache: torch.Tensor | None = None,
    rope_position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool the group closed by each token; return `(pooled, first_positions)`."""
    if raw_key_cache.ndim != 4 or raw_key_cache.shape[2] != 1:
        raise ValueError("raw_key_cache must be [pages, page_size, 1, head_dim]")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    rows = int(token_to_request.shape[0])
    head_dim = raw_key_cache.shape[3]
    device = raw_key_cache.device
    pooled = torch.empty((rows, 1, head_dim), dtype=raw_key_cache.dtype, device=device)
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=device)
    if rows == 0:
        return pooled, first_positions

    load_positions = position_cache is not None
    if load_positions:
        if position_cache.ndim != 4 or position_cache.shape[3] < 3:
            raise ValueError("position_cache must be [pages, page_size, 1, >=3]")
        position_strides = (
            position_cache.stride(0),
            position_cache.stride(1),
            position_cache.stride(3),
        )
    else:
        position_cache = raw_key_cache
        position_strides = (0, 0, 0)

    _qsa_compress_groups_kernel[(rows,)](
        raw_key_cache,
        position_cache,
        page_table,
        token_to_request,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_key_cache.stride(0),
        raw_key_cache.stride(1),
        raw_key_cache.stride(3),
        *position_strides,
        page_table.stride(0),
        page_table.stride(1),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        raw_key_cache.shape[0],
        page_table.shape[0],
        PAGE_SIZE=raw_key_cache.shape[1],
        PAGE_TABLE_WIDTH=_table_width_bucket(page_table.shape[1]),
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        LOAD_POSITIONS=load_positions,
        ROPE_POSITION_OFFSET=0 if load_positions else rope_position_offset,
        num_warps=4,
    )
    return pooled, first_positions
