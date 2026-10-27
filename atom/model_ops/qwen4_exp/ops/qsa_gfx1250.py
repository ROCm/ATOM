# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 (MI450 / MI455) sparse paged GQA for QSA.

Same contract and split-K partial format as `_qsa_sparse_paged_gqa_kernel` in
`qsa.py`, rebuilt around the gfx1250 data movers:

* The paged BF16 K/V caches are viewed as one `[rows, head_dim]` table, where a
  row is one (physical slot, KV head) pair of `head_dim * 2` contiguous bytes.
  The selected tokens of a tile become row ids, and TDM gather mode
  (`tdm.async_gather`, 8 int32 row ids per TENSOR_LOAD_TO_LDS) copies those
  rows straight into LDS. K/V never pass through VGPRs, so the register file
  holds only Q, the accumulator and one tile of WMMA operands.
* Tiles are gathered into an `NUM_STAGES`-deep LDS ring ahead of the math, so
  `NUM_STAGES - 1` tiles of K and V are in flight while the current tile runs.
  The kernel is bandwidth/latency bound (12 FLOP per K/V byte), so bytes in
  flight per CU is what sets its throughput.
* Each LDS row is padded by 16 bytes. The 512-byte gather rows keep the TDM
  direct-copy conditions (first tile dimension >= 256 B, 128 B aligned global
  rows, padding interval a multiple of 128 B) while the padding spreads the
  transposed K reads over LDS banks.
"""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _qsa_tile_rows(
    index_base,
    table_base,
    col0,
    col_end,
    columns,
    request_valid,
    kv_head,
    num_cache_pages,
    PAGE_SIZE: gl.constexpr,
    PAGE_TABLE_WIDTH: gl.constexpr,
    ROWS_PER_PAGE: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
):
    """Row ids into the `[rows, head_dim]` cache view for one tile.

    Invalid columns map to row 0 so every gathered row stays in bounds; TDM
    gather only handles out-of-bounds rows correctly for non-decreasing ids,
    which a top-k selection does not provide. The returned mask hides them.
    """
    cols = col0 + columns
    logical = gl.load(index_base + cols, mask=cols < col_end, other=-1)
    safe_logical = gl.maximum(logical, 0)
    logical_page = safe_logical // PAGE_SIZE
    page_offset = safe_logical % PAGE_SIZE
    valid = request_valid & (logical >= 0) & (logical_page < PAGE_TABLE_WIDTH)
    physical_page = gl.load(
        table_base + gl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid = valid & (physical_page >= 0) & (physical_page < num_cache_pages)
    rows = physical_page * ROWS_PER_PAGE + page_offset * NUM_KV_HEADS + kv_head
    return gl.where(valid, rows, 0), valid


@gluon.jit
def _qsa_sparse_paged_gqa_gfx1250_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    logical_indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    widths_ptr,
    stride_q_token,
    stride_q_head,
    stride_indices_token,
    stride_table_request,
    stride_output_token,
    stride_output_head,
    num_cache_rows,
    num_cache_pages,
    num_requests,
    softmax_scale,
    TOPK: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    PAGE_TABLE_WIDTH: gl.constexpr,
    ROWS_PER_PAGE: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    GROUP_SIZE: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
    KV_SPLITS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    num_warps: gl.constexpr,
):
    WARP_SIZE: gl.constexpr = 32
    LOG2E: gl.constexpr = 1.4426950408889634

    # Warps tile the N axis of both dots: QK columns and PV head dims.
    if num_warps == 1:
        warp_bases: gl.constexpr = []
    elif num_warps == 2:
        warp_bases: gl.constexpr = [[0, 1]]
    else:
        warp_bases: gl.constexpr = [[0, 1], [0, 2]]
    WMMA: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, instr_shape=[16, 16, 32], warp_bases=warp_bases
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=WMMA, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=WMMA, k_width=8)
    ROW: gl.constexpr = gl.SliceLayout(1, WMMA)
    COL: gl.constexpr = gl.SliceLayout(0, WMMA)

    TPW_D: gl.constexpr = HEAD_DIM // 8 if HEAD_DIM // 8 < 16 else 16
    IO: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[WARP_SIZE // TPW_D, TPW_D],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    INDEX: gl.constexpr = gl.BlockedLayout([1], [WARP_SIZE], [num_warps], [0])
    # The TDM descriptor is wave-uniform, so every lane carries the tile's ids.
    GATHER: gl.constexpr = gl.BlockedLayout([BLOCK_K], [WARP_SIZE], [num_warps], [0])
    KV_SHARED: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_DIM, 8]], [BLOCK_K, HEAD_DIM], [1, 0]
    )

    token = gl.program_id(0)
    kv_head = gl.program_id(1)
    part = gl.program_id(2)
    if KV_SPLITS > 1:  # noqa: SIM102 -- compile-time guard for widths_ptr=None
        if kv_head == 0:
            if part == 0:
                gl.store(widths_ptr + token, TOPK)

    PARTITION: gl.constexpr = (
        (TOPK + KV_SPLITS * BLOCK_K - 1) // (KV_SPLITS * BLOCK_K)
    ) * BLOCK_K
    col_begin = part * PARTITION
    if col_begin >= TOPK:
        return
    col_end = gl.minimum(col_begin + PARTITION, TOPK)
    num_tiles = gl.cdiv(col_end - col_begin, BLOCK_K)

    request = gl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = gl.minimum(gl.maximum(request, 0), num_requests - 1)
    index_base = logical_indices_ptr + token * stride_indices_token
    table_base = block_table_ptr + safe_request * stride_table_request
    columns = gl.arange(0, BLOCK_K, layout=INDEX)

    first_q_head = kv_head * GROUP_SIZE
    heads = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, IO))
    dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, IO))
    head_mask = (heads < GROUP_SIZE)[:, None]
    q = gl.amd.cdna4.buffer_load(
        ptr=q_ptr + token * stride_q_token,
        offsets=((first_q_head + heads)[:, None] * stride_q_head + dims[None, :]).to(
            gl.int32
        ),
        mask=head_mask,
        other=0.0,
    )
    # Fold the base-2 softmax scale into Q, as the portable kernel does.
    q = (q.to(gl.float32) * (softmax_scale * LOG2E)).to(q_ptr.dtype.element_ty)
    q_op = gl.convert_layout(q, dot_a)

    k_bufs = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_K, HEAD_DIM], KV_SHARED
    )
    v_bufs = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_K, HEAD_DIM], KV_SHARED
    )
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_cache_ptr,
        shape=[num_cache_rows, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_K, HEAD_DIM],
        layout=KV_SHARED,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_cache_ptr,
        shape=[num_cache_rows, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_K, HEAD_DIM],
        layout=KV_SHARED,
    )

    # Prologue: put tiles 0 .. NUM_STAGES-2 in flight. Tiles past the end of
    # the partition gather row 0 and are never consumed, which keeps the
    # number of outstanding TDM operations per iteration a compile-time value.
    rows, valid_0 = _qsa_tile_rows(
        index_base, table_base, col_begin, col_end, columns, request_valid,
        kv_head, num_cache_pages, PAGE_SIZE, PAGE_TABLE_WIDTH, ROWS_PER_PAGE,
        NUM_KV_HEADS,
    )
    rows = gl.convert_layout(rows, GATHER)
    gl.amd.gfx1250.tdm.async_gather(k_desc, rows, k_bufs.index(0))
    gl.amd.gfx1250.tdm.async_gather(v_desc, rows, v_bufs.index(0))
    valid_1 = valid_0
    if NUM_STAGES == 3:
        rows, valid_1 = _qsa_tile_rows(
            index_base, table_base, col_begin + BLOCK_K, col_end, columns,
            request_valid, kv_head, num_cache_pages, PAGE_SIZE,
            PAGE_TABLE_WIDTH, ROWS_PER_PAGE, NUM_KV_HEADS,
        )
        rows = gl.convert_layout(rows, GATHER)
        gl.amd.gfx1250.tdm.async_gather(k_desc, rows, k_bufs.index(1))
        gl.amd.gfx1250.tdm.async_gather(v_desc, rows, v_bufs.index(1))

    m_i = gl.full([BLOCK_H], -1.0e20, gl.float32, layout=ROW)
    l_i = gl.zeros([BLOCK_H], gl.float32, layout=ROW)
    acc = gl.zeros([BLOCK_H, HEAD_DIM], gl.float32, layout=WMMA)

    for i in range(0, num_tiles):
        ahead = i + NUM_STAGES - 1
        rows, valid_n = _qsa_tile_rows(
            index_base, table_base, col_begin + ahead * BLOCK_K, col_end,
            columns, request_valid, kv_head, num_cache_pages, PAGE_SIZE,
            PAGE_TABLE_WIDTH, ROWS_PER_PAGE, NUM_KV_HEADS,
        )
        rows = gl.convert_layout(rows, GATHER)
        stage_n = ahead % NUM_STAGES
        gl.amd.gfx1250.tdm.async_gather(k_desc, rows, k_bufs.index(stage_n))
        gl.amd.gfx1250.tdm.async_gather(v_desc, rows, v_bufs.index(stage_n))
        # Ops complete in issue order: leaving the 2 * (NUM_STAGES - 1) newest
        # outstanding retires both gathers of tile i.
        gl.amd.gfx1250.tdm.async_wait(2 * (NUM_STAGES - 1))

        stage = i % NUM_STAGES
        k_t = k_bufs.index(stage).permute([1, 0]).load(dot_b)
        scores = gl.amd.gfx1250.wmma(
            q_op, k_t, gl.zeros([BLOCK_H, BLOCK_K], gl.float32, layout=WMMA)
        )
        valid = gl.convert_layout(valid_0, COL)[None, :]
        scores = gl.where(valid, scores, -1.0e20)
        m_new = gl.maximum(m_i, gl.max(scores, axis=1))
        alpha = gl.exp2(m_i - m_new)
        probabilities = gl.where(valid, gl.exp2(scores - m_new[:, None]), 0.0)
        l_i = l_i * alpha + gl.sum(probabilities, axis=1)
        v_t = v_bufs.index(stage).load(dot_b)
        p_op = gl.convert_layout(probabilities.to(q_ptr.dtype.element_ty), dot_a)
        acc = gl.amd.gfx1250.wmma(p_op, v_t, acc * alpha[:, None])
        m_i = m_new

        if NUM_STAGES == 3:
            valid_0 = valid_1
            valid_1 = valid_n
        else:
            valid_0 = valid_n

    # Drain the look-ahead gathers before the ring is released.
    gl.amd.gfx1250.tdm.async_wait(0)

    out_heads = first_q_head + heads
    if KV_SPLITS > 1:
        # Unnormalized [token, head, split, dim] partials with each split's
        # base-2 running maximum and sum, as `_mla_decode_fwd_reduce_kernel`
        # consumes them.
        gl.amd.cdna4.buffer_store(
            gl.convert_layout(acc, IO),
            ptr=output_ptr + token * stride_output_token + part * HEAD_DIM,
            offsets=(out_heads[:, None] * stride_output_head + dims[None, :]).to(
                gl.int32
            ),
            mask=head_mask,
        )
        stat_heads = gl.arange(0, BLOCK_H, layout=ROW)
        stat_offset = (
            token * NUM_KV_HEADS * GROUP_SIZE + first_q_head + stat_heads
        ).to(gl.int64) * KV_SPLITS + part
        stat_mask = stat_heads < GROUP_SIZE
        gl.store(partial_max_ptr + stat_offset, m_i, mask=stat_mask)
        gl.store(partial_sum_ptr + stat_offset, l_i, mask=stat_mask)
    else:
        output = gl.where(
            l_i[:, None] > 0,
            acc / gl.maximum(l_i, 1.0e-20)[:, None],
            0.0,
        )
        gl.amd.cdna4.buffer_store(
            gl.convert_layout(output.to(output_ptr.dtype.element_ty), IO),
            ptr=output_ptr + token * stride_output_token,
            offsets=(out_heads[:, None] * stride_output_head + dims[None, :]).to(
                gl.int32
            ),
            mask=head_mask,
        )


def qsa_gfx1250_supports(q, k_cache, v_cache) -> bool:
    """Whether the gfx1250 kernel can serve these operands.

    The caches must be viewable as one `[rows, head_dim]` row table: unit dim
    stride, heads and page slots packed, page stride a whole number of rows,
    and K/V sharing that layout.
    """
    import torch

    if q.dtype != torch.bfloat16 or k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        return False
    tokens, q_heads, head_dim = q.shape
    pages, page_size, kv_heads, _ = k_cache.shape
    group_size = q_heads // kv_heads
    if head_dim not in (64, 128, 256) or group_size > 16 or q.stride(2) != 1:
        return False
    strides = k_cache.stride()
    if v_cache.stride() != strides:
        return False
    if strides[3] != 1 or strides[2] != head_dim or strides[1] != kv_heads * head_dim:
        return False
    if strides[0] % head_dim:
        return False
    rows = (pages - 1) * (strides[0] // head_dim) + page_size * kv_heads
    if rows >= 2**31:
        return False
    # Q and the output are addressed with 32-bit buffer offsets per token.
    return q_heads * head_dim < 2**31 and tokens * q.stride(0) < 2**31


def qsa_gfx1250_launch_config(tokens: int, kv_heads: int, topk: int, num_cu: int):
    """(BLOCK_K, NUM_STAGES, num_warps, kv_splits) for one launch.

    Measured on MI450 at the AgentX shapes with a 2-deep ring and one wave
    per CTA: deeper rings and 64-row tiles lose, because LDS (~34 KB per CTA
    at 16 rows, ~68 KB at 32) already caps the CTAs a CU holds.
    * Large launches (16K-token prefill) need no split and run 32-row tiles.
    * Decode is latency bound and wants more CTAs in flight: 16-row tiles
      split to about 8 CTAs per CU (T=64 -> 16 splits).
    """
    base = max(1, tokens * kv_heads)
    if topk < 512 or base >= 4 * num_cu:
        return 32, 2, 1, 1
    block_k = 16
    splits = min((8 * num_cu) // base, 64, triton.cdiv(topk, block_k))
    return block_k, 2, 1, 1 << (splits.bit_length() - 1)


def qsa_sparse_paged_gqa_gfx1250(
    q,
    k_cache,
    v_cache,
    logical_indices,
    block_table,
    token_to_request,
    scale: float,
    page_table_width: int,
    num_cu: int,
    kv_splits: int | None = None,
    block_k: int | None = None,
    num_stages: int | None = None,
    num_warps: int | None = None,
):
    import torch

    tokens, q_heads, head_dim = q.shape
    pages, page_size, kv_heads, _ = k_cache.shape
    group_size = q_heads // kv_heads
    topk = logical_indices.shape[1]
    cfg = qsa_gfx1250_launch_config(tokens, kv_heads, topk, num_cu)
    block_k = cfg[0] if block_k is None else block_k
    num_stages = cfg[1] if num_stages is None else num_stages
    num_warps = cfg[2] if num_warps is None else num_warps
    if kv_splits is None:
        kv_splits = cfg[3]
    if num_stages not in (2, 3):
        raise ValueError("num_stages must be 2 or 3")
    if kv_splits < 1 or kv_splits & (kv_splits - 1):
        raise ValueError("kv_splits must be a positive power of two")

    rows_per_page = k_cache.stride(0) // head_dim
    num_cache_rows = (pages - 1) * rows_per_page + page_size * kv_heads
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    target = out
    partial_max = partial_sum = out
    widths = None
    if kv_splits > 1:
        shape = (tokens, q_heads, kv_splits)
        partial_max = torch.empty(shape, device=q.device, dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)
        target = torch.empty((*shape, head_dim), device=q.device, dtype=torch.float32)
        widths = torch.empty(tokens, dtype=torch.int32, device=q.device)

    _qsa_sparse_paged_gqa_gfx1250_kernel[(tokens, kv_heads, kv_splits)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        target,
        partial_max,
        partial_sum,
        widths,
        q.stride(0),
        q.stride(1),
        logical_indices.stride(0),
        block_table.stride(0),
        target.stride(0),
        target.stride(1),
        num_cache_rows,
        pages,
        block_table.shape[0],
        float(scale),
        TOPK=topk,
        PAGE_SIZE=page_size,
        PAGE_TABLE_WIDTH=page_table_width,
        ROWS_PER_PAGE=rows_per_page,
        NUM_KV_HEADS=kv_heads,
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_H=16,
        BLOCK_K=block_k,
        KV_SPLITS=kv_splits,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
    )
    if kv_splits > 1:
        from aiter.ops.triton._triton_kernels.attention.mla import (
            _mla_decode_fwd_reduce_kernel,
        )

        _mla_decode_fwd_reduce_kernel[(tokens, q_heads)](
            out,
            target,
            partial_max,
            partial_sum,
            widths,
            None,
            tokens,
            q_heads,
            out.stride(0),
            out.stride(1),
            1,
            1,
            tokens,
            TILE_SIZE=block_k,
            KV_LORA_RANK=head_dim,
            query_start_len_ptr=None,
            BLOCK_Q=1,
            NUM_SEGMENTS_PER_SEQ=kv_splits,
            ALL_DECODE=True,
        )
    return out
