# SPDX-License-Identifier: MIT
"""Fused history snapshot, RoPE/quantization and SWA commit for native prefill.

Requires contiguous positions per request, unique valid slots for live requests,
and a fixed-address cached boundary published before launch. Query and KV math
comes directly from the current source's RoPE helpers.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE
from atom.model_ops.deepseek_v41.rope_window import (
    _fp8_heads,
    _quantize_heads,
    _quantized_row,
)


@triton.jit
def _prefill_stage_rope_commit(
    query,
    kv,
    cos,
    sin,
    positions,
    query_values,
    query_scales,
    history,
    history_scale,
    staged,
    staged_scale,
    current,
    current_scale,
    current_stride,
    current_scale_stride,
    batches,
    cu,
    slots,
    cached,
    kv_stride,
    tokens,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    RING: tl.constexpr,
    STAGE_ROWS: tl.constexpr,
    FP8_HEADS: tl.constexpr,
):
    pid = tl.program_id(0)
    QUERY_PROGRAMS: tl.constexpr = HEADS // FP8_HEADS
    query_end = tokens * QUERY_PROGRAMS
    is_query = pid < query_end
    token = tl.where(is_query, pid // QUERY_PROGRAMS, pid - query_end)
    position = tl.load(positions + token)
    if is_query:
        head_group = pid % QUERY_PROGRAMS
        row0 = token * HEADS + head_group * FP8_HEADS
        if FP8_HEADS > 1:
            _quantize_heads(
                query,
                query_values,
                query_scales,
                row0,
                position,
                cos,
                sin,
                FP8_HEADS,
                D,
                ROPE,
            )
        else:
            quantized, code, _ = _quantized_row(
                query + row0 * D, position, cos, sin, D, ROPE
            )
            tl.store(
                query_values + row0 * D + tl.arange(0, D), tl.reshape(quantized, (D,))
            )
            tl.store(query_scales + row0 * (D // 32) + tl.arange(0, D // 32), code)
    else:
        batch = tl.load(batches + token)
        slot = tl.full((), -1, tl.int64)
        commit = tl.full((), False, tl.int1)
        if batch >= 0:
            slot = tl.load(slots + batch).to(tl.int64)
            begin = tl.load(cu + batch)
            end = tl.load(cu + batch + 1)
            cached_until = tl.load(cached + batch)
            if slot >= 0 and token == begin:
                # First KV CTA copies the complementary, never-overwritten
                # rows. No extra history CTAs, including graph padding CTAs.
                final_position = tl.load(positions + end - 1)
                writes = tl.maximum(
                    0,
                    tl.minimum(
                        tl.minimum(end - begin, RING), final_position - cached_until + 1
                    ),
                )
                remaining = RING - writes
                for offset in range(0, remaining, STAGE_ROWS):
                    index = offset + tl.arange(0, STAGE_ROWS)
                    history_row = (final_position + 1 + index) % RING
                    mask = index < remaining
                    hd = tl.arange(0, D)
                    hg = tl.arange(0, D // 32)
                    history_src = slot * RING + history_row
                    history_dst = batch.to(tl.int64) * RING + history_row
                    prior_data = tl.load(
                        history + history_src[:, None] * D + hd[None, :],
                        mask[:, None],
                        other=0,
                    )
                    prior_scale = tl.load(
                        history_scale + history_src[:, None] * (D // 32) + hg[None, :],
                        mask[:, None],
                        other=0,
                    )
                    tl.store(
                        staged + history_dst[:, None] * D + hd[None, :],
                        prior_data,
                        mask[:, None],
                    )
                    tl.store(
                        staged_scale + history_dst[:, None] * (D // 32) + hg[None, :],
                        prior_scale,
                        mask[:, None],
                    )
            commit = (slot >= 0) & (token >= end - RING) & (position >= cached_until)
        row = position % RING
        d = tl.arange(0, D)
        g = tl.arange(0, D // 32)
        if commit:
            # The sole writer owns the snapshot too. No other program reads
            # this persistent row, and both old planes are loaded before stores.
            old_data = tl.load(history + (slot * RING + row) * D + d)
            old_scale = tl.load(history_scale + (slot * RING + row) * (D // 32) + g)
            dst = batch.to(tl.int64) * RING + row
            tl.store(staged + dst * D + d, old_data)
            tl.store(staged_scale + dst * (D // 32) + g, old_scale)
        quantized, code, _ = _quantized_row(
            kv + token * kv_stride, position, cos, sin, D, ROPE
        )
        tl.store(current + token * current_stride + d, tl.reshape(quantized, (D,)))
        tl.store(current_scale + token * current_scale_stride + g, code)
        if commit:
            dst = slot * RING + row
            tl.store(
                history + dst * D + d,
                tl.reshape(quantized, (D,)).to(tl.uint8, bitcast=True),
            )
            tl.store(history_scale + dst * (D // 32) + g, code)


def prefill_rope_window(
    query,
    kv,
    cos,
    sin,
    positions,
    query_fp8,
    history,
    staged,
    batch_ids,
    cu,
    slots,
    cached_until,
    *,
    ring,
    rope_dim,
    stage_rows=16,
):
    """Produce all staged operands and commit final ring rows in one launch.

    All tensors and metadata are caller-owned. No allocation or H2D publication is
    performed here. The current KV starts after the full captured request capacity.
    This producer supports only the quantized native prefill path.
    """
    tokens, heads, dim = query.shape
    capacity = cu.numel() - 1
    assert tokens > 0 and heads > 0 and dim % 32 == 0
    assert slots.numel() >= capacity and cached_until.numel() >= capacity
    assert positions.numel() >= tokens and batch_ids.numel() >= tokens
    assert kv.shape == (tokens, dim) and kv.stride(-1) == 1
    assert query.is_contiguous() and stage_rows in (4, 8, 16)
    assert all(p.is_contiguous() for p in (*history, *staged, *query_fp8))
    assert all(p.dtype == torch.uint8 for p in (*history, *staged))
    assert query_fp8[0].dtype == FP8_DTYPE
    assert staged[0].shape == (capacity * ring + tokens, dim)
    assert staged[1].shape == (capacity * ring + tokens, dim // 32)
    fp8_heads = _fp8_heads(tokens, heads, query.device)
    current = staged[0][capacity * ring :].view(FP8_DTYPE)
    current_scale = staged[1][capacity * ring :]
    grid = tokens * (heads // fp8_heads + 1)
    return _prefill_stage_rope_commit[(grid,)](
        query,
        kv,
        cos,
        sin,
        positions,
        query_fp8[0],
        query_fp8[1].view(torch.uint8),
        *history,
        *staged,
        current,
        current_scale,
        current.stride(0),
        current_scale.stride(0),
        batch_ids,
        cu,
        slots,
        cached_until,
        kv.stride(0),
        tokens,
        HEADS=heads,
        D=dim,
        ROPE=rope_dim,
        RING=ring,
        STAGE_ROWS=stage_rows,
        FP8_HEADS=fp8_heads,
        num_warps=(
            2 if fp8_heads > 1 else max(1, dim * query.element_size() // (64 * 16))
        ),
    )
