# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sampled DSpark Markov step (`dspark_markov_sample`, the lossless draft).

The greedy op's bias GEMV (`_markov_logits_tile`), then ``x_k`` drawn from the
top K of ``softmax(logits_k / T)``, stored as q (`sparse_q_store`). q is 0 off
the stored set, so truncation costs acceptance, never exactness (`Q_WIDTH`).
Six launches per position: GEMV + 1/T into the fp32 ``s`` scratch with tile
partials; `softmax_combine_kernel` (row max only: finalize normalizes over the
stored set); then histogram, cut, compact and finalize (`sparse_q_select`).
Fixed-order reductions, no atomics: TP ranks must store and draw alike.
CUDA-graph safe: seed, row bases and filters are device tensors; position and
noise offset are static host ints; no data-dependent grid.
Philox key high word 0; the verify's draws use other words (`ratio_verify`).
int64 addressing: ``[max_bs, T, V]`` logits and the store pass 2**31 elements.
"""

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.model_ops.dspark_markov_sample import (
    _BLOCK_K,
    _BLOCK_V,
    _ID_DTYPES,
    _MAX_BLOCK_ROW,
    _MIN_BLOCK_ROW,
    _markov_logits_tile,
)
from atom.model_ops.sampling_scratch import SamplingScratch, take
from atom.model_ops.softmax_tiles import argmax_lowest_id, softmax_combine_kernel
from atom.model_ops.sparse_q_select import q_select_scratch_shapes, select_and_draw_q


@triton.jit
def _dspark_markov_sample_stage1(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    temperatures_ptr,
    q_row_base_ptr,
    embed_out_ptr,
    s_ptr,
    part_max_ptr,
    part_amax_val_ptr,
    part_amax_idx_ptr,
    num_rows,
    vocab_size,
    rank,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    stride_s_row,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One (V tile, row tile): bias GEMV, add base, scale by 1/T, write ``s``
    into the row's scratch row and publish this tile's max of ``s``, plus the
    unscaled logits' (max, lowest id) a greedy row drafts from.
    """
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    logits = _markov_logits_tile(
        base_ptr,
        prev_ids_ptr,
        w1_ptr,
        w2_ptr,
        embed_out_ptr,
        tile,
        offs_row,
        offs_v,
        row_mask,
        v_mask,
        rank,
        stride_base_row,
        stride_prev,
        stride_w1_vocab,
        stride_embed_row,
        stride_w2_vocab,
        0,
        BLOCK_ROW,
        BLOCK_V,
        BLOCK_K,
    )
    # The greedy op's own (max, lowest id), taken before the division: a
    # near-zero temperature can round distinct logits onto one value.
    raw_max, raw_idx = argmax_lowest_id(
        logits, offs_v[None, :], v_mask[None, :], vocab_size
    )
    tl.store(part_amax_val_ptr + tile * num_rows + offs_row, raw_max, mask=row_mask)
    tl.store(part_amax_idx_ptr + tile * num_rows + offs_row, raw_idx, mask=row_mask)
    # Pre-clamped by the caller to SAMPLER_EPS, so no row divides by zero. A
    # greedy row's scores are only scratch: finalize publishes the exact delta
    # on its argmax instead.
    temperature = tl.load(temperatures_ptr + offs_row, mask=row_mask, other=1.0)
    scaled = logits / temperature[:, None]
    # Only a row with a destination is read back; the rest draft the argmax.
    q_base = tl.load(q_row_base_ptr + offs_row, mask=row_mask, other=-1)
    tl.store(
        s_ptr + offs_row[:, None].to(tl.int64) * stride_s_row + offs_v[None, :],
        scaled,
        mask=(row_mask & (q_base >= 0))[:, None] & v_mask[None, :],
    )

    # Masked lanes are already -inf.
    tile_max = tl.max(scaled, axis=1)
    tl.store(part_max_ptr + tile * num_rows + offs_row, tile_max, mask=row_mask)


def _dspark_markov_sample_fake(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
    q_ids: torch.Tensor,
    q_probs: torch.Tensor,
    q_meta: torch.Tensor,
    s_scratch: torch.Tensor,
    q_row_base: torch.Tensor,
    position: int,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    noise_offset: int,
    scratch_f32: torch.Tensor,
    scratch_i32: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        base_logits.shape[0],
        markov_w1.shape[1],
        dtype=markov_w1.dtype,
        device=base_logits.device,
    )


def dspark_markov_sample_unsupported(
    markov_w1: torch.Tensor, markov_w2: torch.Tensor
) -> str | None:
    """Why the fused sampled op cannot run these weights, or None if it can.

    Checked once, when lossless sampling is armed, so an unsupported head
    keeps the greedy pair rather than failing on the first sampled step.
    """
    if not markov_w1.is_cuda:
        return "the markov head is not on a GPU"
    # tl.dot wants one dtype for both operands, and bf16/fp16 are the only ones
    # whose products are exactly representable in the fp32 accumulator.
    if markov_w1.dtype is not markov_w2.dtype or markov_w1.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return f"markov weights are {markov_w1.dtype}/{markov_w2.dtype}"
    if markov_w1.stride(1) != 1 or markov_w2.stride(1) != 1:
        return "markov weights are not contiguous along the rank axis"
    return None


def _sample_scratch_shapes(num_rows: int, vocab_size: int):
    """fp32 and int32 shapes one `dspark_markov_sample` call needs, in order."""
    gemv_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    f32 = [
        (gemv_tiles, num_rows),  # part_max
        (gemv_tiles, num_rows),  # part_amax_val
        (num_rows,),  # row_max
    ]
    i32 = [
        (gemv_tiles, num_rows),  # part_amax_idx
        (num_rows,),  # row_argmax
        *q_select_scratch_shapes(num_rows, vocab_size),
    ]
    return f32, i32


def dspark_markov_sample_scratch(max_rows: int, vocab_size: int, device):
    """Scratch for any `dspark_markov_sample` call of up to ``max_rows`` rows,
    allocated once. One call runs at a time, so a block's positions share it.
    The dense ``s`` scratch is not part of it: it is the caller's, sized
    ``[max_rows, V]``."""
    f32, i32 = _sample_scratch_shapes(max_rows, vocab_size)
    return SamplingScratch.for_shapes(f32, i32, device)


def _check_rows(name, t, rows, dtype):
    if t.shape != (rows,) or t.dtype is not dtype or t.stride(0) != 1:
        raise ValueError(
            f"{name} {tuple(t.shape)} {t.dtype} is not a contiguous [{rows}] {dtype}"
        )


@torch_compile_guard(
    gen_fake=_dspark_markov_sample_fake,
    mutates_args=[
        "out",
        "q_ids",
        "q_probs",
        "q_meta",
        "s_scratch",
        "scratch_f32",
        "scratch_i32",
    ],
)
def dspark_markov_sample(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
    q_ids: torch.Tensor,
    q_probs: torch.Tensor,
    q_meta: torch.Tensor,
    s_scratch: torch.Tensor,
    q_row_base: torch.Tensor,
    position: int,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    noise_offset: int,
    scratch_f32: torch.Tensor,
    scratch_i32: torch.Tensor,
) -> torch.Tensor:
    """Draw ``x_k`` from the top of ``softmax((base_logits + W1[prev_ids] @
    W2^T) / T)`` and publish exactly the q it was drawn from.

    Args:
        base_logits:  [B, V]  this block position's base logits.
        prev_ids:     [B]     ids of the previously sampled token x_{k-1}.
        markov_w1:    [V, r]  the table the bias row is gathered from.
        markov_w2:    [V, r]  the shared projection table.
        out:          [B]     int32/int64 destination for the ids. A strided
                              view is fine -- that is how the caller's own
                              `[B, T + 1]` block becomes the destination.
        q_ids, q_probs, q_meta:
                              a `SparseQStore`'s three tensors; row
                              ``q_row_base[b] + position`` receives row b's q.
                              Their width K bounds how many ids a row stores.
        s_scratch:    [>=B, V] fp32, the scaled scores between launches.
        q_row_base:   [B]     int32; negative writes nothing.
        position:             host int, this block position.
        temperatures: [B]     fp32, pre-clamped away from 0 by the caller.
        greedy_mask:  [B]     int32, nonzero where the draw is an argmax.
        top_k:        [B]     int32, the request's top-k; 0 is off.
        top_p:        [B]     fp32, the request's top-p; 1.0 is off.
        seed:         [1]     int32 device scalar. Must hold the same value on
                              every TP rank: each rank runs this op over the
                              full vocab and must draw, and store, the same
                              ids.
        noise_offset:         host int, the Philox counter base for this block
                              position. Static (position * max_B * V), so it
                              does not make the launch data-dependent.
        scratch_f32, scratch_i32: flat buffers the call's partials are carved
                              from (`dspark_markov_sample_scratch`).

    Returns:
        ``[B, r]`` ``W1[prev_ids]``, for the same reason the greedy op returns it.

    Raises rather than falling back: an eager fallback would draw from the
    global RNG, which TP ranks do not share, and the drafts would fork.
    """
    if base_logits.dim() != 2 or markov_w1.dim() != 2 or markov_w2.dim() != 2:
        raise ValueError("dspark_markov_sample expects 2-D base_logits/w1/w2")
    num_rows, vocab_size = base_logits.shape
    rank = markov_w2.shape[1]
    if prev_ids.shape != (num_rows,) or prev_ids.dtype not in _ID_DTYPES:
        raise ValueError(
            f"prev_ids {tuple(prev_ids.shape)} {prev_ids.dtype} is not the "
            f"[{num_rows}] int32 or int64 an id block is made of"
        )
    if out.shape != (num_rows,) or out.dtype not in _ID_DTYPES:
        raise ValueError(
            f"out {tuple(out.shape)} {out.dtype} is not the [{num_rows}] int32 "
            "or int64 an id block is made of"
        )
    store_rows, store_width = q_ids.shape if q_ids.dim() == 2 else (0, 0)
    if (
        store_width < 1
        or q_ids.dtype is not torch.int32
        or q_probs.shape != q_ids.shape
        or q_probs.dtype is not torch.float32
        or q_meta.shape != (store_rows, 2)
        or q_meta.dtype is not torch.int32
        or not (q_ids.is_contiguous() and q_probs.is_contiguous())
        or not q_meta.is_contiguous()
    ):
        raise ValueError(
            f"q_ids {tuple(q_ids.shape)} {q_ids.dtype}, q_probs "
            f"{tuple(q_probs.shape)} {q_probs.dtype}, q_meta "
            f"{tuple(q_meta.shape)} {q_meta.dtype} are not a contiguous "
            "[N, K] int32 / [N, K] fp32 / [N, 2] int32 SparseQStore"
        )
    if (
        s_scratch.dim() != 2
        or s_scratch.shape[0] < num_rows
        or s_scratch.shape[1] != vocab_size
        or s_scratch.dtype is not torch.float32
        or not s_scratch.is_contiguous()
    ):
        raise ValueError(
            f"s_scratch {tuple(s_scratch.shape)} {s_scratch.dtype} is not a "
            f"contiguous [>= {num_rows}, {vocab_size}] fp32"
        )
    _check_rows("q_row_base", q_row_base, num_rows, torch.int32)
    _check_rows("top_k", top_k, num_rows, torch.int32)
    _check_rows("top_p", top_p, num_rows, torch.float32)
    if base_logits.stride(1) != 1:
        raise ValueError("base_logits rows must be contiguous over the vocab axis")
    why = dspark_markov_sample_unsupported(markov_w1, markov_w2)
    if why is not None:
        raise ValueError(f"dspark_markov_sample cannot run: {why}")
    device = base_logits.device
    markov_embed = torch.empty(num_rows, rank, dtype=markov_w1.dtype, device=device)
    if num_rows == 0:
        return markov_embed
    block_row = min(
        _MAX_BLOCK_ROW, max(_MIN_BLOCK_ROW, triton.next_power_of_2(num_rows))
    )
    row_tiles = triton.cdiv(num_rows, block_row)

    # Stage 1's GEMV keeps the greedy op's narrow V tile (its accumulator is the
    # MFMA one); the q stages tile wide, because they only stream.
    gemv_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    f32_shapes, i32_shapes = _sample_scratch_shapes(num_rows, vocab_size)
    (
        (part_max, part_amax_val, row_max),
        (part_amax_idx, row_argmax, *select_scratch),
    ) = take(f32_shapes, i32_shapes, scratch_f32, scratch_i32)

    _dspark_markov_sample_stage1[(gemv_tiles, row_tiles)](
        base_logits,
        prev_ids,
        markov_w1,
        markov_w2,
        temperatures,
        q_row_base,
        markov_embed,
        s_scratch,
        part_max,
        part_amax_val,
        part_amax_idx,
        num_rows,
        vocab_size,
        rank,
        stride_base_row=base_logits.stride(0),
        stride_prev=prev_ids.stride(0),
        stride_w1_vocab=markov_w1.stride(0),
        stride_embed_row=markov_embed.stride(0),
        stride_w2_vocab=markov_w2.stride(0),
        stride_s_row=s_scratch.stride(0),
        BLOCK_ROW=block_row,
        BLOCK_V=_BLOCK_V,
        BLOCK_K=_BLOCK_K,
        num_warps=8 if block_row >= 32 else 4,
        num_stages=2,
    )
    softmax_combine_kernel[(num_rows,)](
        part_max,
        None,
        row_max,
        None,
        part_amax_val,
        part_amax_idx,
        row_argmax,
        num_rows,
        gemv_tiles,
        BLOCK_TILE=triton.next_power_of_2(gemv_tiles),
        WITH_ARGMAX=True,
        WITH_INV_Z=False,
        num_warps=4,
        num_stages=1,
    )
    select_and_draw_q(
        s_scratch,
        row_max,
        row_argmax,
        q_row_base,
        greedy_mask,
        top_k,
        top_p,
        seed,
        out,
        q_ids,
        q_probs,
        q_meta,
        select_scratch,
        position,
        noise_offset,
    )
    return markov_embed
