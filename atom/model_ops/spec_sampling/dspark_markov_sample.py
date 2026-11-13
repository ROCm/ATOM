# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sampled DSpark Markov step (`dspark_markov_sample`, the lossless draft).

The greedy op's stage-1 GEMV (`dspark_markov_stage1`), then ``x_k`` drawn from the
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

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.model_ops.dspark_markov_argmax import (
    check_markov_inputs,
    markov_weights_unsupported,
)
from atom.model_ops.dspark_markov_stage1 import (
    BLOCK_K,
    BLOCK_V,
    markov_launch_geometry,
    markov_logits_tile,
    publish_tile_argmax,
)
from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.sampling_scratch import (
    Region,
    carve,
    scratch_buffer,
)
from atom.model_ops.spec_sampling.softmax_tiles import softmax_combine_kernel
from atom.model_ops.spec_sampling.sparse_q_select import (
    QSelectScratch,
    q_select_scratch,
    select_and_draw_q,
)


@triton.jit
def _markov_sample_stage1(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    part_val_ptr,
    part_idx_ptr,
    temperatures_ptr,
    q_row_base_ptr,
    s_ptr,
    part_smax_ptr,
    num_rows,
    vocab_size,
    rank,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_w2_vocab,
    stride_s_row,
    min_temperature,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One (V tile, row tile): the greedy op's stage 1 (bias GEMV, base
    logits, the tile's (max, lowest id) of the logits), then scale by 1/T,
    write the scores ``s`` of rows with a q destination into their scratch
    row, and publish the tile's max of ``s``. The (max, id) is taken before
    the division: a near-zero temperature can round distinct logits onto one
    value. The whole vocab, from id 0."""
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    vals = markov_logits_tile(
        base_ptr,
        prev_ids_ptr,
        w1_ptr,
        w2_ptr,
        None,
        tile,
        offs_row,
        offs_v,
        row_mask,
        v_mask,
        rank,
        stride_base_row,
        stride_prev,
        stride_w1_vocab,
        0,
        stride_w2_vocab,
        0,
        BLOCK_ROW,
        BLOCK_V,
        BLOCK_K,
        False,
    )
    publish_tile_argmax(
        vals,
        tile,
        offs_row,
        offs_v,
        row_mask,
        v_mask,
        num_rows,
        0,
        part_val_ptr,
        part_idx_ptr,
    )
    # Clamped here too, so a pad row's stale or zero temperature never divides
    # by zero. A greedy row's scores are only scratch: finalize publishes the
    # exact delta on its argmax instead.
    temperature = tl.maximum(
        tl.load(temperatures_ptr + offs_row, mask=row_mask, other=1.0),
        min_temperature,
    )
    scaled = vals / temperature[:, None]
    # Only a row with a destination is read back; the rest draft the argmax.
    q_base = tl.load(q_row_base_ptr + offs_row, mask=row_mask, other=-1)
    tl.store(
        s_ptr + offs_row[:, None].to(tl.int64) * stride_s_row + offs_v[None, :],
        scaled,
        mask=(row_mask & (q_base >= 0))[:, None] & v_mask[None, :],
    )
    # Masked lanes are already -inf.
    tl.store(
        part_smax_ptr + tile * num_rows + offs_row,
        tl.max(scaled, axis=1),
        mask=row_mask,
    )


def _dspark_markov_sample_fake(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
    q_ids: torch.Tensor,
    q_probs: torch.Tensor,
    q_meta: torch.Tensor,
    q_row_base: torch.Tensor,
    position: int,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    noise_offset: int,
    scratch: torch.Tensor,
) -> None:
    return None


class SampleScratch(NamedTuple):
    """One `dspark_markov_sample` call's work buffers."""

    s: torch.Tensor  # [B, V] fp32, the scaled scores between launches
    part_max: torch.Tensor  # [gemv tiles, B] fp32, each tile's max of s
    part_amax_val: torch.Tensor  # [gemv tiles, B] fp32, raw max
    part_amax_idx: torch.Tensor  # [gemv tiles, B] int32, its lowest id
    row_max: torch.Tensor  # [B] fp32
    row_argmax: torch.Tensor  # [B] int32
    select: QSelectScratch


def _sample_scratch(num_rows: int, vocab_size: int) -> SampleScratch:
    """The regions one `dspark_markov_sample` call of ``num_rows`` needs."""
    gemv_tiles = triton.cdiv(vocab_size, BLOCK_V)
    f32, i32 = torch.float32, torch.int32
    return SampleScratch(
        Region((num_rows, vocab_size), f32),
        Region((gemv_tiles, num_rows), f32),
        Region((gemv_tiles, num_rows), f32),
        Region((gemv_tiles, num_rows), i32),
        Region((num_rows,), f32),
        Region((num_rows,), i32),
        q_select_scratch(num_rows, vocab_size),
    )


def dspark_markov_sample_scratch(max_rows: int, vocab_size: int, device):
    """Scratch for any `dspark_markov_sample` call of up to ``max_rows`` rows,
    allocated once. One call runs at a time, so a block's positions share it."""
    return scratch_buffer(_sample_scratch(max_rows, vocab_size), device)


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
        "scratch",
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
    q_row_base: torch.Tensor,
    position: int,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    noise_offset: int,
    scratch: torch.Tensor,
) -> None:
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
        q_row_base:   [B]     int32; negative writes nothing.
        position:             host int, this block position.
        temperatures: [B]     fp32, clamped to ``SAMPLER_EPS`` from below as
                              `ModelRunner.prepare_sample` clamps it; a row
                              at ``SAMPLER_EPS`` is greedy (`Sampler`'s
                              predicate) and drafts its argmax.
        top_k:        [B]     int32, the request's top-k; 0 or -1 is off.
        top_p:        [B]     fp32, the request's top-p; 1.0 is off.
        seed:         [1]     int32 device scalar. Must hold the same value on
                              every TP rank: each rank runs this op over the
                              full vocab and must draw, and store, the same
                              ids.
        noise_offset:         host int, the Philox counter base for this block
                              position. Static (position * max_B * V), so it
                              does not make the launch data-dependent.
        scratch:              the byte buffer of a
                              `dspark_markov_sample_scratch`, which the call's
                              scaled scores and partials are carved from.

    Raises rather than falling back: an eager fallback would draw from the
    global RNG, which TP ranks do not share, and the drafts would fork.
    """
    check_markov_inputs(
        "dspark_markov_sample", base_logits, prev_ids, markov_w1, markov_w2, out
    )
    num_rows, vocab_size = base_logits.shape
    rank = markov_w2.shape[1]
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
    _check_rows("q_row_base", q_row_base, num_rows, torch.int32)
    _check_rows("temperatures", temperatures, num_rows, torch.float32)
    _check_rows("top_k", top_k, num_rows, torch.int32)
    _check_rows("top_p", top_p, num_rows, torch.float32)
    if base_logits.stride(1) != 1:
        raise ValueError("base_logits rows must be contiguous over the vocab axis")
    why = markov_weights_unsupported(markov_w1, markov_w2)
    if why is not None:
        raise ValueError(f"dspark_markov_sample cannot run: {why}")
    if num_rows == 0:
        return
    # Stage 1's GEMV keeps the greedy op's narrow V tile (its accumulator is the
    # MFMA one); the q stages tile wide, because they only stream.
    geometry = markov_launch_geometry(num_rows, vocab_size)
    gemv_tiles = geometry.v_tiles
    work = carve(scratch, _sample_scratch(num_rows, vocab_size))

    _markov_sample_stage1[(gemv_tiles, geometry.row_tiles)](
        base_logits,
        prev_ids,
        markov_w1,
        markov_w2,
        work.part_amax_val,
        work.part_amax_idx,
        temperatures,
        q_row_base,
        work.s,
        work.part_max,
        num_rows,
        vocab_size,
        rank,
        stride_base_row=base_logits.stride(0),
        stride_prev=prev_ids.stride(0),
        stride_w1_vocab=markov_w1.stride(0),
        stride_w2_vocab=markov_w2.stride(0),
        stride_s_row=work.s.stride(0),
        min_temperature=SAMPLER_EPS,
        BLOCK_ROW=geometry.block_row,
        BLOCK_V=BLOCK_V,
        BLOCK_K=BLOCK_K,
        num_warps=geometry.num_warps,
        num_stages=2,
    )
    softmax_combine_kernel[(num_rows,)](
        work.part_max,
        None,
        work.row_max,
        None,
        work.part_amax_val,
        work.part_amax_idx,
        work.row_argmax,
        num_rows,
        gemv_tiles,
        BLOCK_TILE=triton.next_power_of_2(gemv_tiles),
        WITH_ARGMAX=True,
        WITH_INV_Z=False,
        num_warps=4,
        num_stages=1,
    )
    select_and_draw_q(
        work.s,
        work.row_max,
        work.row_argmax,
        q_row_base,
        temperatures,
        top_k,
        top_p,
        seed,
        out,
        q_ids,
        q_probs,
        q_meta,
        work.select,
        position,
        noise_offset,
    )
