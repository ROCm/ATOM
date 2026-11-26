# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sampled DSpark Markov step (`dspark_markov_sample`, the lossless draft).

The greedy op's stage-1 GEMV (`dspark_markov_stage1`), then ``x_k`` drawn from the
top K of ``softmax(logits_k / T)``, stored as q (`sparse_q_store`). q is 0 off
the stored set, so truncation costs acceptance, never exactness (`Q_WIDTH`).
Four launches per position: GEMV + 1/T into the fp32 ``s`` scratch with tile
partials; then histogram, compact and finalize (`sparse_q_select`), which
reduce the row max, the cut and the raw argmax from those partials and tables
themselves (finalize normalizes over the stored set). Past 16 rows, whole vocab
or TP, the softmax combine and the cut run as their own launches (six).
Fixed-order reductions, no atomics: TP ranks must store and draw alike.
CUDA-graph safe: seed, row bases and filters are device tensors; position and
noise offset are static host ints; no data-dependent grid.
Philox key high word 0; the verify's draws use other words (`ratio_verify`).
int64 addressing: ``[max_bs, T, V]`` logits and the store pass 2**31 elements.
`dspark_markov_sample_tp` is the same op over a TP-sharded LM head: the same
stored q and draw, with tile partials, histogram tables and stored entries
gathered across ranks instead of the logits (`sparse_q_select`).
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
    scratch_bytes,
)
from atom.model_ops.spec_sampling.sparse_q_select import (
    QSelectScratch,
    QSelectTPScratch,
    q_select_scratch,
    q_select_tp_scratch,
    select_and_draw_q,
    select_and_draw_q_tp,
)
from atom.model_ops.spec_sampling.sparse_q_store import store_width


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
    stride_part,
    vocab_offset,
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
    value.

    ``vocab_size`` ids from ``vocab_offset``: the whole vocab from 0, or one TP
    rank's shard of it (`dspark_markov_sample_tp`), whose published ids are
    global. The partials are ``[tiles, ...]`` with tiles ``stride_part`` apart
    (``num_rows`` unless they are interleaved for one gather).
    """
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
        vocab_offset,
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
        stride_part,
        vocab_offset,
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
        part_smax_ptr + tile * stride_part + offs_row,
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
    # [B] fp32, from the histogram up to 16 rows, the combine past that
    row_max: torch.Tensor
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
        q_select_scratch(num_rows, vocab_size),
    )


def dspark_markov_sample_scratch(
    max_rows: int, vocab_size: int, device, tp_size: int = 1
):
    """Scratch for any `dspark_markov_sample` call of up to ``max_rows`` rows,
    and with ``tp_size`` > 1 any `dspark_markov_sample_tp` call over that many
    shards too, allocated once. One call runs at a time, so a block's
    positions share it."""
    spec = _sample_scratch(max_rows, vocab_size)
    if tp_size > 1:
        sharded = _sample_tp_scratch(
            max_rows, vocab_size, tp_size, store_width(vocab_size)
        )
        if scratch_bytes(sharded) > scratch_bytes(spec):
            spec = sharded
    return scratch_buffer(spec, device)


def _check_store(q_ids, q_probs, q_meta):
    store_rows, width = q_ids.shape if q_ids.dim() == 2 else (0, 0)
    if (
        width < 1
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
    _check_store(q_ids, q_probs, q_meta)
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
        stride_part=num_rows,
        vocab_offset=0,
        min_temperature=SAMPLER_EPS,
        BLOCK_ROW=geometry.block_row,
        BLOCK_V=BLOCK_V,
        BLOCK_K=BLOCK_K,
        num_warps=geometry.num_warps,
        num_stages=2,
    )
    select_and_draw_q(
        work.s,
        work.part_max,
        work.part_amax_val,
        work.part_amax_idx,
        work.row_max,
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


class SampleTPScratch(NamedTuple):
    """One `dspark_markov_sample_tp` call's work buffers."""

    s: torch.Tensor  # [B, V / tp] fp32, this rank's scaled scores
    # [gemv tiles, 3, B] fp32: each tile's max of s, raw max and lowest id
    # (int32 bits), tile-major so that one gather is the whole row's tiles.
    parts: torch.Tensor
    # [B] fp32, from the histogram up to 16 rows, the combine past that
    row_max: torch.Tensor
    select: QSelectTPScratch


def _sample_tp_scratch(
    num_rows: int, vocab_size: int, world: int, width: int
) -> SampleTPScratch:
    """The regions one `dspark_markov_sample_tp` call of ``num_rows`` over a
    ``vocab_size`` vocab sharded ``world`` ways needs, for a store ``width``
    entries wide."""
    local_vocab = triton.cdiv(vocab_size, world)
    gemv_tiles = triton.cdiv(local_vocab, BLOCK_V)
    return SampleTPScratch(
        Region((num_rows, local_vocab), torch.float32),
        Region((gemv_tiles, 3, num_rows), torch.float32),
        Region((num_rows,), torch.float32),
        q_select_tp_scratch(num_rows, local_vocab, world, width),
    )


def dspark_markov_sample_tp(
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
    vocab_start: int,
    tp_group,
    use_custom: bool = True,
) -> None:
    """`dspark_markov_sample` over a TP-sharded vocab: the same draw and the
    same stored q, bit for bit, without gathering the logits.

    ``base_logits`` is this rank's ``[B, V / tp]`` slice of the LM head, ids
    ``vocab_start`` on; the Markov tables are whole, and so are the store and
    the noise counters (``noise_offset`` is the whole-vocab op's). ``scratch``
    is a `dspark_markov_sample_scratch` sized for this TP degree. Each rank
    runs stage 1 on its slice and the tile partials are gathered, so the row
    max and raw argmax are the whole row's; the selection then crosses ranks
    as `select_and_draw_q_tp` describes. Every column's score is the same
    ``r``-term dot product in the same K order whichever tile holds it, so it
    is the whole-vocab op's value. Per position, three gathers cross ranks:
    the ``[V / (128 tp), 3, B]`` partials, the ``[V / (4096 tp), B, 256]``
    histogram tables and the ``[2, B, K]`` staging pair.
    """
    if base_logits.dim() != 2 or markov_w1.dim() != 2 or markov_w2.dim() != 2:
        raise ValueError("dspark_markov_sample_tp expects 2-D base_logits/w1/w2")
    num_rows, local_vocab = base_logits.shape
    world = tp_group.world_size
    vocab_size = markov_w2.shape[0]
    rank = markov_w2.shape[1]
    if local_vocab * world != vocab_size or markov_w1.shape != markov_w2.shape:
        raise ValueError(
            f"base_logits holds {local_vocab} of {vocab_size} ids on each of "
            f"{world} ranks, or markov_w1 {tuple(markov_w1.shape)} does not match "
            f"markov_w2 {tuple(markov_w2.shape)}"
        )
    for name, ids in (("prev_ids", prev_ids), ("out", out)):
        if ids.shape != (num_rows,) or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"{name} {tuple(ids.shape)} {ids.dtype} is not a [{num_rows}] "
                "int32 or int64 id column"
            )
    _check_store(q_ids, q_probs, q_meta)
    _check_rows("q_row_base", q_row_base, num_rows, torch.int32)
    _check_rows("temperatures", temperatures, num_rows, torch.float32)
    _check_rows("top_k", top_k, num_rows, torch.int32)
    _check_rows("top_p", top_p, num_rows, torch.float32)
    if base_logits.stride(1) != 1:
        raise ValueError("base_logits rows must be contiguous over the vocab axis")
    why = markov_weights_unsupported(markov_w1, markov_w2)
    if why is not None:
        raise ValueError(f"dspark_markov_sample_tp cannot run: {why}")
    if num_rows == 0:
        return
    geometry = markov_launch_geometry(num_rows, local_vocab)
    gemv_tiles = geometry.v_tiles
    work = carve(
        scratch, _sample_tp_scratch(num_rows, vocab_size, world, q_ids.shape[1])
    )
    parts = work.parts
    _markov_sample_stage1[(gemv_tiles, geometry.row_tiles)](
        base_logits,
        prev_ids,
        markov_w1,
        markov_w2,
        parts[:, 1],
        parts[:, 2].view(torch.int32),
        temperatures,
        q_row_base,
        work.s,
        parts[:, 0],
        num_rows,
        local_vocab,
        rank,
        stride_base_row=base_logits.stride(0),
        stride_prev=prev_ids.stride(0),
        stride_w1_vocab=markov_w1.stride(0),
        stride_w2_vocab=markov_w2.stride(0),
        stride_s_row=work.s.stride(0),
        stride_part=parts.stride(0),
        vocab_offset=vocab_start,
        min_temperature=SAMPLER_EPS,
        BLOCK_ROW=geometry.block_row,
        BLOCK_V=BLOCK_V,
        BLOCK_K=BLOCK_K,
        num_warps=geometry.num_warps,
        num_stages=2,
    )
    # Rank-major is id order, so the gathered partials are the whole row's
    # tiles, still `stride_part` apart.
    gathered = tp_group.all_gather(parts, use_custom=use_custom, dim=0)
    select_and_draw_q_tp(
        work.s,
        gathered[:, 0],
        gathered[:, 1],
        gathered[:, 2].view(torch.int32),
        work.row_max,
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
        vocab_size,
        vocab_start,
        tp_group,
        use_custom,
    )
