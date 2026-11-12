# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Same-proposal probabilistic Markov chain with one packed maximum per row.

Stage k+1 reads the winning sampled id directly from stage k's uint64 key.
The final kernel writes all ids and resets the keys for the next invocation.
No vocabulary partial arrays or separate per-position reduction launches are
needed. The bias dot product, fp32 q logits and Philox counter mapping match
``prob_markov_sample``; this does not change the proposal algorithm.

Only the full-vocabulary path is integrated. A sharded path still needs the
per-position inter-rank winner exchange and uses the existing implementation.
"""

from functools import cache

import torch
import triton
import triton.language as tl

from atom.model_ops import dspark_markov_stage1 as _markov_stage1
from atom.model_ops.dspark_prob_sample import (
    _NO_HIT,
    _draft_row_temps,
    _sampled_tile_argmax,
)

_BLOCK_K = _markov_stage1.BLOCK_K

# Batches above this many rows take 16-row tiles with two 32-column vocab
# tiles per program, which keeps many programs in flight; smaller batches
# keep the Markov stage-1 tiling, already a single row tile there.
_WIDE_MIN_ROWS = 16
_WIDE_BLOCK_ROW = 16
_WIDE_BLOCK_V = 32
_WIDE_TILES_PER_PROGRAM = 2


def _chain_launch(num_rows: int, vocab_size: int) -> _markov_stage1.MarkovGeometry:
    """`markov_launch_geometry`, with the wide override for larger batches."""
    if num_rows <= _WIDE_MIN_ROWS:
        return _markov_stage1.markov_launch_geometry(num_rows, vocab_size)
    return _markov_stage1.MarkovGeometry(
        block_row=_WIDE_BLOCK_ROW,
        row_tiles=triton.cdiv(num_rows, _WIDE_BLOCK_ROW),
        v_tiles=triton.cdiv(vocab_size, _WIDE_BLOCK_V * _WIDE_TILES_PER_PROGRAM),
        num_warps=2,
        block_v=_WIDE_BLOCK_V,
        tiles_per_program=_WIDE_TILES_PER_PROGRAM,
        num_stages=1,
    )


@triton.jit
def _pack_key(tile_max, tile_id):
    """uint64 key whose maximum is the (max score, lowest id) pair: the
    ordered fp32 bits are the high word and the inverted id the low word,
    so smaller ids win exact score ties. Zero is canonicalized so -0 and +0
    compare equal. Every valid score, -inf included, packs to a key above
    zero, which is what an empty row keeps."""
    tile_max = tl.where(tile_max == 0.0, 0.0, tile_max)
    bits = tile_max.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    return (ordered.to(tl.uint64) << 32) | (~tile_id.to(tl.uint32)).to(tl.uint64)


@triton.jit
def _prob_markov_chain_stage1(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    embed_out_ptr,
    q_out_ptr,
    temp_ptr,
    seed_ptr,
    winner_ptr,
    num_rows,
    vocab_size,  # columns of base_logits (this rank's shard)
    vocab_global,
    vocab_offset,  # global id of column 0
    pos_k,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    stride_q_row,
    RANK: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_V_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FROM_PACKED: tl.constexpr,
):
    """Native bias/q/RNG arithmetic (`_sampled_tile_argmax`, as
    `prob_markov_sample`); publish a packed per-row maximum. A program walks
    NUM_V_TILES consecutive vocab tiles and folds their keys in registers
    before a single atomic; the key maximum is order independent."""
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    row_mask = offs_row < num_rows
    greedy, inv_t, seed = _draft_row_temps(temp_ptr, seed_ptr, offs_row, row_mask)
    best = tl.zeros((BLOCK_ROW,), dtype=tl.uint64)
    for j in range(NUM_V_TILES):
        tile = tl.program_id(0) * NUM_V_TILES + j
        offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
        v_mask = offs_v < vocab_size
        tile_max, tile_idx = _sampled_tile_argmax(
            base_ptr,
            prev_ids_ptr,
            w1_ptr,
            w2_ptr,
            embed_out_ptr,
            q_out_ptr,
            tile,
            offs_row,
            offs_v,
            row_mask,
            v_mask,
            greedy,
            inv_t,
            seed,
            num_rows,
            vocab_global,
            vocab_offset,
            RANK,
            pos_k,
            stride_base_row,
            stride_prev,
            stride_w1_vocab,
            stride_embed_row,
            stride_w2_vocab,
            stride_q_row,
            BLOCK_ROW,
            BLOCK_V,
            BLOCK_K,
            FROM_PACKED,
        )
        # On gfx1250, native max ignores an all-NaN tile if a numeric tile
        # exists. Such a tile has no valid argmax index and contributes no
        # key; neither do a tile past the vocabulary and its padded lanes.
        key = _pack_key(tile_max, tile_idx + vocab_offset)
        best = tl.maximum(best, tl.where(tile_idx != _NO_HIT, key, 0))
    # The next launch on this stream sees the completed reduction. No CTA
    # spins or attempts a device-wide barrier inside this kernel. An empty
    # row retains zero and decodes to safe id 0.
    tl.atomic_max(
        winner_ptr + offs_row, best, mask=row_mask & (best != 0), sem="relaxed"
    )


@triton.jit
def _finish_prob_markov_chain(
    keys_ptr,
    out_ptr,
    saved_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    stride_key_pos,
    stride_out_row,
    stride_out_pos,
    stride_saved_row,
    stride_saved_pos,
    SAVE_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < B * T
    pos, row = i // B, i % B
    ptr = keys_ptr + pos * stride_key_pos + row
    key = tl.load(ptr, mask=mask, other=0)
    token = tl.where(key == 0, 0, ~key.to(tl.uint32))
    tl.store(out_ptr + row * stride_out_row + pos * stride_out_pos, token, mask=mask)
    if SAVE_IDS:
        tl.store(
            saved_ptr + row * stride_saved_row + pos * stride_saved_pos,
            token,
            mask=mask,
        )
    # All scoring kernels have completed before this launch. Each key has
    # exactly one owner here, so reset needs neither atomics nor a barrier.
    tl.store(ptr, tl.full((BLOCK,), 0, tl.uint64), mask=mask)


@cache
def _is_gfx1250(device):
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return arch.split(":")[0] == "gfx1250"


def can_use_prob_markov_chain(base_logits, markov_w1, markov_w2, keys):
    """Static metadata gate for the measured gfx1250 V4.1 workload."""
    if keys is None or base_logits.ndim != 3:
        return False
    b, t, v = base_logits.shape
    return (
        base_logits.is_cuda
        and base_logits.dtype == torch.bfloat16
        and b >= 1
        and t == 5
        and v == 129280
        and markov_w1.shape == markov_w2.shape == (v, 256)
        and markov_w1.dtype == markov_w2.dtype == torch.bfloat16
        and base_logits.stride(2) == 1
        and markov_w1.stride(1) == markov_w2.stride(1) == 1
        and keys.dtype == torch.uint64
        and keys.ndim == 2
        and keys.shape[0] >= t
        and keys.shape[1] >= b
        and keys.stride(1) == 1
        and markov_w1.device == markov_w2.device == keys.device == base_logits.device
        and _is_gfx1250(base_logits.device)
    )


def prob_markov_sample_chain(
    base_logits,
    prev_ids,
    markov_w1,
    markov_w2,
    out,
    q_out,
    temps,
    seed,
    embed_out,
    keys,
    saved_ids=None,
):
    """Draw a full-vocabulary sampled-prefix block with native RNG and q.

    ``keys`` is a dedicated, initially-zero uint64 [>=T, >=B] workspace.
    Allocate it once before graph capture, and use it on one ordered stream.
    It is zero again after every completed invocation, including replay;
    different graph batch sizes can therefore reuse the same allocation.
    Do not share it between simultaneous invocations on different streams.

    ``q_out`` remains fp32, before temperature scaling. ``saved_ids`` can
    point to the verifier's persistent id buffer, fusing its existing copy
    into the final write. All row/position strides are respected.
    """
    b, t, v = base_logits.shape
    rank = markov_w2.shape[1]
    assert b > 0 and t > 0 and v > 0
    assert markov_w1.shape == markov_w2.shape == (v, rank)
    assert markov_w1.dtype == markov_w2.dtype == torch.bfloat16
    assert base_logits.stride(2) == q_out.stride(2) == 1
    assert markov_w1.stride(1) == markov_w2.stride(1) == embed_out.stride(2) == 1
    assert q_out.dtype == torch.float32 and q_out.shape == (b, t, v)
    assert out.shape == (b, t) and out.dtype in (torch.int32, torch.int64)
    assert prev_ids.shape == (b,) and prev_ids.dtype in (torch.int32, torch.int64)
    assert embed_out.shape == (b, t, rank) and embed_out.dtype == markov_w1.dtype
    assert temps.shape == (b,) and temps.dtype == torch.float32
    assert seed.numel() == 1 and seed.dtype == torch.int64
    assert keys.dtype == torch.uint64 and keys.ndim == 2
    assert keys.shape[0] >= t and keys.shape[1] >= b and keys.stride(1) == 1
    if saved_ids is not None:
        assert saved_ids.shape == (b, t) and saved_ids.dtype in (
            torch.int32,
            torch.int64,
        )
    launch = _chain_launch(b, v)
    grid = (launch.v_tiles, launch.row_tiles)
    for k in range(t):
        previous = prev_ids if k == 0 else keys[k - 1]
        _prob_markov_chain_stage1[grid](
            base_logits[:, k],
            previous,
            markov_w1,
            markov_w2,
            embed_out[:, k],
            q_out[:, k],
            temps,
            seed,
            keys[k],
            b,
            v,
            v,
            0,
            k,
            stride_base_row=base_logits.stride(0),
            stride_prev=previous.stride(0),
            stride_w1_vocab=markov_w1.stride(0),
            stride_embed_row=embed_out.stride(0),
            stride_w2_vocab=markov_w2.stride(0),
            stride_q_row=q_out.stride(0),
            RANK=rank,
            BLOCK_ROW=launch.block_row,
            BLOCK_V=launch.block_v,
            NUM_V_TILES=launch.tiles_per_program,
            BLOCK_K=_BLOCK_K,
            FROM_PACKED=k > 0,
            num_warps=launch.num_warps,
            num_stages=launch.num_stages,
        )
    saved = out if saved_ids is None else saved_ids
    _finish_prob_markov_chain[(triton.cdiv(b * t, 256),)](
        keys,
        out,
        saved,
        b,
        t,
        keys.stride(0),
        out.stride(0),
        out.stride(1),
        saved.stride(0),
        saved.stride(1),
        SAVE_IDS=saved_ids is not None,
        BLOCK=256,
        num_warps=4,
    )
    return embed_out
