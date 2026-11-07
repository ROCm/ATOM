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

from atom.model_ops.dspark_markov_sample import (
    _BLOCK_K,
    _BLOCK_V,
    _MAX_BLOCK_ROW,
    _MIN_BLOCK_ROW,
)
from atom.model_ops.dspark_prob_sample import _GREEDY_T, _gumbel


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
    rank,
    pos_k,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    stride_q_row,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FROM_PACKED: tl.constexpr,
):
    """Native bias/q/RNG arithmetic; publish a packed per-row maximum."""
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    previous = tl.load(prev_ids_ptr + offs_row * stride_prev, mask=row_mask, other=0)
    if FROM_PACKED:
        prev = tl.where(previous == 0, 0, ~previous.to(tl.uint32)).to(tl.int64)
    else:
        prev = previous

    acc = tl.zeros((BLOCK_ROW, BLOCK_V), dtype=tl.float32)
    for k in range(tl.cdiv(rank, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < rank
        embed = tl.load(
            w1_ptr + prev[:, None] * stride_w1_vocab + offs_k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        if tile == 0:
            tl.store(
                embed_out_ptr + offs_row[:, None] * stride_embed_row + offs_k[None, :],
                embed,
                mask=row_mask[:, None] & k_mask[None, :],
            )
        w2 = tl.load(
            w2_ptr
            + (vocab_offset + offs_v[None, :]) * stride_w2_vocab
            + offs_k[:, None],
            mask=v_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(embed, w2, acc)

    mask2 = row_mask[:, None] & v_mask[None, :]
    base = tl.load(
        base_ptr + offs_row[:, None] * stride_base_row + offs_v[None, :],
        mask=mask2,
        other=0.0,
    )
    z = base.to(tl.float32) + acc
    # q's logits, pre-temperature: verification rebuilds q from exactly these.
    tl.store(
        q_out_ptr + offs_row[:, None] * stride_q_row + offs_v[None, :], z, mask=mask2
    )

    temp = tl.load(temp_ptr + offs_row, mask=row_mask, other=1.0)
    greedy = temp <= _GREEDY_T
    inv_t = 1.0 / tl.where(greedy, 1.0, temp)
    seed = tl.load(seed_ptr)
    noise_offs = (pos_k * num_rows + offs_row)[:, None] * vocab_global + (
        vocab_offset + offs_v
    )[None, :]
    g = _gumbel(seed, noise_offs)
    score = tl.where(greedy[:, None], z, z * inv_t[:, None] + g)
    vals = tl.where(v_mask[None, :], score, float("-inf"))
    tile_max = tl.max(vals, axis=1)
    cand = tl.where(
        (vals == tile_max[:, None]) & v_mask[None, :], offs_v[None, :], vocab_size
    )
    tile_idx = tl.min(cand, axis=1)
    tile_idx = tl.where(tile_idx < vocab_size, tile_idx + vocab_offset, vocab_size)
    # Ordered fp32 bits are the high word; inverse id makes smaller ids
    # win exact score ties. Canonicalize zero so -0 and +0 compare equally.
    # On gfx1250, native max ignores an all-NaN tile if a numeric tile
    # exists. Such a tile has no valid argmax index and contributes no key.
    tile_max = tl.where(tile_max == 0.0, 0.0, tile_max)
    bits = tile_max.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    key = (ordered.to(tl.uint64) << 32) | (~tile_idx.to(tl.uint32)).to(tl.uint64)
    # Every valid score, including -inf, has a key greater than zero.
    # The next launch on this stream sees the completed reduction. No CTA
    # spins or attempts a device-wide barrier inside this kernel.
    # The index mask also excludes a partial tile with only NaNs and padded
    # -inf lanes. An empty row retains zero and decodes to safe id 0.
    tl.atomic_max(
        winner_ptr + offs_row,
        key,
        mask=row_mask & (tile_idx < vocab_size),
        sem="relaxed",
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
        and 1 <= b <= 32
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
    block_row = min(_MAX_BLOCK_ROW, max(_MIN_BLOCK_ROW, triton.next_power_of_2(b)))
    num_tiles = triton.cdiv(v, _BLOCK_V)
    for k in range(t):
        previous = prev_ids if k == 0 else keys[k - 1]
        _prob_markov_chain_stage1[(num_tiles, triton.cdiv(b, block_row))](
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
            rank,
            k,
            stride_base_row=base_logits.stride(0),
            stride_prev=previous.stride(0),
            stride_w1_vocab=markov_w1.stride(0),
            stride_embed_row=embed_out.stride(0),
            stride_w2_vocab=markov_w2.stride(0),
            stride_q_row=q_out.stride(0),
            BLOCK_ROW=block_row,
            BLOCK_V=_BLOCK_V,
            BLOCK_K=_BLOCK_K,
            FROM_PACKED=k > 0,
            num_warps=8 if block_row >= 32 else 4,
            num_stages=2,
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
