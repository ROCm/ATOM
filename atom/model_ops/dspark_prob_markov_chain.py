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

import os
from functools import cache

import torch


def _fits_int32_offsets(tensor):
    # Triton forms the element offsets in int32 before adding them to a
    # pointer. The tensor's storage offset is already part of that pointer.
    strides = tensor.stride()
    return (
        all(s >= 0 for s in strides)
        and sum((n - 1) * s for n, s in zip(tensor.shape, strides)) <= 2**31 - 1
    )


@cache
def _supports_packed_chain(device):
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    arch = arch.split(":")[0]
    # Dedicated PR2877 CI branch only: numerical tests gate this experiment
    # before the existing gfx950 runners measure the fused implementation.
    return arch == "gfx1250" or (
        arch == "gfx950" and os.getenv("ATOM_PR2877_TEST_GFX950", "0") == "1"
    )


def can_use_prob_markov_chain(base_logits, markov_w1, markov_w2, keys):
    """Static metadata gate; gfx950 is opt-in on the PR2877 CI branch."""
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
        and keys.stride(0) >= b
        and markov_w1.device == markov_w2.device == keys.device == base_logits.device
        and all(
            _fits_int32_offsets(tensor)
            for tensor in (base_logits, markov_w1, markov_w2, keys)
        )
        and _supports_packed_chain(base_logits.device)
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
    assert base_logits.ndim == 3 and markov_w2.ndim == 2
    b, t, v = base_logits.shape
    rank = markov_w2.shape[1]
    assert b > 0 and t > 0 and v > 0
    assert markov_w1.shape == markov_w2.shape == (v, rank)
    assert base_logits.dtype == torch.bfloat16
    assert markov_w1.dtype == markov_w2.dtype == torch.bfloat16
    assert q_out.dtype == torch.float32 and q_out.shape == (b, t, v)
    assert out.shape == (b, t) and out.dtype in (torch.int32, torch.int64)
    assert prev_ids.shape == (b,) and prev_ids.dtype in (torch.int32, torch.int64)
    assert embed_out.shape == (b, t, rank) and embed_out.dtype == markov_w1.dtype
    assert temps.shape == (b,) and temps.dtype == torch.float32
    assert temps.stride(0) == 1
    assert seed.numel() == 1 and seed.dtype == torch.int64
    assert keys.dtype == torch.uint64 and keys.ndim == 2
    assert keys.shape[0] >= t and keys.shape[1] >= b and keys.stride(1) == 1
    assert keys.stride(0) >= b
    assert base_logits.stride(2) == q_out.stride(2) == 1
    assert markov_w1.stride(1) == markov_w2.stride(1) == embed_out.stride(2) == 1
    if saved_ids is not None:
        assert saved_ids.shape == (b, t) and saved_ids.dtype in (
            torch.int32,
            torch.int64,
        )
    tensors = (
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
    ) + (() if saved_ids is None else (saved_ids,))
    assert all(tensor.device == base_logits.device for tensor in tensors)
    assert all(_fits_int32_offsets(tensor) for tensor in tensors)

    # Keep the metadata gate importable in CPU CI without Triton or AITER.
    import triton

    from atom.model_ops.dspark_prob_sample import (
        _BLOCK_K,
        _BLOCK_V,
        _MAX_BLOCK_ROW,
        _MIN_BLOCK_ROW,
        _finish_prob_markov_chain,
        _prob_markov_stage1,
    )

    block_row = min(_MAX_BLOCK_ROW, max(_MIN_BLOCK_ROW, triton.next_power_of_2(b)))
    num_tiles = triton.cdiv(v, _BLOCK_V)
    for k in range(t):
        previous = prev_ids if k == 0 else keys[k - 1]
        _prob_markov_stage1[(num_tiles, triton.cdiv(b, block_row))](
            base_logits[:, k],
            previous,
            markov_w1,
            markov_w2,
            embed_out[:, k],
            q_out[:, k],
            temps,
            seed,
            keys[k],
            keys[k],  # unused partial-index pointer in packed mode
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
            PACKED_CHAIN=True,
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
