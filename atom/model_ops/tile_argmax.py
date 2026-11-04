# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The lowest-id tile argmax every argmax and race kernel shares: the greedy
DSpark draft (`dspark_markov_argmax`) and the lossless sampling kernels
(`spec_sampling`), so their picks agree bit for bit."""

import triton
import triton.language as tl


@triton.jit
def argmax_lowest_id(key, ids, mask, sentinel):
    """``(max, lowest id holding it)`` of a ``[BLOCK]`` or ``[ROWS, BLOCK]``
    tile along its last axis, over the lanes ``mask`` keeps.

    The tie rule ``torch.argmax`` and ``topk_select(..., tie="low")`` make. A
    tile no lane of which equals its max (masked out throughout, or all NaN,
    since NaN equals nothing) answers ``sentinel``; callers pick one past
    every real id and map it back into range themselves.
    """
    best = tl.max(key, axis=-1)
    if len(key.shape) == 1:
        hit = key == best
    else:
        hit = key == best[:, None]
    return best, tl.min(tl.where(hit & mask, ids, sentinel), axis=-1)
