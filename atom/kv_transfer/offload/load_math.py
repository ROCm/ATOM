# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How many tokens an offload tier moves for a request, in one place.

The offload scheduler decides loads with these functions after it allocated a
request, and DP routing (`OffloadTierProfile.loadable_tokens`) uses them to
predict what the same scheduler will load. Pure integer arithmetic, so the two
cannot drift apart.
"""

from __future__ import annotations


def chunk_floor(tokens: int, chunk: int) -> int:
    """`tokens` rounded down to a whole number of `chunk`s."""
    return (max(0, int(tokens)) // int(chunk)) * int(chunk)


def loadable_hit(hit: int, num_prompt: int | None, chunk: int) -> int:
    """A lookup `hit` as a length the tier can serve.

    A hit covering the whole prompt leaves nothing to compute, so it steps back
    one token; then it floors to a chunk (`retrieve` resolves the tier at chunk
    granularity). The floor comes second: stepping back from a chunk multiple
    walks off the boundary.
    """
    hit = int(hit)
    if num_prompt is not None and hit == int(num_prompt):
        hit -= 1
    return chunk_floor(hit, chunk)


def tier_load_tokens(hbm: int, loadable: int, chunk: int, min_load: int) -> int:
    """Tokens the tier moves for a request with `hbm` tokens cached locally and
    a servable tier hit of `loadable` (`loadable_hit`); 0 if the load is too
    small to be worth it.

    A load starts on a chunk boundary: from `hbm` when it is one, else from the
    next boundary (the tokens in between are recomputed first).
    """
    chunk = int(chunk)
    start = -(-int(hbm) // chunk) * chunk
    moved = int(loadable) - start
    return moved if moved > 0 and moved >= int(min_load) else 0
