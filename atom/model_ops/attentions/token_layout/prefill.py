# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a prefill step's tokens sit in their own sequences.

Sequence `i` forwards tokens `[num_cached_tokens[i], context_lens[i])`, so the
axis is that ragged range flattened in sequence order -- the layout attention
reads through `cu_seqlens_q`. Everything per-sequence is an input: this file
derives nothing the step already knows. The slot each of these positions maps
to is `.slots`, shared with `.decode`.
"""

from __future__ import annotations

import numpy as np


def prefill_positions(
    token_offsets: np.ndarray,
    cached_lens: np.ndarray,
    cu_seqlens_q: np.ndarray,
    seqlens_q: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Each token's absolute index in its OWN sequence.

    Token `t` of sequence `i` is at `cached_lens[i] + (t - cu_seqlens_q[i])`, so
    a chunked prefill resumes at the cached prefix rather than at zero. The
    whole per-sequence part is one repeat, leaving the token axis read once and
    written once. `token_offsets` is the flat axis, `arange(sum(seqlens_q))`,
    passed in because the caller already owns a resident one.
    """
    starts = np.repeat(cached_lens - cu_seqlens_q[:-1], seqlens_q)
    return np.add(token_offsets, starts, out=out)
