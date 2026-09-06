# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The map between the token axis and the sequence axis.

Every other per-token array in this package is indexed by token; almost every
quantity the scheduler hands over is indexed by sequence. This is the one thing
that converts between them, so a kernel can resolve `per_seq[batch_id[t]]`.

It is its own module because it is the only piece here that both sides of the
forward need -- prefill builds it from the chunk's token counts, decode from a
rectangular one-row-per-sequence step -- and because a second transcription of
it goes wrong silently: a wrong id does not fault, it reads another request's
row.
"""

from __future__ import annotations

import numpy as np


def batch_id_per_token(
    seqlens_q: np.ndarray,
    pad_to: int | None = None,
    pad: int = -1,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Which sequence each token of the flat token axis belongs to.

    Sequence `i` contributes `seqlens_q[i]` consecutive entries of value `i`,
    in sequence order -- the layout attention reads through `cu_seqlens_q`.

    `pad_to` widens the result to a CUDAGraph bucket, filling the tail with
    `pad`. The tail is NOT sequence 0: a captured step runs at the bucket width
    whatever the batch, so a fabricated token naming a real sequence would have
    every consumer resolve it to that request's row and quietly attend on its
    behalf. `-1` is what the kernels bail on.
    """
    total = int(seqlens_q.sum())
    width = total if pad_to is None else pad_to
    # Asserted, not raised: numpy refuses the write below on its own, so this
    # only trades its message for a better one. The dtype guard in `.prefill`
    # is a `raise` because there numpy would answer, wrongly and in silence.
    assert width >= total, f"pad_to={pad_to} is under the {total} tokens scheduled"
    out = np.empty(width, dtype=np.int32) if out is None else out[:width]
    out[:total] = np.repeat(np.arange(len(seqlens_q), dtype=np.int32), seqlens_q)
    out[total:] = pad
    return out
