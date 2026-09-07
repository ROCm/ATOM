# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The per-token index arrays a decode step has to derive.

The counterpart of `.prefill`, on a rectangular token axis: a decode step runs
the same `max_seqlen_q` rows for every sequence, so sequence `i` forwards
`[context_lens[i] - max_seqlen_q, context_lens[i])` and the flat axis is that
range concatenated in sequence order.

The slot each of these positions maps to is `.slots`, shared with `.prefill`:
the gather reads a position and a packed table, and neither cares which side
shaped the axis. What stays with the builders is the one-token step, which they
derive from `last_block_num_tokens` and the last block-table entry rather than
from a position at all, and around which the MLA one wraps a DCP variant.
"""

from __future__ import annotations

import numpy as np


def decode_positions(
    context_lens: np.ndarray, max_seqlen_q: int, out: np.ndarray | None = None
) -> np.ndarray:
    """Each token's absolute index in its OWN sequence.

    Broadcast rather than `tile(ramp) + repeat(start)`: the rectangle IS a 2-D
    shape, so writing it as one leaves the token axis touched once instead of
    three times (1.4-2.2x, `/app/logs_claude/tool/decode_positions_perf.py`).
    """
    bs = len(context_lens)
    out = np.empty(bs * max_seqlen_q, dtype=np.int32) if out is None else out
    np.add(
        (context_lens - max_seqlen_q)[:, None],
        np.arange(max_seqlen_q, dtype=np.int32),
        out=out.reshape(bs, max_seqlen_q),
    )
    return out
