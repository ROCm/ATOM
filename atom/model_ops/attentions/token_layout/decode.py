# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The per-token index arrays a decode step has to derive.

The counterpart of `.prefill`: sequence `i` forwards its last `lengths[i]`
context tokens, `[context_lens[i] - lengths[i], context_lens[i])`, and the flat
axis is those ranges concatenated in sequence order (`cu_seqlens_q`).

The slot each of these positions maps to is `.slots`, shared with `.prefill`:
the gather reads a position and a packed table, and neither cares which side
shaped the axis. What stays with the builders is the one-token step, one
position per row computed inline from the ragged rows, around which the MLA one
wraps a DCP variant.
"""

from __future__ import annotations

import numpy as np


def decode_positions(
    context_lens: np.ndarray,
    lengths: np.ndarray,
    cu_seqlens_q: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Each token's absolute index in its OWN sequence, for sequences of query
    ``lengths`` laid out at ``cu_seqlens_q``."""
    total = int(cu_seqlens_q[len(lengths)])
    out = np.empty(total, dtype=np.int32) if out is None else out[:total]
    np.add(
        np.arange(total, dtype=np.int32),
        np.repeat(context_lens - lengths - cu_seqlens_q[: len(lengths)], lengths),
        out=out,
    )
    return out


def query_is_uniform(lengths: np.ndarray, max_seqlen_q: int) -> bool:
    """Whether every scheduled sequence runs ``max_seqlen_q`` query tokens: the
    one shape a decode kernel taking a single query length a sequence reads."""
    return bool((lengths == max_seqlen_q).all())


def ragged_decode_refusal(
    owner: str, lengths: np.ndarray, max_seqlen_q: int
) -> ValueError:
    """The error refusing a decode step whose sequences run different query
    ``lengths`` on ``owner``, a kernel that lays each sequence's queries
    ``max_seqlen_q`` rows apart: run, it would read the wrong rows."""
    return ValueError(
        f"{owner} decodes one query length a sequence; this decode step's are "
        f"{lengths.tolist()} against max_seqlen_q={max_seqlen_q}"
    )
