# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
TBO prefill token-split — shared straddle geometry.
=====================================================================

This module backs the ``ATOM_TBO_PREFILL_TOKEN_SPLIT`` path: when TBO splits
a prefill batch at the exact *token* midpoint (rather than on a request
boundary), the cut can land **inside a single request**. The request is then
processed across two micro-batches:

    req R, tokens [0 ............ M ............ L)
                  |---- ubatch 0 ----|--- ubatch 1 ---|
                         (prefix)          (this ubatch)

"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "StraddleSplitInfo",
    "TokenSplitPrefillState",
    "compute_straddle_split_info",
]


@dataclass
class TokenSplitPrefillState:
    block_tables: list
    cu_tokens: np.ndarray
    num_cached: Optional[np.ndarray]


@dataclass
class StraddleSplitInfo:
    """Geometry of a token-midpoint prefill cut for one micro-batch.

    Attributes:
        is_straddling: True iff this ubatch's first request was cut from a
            previous ubatch (i.e. ``prefix_len > 0``). When False the ubatch
            starts on a clean request boundary and no prefix attach is needed.
        first_req: Index (into the full prefill batch) of this ubatch's first
            request — the only request that can carry a straddle prefix.
        ub_num_reqs: Number of requests in this ubatch.
        ub_num_tokens: Number of *new* tokens in this ubatch.
        prefix_len: Tokens of ``first_req`` already processed (and KV-cached)
            by the previous ubatch. Zero when not straddling.
        req_global_start: Global token offset where ``first_req`` began.
    """

    is_straddling: bool
    first_req: int
    ub_num_reqs: int
    ub_num_tokens: int
    prefix_len: int
    req_global_start: int


def compute_straddle_split_info(cu_tokens, ub_slice) -> StraddleSplitInfo:
    """Compute the token-midpoint straddle geometry for one micro-batch.

    Args:
        cu_tokens: Exclusive-prefix cumulative q-seqlens over the full prefill
            batch (``cu_tokens[i]`` = global first-token offset of request i).
            Accepts a numpy array or any indexable of ints.
        ub_slice: The :class:`UBatchSlice` describing this micro-batch's
            request/token spans.

    Returns:
        A :class:`StraddleSplitInfo`. Check ``.is_straddling`` first; the other
        fields are always populated but only meaningful when straddling.
    """
    rs = ub_slice.request_slice
    ts = ub_slice.token_slice
    first_req = rs.start
    req_global_start = int(cu_tokens[first_req])
    prefix_len = ts.start - req_global_start
    return StraddleSplitInfo(
        is_straddling=prefix_len > 0,
        first_req=first_req,
        ub_num_reqs=rs.stop - rs.start,
        ub_num_tokens=ts.stop - ts.start,
        prefix_len=int(prefix_len),
        req_global_start=req_global_start,
    )
