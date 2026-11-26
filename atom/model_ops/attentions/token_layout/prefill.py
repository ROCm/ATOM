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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrefillBatch:
    """A prefill batch as its metadata reads it: request `i` forwards the
    `seqlens_q[i]` tokens of its sequence ending at `context_lens[i]`, paged
    by `block_tables[i]`. `is_dummy_run` marks a warmup's, which no request
    runs."""

    context_lens: np.ndarray
    seqlens_q: np.ndarray
    block_tables: Sequence[Sequence[int]]
    is_dummy_run: bool = False

    @classmethod
    def padding(cls, tokens: int, is_dummy_run: bool) -> PrefillBatch:
        """A prefill of ``tokens`` padding tokens and no request (an idle DP
        rank's in its peers' prefill), run as one request of its own over no
        page: its metadata describes a request, as every kernel's takes one."""
        lens = np.asarray([tokens], dtype=np.int32)
        return cls(
            context_lens=lens,
            seqlens_q=lens,
            block_tables=[],
            is_dummy_run=is_dummy_run,
        )

    @property
    def scheduled_bs(self) -> int:
        return len(self.seqlens_q)

    @property
    def scheduled_tokens(self) -> int:
        return int(self.seqlens_q.sum())

    def head(self, count: int) -> PrefillBatch:
        """Its first ``count`` requests."""
        return PrefillBatch(
            context_lens=self.context_lens[:count],
            seqlens_q=self.seqlens_q[:count],
            block_tables=self.block_tables[:count],
            is_dummy_run=self.is_dummy_run,
        )

    @property
    def cached_lens(self) -> np.ndarray:
        """Each request's tokens already in its KV cache: where it resumes."""
        return self.context_lens - self.seqlens_q

    def cut(
        self, requests: slice, tokens: slice, cu_seqlens_q: np.ndarray
    ) -> PrefillBatch:
        """The batch of the flat axis' ``tokens`` (laid out by
        `cu_seqlens_q`), over ``requests``, the requests they fall in -- a
        TBO micro-batch. A request the cut halves forwards its share, resuming
        where the tokens before the cut end: its context stops at the cut's
        end."""
        bounds = np.clip(
            cu_seqlens_q[requests.start : requests.stop + 1], tokens.start, tokens.stop
        )
        later = cu_seqlens_q[requests.start + 1 : requests.stop + 1] - bounds[1:]
        return PrefillBatch(
            context_lens=(self.context_lens[requests] - later).astype(np.int32),
            seqlens_q=np.diff(bounds).astype(np.int32),
            block_tables=self.block_tables[requests],
            is_dummy_run=self.is_dummy_run,
        )


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
