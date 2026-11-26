# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass

import numpy as np

# The micro-batches a TBO step splits into.
NUM_TBO_UBATCHES = 2


@dataclass
class UBatchSlice:
    """Describes which portion of a batch belongs to a micro-batch."""

    request_slice: slice  # the step's requests this micro-batch runs
    # The running tokens it runs: the micro-batches lay out end to end, none
    # overlapping.
    token_slice: slice
    # The rows of `token_slice` its requests' tokens fill, the rest padding:
    # its first rows (`cut_decode_ubatches`, `cut_prefill_ubatches`),
    # or all of them.
    scheduled_token_slice: slice | None = None
    # The requests its per-request metadata spans: its share of the step's
    # `running_bs` (`cut_decode_ubatches`), or `request_slice`'s.
    running_bs: int | None = None

    def __post_init__(self):
        if self.scheduled_token_slice is None:
            self.scheduled_token_slice = self.token_slice
        if self.running_bs is None:
            self.running_bs = self.request_slice.stop - self.request_slice.start

    @property
    def running_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start

    @property
    def scheduled_tokens(self) -> int:
        scheduled = self.scheduled_token_slice
        return scheduled.stop - scheduled.start

    def scheduled_bs(self, step_scheduled_bs: int) -> int:
        """This micro-batch's scheduled requests: the step's first
        `step_scheduled_bs` falling in `request_slice`, the rest padding."""
        rs = self.request_slice
        return max(0, min(rs.stop, step_scheduled_bs) - rs.start)


def decode_bucket_splits(bs: int) -> bool:
    """Whether a decode batch of ``bs`` requests is split into micro-batches.

    One threshold for both places a decode split is decided: a step's
    `running_bs` (`ForwardMode.decide`) and a recorded request bucket
    (`ModelRunner.capture_cudagraph`, which records a bucket that splits as a
    TBO graph). Such a bucket is even (`Config.capture_sizes`), its two halves
    as many requests."""
    return bs > 2


def cut_decode_ubatches(
    running_bs: int, running_tokens: int, lengths: np.ndarray
) -> tuple[UBatchSlice, ...]:
    """A split decode's two micro-batches, cut by request as its graphs are
    recorded (`ReqGraph`). Cut once a step (`ModelRunner._cut_ubatches`, or
    the capture of a TBO graph) and handed to every consumer.

    One cut through both of the step's axes, the same on every step of the
    bucket: its `running_bs` requests in half, the first taking the first
    half of this rank's scheduled requests (``lengths``, their query tokens),
    and its `running_tokens` in half. Each half runs its own requests' packed
    tokens first, then its share of the padding
    (`decode_ubatch_token_order`)."""
    scheduled_bs = len(lengths)
    first_bs = running_bs // 2
    split = min(first_bs, scheduled_bs)
    half = running_tokens // 2
    first_tokens = int(lengths[:split].sum())
    second_tokens = int(lengths[split:].sum())
    if first_tokens > half or second_tokens > running_tokens - half:
        raise ValueError(
            f"{first_tokens} and {second_tokens} tokens of {first_bs} and "
            f"{running_bs - first_bs} requests do not fit two micro-batches of "
            f"{half} and {running_tokens - half} tokens"
        )
    return (
        UBatchSlice(
            request_slice=slice(0, split),
            token_slice=slice(0, half),
            scheduled_token_slice=slice(0, first_tokens),
            running_bs=first_bs,
        ),
        UBatchSlice(
            request_slice=slice(split, scheduled_bs),
            token_slice=slice(half, running_tokens),
            scheduled_token_slice=slice(first_tokens, first_tokens + second_tokens),
            running_bs=running_bs - first_bs,
        ),
    )


def decode_ubatch_token_order(
    slices: tuple[UBatchSlice, ...],
    scheduled_tokens: int,
    token_order: np.ndarray,
) -> int:
    """The step's packed tokens in the order its micro-batches run them:
    ``token_order[t]`` is the packed token that running token ``t`` reads and
    writes back. Each micro-batch's `token_slice` takes its own tokens first,
    then its share of the packed padding past ``scheduled_tokens``: a
    permutation of the running tokens, so each packed token is written back
    once. Returns the running tokens."""
    padding = scheduled_tokens
    for ub in slices:
        start, scheduled = ub.token_slice.start, ub.scheduled_token_slice
        token_order[start : start + ub.scheduled_tokens] = np.arange(
            scheduled.start, scheduled.stop
        )
        pad = ub.running_tokens - ub.scheduled_tokens
        token_order[start + ub.scheduled_tokens : ub.token_slice.stop] = np.arange(
            padding, padding + pad
        )
        padding += pad
    return slices[-1].token_slice.stop


def prefill_token_halves(tokens: int) -> tuple[int, int] | None:
    """A prefill's ``tokens`` cut at the midpoint: each micro-batch's count,
    or None when fewer than two would leave one empty. The one cut a rank's
    vote (`prefill_tbo_vote`) gathers and its split (`cut_prefill_ubatches`)
    runs: were they two, the micro-batches' collectives would gather sizes the
    slices do not run."""
    if tokens < 2:
        return None
    half = tokens // 2
    return half, tokens - half


def cut_prefill_ubatches(
    num_reqs: int,
    num_scheduled_tokens: np.ndarray,
    running_tokens: int,
) -> list[UBatchSlice] | None:
    """A prefill's two micro-batches, its ``running_tokens`` cut at the
    midpoint (`prefill_token_halves`), through a request if one straddles it,
    so a single request still splits; None when it does not cut. Tokens past
    the requests' are padding and belong to no request.

    Whether the step splits at all is the group's decision (`prefill_tbo_vote`,
    OR-reduced): a rank below the min-token bar still splits once a peer
    cleared it, to keep the per-ubatch collectives size-aligned.
    """
    halves = prefill_token_halves(running_tokens)
    if halves is None:
        return None
    toks = np.asarray(num_scheduled_tokens[:num_reqs], dtype=np.int64)
    # Exclusive-prefix cumulative token offsets: cu[i] = first token of req i.
    cu = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(toks, out=cu[1:])

    slices: list[UBatchSlice] = []
    tok_start = 0
    for tok_end in (halves[0], running_tokens):
        # Requests overlapping [tok_start, tok_end):
        #   first req = the one containing tok_start
        #   last req  = the one containing tok_end - 1
        req_start = min(int(np.searchsorted(cu, tok_start, side="right") - 1), num_reqs)
        req_stop = min(int(np.searchsorted(cu, tok_end - 1, side="right")), num_reqs)
        slices.append(
            UBatchSlice(
                request_slice=slice(req_start, req_stop),
                token_slice=slice(tok_start, tok_end),
                scheduled_token_slice=slice(
                    tok_start, max(tok_start, min(tok_end, int(cu[-1])))
                ),
                # A half of nothing but padding still forwards one request.
                running_bs=max(req_stop - req_start, 1),
            )
        )
        tok_start = tok_end
    return slices
