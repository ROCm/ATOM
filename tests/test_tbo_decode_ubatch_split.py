# SPDX-License-Identifier: MIT
"""A TBO decode step's micro-batches are cut once (`decode_ubatch_request_ranges`):
the slices the wrapper runs, the per-ubatch token counts the DP sync agrees on,
and the requests every attention backend lays each ubatch's metadata out for
must all be that split. A backend that split the padded batch instead handed a
micro-batch's rows another request's context and KV slots."""

from types import SimpleNamespace

import pytest

from atom.utils.tbo.ubatch_splitting import (
    decode_ubatch_request_ranges,
    maybe_create_ubatch_slices,
)
from atom.utils.tbo.ubatching import _precompute_decode


@pytest.mark.parametrize("scheduled_bs", [3, 4, 5, 25, 29, 32])
@pytest.mark.parametrize("q", [1, 3])
def test_slices_and_dp_sync_cut_the_same_requests(scheduled_bs, q):
    ranges = decode_ubatch_request_ranges(scheduled_bs, 2)
    assert ranges[0][0] == 0 and ranges[-1][1] == scheduled_bs
    assert ranges[0][1] == ranges[1][0]

    slices = maybe_create_ubatch_slices(
        num_reqs=scheduled_bs, scheduled_tokens=scheduled_bs * q, num_ubatches=2
    )
    assert [(s.request_slice.start, s.request_slice.stop) for s in slices] == ranges
    assert [(s.token_slice.start, s.token_slice.stop) for s in slices] == [
        (start * q, end * q) for start, end in ranges
    ]

    batch = SimpleNamespace(
        total_seqs_num_decode=scheduled_bs, total_tokens_num=scheduled_bs * q
    )
    _, _, ub0, ub1 = _precompute_decode(batch)
    assert (ub0, ub1) == tuple((end - start) * q for start, end in ranges)


@pytest.mark.parametrize("scheduled_bs,running_bs", [(25, 32), (29, 32), (3, 4)])
def test_requests_fit_the_padded_halves_backends_lay_them_into(
    scheduled_bs, running_bs
):
    """Backends pad each ubatch to half the padded batch: the requests it
    holds always fit."""
    half = running_bs // 2
    widths = [half, running_bs - half]
    for (start, end), width in zip(
        decode_ubatch_request_ranges(scheduled_bs, 2), widths
    ):
        assert end - start <= width
