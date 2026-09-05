# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The M3 decode path may only be handed a batch whose query lengths agree.

Its index-topk kernels recover a request from a flat query row as
``row // max_query_len`` and the row's causal cutoff as ``seq_len -
max_query_len + tok + 1``. Both are only true when every decode request
contributes exactly ``max_query_len`` rows.

Speculative decode breaks that: a request that joins the batch without draft
tokens contributes one row while its neighbours contribute ``num_spec + 1``,
and vLLM keeps all of them in the decode segment because each query length is
still within the reorder threshold. The batch then has ``total_q`` rows that no
longer divide by ``max_query_len`` -- which is how this surfaced in practice,
as ``total_q 121 not divisible by max_query_len 4`` after a few hundred gsm8k
requests, while short runs happened to stay uniform and passed.

So the property under test is not "does the assert fire" but "does the builder
recognise a ragged decode segment before the kernels see it".
"""

import pytest
import torch

pytest.importorskip("vllm")

from atom.plugin.vllm.attention.metadata import _uniform_decode_query_len


def _starts(query_lens):
    """cu_seqlens_q for the given per-request query lengths."""
    return torch.tensor(
        [0] + list(torch.tensor(query_lens).cumsum(0)), dtype=torch.int32
    )


@pytest.mark.parametrize("query_len", [1, 2, 4, 8])
def test_uniform_segment_reports_its_query_len(query_len):
    starts = _starts([query_len] * 6)
    assert _uniform_decode_query_len(starts, 6) == query_len


def test_single_request_is_uniform():
    assert _uniform_decode_query_len(_starts([4]), 1) == 4


@pytest.mark.parametrize(
    "query_lens",
    [
        [4, 4, 1, 4],  # a request joined without draft tokens
        [1, 4, 4, 4],  # ... at the head of the segment
        [4, 4, 4, 1],  # ... at the tail
        [4, 2, 4, 4],  # partially accepted drafts
    ],
)
def test_ragged_segment_is_rejected(query_lens):
    assert _uniform_decode_query_len(_starts(query_lens), len(query_lens)) is None


def test_ragged_segment_whose_total_still_divides_is_rejected():
    """total_q % max_query_len == 0 is not enough to make a segment uniform.

    [4, 4, 6, 2] sums to 16 == 4 * 4, so the kernel's own assert would pass and
    it would then read rows 8..11 as request 2 -- silently wrong instead of
    loud. The builder has to reject on the query lengths themselves.
    """
    query_lens = [4, 4, 6, 2]
    starts = _starts(query_lens)
    assert int(starts[-1]) % 4 == 0
    assert _uniform_decode_query_len(starts, len(query_lens)) is None


def test_only_the_decode_prefix_is_inspected():
    """Trailing prefill requests must not make a uniform decode segment look ragged."""
    starts = _starts([4, 4, 4, 512, 300])
    assert _uniform_decode_query_len(starts, 3) == 4


def test_no_decode_requests():
    assert _uniform_decode_query_len(_starts([512]), 0) is None
    assert _uniform_decode_query_len(None, 4) is None


def test_truncated_query_start_loc_is_rejected():
    """Fewer offsets than num_decodes+1 means the caller cannot be trusted."""
    assert _uniform_decode_query_len(_starts([4, 4]), 5) is None
