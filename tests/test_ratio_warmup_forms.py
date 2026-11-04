# SPDX-License-Identifier: MIT
"""The ratio verify's startup warmup covers every filter variant a caller can
hand over: `warmup_forms` against `row_filters` on the host, no GPU."""

import itertools

import numpy as np
import pytest
import torch

pytest.importorskip("triton")

from atom.model_ops.spec_sampling.ratio_verify import (
    filter_variant,
    warmup_forms,
)
from atom.model_ops.spec_sampling.sampling_filter import row_filters


def caller_top_ks(bs):
    """Every kind of top-k `prepare_sample` or a direct caller hands over,
    at several values, with and without its host copy."""
    yield None, None
    for k in (1, 20, 256, 257, 5000):
        yield k, None
    for values in ([20] * bs, [300] * bs, [20, -1, 300][:bs], [-1, 5000, 7][:bs]):
        host = np.array(values, np.int32)
        yield torch.tensor(values, dtype=torch.int32), host
        yield torch.tensor(values, dtype=torch.int32), None


def caller_top_ps(bs):
    yield None
    yield from (0.5, 0.95, 1.0)
    yield torch.full((bs,), 0.9)
    yield torch.tensor([0.9, 1.0, 0.5][:bs])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bs", [1, 3])
def test_every_filter_form_a_caller_hands_over_is_warmed(dtype, bs):
    warmed = {
        filter_variant(row_filters(k, p, 1, dtype, host))
        for k, p, host in warmup_forms(dtype, "cpu")
    }
    for (k, host), p in itertools.product(caller_top_ks(bs), caller_top_ps(bs)):
        variant = filter_variant(row_filters(k, p, bs, dtype, host))
        assert variant in warmed, (k, host, p)


def test_a_scalar_k_beside_a_per_request_p_is_warmed():
    """The form a uniform top-k with per-request top-p takes: counted, k one
    value, p per row."""
    forms = warmup_forms(torch.bfloat16, "cpu")
    assert any(
        isinstance(k, int) and k <= 256 and isinstance(p, torch.Tensor)
        for k, p, _ in forms
    )


def test_a_per_request_k_with_no_countable_row_skips_the_count():
    rows = torch.tensor([1000, -1], dtype=torch.int32)
    filters = row_filters(rows, None, 2, torch.bfloat16, np.array([1000, -1]))
    assert not filters.counts and filters.descends
    # Without the host values the count is assumed.
    assert row_filters(rows, None, 2, torch.bfloat16).counts
