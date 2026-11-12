# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""How a replayed eager region hands its result to the next segment, on the
CPU: a region running a prefill's scheduled rows writes them straight into the
tensor the next segment reads and zeroes only that tensor's pad rows; outside
a replay it pads as before. The recordings themselves are covered on a GPU in
`test_region_graph.py`."""

import pytest
import torch

from atom.utils import region_graph
from atom.utils.region_graph import _EagerStep, on_scheduled_rows

BUCKET, ROWS, WIDTH = 16, 11, 8


@pytest.fixture
def delivery():
    """Stands in for `_EagerStep`: the tensor a replayed region delivers to,
    full of NaN as recycled graph-pool memory may be."""
    target = torch.full((BUCKET, WIDTH), float("nan"))
    region_graph._local.delivery = target
    yield target
    region_graph._local.delivery = None


def _bucket():
    return torch.randn(BUCKET, WIDTH)


def _claimed() -> bool:
    return getattr(region_graph._local, "delivery", None) is None


def test_scheduled_rows_are_written_into_the_delivery_target(delivery):
    x = _bucket()
    seen = {}

    def double(h, out=None):
        seen["out"] = out
        return torch.mul(h, 2.0, out=out)

    got = on_scheduled_rows(double, ROWS, x, writes_out=True)

    assert got is delivery and _claimed()
    # fn wrote the target's scheduled rows itself: nothing to copy.
    assert seen["out"].data_ptr() == delivery.data_ptr()
    assert seen["out"].shape == (ROWS, WIDTH)
    torch.testing.assert_close(got[:ROWS], x[:ROWS] * 2.0, rtol=0, atol=0)
    # Only the pad rows are zeroed, NaN gone.
    assert torch.equal(got[ROWS:], torch.zeros(BUCKET - ROWS, WIDTH))


def test_a_result_returned_elsewhere_is_copied_into_the_target(delivery):
    x = _bucket()
    got = on_scheduled_rows(lambda h: h.cumsum(0), ROWS, x)

    assert got is delivery and _claimed()
    torch.testing.assert_close(got[:ROWS], x[:ROWS].cumsum(0), rtol=0, atol=0)
    assert torch.equal(got[ROWS:], torch.zeros(BUCKET - ROWS, WIDTH))


def test_writes_out_falls_back_to_a_copy_when_fn_ignores_out(delivery):
    x = _bucket()
    got = on_scheduled_rows(lambda h, out=None: h + 1.0, ROWS, x, writes_out=True)

    assert got is delivery
    torch.testing.assert_close(got[:ROWS], x[:ROWS] + 1.0, rtol=0, atol=0)
    assert torch.equal(got[ROWS:], torch.zeros(BUCKET - ROWS, WIDTH))


def test_a_full_bucket_writes_the_whole_target_and_zeroes_nothing(delivery):
    x = _bucket()
    seen = {}

    def double(h, out=None):
        seen["out"] = out
        return torch.mul(h, 2.0, out=out)

    got = on_scheduled_rows(double, None, x, writes_out=True)

    assert got is delivery and seen["out"] is delivery
    torch.testing.assert_close(got, x * 2.0, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [ROWS, None])
def test_outside_a_replay_the_region_pads_as_before(rows):
    x = _bucket()
    seen = {}

    def double(h, out=None):
        seen["out"] = out
        return h * 2.0

    got = on_scheduled_rows(double, rows, x, writes_out=True)

    assert seen["out"] is None  # no target: fn allocates its own result
    n = BUCKET if rows is None else rows
    torch.testing.assert_close(got[:n], x[:n] * 2.0, rtol=0, atol=0)
    assert got.shape == (BUCKET, WIDTH)
    assert torch.equal(got[n:], torch.zeros(BUCKET - n, WIDTH))


def test_a_target_of_another_height_is_not_claimed(delivery):
    x = torch.randn(BUCKET // 2, WIDTH)
    got = on_scheduled_rows(lambda h: h * 2.0, ROWS // 2, x)

    assert got is not delivery and not _claimed()
    assert got.shape == x.shape


def _step(fn, target):
    return _EagerStep(fn, {"x": _bucket()}, None, None, target)


def test_a_replayed_step_reads_a_claimed_target_without_a_copy(monkeypatch):
    target = torch.full((BUCKET, WIDTH), float("nan"))
    copies = []
    monkeypatch.setattr(
        torch.Tensor, "copy_", lambda self, src: copies.append(1) or self
    )

    def region(x):
        return on_scheduled_rows(
            lambda h, out=None: torch.mul(h, 2.0, out=out), ROWS, x, writes_out=True
        )

    _step(region, target)()
    assert not copies
    assert torch.equal(target[ROWS:], torch.zeros(BUCKET - ROWS, WIDTH))
    assert not torch.isnan(target).any()
    assert getattr(region_graph._local, "delivery", None) is None


def test_a_step_still_copies_the_result_of_a_region_that_does_not_claim():
    target = torch.zeros(BUCKET, WIDTH)
    step = _step(lambda x: x * 3.0, target)
    step()
    torch.testing.assert_close(target, step._arguments["x"] * 3.0, rtol=0, atol=0)


def test_a_region_that_claims_the_target_must_return_it():
    target = torch.zeros(BUCKET, WIDTH)

    def region(x):
        on_scheduled_rows(lambda h: h, None, x)  # claims, then drops it
        return x * 2.0

    with pytest.raises(RuntimeError, match="claimed"):
        _step(region, target)()
    assert getattr(region_graph._local, "delivery", None) is None
