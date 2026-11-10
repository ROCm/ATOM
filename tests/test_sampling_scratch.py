# SPDX-License-Identifier: MIT
"""`sampling_scratch`: named per-call views carved from one buffer allocated once."""

from typing import NamedTuple

import pytest
import torch

from atom.model_ops.spec_sampling.sampling_scratch import (
    SCRATCH_ALIGN,
    Region,
    carve,
    scratch_buffer,
)


class Inner(NamedTuple):
    c: object


class Spec(NamedTuple):
    a: object
    b: object
    inner: Inner


def spec(rows=2):
    return Spec(
        Region((rows, 3), torch.float32),
        Region((4,), torch.int32),
        Inner(Region((5,), torch.float32)),
    )


def test_views_come_back_named_in_field_order_with_their_shapes_and_dtypes():
    scratch = scratch_buffer(spec(), "cpu")
    views = carve(scratch, spec())
    assert isinstance(views, Spec) and isinstance(views.inner, Inner)
    assert views.a.shape == (2, 3) and views.a.dtype is torch.float32
    assert views.b.shape == (4,) and views.b.dtype is torch.int32
    assert views.inner.c.shape == (5,)
    base = scratch.data_ptr()
    assert views.a.data_ptr() == base
    assert views.b.data_ptr() == base + SCRATCH_ALIGN
    assert views.inner.c.data_ptr() == base + 2 * SCRATCH_ALIGN
    assert all(v.is_contiguous() for v in (views.a, views.b, views.inner.c))


@pytest.mark.parametrize("rows", range(1, 9))
def test_every_view_is_aligned_whatever_the_batch(rows):
    # Triton specializes on each pointer's 16-byte alignment: views whose
    # alignment moved with the batch would compile a kernel per pattern.
    scratch = scratch_buffer(spec(8), "cpu")
    views = carve(scratch, spec(rows))
    for view in (views.a, views.b, views.inner.c):
        assert (view.data_ptr() - scratch.data_ptr()) % SCRATCH_ALIGN == 0


def test_a_batch_larger_than_the_scratch_is_refused():
    scratch = scratch_buffer(spec(2), "cpu")
    with pytest.raises(ValueError, match="cannot hold"):
        carve(scratch, spec(200))
