# SPDX-License-Identifier: MIT
"""`sampling_scratch`: per-call views carved from buffers allocated once."""

import pytest
import torch

from atom.model_ops.sampling_scratch import SamplingScratch, take


def test_views_are_laid_out_in_order_with_the_requested_shapes():
    scratch = SamplingScratch(20, 10, "cpu")
    (a, b), (c,) = take([(2, 3), (4,)], [(5,)], "cpu", scratch.f32, scratch.i32)
    assert a.shape == (2, 3) and b.shape == (4,) and c.shape == (5,)
    assert a.dtype is torch.float32 and c.dtype is torch.int32
    assert a.data_ptr() == scratch.f32.data_ptr()
    # 6 elements, then up to the next 16-byte boundary.
    assert b.data_ptr() == scratch.f32.data_ptr() + 8 * 4
    assert a.is_contiguous() and b.is_contiguous()


@pytest.mark.parametrize("rows", range(1, 9))
def test_every_view_is_16_byte_aligned_whatever_the_batch(rows):
    # Triton specializes on each pointer's 16-byte alignment: views whose
    # alignment moved with the batch would compile a kernel per pattern.
    f32 = [(rows,), (3, rows), (rows, 5)]
    i32 = [(rows,), (rows, 7), (2,)]
    scratch = SamplingScratch.for_shapes(
        [(8,), (3, 8), (8, 5)], [(8,), (8, 7), (2,)], "cpu"
    )
    fv, iv = take(f32, i32, "cpu", scratch.f32, scratch.i32)
    for view, base in [(v, scratch.f32) for v in fv] + [(v, scratch.i32) for v in iv]:
        assert (view.data_ptr() - base.data_ptr()) % 16 == 0


def test_no_buffer_allocates_per_call():
    (a,), (b,) = take([(3,)], [(2, 2)], "cpu")
    assert a.shape == (3,) and b.shape == (2, 2)


def test_a_batch_larger_than_the_scratch_is_refused():
    scratch = SamplingScratch(5, 5, "cpu")
    with pytest.raises(ValueError, match="sized for a smaller batch"):
        take([(2, 3)], [], "cpu", scratch.f32, scratch.i32)
