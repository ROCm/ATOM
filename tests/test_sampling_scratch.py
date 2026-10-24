# SPDX-License-Identifier: MIT
"""`sampling_scratch`: per-call views carved from buffers allocated once."""

import pytest
import torch

from atom.model_ops.sampling_scratch import SamplingScratch, take


def test_views_are_laid_out_back_to_back_with_the_requested_shapes():
    scratch = SamplingScratch(20, 10, "cpu")
    (a, b), (c,) = take([(2, 3), (4,)], [(5,)], "cpu", scratch.f32, scratch.i32)
    assert a.shape == (2, 3) and b.shape == (4,) and c.shape == (5,)
    assert a.dtype is torch.float32 and c.dtype is torch.int32
    assert a.data_ptr() == scratch.f32.data_ptr()
    assert b.data_ptr() == scratch.f32.data_ptr() + 6 * 4
    assert a.is_contiguous() and b.is_contiguous()


def test_no_buffer_allocates_per_call():
    (a,), (b,) = take([(3,)], [(2, 2)], "cpu")
    assert a.shape == (3,) and b.shape == (2, 2)


def test_a_batch_larger_than_the_scratch_is_refused():
    scratch = SamplingScratch(5, 5, "cpu")
    with pytest.raises(ValueError, match="sized for a smaller batch"):
        take([(2, 3)], [], "cpu", scratch.f32, scratch.i32)
