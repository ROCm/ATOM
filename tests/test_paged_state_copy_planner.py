# SPDX-License-Identifier: MIT

"""Cutting one segmented byte stream against another, and issuing the result.

The plan is addressless on purpose (`SegmentedCopyPlan`), so these tests are in
two halves: that the cut lands where it should, and that feeding it a pair of
base-address vectors reconstitutes the copy those cuts describe — including
backwards, which is what a restore is.
"""

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.paged_state_copy import (
    launch_copy_descriptor,
    plan_segmented_copy,
)


def describe(plan, src_bases, dst_bases, forward=True):
    """The plan at concrete addresses, as `(src, dst, length)` triples."""
    out = np.empty((plan.num_spans, 3), dtype=np.int64)
    plan.write_descriptor(
        out,
        np.array(src_bases, dtype=np.int64),
        np.array(dst_bases, dtype=np.int64),
        forward=forward,
    )
    return [tuple(int(x) for x in row) for row in out]


def test_segmented_stream_intersection_preserves_wire_order():
    plan = plan_segmented_copy([5, 7], [3, 4, 5], total_bytes=12)

    assert describe(plan, [1000, 2000], [3000, 4000, 5000]) == [
        (1000, 3000, 3),
        (1003, 4000, 2),
        (2000, 4002, 2),
        (2002, 5000, 5),
    ]


def test_a_reversed_descriptor_is_the_same_cut_the_other_way():
    """A restore reuses its store's plan, so the two must mirror exactly."""
    plan = plan_segmented_copy([5, 7], [3, 4, 5], total_bytes=12)

    forward = describe(plan, [1000, 2000], [3000, 4000, 5000])
    backward = describe(plan, [1000, 2000], [3000, 4000, 5000], forward=False)

    assert backward == [(dst, src, n) for src, dst, n in forward]


def test_the_plan_does_not_depend_on_the_addresses():
    """The same geometry at two sets of bases differs only by the bases."""
    plan = plan_segmented_copy([5, 7], [3, 4, 5], total_bytes=12)

    here = describe(plan, [1000, 2000], [3000, 4000, 5000])
    there = describe(plan, [1_000_000, 2000], [3000, 4000, 5000])

    assert [n for _, _, n in here] == [n for _, _, n in there]
    assert [d for _, d, _ in here] == [d for _, d, _ in there]


def test_partial_tail_stops_before_unused_unit_capacity():
    plan = plan_segmented_copy([13], [5, 5, 5], total_bytes=13)
    spans = describe(plan, [1000], [2000, 3000, 4000])

    assert sum(n for _, _, n in spans) == 13
    assert spans[-1][1] == 4000
    assert spans[-1][2] == 3


def test_the_widest_span_is_what_the_grid_has_to_cover():
    plan = plan_segmented_copy([13], [5, 5, 5], total_bytes=13)

    assert plan.widest == 5
    assert plan.num_spans == 3


@pytest.mark.parametrize(
    "src, dst, total, message",
    [
        ([5], [5], -1, "non-negative"),
        ([3], [5], 5, "source segmented stream is shorter"),
        ([5], [3], 5, "destination segmented stream is shorter"),
        ([5, 0], [5], 5, "cannot contain empty segments"),
    ],
)
def test_an_impossible_copy_is_refused(src, dst, total, message):
    with pytest.raises(ValueError, match=message):
        plan_segmented_copy(src, dst, total)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_descriptor_kernel_round_trips_random_bytes_with_partial_tail():
    device = torch.device("cuda")
    original = torch.randint(0, 256, (13_117,), dtype=torch.uint8, device=device)
    image = torch.full((14_000,), 0xA5, dtype=torch.uint8, device=device)
    restored = torch.zeros_like(original)

    units = [4096, 4096, image.numel() - 8192]
    unit_bases = np.array(
        [image.data_ptr(), image.data_ptr() + 4096, image.data_ptr() + 8192],
        dtype=np.int64,
    )
    plan = plan_segmented_copy([original.numel()], units, original.numel())

    for slot_ptr, forward in (
        (original.data_ptr(), True),
        (restored.data_ptr(), False),
    ):
        descriptor = np.empty((plan.num_spans, 3), dtype=np.int64)
        plan.write_descriptor(
            descriptor,
            np.array([slot_ptr], dtype=np.int64),
            unit_bases,
            forward=forward,
        )
        launch_copy_descriptor(descriptor, plan.widest, device)

    torch.cuda.synchronize()
    assert torch.equal(restored, original)
    # Bytes beyond total_bytes in the final unit are never touched.
    assert torch.all(image[original.numel() :] == 0xA5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_several_copies_ride_in_one_descriptor():
    """Production batches every op of a step into a single launch."""
    device = torch.device("cuda")
    sources = [
        torch.randint(0, 256, (5_000,), dtype=torch.uint8, device=device)
        for _ in range(3)
    ]
    images = [torch.zeros(6_000, dtype=torch.uint8, device=device) for _ in range(3)]
    plan = plan_segmented_copy([5_000], [2_048, 2_048, 1_904], 5_000)

    descriptor = np.empty((3 * plan.num_spans, 3), dtype=np.int64)
    for i, (src, image) in enumerate(zip(sources, images, strict=True)):
        plan.write_descriptor(
            descriptor[i * plan.num_spans : (i + 1) * plan.num_spans],
            np.array([src.data_ptr()], dtype=np.int64),
            np.array(
                [image.data_ptr(), image.data_ptr() + 2_048, image.data_ptr() + 4_096],
                dtype=np.int64,
            ),
        )
    launch_copy_descriptor(descriptor, plan.widest, device)

    torch.cuda.synchronize()
    for src, image in zip(sources, images, strict=True):
        assert torch.equal(image[:5_000], src)
