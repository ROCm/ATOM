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

from atom.model_ops.attentions.pool_layout.paged_state_copy import (
    launch_copy_descriptor,
    plan_segmented_copy,
)


def describe(plan, src_bases, dst_bases, forward=True):
    """The plan at concrete addresses, as `(src, dst, length)` triples."""
    out = np.empty((plan.num_spans, 3), dtype=np.int64)
    plan.write_descriptor(
        out,
        np.array([src_bases], dtype=np.int64),
        np.array([dst_bases], dtype=np.int64),
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
    """The same cut at two sets of bases differs by exactly the bases.

    Asserting only that the *unchanged* columns stayed put would pass for a
    `write_descriptor` that ignored `src_bases` altogether, so the moved
    column is checked against the delta rather than merely for inequality.
    """
    plan = plan_segmented_copy([5, 7], [3, 4, 5], total_bytes=12)
    delta = 999_000

    here = describe(plan, [1000, 2000], [3000, 4000, 5000])
    moved = describe(plan, [1000 + delta, 2000], [3000, 4000, 5000])

    assert [n for _, _, n in here] == [n for _, _, n in moved]
    assert [d for _, d, _ in here] == [d for _, d, _ in moved]
    # Only spans out of source segment 0 move, and each by exactly `delta`.
    assert [s for s, _, _ in moved] == [
        src + (delta if seg == 0 else 0)
        for (src, _, _), seg in zip(here, plan.src_seg, strict=True)
    ]


def test_partial_tail_stops_before_unused_unit_capacity():
    plan = plan_segmented_copy([13], [5, 5, 5], total_bytes=13)
    spans = describe(plan, [1000], [2000, 3000, 4000])

    assert sum(n for _, _, n in spans) == 13
    assert spans[-1][1] == 4000
    assert spans[-1][2] == 3


def test_the_tiling_covers_every_span_exactly_once():
    """The grid is one program per tile, so the tiling is the copy's extent."""
    plan = plan_segmented_copy([13], [5, 5, 5], total_bytes=13)

    assert plan.num_spans == 3
    # Each of the three spans is under one tile, so one tile each, all at
    # offset zero inside their span.
    assert plan.num_tiles == 3
    assert list(plan.span_of_tile) == [0, 1, 2]
    assert list(plan.tile_start) == [0, 0, 0]


def test_a_long_span_is_cut_into_consecutive_tiles():
    plan = plan_segmented_copy([10_000], [10_000], total_bytes=10_000)

    assert plan.num_spans == 1
    assert plan.num_tiles == 3  # 4096 + 4096 + 1808
    assert list(plan.span_of_tile) == [0, 0, 0]
    assert list(plan.tile_start) == [0, 4096, 8192]


def test_the_tiling_reaches_the_end_of_every_span():
    """Nothing is dropped off a span's tail, whatever the sizes are."""
    plan = plan_segmented_copy([9_000, 300, 20_000], [29_300], total_bytes=29_300)

    covered = {}
    for span, start in zip(plan.span_of_tile, plan.tile_start, strict=True):
        covered.setdefault(int(span), []).append(int(start))
    for span, starts in covered.items():
        assert starts == list(range(0, int(plan.length[span]), 4096))


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
            np.array([[slot_ptr]], dtype=np.int64),
            unit_bases[None],
            forward=forward,
        )
        launch_copy_descriptor(torch.from_numpy(descriptor).to(device), plan)

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
            np.array([[src.data_ptr()]], dtype=np.int64),
            np.array(
                [
                    [
                        image.data_ptr(),
                        image.data_ptr() + 2_048,
                        image.data_ptr() + 4_096,
                    ]
                ],
                dtype=np.int64,
            ),
        )
    launch_copy_descriptor(torch.from_numpy(descriptor).to(device), plan)

    torch.cuda.synchronize()
    for src, image in zip(sources, images, strict=True):
        assert torch.equal(image[:5_000], src)


def test_a_batch_describes_each_copy_the_way_one_call_would():
    """The vectorised fill has to be indistinguishable from a loop.

    `write_descriptor` fills every copy in one pass because doing it per copy
    was a quarter of the whole copy path. That is only allowed to be faster,
    so the batch is checked against the copies written one at a time -- an
    axis dropped or an offset broadcast the wrong way would otherwise be a
    descriptor full of plausible addresses.
    """
    plan = plan_segmented_copy([5, 7], [3, 4, 5], total_bytes=12)
    src_bases = np.array([[1000, 2000], [1100, 2100], [1200, 2200]], dtype=np.int64)
    dst_bases = np.array(
        [[7000, 8000, 9000], [7100, 8100, 9100], [7200, 8200, 9200]], dtype=np.int64
    )

    for forward in (True, False):
        batched = np.empty((3 * plan.num_spans, 3), dtype=np.int64)
        plan.write_descriptor(batched, src_bases, dst_bases, forward=forward)

        one_at_a_time = np.empty_like(batched)
        for i in range(3):
            plan.write_descriptor(
                one_at_a_time[i * plan.num_spans : (i + 1) * plan.num_spans],
                src_bases[i : i + 1],
                dst_bases[i : i + 1],
                forward=forward,
            )

        assert np.array_equal(batched, one_at_a_time), f"forward={forward}"


def test_a_zero_byte_copy_is_an_empty_plan_not_a_broadcast_error():
    """0 is a legal `total_bytes` by this function's own contract.

    It validates negatives and empty segments, so a caller reads 0 as allowed
    -- and the tiling used to build its exclusive prefix sum by prepending a
    zero, which has no answer when there are no spans and failed to broadcast
    instead. An empty plan is the answer; `launch_copy_descriptor` returns on
    a zero-row descriptor before it can divide by `num_spans`.
    """
    plan = plan_segmented_copy([5], [5], total_bytes=0)

    assert plan.num_spans == 0
    assert plan.num_tiles == 0


def test_bases_that_do_not_cover_every_copy_are_refused():
    """Numpy would broadcast them, and every copy would go to image zero.

    Verified before the guard: a one-row `dst_bases` against three copies
    produced three identical destination rows -- three GPU copies into the
    same image, no exception, two images left holding stale bytes their
    checkpoint records still claim. This is where host arithmetic becomes a
    raw pointer, so it is where the shape is checked.
    """
    plan = plan_segmented_copy([5, 7], [3, 4, 5], total_bytes=12)
    out = np.empty((3 * plan.num_spans, 3), dtype=np.int64)
    three = np.zeros((3, 2), dtype=np.int64)

    with pytest.raises(ValueError, match="one row of bases per copy"):
        plan.write_descriptor(out, three, np.zeros((1, 3), dtype=np.int64))

    with pytest.raises(ValueError, match="must be"):
        plan.write_descriptor(
            np.empty((2, 3), dtype=np.int64), three, np.zeros((3, 3), dtype=np.int64)
        )


def test_a_flat_dst_bases_is_refused_by_this_guard_not_by_numpy():
    """Both sides are asked the same way, down to the rank.

    A guard that checked only the leading axis of `dst_bases` accepted a 1-D
    one -- three entries, three copies, the shapes agree -- and it then failed
    two lines later inside `dst_bases[:, dst_seg]`, raising an index error
    about an array the caller never passed. Same rejection, unreadable reason.
    """
    plan = plan_segmented_copy([5, 7], [3, 4, 5], total_bytes=12)
    out = np.empty((3 * plan.num_spans, 3), dtype=np.int64)
    three = np.zeros((3, 2), dtype=np.int64)

    with pytest.raises(ValueError, match="one row of bases per copy"):
        plan.write_descriptor(out, three, np.zeros(3, dtype=np.int64))
