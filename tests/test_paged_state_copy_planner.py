# SPDX-License-Identifier: MIT

import pytest
import torch

from atom.model_ops.attentions.paged_state_copy import (
    ByteSegment,
    launch_copy_spans,
    plan_segmented_copy,
)


def test_segmented_stream_intersection_preserves_wire_order():
    src = [ByteSegment(1000, 5), ByteSegment(2000, 7)]
    dst = [ByteSegment(3000, 3), ByteSegment(4000, 4), ByteSegment(5000, 5)]

    spans = plan_segmented_copy(src, dst, total_bytes=12)

    assert [(s.src_ptr, s.dst_ptr, s.num_bytes) for s in spans] == [
        (1000, 3000, 3),
        (1003, 4000, 2),
        (2000, 4002, 2),
        (2002, 5000, 5),
    ]


def test_partial_tail_stops_before_unused_unit_capacity():
    src = [ByteSegment(1000, 13)]
    dst = [ByteSegment(2000, 5), ByteSegment(3000, 5), ByteSegment(4000, 5)]
    spans = plan_segmented_copy(src, dst, total_bytes=13)
    assert sum(span.num_bytes for span in spans) == 13
    assert spans[-1].dst_ptr == 4000
    assert spans[-1].num_bytes == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_descriptor_kernel_round_trips_random_bytes_with_partial_tail():
    device = torch.device("cuda")
    original = torch.randint(0, 256, (13_117,), dtype=torch.uint8, device=device)
    image = torch.full((14_000,), 0xA5, dtype=torch.uint8, device=device)
    restored = torch.zeros_like(original)

    slot = [ByteSegment(original.data_ptr(), original.numel())]
    units = [
        ByteSegment(image.data_ptr(), 4096),
        ByteSegment(image.data_ptr() + 4096, 4096),
        ByteSegment(image.data_ptr() + 8192, image.numel() - 8192),
    ]
    scatter = plan_segmented_copy(slot, units, original.numel())
    launch_copy_spans(scatter, device)
    gather = plan_segmented_copy(
        units,
        [ByteSegment(restored.data_ptr(), restored.numel())],
        original.numel(),
    )
    launch_copy_spans(gather, device)

    torch.cuda.synchronize()
    assert torch.equal(restored, original)
    # Bytes beyond total_bytes in the final unit are never touched.
    assert torch.all(image[original.numel() :] == 0xA5)
