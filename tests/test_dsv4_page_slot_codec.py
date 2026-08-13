# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import pytest
import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.hybrid.dsv4.codec import (
    DSV4CopyPlan,
    DSV4PageSlotCodec,
    DSV4PayloadKind,
    DSV4PayloadSection,
)


def _region(
    base_addr: int,
    unit_bytes: int,
    item_count: int,
    *,
    reverse_indexed: bool,
    padding_bytes: int = 0,
) -> KVTransferRegion:
    return KVTransferRegion(
        base_addr=base_addr,
        total_bytes=unit_bytes * item_count + padding_bytes,
        unit_bytes=unit_bytes,
        reverse_indexed=reverse_indexed,
    )


def _codec() -> DSV4PageSlotCodec:
    return DSV4PageSlotCodec(
        page_regions=[
            _region(1_000, 4, 4, reverse_indexed=False),
            _region(2_000, 2, 4, reverse_indexed=False),
        ],
        slot_regions=[
            _region(3_000, 5, 3, reverse_indexed=True, padding_bytes=5),
            _region(4_000, 3, 3, reverse_indexed=True, padding_bytes=3),
        ],
        num_blocks=4,
        num_slots=3,
        device="cpu",
    )


def test_page_plan_is_block_major_region_minor_and_excludes_slot_width():
    codec = _codec()

    plan = codec.page_plan([2, 0], buffer_offset=7)
    spans = list(codec.iter_reference_spans(plan))

    assert codec.page_bytes_per_block == 6
    assert codec.bytes_per_block == 6
    assert codec.slot_bytes == 8
    assert plan.payload_bytes == 12
    assert plan.required_buffer_bytes == 19
    assert [
        (
            span.kind,
            span.item_id,
            span.region_index,
            span.device_addr,
            span.buffer_offset,
            span.nbytes,
        )
        for span in spans
    ] == [
        (DSV4PayloadKind.PAGE, 2, 0, 1_008, 7, 4),
        (DSV4PayloadKind.PAGE, 2, 1, 2_004, 11, 2),
        (DSV4PayloadKind.PAGE, 0, 0, 1_000, 13, 4),
        (DSV4PayloadKind.PAGE, 0, 1, 2_000, 17, 2),
    ]


def test_slot_plan_uses_reverse_addresses_and_region_minor_layout():
    codec = _codec()

    plan = codec.slot_plan(1, buffer_offset=19)
    spans = list(codec.iter_reference_spans(plan))

    assert plan.payload_bytes == 8
    assert plan.required_buffer_bytes == 27
    assert [
        (span.kind, span.item_id, span.device_addr, span.buffer_offset, span.nbytes)
        for span in spans
    ] == [
        (DSV4PayloadKind.SLOT, 1, 3_010, 19, 5),
        (DSV4PayloadKind.SLOT, 1, 4_006, 24, 3),
    ]


def test_checkpoint_plan_places_slot_immediately_after_all_page_bytes():
    codec = _codec()

    plan = codec.checkpoint_plan([2, 0], 1, buffer_offset=7)

    assert plan.payload_bytes == 20
    assert plan.required_buffer_bytes == 27
    assert [
        (section.kind, section.item_ids, section.buffer_offset, section.nbytes)
        for section in plan.sections
    ] == [
        (DSV4PayloadKind.PAGE, (2, 0), 7, 12),
        (DSV4PayloadKind.SLOT, (1,), 19, 8),
    ]
    assert [span.buffer_offset for span in codec.iter_reference_spans(plan)] == [
        7,
        11,
        13,
        17,
        19,
        24,
    ]


@pytest.mark.parametrize(
    ("page_reverse", "slot_reverse"),
    [(True, True), (False, False)],
)
def test_codec_rejects_wrong_address_direction(
    page_reverse: bool,
    slot_reverse: bool,
):
    with pytest.raises(ValueError, match="reverse_indexed"):
        DSV4PageSlotCodec(
            [_region(1_000, 4, 2, reverse_indexed=page_reverse)],
            [_region(2_000, 8, 2, reverse_indexed=slot_reverse)],
            num_blocks=2,
            num_slots=2,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("method", "item_ids", "message"),
    [
        ("page_plan", [0, 0], "duplicate block ids"),
        ("page_plan", [4], "outside pool"),
        ("slot_plan", 3, "outside pool"),
        ("slot_plan", -1, "group id"),
    ],
)
def test_plan_validation_rejects_invalid_ids(method, item_ids, message):
    codec = _codec()

    with pytest.raises(ValueError, match=message):
        getattr(codec, method)(item_ids)


def test_codec_rejects_region_smaller_than_declared_pool():
    too_small = KVTransferRegion(
        base_addr=1_000,
        total_bytes=11,
        unit_bytes=4,
        reverse_indexed=False,
    )

    with pytest.raises(ValueError, match=r"total_bytes.*need 12"):
        DSV4PageSlotCodec(
            [too_small],
            [],
            num_blocks=3,
            num_slots=0,
            device="cpu",
        )


def test_gather_validates_staging_buffer_before_cpu_runtime_rejection():
    codec = _codec()
    plan = codec.page_plan([0])

    with pytest.raises(TypeError, match="uint8"):
        codec.gather(plan, torch.empty(6, dtype=torch.float32))
    with pytest.raises(ValueError, match="too small"):
        codec.gather(plan, torch.empty(5, dtype=torch.uint8))
    with pytest.raises(RuntimeError, match="requires CUDA/HIP"):
        codec.gather(plan, torch.empty(6, dtype=torch.uint8))


@pytest.mark.parametrize(
    "section",
    [
        DSV4PayloadSection(DSV4PayloadKind.PAGE, (4,), 0, 6),
        DSV4PayloadSection(DSV4PayloadKind.PAGE, (0,), 0, 5),
    ],
)
def test_external_copy_plan_is_revalidated_before_raw_pointer_use(section):
    codec = _codec()
    plan = DSV4CopyPlan(
        sections=(section,),
        payload_bytes=section.nbytes,
        required_buffer_bytes=section.nbytes,
    )

    with pytest.raises(ValueError):
        codec.gather(plan, torch.empty(section.nbytes, dtype=torch.uint8))
