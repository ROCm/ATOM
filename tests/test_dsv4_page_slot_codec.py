# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

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
    semantic_role: str | None = None,
) -> KVTransferRegion:
    return KVTransferRegion(
        base_addr=base_addr,
        total_bytes=unit_bytes * item_count + padding_bytes,
        unit_bytes=unit_bytes,
        reverse_indexed=reverse_indexed,
        semantic_role=semantic_role,
    )


def _codec() -> DSV4PageSlotCodec:
    return DSV4PageSlotCodec(
        page_regions=[
            _region(1_000, 4, 4, reverse_indexed=False),
            _region(2_000, 2, 4, reverse_indexed=False),
        ],
        slot_regions=[
            _region(
                3_000,
                5,
                3,
                reverse_indexed=True,
                padding_bytes=5,
                semantic_role="dsv4.main_kv.nope",
            ),
            _region(
                4_000,
                3,
                3,
                reverse_indexed=True,
                padding_bytes=3,
                semantic_role="dsv4.main_kv.rope",
            ),
        ],
        num_blocks=4,
        num_slots=3,
        device="cpu",
    )


def _reference_spans(codec: DSV4PageSlotCodec, plan: DSV4CopyPlan):
    """Test-only expansion of the public plan and immutable region snapshots."""

    for section in plan.sections:
        regions = (
            codec.page_regions
            if section.kind is DSV4PayloadKind.PAGE
            else codec.slot_regions
        )
        bytes_per_item = sum(region.unit_bytes for region in regions)
        for item_pos, item_id in enumerate(section.item_ids):
            offset = section.buffer_offset + item_pos * bytes_per_item
            for region_index, region in enumerate(regions):
                yield SimpleNamespace(
                    kind=section.kind,
                    item_id=item_id,
                    region_index=region_index,
                    device_addr=region.unit_addr(item_id),
                    buffer_offset=offset,
                    nbytes=region.unit_bytes,
                )
                offset += region.unit_bytes


def test_page_plan_is_block_major_region_minor_and_excludes_slot_width():
    codec = _codec()

    plan = codec.page_plan([2, 0], buffer_offset=7)
    spans = list(_reference_spans(codec, plan))

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


def test_cpu_codec_does_not_advertise_fused_triton_staging():
    assert _codec().has_fused_chunk_major_staging is False


def test_slot_plan_uses_reverse_addresses_and_region_minor_layout():
    codec = _codec()

    plan = codec.slot_plan(1, buffer_offset=19)
    spans = list(_reference_spans(codec, plan))

    assert plan.payload_bytes == 8
    assert plan.required_buffer_bytes == 27
    assert [
        (span.kind, span.item_id, span.device_addr, span.buffer_offset, span.nbytes)
        for span in spans
    ] == [
        (DSV4PayloadKind.SLOT, 1, 3_010, 19, 5),
        (DSV4PayloadKind.SLOT, 1, 4_006, 24, 3),
    ]


def test_page_and_slot_plans_compose_with_an_explicit_prefix_offset():
    codec = _codec()

    page = codec.page_plan([2, 0], buffer_offset=7)
    slot = codec.slot_plan(1, buffer_offset=page.required_buffer_bytes)

    assert [
        (section.kind, section.item_ids, section.buffer_offset, section.nbytes)
        for section in page.sections + slot.sections
    ] == [
        (DSV4PayloadKind.PAGE, (2, 0), 7, 12),
        (DSV4PayloadKind.SLOT, (1,), 19, 8),
    ]
    assert page.payload_bytes + slot.payload_bytes == 20
    assert slot.required_buffer_bytes == 27
    assert [
        span.buffer_offset
        for plan in (page, slot)
        for span in _reference_spans(codec, plan)
    ] == [
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
    ("page_reverse", "slot_reverse"),
    [(0, True), (False, 1)],
)
def test_codec_rejects_non_boolean_address_direction(
    page_reverse,
    slot_reverse,
):
    with pytest.raises(ValueError, match="reverse_indexed must be a boolean"):
        DSV4PageSlotCodec(
            [_region(1_000, 4, 2, reverse_indexed=page_reverse)],
            [
                _region(
                    2_000,
                    8,
                    2,
                    reverse_indexed=slot_reverse,
                    semantic_role="dsv4.main_kv.nope",
                )
            ],
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


def test_codec_requires_stable_semantic_role_for_slot_regions():
    with pytest.raises(ValueError, match="semantic_role is required"):
        DSV4PageSlotCodec(
            [_region(1_000, 4, 2, reverse_indexed=False)],
            [_region(2_000, 8, 2, reverse_indexed=True)],
            num_blocks=2,
            num_slots=2,
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
