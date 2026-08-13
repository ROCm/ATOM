# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import FrozenInstanceError

import pytest
import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.hybrid.admission import SlotSidecarAdmission
from atom.kv_transfer.offload.hybrid.slot_codec import ATOMSlotSidecarCodec


def _invalid_integer_scalars():
    np = pytest.importorskip("numpy")
    return [
        True,
        np.bool_(True),
        torch.tensor(True),
        torch.tensor(1),
    ]


def _region(
    base_addr: int,
    unit_bytes: int,
    num_slots: int,
    *,
    total_bytes: int | None = None,
    reverse_indexed: bool = True,
) -> KVTransferRegion:
    return KVTransferRegion(
        base_addr=base_addr,
        total_bytes=(unit_bytes * num_slots if total_bytes is None else total_bytes),
        unit_bytes=unit_bytes,
        reverse_indexed=reverse_indexed,
    )


def _codec(
    region_units: list[int],
    *,
    num_slots: int = 3,
    staging_slots: int = 1,
    device: torch.device | str = "cpu",
) -> ATOMSlotSidecarCodec:
    return ATOMSlotSidecarCodec(
        [
            _region(1_000 * (region_index + 1), unit_bytes, num_slots)
            for region_index, unit_bytes in enumerate(region_units)
        ],
        num_slots=num_slots,
        staging_slots=staging_slots,
        device=device,
    )


def test_payload_is_region_unit_sum_and_staging_has_declared_layout():
    codec = _codec([513, 257, 64], num_slots=4, staging_slots=3)

    assert codec.payload_bytes == 834
    assert codec.num_slots == 4
    assert codec.staging_slots == 3
    assert codec.device == torch.device("cpu")
    assert codec.staging.shape == (3, 834)
    assert codec.staging.dtype == torch.uint8
    assert codec.staging.device == torch.device("cpu")
    assert codec.staging.is_contiguous()


def test_copy_plan_uses_reverse_group_addresses_in_region_order():
    codec = ATOMSlotSidecarCodec(
        [
            _region(1_000, 4, 3),
            _region(2_000, 6, 3),
        ],
        num_slots=3,
        staging_slots=2,
        device="cpu",
    )

    group_zero = codec.copy_plan(0)
    assert [(copy.src_addr, copy.dst_offset, copy.nbytes) for copy in group_zero] == [
        (1_008, 0, 4),
        (2_012, 4, 6),
    ]

    group_one = codec.copy_plan(1)
    assert [(copy.src_addr, copy.dst_offset, copy.nbytes) for copy in group_one] == [
        (1_004, 0, 4),
        (2_006, 4, 6),
    ]


def test_region_geometry_is_snapshotted_and_immutable():
    region = _region(1_000, 8, 3, total_bytes=32)
    codec = ATOMSlotSidecarCodec(
        [region],
        num_slots=3,
        staging_slots=1,
        device="cpu",
    )

    region.base_addr = 9_000
    region.total_bytes = 300
    region.unit_bytes = 100
    region.reverse_indexed = False

    copy = codec.copy_plan(0)[0]
    assert (copy.src_addr, copy.dst_offset, copy.nbytes) == (1_024, 0, 8)
    assert codec.payload_bytes == 8
    assert codec.regions[0].reverse_indexed is True
    with pytest.raises(FrozenInstanceError):
        codec.regions[0].unit_bytes = 16


def test_codec_rejects_empty_slot_regions():
    with pytest.raises(ValueError, match="at least one"):
        ATOMSlotSidecarCodec(
            [],
            num_slots=3,
            staging_slots=1,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_addr", 0),
        ("base_addr", -1),
        ("unit_bytes", 0),
        ("unit_bytes", -1),
        ("total_bytes", 0),
        ("total_bytes", -1),
    ],
)
def test_codec_rejects_nonpositive_region_geometry(field: str, value: int):
    region = _region(1_000, 8, 3)
    setattr(region, field, value)

    with pytest.raises(ValueError, match=rf"{field} must be > 0"):
        ATOMSlotSidecarCodec(
            [region],
            num_slots=3,
            staging_slots=1,
            device="cpu",
        )


@pytest.mark.parametrize("field", ["base_addr", "unit_bytes", "total_bytes"])
def test_codec_geometry_rejects_boolean_and_tensor_scalars(field: str):
    for value in _invalid_integer_scalars():
        region = _region(1_000, 8, 3)
        setattr(region, field, value)

        with pytest.raises(ValueError, match=rf"{field} must be an integer"):
            ATOMSlotSidecarCodec(
                [region],
                num_slots=3,
                staging_slots=1,
                device="cpu",
            )


def test_codec_rejects_region_too_small_for_all_slots():
    region = _region(1_000, 8, 3, total_bytes=23)

    with pytest.raises(ValueError, match=r"total_bytes.*need 24"):
        ATOMSlotSidecarCodec(
            [region],
            num_slots=3,
            staging_slots=1,
            device="cpu",
        )


def test_codec_requires_reverse_indexed_slot_regions():
    region = _region(1_000, 8, 3, reverse_indexed=False)

    with pytest.raises(ValueError, match="reverse_indexed=True"):
        ATOMSlotSidecarCodec(
            [region],
            num_slots=3,
            staging_slots=1,
            device="cpu",
        )


@pytest.mark.parametrize("num_slots", [0, -1])
def test_codec_rejects_nonpositive_num_slots(num_slots: int):
    with pytest.raises(ValueError, match="num_slots must be > 0"):
        ATOMSlotSidecarCodec(
            [_region(1_000, 8, 3)],
            num_slots=num_slots,
            staging_slots=1,
            device="cpu",
        )


@pytest.mark.parametrize("staging_slots", [0, -1])
def test_codec_rejects_nonpositive_staging_slots(staging_slots: int):
    with pytest.raises(ValueError, match="staging_slots must be > 0"):
        ATOMSlotSidecarCodec(
            [_region(1_000, 8, 3)],
            num_slots=3,
            staging_slots=staging_slots,
            device="cpu",
        )


@pytest.mark.parametrize("field", ["num_slots", "staging_slots"])
def test_codec_counts_reject_boolean_and_tensor_scalars(field: str):
    for value in _invalid_integer_scalars():
        kwargs = {
            "num_slots": 3,
            "staging_slots": 1,
            "device": "cpu",
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=rf"{field} must be an integer"):
            ATOMSlotSidecarCodec([_region(1_000, 8, 3)], **kwargs)


@pytest.mark.parametrize("group", [-1, 3])
def test_copy_plan_rejects_group_outside_slot_pool(group: int):
    codec = _codec([8], num_slots=3)

    with pytest.raises(ValueError, match="group"):
        codec.copy_plan(group)


def test_group_id_rejects_boolean_and_tensor_scalars():
    codec = _codec([8], num_slots=3)

    for value in _invalid_integer_scalars():
        with pytest.raises(ValueError, match="group id must be an integer"):
            codec.copy_plan(value)


@pytest.mark.parametrize("staging_id", [-1, 2])
def test_staging_view_rejects_id_outside_staging_pool(staging_id: int):
    codec = _codec([8], staging_slots=2)

    with pytest.raises(ValueError, match="staging"):
        codec.staging_view(staging_id)


def test_staging_id_rejects_boolean_and_tensor_scalars():
    codec = _codec([8], staging_slots=2)

    for value in _invalid_integer_scalars():
        with pytest.raises(ValueError, match="staging id must be an integer"):
            codec.staging_view(value)


def test_codec_accepts_numpy_integer_scalars():
    np = pytest.importorskip("numpy")
    region = KVTransferRegion(
        base_addr=np.int64(1_000),
        total_bytes=np.int64(24),
        unit_bytes=np.int64(8),
        reverse_indexed=True,
    )

    codec = ATOMSlotSidecarCodec(
        [region],
        num_slots=np.int64(3),
        staging_slots=np.int64(2),
        device="cpu",
    )

    assert codec.copy_plan(np.int64(1))[0].src_addr == 1_008
    assert codec.staging_view(np.int64(1)).data_ptr() == codec.staging[1].data_ptr()


@pytest.mark.parametrize("method_name", ["snapshot_to_staging", "restore_from_staging"])
def test_copy_methods_validate_group_and_staging_ids(method_name: str):
    codec = _codec([8], num_slots=3, staging_slots=2)
    method = getattr(codec, method_name)

    with pytest.raises(ValueError, match="group"):
        method(3, 0)
    with pytest.raises(ValueError, match="staging"):
        method(0, 2)


def test_staging_view_aliases_only_the_requested_row():
    codec = _codec([8], staging_slots=2)
    codec.staging.zero_()

    row_one = codec.staging_view(1)
    row_one.fill_(91)

    assert row_one.shape == (codec.payload_bytes,)
    assert row_one.is_contiguous()
    assert row_one.data_ptr() == codec.staging[1].data_ptr()
    assert torch.count_nonzero(codec.staging_view(0)) == 0
    assert torch.equal(
        codec.staging_view(1),
        torch.full((codec.payload_bytes,), 91, dtype=torch.uint8),
    )


def test_snapshot_and_restore_reuse_copy_plan_staging_apis():
    codec = _codec([4, 2], num_slots=3, staging_slots=2)
    codec.staging.zero_()
    calls = []

    class _FakeStaging:
        @staticmethod
        def gather_copy_plan(plan, dst, *, stream=None):
            calls.append(("gather", plan, dst, stream))
            dst.fill_(17)

        @staticmethod
        def scatter_copy_plan(src, plan, *, stream=None):
            calls.append(("scatter", plan, src, stream))

    codec._fused_kv_staging = _FakeStaging()
    stream = None

    codec.snapshot_to_staging(1, 1, stream=stream)
    codec.restore_from_staging(1, 1, stream=stream)

    expected_plan = codec.copy_plan(1)
    assert [call[0] for call in calls] == ["gather", "scatter"]
    assert calls[0][1] == expected_plan
    assert calls[1][1] == expected_plan
    assert calls[0][2].data_ptr() == codec.staging_view(1).data_ptr()
    assert calls[1][2].data_ptr() == codec.staging_view(1).data_ptr()
    assert calls[0][3] is stream
    assert calls[1][3] is stream
    assert torch.count_nonzero(codec.staging_view(0)) == 0
    assert torch.all(codec.staging_view(1) == 17)


class _FakeStream:
    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)


@pytest.mark.parametrize("method_name", ["snapshot_to_staging", "restore_from_staging"])
def test_copy_methods_reject_stream_from_another_device_before_dispatch(method_name):
    codec = _codec([8])
    codec.device = torch.device("cuda:1")
    codec._device_ctx = nullcontext
    calls = []

    class _FakeStaging:
        @staticmethod
        def gather_copy_plan(plan, dst, *, stream=None):
            calls.append(("gather", stream))

        @staticmethod
        def scatter_copy_plan(src, plan, *, stream=None):
            calls.append(("scatter", stream))

    codec._fused_kv_staging = _FakeStaging()
    method = getattr(codec, method_name)

    with pytest.raises(ValueError, match="stream device.*codec device"):
        method(0, 0, stream=_FakeStream("cuda:0"))

    assert calls == []


@pytest.mark.parametrize("method_name", ["snapshot_to_staging", "restore_from_staging"])
@pytest.mark.parametrize("stream_device", [None, "cuda:1"])
def test_copy_methods_preserve_none_and_same_device_streams(
    method_name,
    stream_device,
):
    codec = _codec([8])
    codec.device = torch.device("cuda:1")
    codec._device_ctx = nullcontext
    calls = []

    class _FakeStaging:
        @staticmethod
        def gather_copy_plan(plan, dst, *, stream=None):
            calls.append(stream)

        @staticmethod
        def scatter_copy_plan(src, plan, *, stream=None):
            calls.append(stream)

    codec._fused_kv_staging = _FakeStaging()
    stream = None if stream_device is None else _FakeStream(stream_device)

    getattr(codec, method_name)(0, 0, stream=stream)

    assert calls == [stream]


def test_admission_acquires_refuses_releases_and_reuses_smallest_id():
    admission = SlotSidecarAdmission(3)

    assert admission.capacity == 3
    assert admission.num_free == 3
    assert [admission.try_acquire() for _ in range(4)] == [0, 1, 2, None]
    assert admission.num_free == 0

    admission.release(1)
    admission.release(0)

    assert admission.num_free == 2
    assert admission.try_acquire() == 0
    assert admission.try_acquire() == 1
    assert admission.try_acquire() is None


def test_admission_quarantine_permanently_removes_acquired_id():
    admission = SlotSidecarAdmission(1)
    slot_id = admission.try_acquire()

    admission.quarantine(slot_id)

    assert admission.num_free == 0
    assert admission.try_acquire() is None
    with pytest.raises(ValueError, match="quarantined"):
        admission.release(slot_id)


@pytest.mark.parametrize("slot_id", [-1, 2, 1.5, True])
def test_admission_rejects_invalid_release(slot_id):
    admission = SlotSidecarAdmission(2)

    with pytest.raises(ValueError, match="slot"):
        admission.release(slot_id)


def test_admission_constructor_rejects_boolean_and_tensor_scalars():
    for value in _invalid_integer_scalars():
        with pytest.raises(ValueError, match="num_slots must be an integer"):
            SlotSidecarAdmission(value)


def test_admission_release_rejects_boolean_and_tensor_scalars():
    admission = SlotSidecarAdmission(2)
    assert admission.try_acquire() == 0

    for value in _invalid_integer_scalars():
        with pytest.raises(ValueError, match="slot id must be an integer"):
            admission.release(value)

    assert admission.num_free == 1
    admission.release(0)


def test_admission_accepts_numpy_integer_scalars():
    np = pytest.importorskip("numpy")
    admission = SlotSidecarAdmission(np.int64(2))

    assert admission.try_acquire() == 0
    admission.release(np.int64(0))
    assert admission.num_free == 2


def test_admission_rejects_double_release():
    admission = SlotSidecarAdmission(1)
    assert admission.try_acquire() == 0
    admission.release(0)

    with pytest.raises(ValueError, match="not acquired"):
        admission.release(0)


@pytest.mark.parametrize("num_slots", [0, -1, 1.5, True])
def test_admission_rejects_invalid_capacity(num_slots):
    with pytest.raises(ValueError, match="num_slots"):
        SlotSidecarAdmission(num_slots)


def test_admission_is_thread_safe_under_basic_contention():
    capacity = 3
    competitors = 12
    admission = SlotSidecarAdmission(capacity)
    rendezvous = threading.Barrier(competitors)

    def compete(_):
        slot_id = admission.try_acquire()
        rendezvous.wait()
        if slot_id is not None:
            admission.release(slot_id)
        return slot_id

    with ThreadPoolExecutor(max_workers=competitors) as executor:
        results = list(executor.map(compete, range(competitors)))

    acquired = [slot_id for slot_id in results if slot_id is not None]
    assert sorted(acquired) == list(range(capacity))
    assert len(acquired) == len(set(acquired))
    assert results.count(None) == competitors - capacity
    assert admission.num_free == capacity


def test_admission_keeps_id_owned_until_caller_synchronizes_and_releases():
    admission = SlotSidecarAdmission(1)
    slot_id = admission.try_acquire()

    class _FakeTransfer:
        synchronized = False

        def synchronize(self):
            self.synchronized = True

    transfer = _FakeTransfer()

    assert slot_id == 0
    assert admission.try_acquire() is None

    transfer.synchronize()

    assert transfer.synchronized
    assert admission.try_acquire() is None
    admission.release(slot_id)
    assert admission.try_acquire() == slot_id


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="full-slot gather/scatter kernels require CUDA/HIP",
)
def test_gpu_full_slot_round_trip_on_nondefault_stream():
    device = torch.device("cuda", torch.cuda.current_device())
    num_slots = 3
    region0_unit = 1_537
    region1_unit = 1_289
    region0 = (
        torch.arange(num_slots * region0_unit, device=device)
        .remainder(251)
        .to(torch.uint8)
        .reshape(num_slots, region0_unit)
    )
    region1 = (
        torch.arange(num_slots * region1_unit, device=device)
        .add(101)
        .remainder(251)
        .to(torch.uint8)
        .reshape(num_slots, region1_unit)
    )
    original0 = region0.clone()
    original1 = region1.clone()
    codec = ATOMSlotSidecarCodec(
        [
            _region(region0.data_ptr(), region0_unit, num_slots),
            _region(region1.data_ptr(), region1_unit, num_slots),
        ],
        num_slots=num_slots,
        staging_slots=2,
        device=device,
    )
    stream = torch.cuda.Stream(device=device)
    assert stream != torch.cuda.default_stream(device)
    codec.staging.zero_()
    stream.wait_stream(torch.cuda.current_stream(device))

    codec.snapshot_to_staging(0, 1, stream=stream)
    stream.synchronize()

    assert torch.count_nonzero(codec.staging_view(0)) == 0
    assert torch.equal(
        codec.staging_view(1),
        torch.cat([original0[-1], original1[-1]]),
    )

    region0[-1].zero_()
    region1[-1].zero_()
    stream.wait_stream(torch.cuda.current_stream(device))
    codec.restore_from_staging(0, 1, stream=stream)
    stream.synchronize()

    assert torch.equal(region0, original0)
    assert torch.equal(region1, original1)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="cross-device stream validation requires at least two GPUs",
)
def test_gpu_rejects_stream_from_another_device():
    codec_device = torch.device("cuda:0")
    other_device = torch.device("cuda:1")
    num_slots = 2
    unit_bytes = 64
    region = torch.zeros(
        (num_slots, unit_bytes),
        dtype=torch.uint8,
        device=codec_device,
    )
    codec = ATOMSlotSidecarCodec(
        [_region(region.data_ptr(), unit_bytes, num_slots)],
        num_slots=num_slots,
        staging_slots=1,
        device=codec_device,
    )
    other_stream = torch.cuda.Stream(device=other_device)

    with pytest.raises(ValueError, match="stream device.*codec device"):
        codec.snapshot_to_staging(0, 0, stream=other_stream)
