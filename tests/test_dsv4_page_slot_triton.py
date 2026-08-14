# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.hybrid.dsv4.codec import (
    DSV4PageSlotCodec,
    DSV4PayloadKind,
)

_TRITON_MODULE_NAME = "atom.kv_transfer.offload.hybrid.dsv4.triton_page_slot"
_REAL_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
_CPU_CONTRACT_MODULE = None


def _triton_page_slot_cpu_contract():
    """Load the real wrapper source with an import-only Triton stub if needed."""

    global _CPU_CONTRACT_MODULE
    if _REAL_TRITON_AVAILABLE:
        return importlib.import_module(_TRITON_MODULE_NAME)
    if _CPU_CONTRACT_MODULE is not None:
        return _CPU_CONTRACT_MODULE

    fake_triton = ModuleType("triton")
    fake_language = ModuleType("triton.language")
    fake_triton.__path__ = []
    fake_triton.language = fake_language
    fake_triton.jit = lambda function: function
    isolated_name = "_dsv4_triton_page_slot_cpu_contract"
    source = (
        Path(__file__).parents[1]
        / "atom/kv_transfer/offload/hybrid/dsv4/triton_page_slot.py"
    )
    spec = importlib.util.spec_from_file_location(isolated_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    missing = object()
    original_triton = sys.modules.get("triton", missing)
    original_language = sys.modules.get("triton.language", missing)
    sys.modules["triton"] = fake_triton
    sys.modules["triton.language"] = fake_language
    sys.modules[isolated_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if original_triton is missing:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = original_triton
        if original_language is missing:
            sys.modules.pop("triton.language", None)
        else:
            sys.modules["triton.language"] = original_language
    _CPU_CONTRACT_MODULE = module
    return module


def _install_contract_module(monkeypatch):
    module = _triton_page_slot_cpu_contract()
    package = importlib.import_module("atom.kv_transfer.offload.hybrid.dsv4")
    monkeypatch.setitem(sys.modules, _TRITON_MODULE_NAME, module)
    monkeypatch.setattr(package, "triton_page_slot", module, raising=False)
    return module


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


def _cuda_contract_codec() -> DSV4PageSlotCodec:
    # An unindexed CUDA device does not initialize CUDA. The launch path is
    # monkeypatched below, so this remains a CPU-only orchestration test.
    return DSV4PageSlotCodec(
        [_region(1_000, 4, 4, reverse_indexed=False)],
        [
            _region(
                2_000,
                6,
                3,
                reverse_indexed=True,
                semantic_role="dsv4.main_kv.nope",
            )
        ],
        num_blocks=4,
        num_slots=3,
        device="cuda",
    )


class _NoOwnershipStream:
    device = torch.device("cuda")

    def synchronize(self):
        raise AssertionError("the DSV4 wrapper must not synchronize its stream")

    def record_event(self, *args, **kwargs):
        raise AssertionError("the DSV4 wrapper must not record an event")


@pytest.mark.parametrize(
    ("method_name", "wrapper_name"),
    [
        ("gather", "_gather_region_items_unchecked"),
        ("scatter", "_scatter_region_items_unchecked"),
    ],
)
def test_page_and_slot_plans_launch_on_the_same_caller_stream(
    monkeypatch,
    method_name: str,
    wrapper_name: str,
):
    triton_page_slot = _install_contract_module(monkeypatch)
    codec = _cuda_contract_codec()
    page_plan = codec.page_plan([2, 0], buffer_offset=7)
    slot_plan = codec.slot_plan(1, buffer_offset=page_plan.required_buffer_bytes)
    stream = _NoOwnershipStream()
    buffer = object()
    page_device_plan = object()
    slot_device_plan = object()
    device_plans = {
        DSV4PayloadKind.PAGE: page_device_plan,
        DSV4PayloadKind.SLOT: slot_device_plan,
    }
    calls = []

    monkeypatch.setattr(codec, "_validate_buffer", lambda *args, **kwargs: None)

    def region_plan(kind, *, stream):
        calls.append(("plan", kind, stream))
        return device_plans[kind]

    def launch(
        device_plan,
        item_ids,
        payload,
        *,
        buffer_offset,
        stream,
    ):
        calls.append(
            (
                wrapper_name,
                device_plan,
                tuple(item_ids),
                payload,
                buffer_offset,
                stream,
            )
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("wrong movement wrapper was called")

    monkeypatch.setattr(codec, "_region_plan", region_plan)
    monkeypatch.setattr(triton_page_slot, wrapper_name, launch)
    other_wrapper = (
        "_scatter_region_items_unchecked"
        if wrapper_name == "_gather_region_items_unchecked"
        else "_gather_region_items_unchecked"
    )
    monkeypatch.setattr(triton_page_slot, other_wrapper, forbidden)
    monkeypatch.setattr(triton_page_slot, "gather_region_items", forbidden)
    monkeypatch.setattr(triton_page_slot, "scatter_region_items", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)

    if method_name == "gather":
        codec.gather(page_plan, buffer, stream=stream)
        codec.gather(slot_plan, buffer, stream=stream)
    else:
        codec.scatter(buffer, page_plan, stream=stream)
        codec.scatter(buffer, slot_plan, stream=stream)

    assert calls == [
        ("plan", DSV4PayloadKind.PAGE, stream),
        (wrapper_name, page_device_plan, (2, 0), buffer, 7, stream),
        ("plan", DSV4PayloadKind.SLOT, stream),
        (wrapper_name, slot_device_plan, (1,), buffer, 15, stream),
    ]


def test_codec_rejects_cross_device_stream_before_unchecked_launch(monkeypatch):
    _install_contract_module(monkeypatch)
    codec = DSV4PageSlotCodec(
        [_region(1_000, 4, 4, reverse_indexed=False)],
        [],
        num_blocks=4,
        num_slots=0,
        device="cuda:0",
    )
    monkeypatch.setattr(codec, "_validate_buffer", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="stream device"):
        codec.gather(
            codec.page_plan([0]),
            object(),
            stream=SimpleNamespace(device=torch.device("cuda:1")),
        )


def test_region_plan_is_published_once_under_concurrent_first_use(monkeypatch):
    triton_page_slot = _install_contract_module(monkeypatch)
    codec = _cuda_contract_codec()
    built = []
    barrier = threading.Barrier(2)

    def build(*args, **kwargs):
        built.append((args, kwargs))
        return object()

    monkeypatch.setattr(triton_page_slot, "build_region_plan", build)

    def get_plan():
        barrier.wait()
        return codec._region_plan(DSV4PayloadKind.PAGE, stream=None)

    results = []

    def run():
        results.append(get_plan())

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(built) == 1
    assert results[0] is results[1]


class _KernelLaunchRecorder:
    def __init__(self, calls):
        self.calls = calls

    def __getitem__(self, grid):
        self.calls.append(("grid", grid))

        def launch(*args, **kwargs):
            self.calls.append(("kernel", args, kwargs))

        return launch


@pytest.mark.parametrize(
    ("wrapper_name", "kernel_name"),
    [
        ("gather_region_items", "_gather_region_items_kernel"),
        ("scatter_region_items", "_scatter_region_items_kernel"),
    ],
)
def test_triton_wrapper_only_enqueues_on_the_supplied_stream(
    monkeypatch,
    wrapper_name: str,
    kernel_name: str,
):
    triton_page_slot = _triton_page_slot_cpu_contract()
    calls = []
    stream = _NoOwnershipStream()
    buffer = SimpleNamespace(device=torch.device("cuda"))
    plan = SimpleNamespace(
        tiles_per_item=3,
        region_base="region-base",
        region_total_bytes="region-total",
        region_unit_bytes="region-unit",
        tile_region="tile-region",
        tile_unit_offset="tile-unit-offset",
        tile_output_offset="tile-output-offset",
        tile_valid_bytes="tile-valid-bytes",
        bytes_per_item=10,
        reverse=True,
    )

    class _StreamContext:
        def __enter__(self):
            calls.append(("stream-enter", stream))

        def __exit__(self, *args):
            calls.append(("stream-exit", stream))

    def validate(
        actual_plan,
        item_ids,
        actual_buffer,
        *,
        buffer_offset,
        stream,
    ):
        calls.append(
            (
                "validate",
                actual_plan,
                tuple(item_ids),
                actual_buffer,
                buffer_offset,
                stream,
            )
        )
        return (3, 1), 5

    def device_ids(ids, device):
        calls.append(("ids", ids, device))
        return "device-ids"

    def forbidden(*args, **kwargs):
        raise AssertionError("the Triton wrapper took ownership of completion")

    monkeypatch.setattr(triton_page_slot, "_validate_launch", validate)
    monkeypatch.setattr(
        triton_page_slot,
        "_stream_context",
        lambda actual_stream: (
            _StreamContext()
            if actual_stream is stream
            else pytest.fail("wrapper changed the supplied stream")
        ),
    )
    monkeypatch.setattr(triton_page_slot, "_device_i64", device_ids)
    monkeypatch.setattr(
        triton_page_slot,
        kernel_name,
        _KernelLaunchRecorder(calls),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)

    getattr(triton_page_slot, wrapper_name)(
        plan,
        [3, 1],
        buffer,
        buffer_offset=5,
        stream=stream,
    )

    assert calls[0] == ("validate", plan, (3, 1), buffer, 5, stream)
    assert calls[1:4] == [
        ("stream-enter", stream),
        ("ids", (3, 1), torch.device("cuda")),
        ("grid", (2, 3)),
    ]
    assert calls[4][0] == "kernel"
    assert calls[5] == ("stream-exit", stream)
    _, kernel_args, kernel_kwargs = calls[4]
    assert kernel_args == (
        buffer,
        "device-ids",
        "region-base",
        "region-total",
        "region-unit",
        "tile-region",
        "tile-unit-offset",
        "tile-output-offset",
        "tile-valid-bytes",
        5,
        10,
    )
    assert kernel_kwargs == {
        "REVERSE": True,
        "TILE": triton_page_slot.TILE_BYTES,
        "num_warps": 8,
    }


@pytest.mark.parametrize("unit_bytes", [1023, 1024, 1025, 2049])
def test_region_plan_tiles_boundary_sized_units_without_gaps(
    monkeypatch,
    unit_bytes: int,
):
    """CPU contract for the exact metadata consumed by the Triton kernels."""

    triton_page_slot = _triton_page_slot_cpu_contract()
    monkeypatch.setattr(
        triton_page_slot,
        "_static_device_i64",
        lambda values, _device: torch.tensor(tuple(values), dtype=torch.int64),
    )
    monkeypatch.setattr(
        triton_page_slot,
        "_static_device_i32",
        lambda values, _device: torch.tensor(tuple(values), dtype=torch.int32),
    )
    region = SimpleNamespace(
        base_addr=0x1000,
        total_bytes=2 * unit_bytes,
        unit_bytes=unit_bytes,
    )

    plan = triton_page_slot.build_region_plan(
        [region],
        item_count=2,
        device="cuda",
        reverse=False,
    )

    expected_offsets = list(range(0, unit_bytes, triton_page_slot.TILE_BYTES))
    expected_valid = [
        min(triton_page_slot.TILE_BYTES, unit_bytes - offset)
        for offset in expected_offsets
    ]
    assert plan.bytes_per_item == unit_bytes
    assert plan.tiles_per_item == len(expected_offsets)
    assert plan.tile_region.tolist() == [0] * len(expected_offsets)
    assert plan.tile_unit_offset.tolist() == expected_offsets
    assert plan.tile_output_offset.tolist() == expected_offsets
    assert plan.tile_valid_bytes.tolist() == expected_valid
    assert sum(expected_valid) == unit_bytes


def _gpu_region(
    tensor: torch.Tensor,
    unit_bytes: int,
    *,
    reverse_indexed: bool,
) -> KVTransferRegion:
    return KVTransferRegion(
        base_addr=tensor.data_ptr(),
        total_bytes=tensor.numel(),
        unit_bytes=unit_bytes,
        reverse_indexed=reverse_indexed,
    )


def _forward_slice(tensor: torch.Tensor, unit_bytes: int, item_id: int):
    start = item_id * unit_bytes
    return tensor[start : start + unit_bytes]


def _reverse_slice(tensor: torch.Tensor, unit_bytes: int, item_id: int):
    start = tensor.numel() - (item_id + 1) * unit_bytes
    return tensor[start : start + unit_bytes]


@pytest.mark.parametrize("unit_bytes", [1023, 1024, 1025, 2049])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not _REAL_TRITON_AVAILABLE,
    reason="requires a real CUDA/HIP device and Triton",
)
def test_page_region_tile_boundaries_round_trip_on_gpu(unit_bytes: int):
    device = torch.device("cuda", torch.cuda.current_device())
    source_cpu = torch.arange(2 * unit_bytes, dtype=torch.int64).to(torch.uint8)
    source = source_cpu.to(device)
    codec = DSV4PageSlotCodec(
        [_gpu_region(source, unit_bytes, reverse_indexed=False)],
        [],
        num_blocks=2,
        num_slots=0,
        device=device,
    )
    staging = torch.empty(unit_bytes, dtype=torch.uint8, device=device)
    plan = codec.page_plan([1])

    codec.gather(plan, staging)
    torch.cuda.synchronize(device)
    assert torch.equal(staging.cpu(), source_cpu[unit_bytes:])

    expected = staging.clone()
    source[unit_bytes:].zero_()
    codec.scatter(expected, plan)
    torch.cuda.synchronize(device)
    assert torch.equal(source.cpu(), source_cpu)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _REAL_TRITON_AVAILABLE,
    reason="requires a real CUDA/HIP device and Triton for the round trip",
)
def test_page_slot_non_default_stream_gather_scatter_round_trip():
    importlib.import_module(_TRITON_MODULE_NAME)
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    page_a_cpu = torch.arange(21, dtype=torch.uint8) + 1
    page_b_cpu = torch.arange(15, dtype=torch.uint8) + 41
    slot_a_cpu = torch.arange(20, dtype=torch.uint8) + 81
    slot_b_cpu = torch.arange(14, dtype=torch.uint8) + 121

    with torch.cuda.stream(stream):
        page_a = page_a_cpu.to(device)
        page_b = page_b_cpu.to(device)
        slot_a = slot_a_cpu.to(device)
        slot_b = slot_b_cpu.to(device)
        codec = DSV4PageSlotCodec(
            [
                _gpu_region(page_a, 7, reverse_indexed=False),
                _gpu_region(page_b, 5, reverse_indexed=False),
            ],
            [
                _gpu_region(slot_a, 6, reverse_indexed=True),
                _gpu_region(slot_b, 4, reverse_indexed=True),
            ],
            num_blocks=3,
            num_slots=3,
            device=device,
            slot_region_roles=("dsv4.main_kv.nope", "dsv4.main_kv.rope"),
        )
        page_plan = codec.page_plan([2, 0], buffer_offset=9)
        slot_plan = codec.slot_plan(
            1,
            buffer_offset=page_plan.required_buffer_bytes,
        )
        staging = torch.full(
            (slot_plan.required_buffer_bytes,),
            0xEE,
            dtype=torch.uint8,
            device=device,
        )
        codec.gather(page_plan, staging, stream=stream)
        codec.gather(slot_plan, staging, stream=stream)
    stream.synchronize()

    expected_payload = torch.cat(
        [
            _forward_slice(page_a_cpu, 7, 2),
            _forward_slice(page_b_cpu, 5, 2),
            _forward_slice(page_a_cpu, 7, 0),
            _forward_slice(page_b_cpu, 5, 0),
            _reverse_slice(slot_a_cpu, 6, 1),
            _reverse_slice(slot_b_cpu, 4, 1),
        ]
    )
    assert torch.equal(staging[9:].cpu(), expected_payload)

    with torch.cuda.stream(stream):
        page_a.zero_()
        page_b.zero_()
        slot_a.zero_()
        slot_b.zero_()
        codec.scatter(staging, page_plan, stream=stream)
        codec.scatter(staging, slot_plan, stream=stream)
    stream.synchronize()

    expected_page_a = torch.zeros_like(page_a_cpu)
    expected_page_b = torch.zeros_like(page_b_cpu)
    expected_slot_a = torch.zeros_like(slot_a_cpu)
    expected_slot_b = torch.zeros_like(slot_b_cpu)
    for block_id in (2, 0):
        _forward_slice(expected_page_a, 7, block_id).copy_(
            _forward_slice(page_a_cpu, 7, block_id)
        )
        _forward_slice(expected_page_b, 5, block_id).copy_(
            _forward_slice(page_b_cpu, 5, block_id)
        )
    _reverse_slice(expected_slot_a, 6, 1).copy_(_reverse_slice(slot_a_cpu, 6, 1))
    _reverse_slice(expected_slot_b, 4, 1).copy_(_reverse_slice(slot_b_cpu, 4, 1))
    assert torch.equal(page_a.cpu(), expected_page_a)
    assert torch.equal(page_b.cpu(), expected_page_b)
    assert torch.equal(slot_a.cpu(), expected_slot_a)
    assert torch.equal(slot_b.cpu(), expected_slot_b)
