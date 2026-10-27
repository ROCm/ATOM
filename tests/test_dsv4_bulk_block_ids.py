# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for batching PAGE IDs without changing the gather/scatter ABI."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec


class _CudaTensor(torch.Tensor):
    """CPU storage reporting a CUDA device for wrapper-only validation."""

    @property
    def device(self):
        return torch.device("cuda:0")


class _OtherCudaTensor(_CudaTensor):
    @property
    def device(self):
        return torch.device("cuda:1")


def _buffer(nbytes=16, *, dtype=torch.uint8):
    return torch.zeros(nbytes, dtype=dtype).as_subclass(_CudaTensor)


@pytest.fixture
def contract(monkeypatch):
    # Import the actual wrapper source with an import-only Triton stub even in
    # a GPU-capable image. Every tensor created by these tests has CPU storage.
    fake_triton = ModuleType("triton")
    fake_language = ModuleType("triton.language")
    fake_triton.__path__ = []
    fake_triton.language = fake_language
    fake_triton.jit = lambda function: function
    source = (
        Path(__file__).parents[1]
        / "atom/kv_transfer/offload/hybrid/dsv4/triton_page_slot.py"
    )
    name = "_dsv4_bulk_ids_cpu_contract"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "triton", fake_triton)
        imports.setitem(sys.modules, "triton.language", fake_language)
        spec.loader.exec_module(module)

    codec = DSV4PageSlotCodec(
        [KVTransferRegion(1000, 32, 4, reverse_indexed=False)],
        [],
        num_blocks=8,
        num_slots=0,
        device="cpu",
    )
    codec.device = torch.device("cuda:0")
    codec._triton_page_slot = module
    stream = SimpleNamespace(device=codec.device)
    uploads = []
    launches = []
    active_streams = []
    plan = SimpleNamespace(
        region_base="base",
        region_total_bytes="total",
        region_unit_bytes="unit",
        tile_region="region",
        tile_unit_offset="unit-offset",
        tile_output_offset="output-offset",
        tile_valid_bytes="valid",
        tiles_per_item=1,
        bytes_per_item=4,
        reverse=False,
    )

    @contextmanager
    def stream_context(actual):
        active_streams.append(actual)
        try:
            yield
        finally:
            assert active_streams.pop() is actual

    def upload(values, device):
        assert device == codec.device
        assert active_streams[-1] is stream
        uploads.append(tuple(values))
        return torch.tensor(values, dtype=torch.int64).as_subclass(_CudaTensor)

    class Kernel:
        def __init__(self, direction):
            self.direction = direction

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                assert active_streams[-1] is stream
                launches.append((self.direction, grid, args, kwargs))

            return launch

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected GPU initialization or synchronization")

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    for api in ("synchronize", "current_stream", "current_device", "Stream", "Event"):
        monkeypatch.setattr(torch.cuda, api, forbidden)
    monkeypatch.setattr(codec, "_region_plan", lambda kind, *, stream: plan)
    monkeypatch.setattr(module, "_static_device_i64", upload)
    monkeypatch.setattr(module, "_device_i64", forbidden)
    monkeypatch.setattr(module, "_gather_region_items_kernel", Kernel("gather"))
    monkeypatch.setattr(module, "_scatter_region_items_kernel", Kernel("scatter"))
    return SimpleNamespace(
        codec=codec,
        module=module,
        stream=stream,
        uploads=uploads,
        launches=launches,
        upload=upload,
    )


def _prepare(contract, groups):
    return contract.codec.prepare_block_id_groups(
        groups, device=contract.codec.device, stream=contract.stream
    )


@pytest.mark.parametrize(
    ("method", "direction"),
    [
        ("gpu_to_chunk_major_device_buffer_prepared", "gather"),
        ("chunk_major_device_buffer_to_gpu_prepared", "scatter"),
    ],
)
def test_one_upload_preserves_group_order_and_kernel_arguments(
    contract, method, direction
):
    # The same block may occur in different groups. Within a group, chunk order
    # and block order are the byte-layout contract, including a shorter tail.
    owner = _prepare(contract, [[[6, 4], [2]], [], [[4], [0]]])
    assert contract.uploads == [(6, 4, 2, 4, 0)]
    assert owner.group_count == 3
    assert owner.upload_count == 1
    payload = _buffer(12)
    for index in range(owner.group_count):
        getattr(contract.codec, method)(payload, owner, index, stream=contract.stream)
    assert contract.uploads == [(6, 4, 2, 4, 0)]
    assert [call[0] for call in contract.launches] == [direction, direction]
    assert [call[1] for call in contract.launches] == [(3, 1), (2, 1)]
    assert [call[2][1].tolist() for call in contract.launches] == [[6, 4, 2], [4, 0]]
    for _, _, args, kwargs in contract.launches:
        assert args[0] is payload
        assert args[9:] == (0, 4)
        assert kwargs == {"REVERSE": False, "TILE": 1024, "num_warps": 8}
        assert (
            args[1].untyped_storage().data_ptr()
            == owner._device_ids.untyped_storage().data_ptr()
        )
    with pytest.raises(FrozenInstanceError):
        owner._device_ids = None


@pytest.mark.parametrize("groups", [[], [[], []]])
def test_empty_groups_do_not_upload_or_launch(contract, groups):
    owner = _prepare(contract, groups)
    assert owner.upload_count == 0
    assert owner.group_count == len(groups)
    for index in range(owner.group_count):
        contract.codec.gpu_to_chunk_major_device_buffer_prepared(
            _buffer(0), owner, index, stream=contract.stream
        )
    assert not contract.uploads
    assert not contract.launches


@pytest.mark.parametrize("invalid", [[1, 1], [-1], [8], [True], [1.5]])
def test_all_groups_are_validated_before_any_upload(contract, invalid):
    with pytest.raises(ValueError):
        _prepare(contract, [[0, 2], invalid])
    assert not contract.uploads
    assert not contract.launches


def test_preparation_rejects_device_or_stream_mismatch(contract):
    for device, stream in [
        ("cpu", contract.stream),
        ("cuda:1", contract.stream),
        ("cuda:0", SimpleNamespace(device=torch.device("cuda:1"))),
        ("cuda:0", None),
    ]:
        with pytest.raises(ValueError):
            contract.codec.prepare_block_id_groups([[0]], device=device, stream=stream)
    assert not contract.uploads


def test_preparation_resolves_cuda_device_alias(contract):
    owner = contract.codec.prepare_block_id_groups(
        [[2, 0]], device="cuda", stream=contract.stream
    )
    assert owner.upload_count == 1
    assert contract.uploads == [(2, 0)]


def test_owner_is_bound_to_codec_and_exact_stream(contract):
    owner = _prepare(contract, [[0]])
    method = contract.codec.gpu_to_chunk_major_device_buffer_prepared
    with pytest.raises(ValueError, match="different codec"):
        method(_buffer(), replace(owner, _codec=object()), 0, stream=contract.stream)
    with pytest.raises(ValueError, match="preparation stream"):
        method(
            _buffer(), owner, 0, stream=SimpleNamespace(device=contract.codec.device)
        )
    with pytest.raises(ValueError, match="preparation stream"):
        method(_buffer(), owner, 0)
    assert not contract.launches


@pytest.mark.parametrize("index", [-1, 1, True, 0.5])
def test_invalid_group_index_never_launches(contract, index):
    owner = _prepare(contract, [[0]])
    with pytest.raises(ValueError):
        contract.codec.gpu_to_chunk_major_device_buffer_prepared(
            _buffer(), owner, index, stream=contract.stream
        )
    assert not contract.launches


@pytest.mark.parametrize(
    "buffer",
    [
        _buffer(3),
        _buffer(dtype=torch.int64),
        torch.zeros(4, dtype=torch.uint8),
        _buffer(8)[::2],
    ],
)
def test_prepared_launch_keeps_payload_validation(contract, buffer):
    owner = _prepare(contract, [[0]])
    with pytest.raises((TypeError, ValueError)):
        contract.codec.gpu_to_chunk_major_device_buffer_prepared(
            buffer, owner, 0, stream=contract.stream
        )
    assert not contract.launches


@pytest.mark.parametrize(
    "ids",
    [
        _buffer(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64).as_subclass(_OtherCudaTensor),
        _buffer(2, dtype=torch.int64),
        _buffer(1, dtype=torch.int64).reshape(1, 1),
    ],
)
def test_prepared_launch_rejects_invalid_id_tensor(contract, ids):
    owner = replace(_prepare(contract, [[0]]), _device_ids=ids)
    with pytest.raises((TypeError, ValueError)):
        contract.codec.gpu_to_chunk_major_device_buffer_prepared(
            _buffer(), owner, 0, stream=contract.stream
        )
    assert not contract.launches


@pytest.mark.parametrize(
    "method", ["gpu_to_chunk_major_device_buffer", "chunk_major_device_buffer_to_gpu"]
)
def test_legacy_launch_still_uploads_each_group(contract, monkeypatch, method):
    monkeypatch.setattr(contract.module, "_device_i64", contract.upload)
    for group in [[[5], [1]], [[2]]]:
        getattr(contract.codec, method)(_buffer(), group, stream=contract.stream)
    assert contract.uploads == [(5, 1), (2,)]
    assert [call[2][1].tolist() for call in contract.launches] == [[5, 1], [2]]
