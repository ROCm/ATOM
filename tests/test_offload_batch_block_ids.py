# SPDX-License-Identifier: MIT
"""CPU contract checks for preparation, staging order, and metadata ownership."""

import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector


class _Tensor:
    """Record host-copy submissions without allocating on a CUDA device."""

    dtype = torch.uint8

    def __init__(self, case, name, device, nbytes):
        self.case = case
        self.name = name
        self.device = SimpleNamespace(type=device)
        self.nbytes = nbytes

    def numel(self):
        return self.nbytes

    def is_contiguous(self):
        return True

    def is_pinned(self):
        return self.device.type == "cpu"

    def reshape(self, *_shape):
        return self

    def __getitem__(self, key):
        start, stop, step = key.indices(self.nbytes)
        assert step == 1
        return _Tensor(self.case, self.name, self.device.type, stop - start)

    def copy_(self, source, *, non_blocking):
        self.case.trace.append(("copy", self.name, source.name, non_blocking))


class _Owner:
    def __init__(self, groups):
        self.groups = tuple(tuple(tuple(chunk) for chunk in group) for group in groups)
        self.group_count = len(groups)
        self.upload_count = int(any(chunk for group in groups for chunk in group))


class _Case:
    def __init__(
        self,
        monkeypatch,
        *,
        supported="both",
    ):
        monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "2")
        monkeypatch.delenv("OFFLOAD_GPU_STAGING_MAX_BYTES", raising=False)
        monkeypatch.delenv("OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER", raising=False)
        self.trace = []
        self.owner_ref = None
        self.prepare_error = False
        self.fail_group = None
        self.fail_stream = None
        self.fail_allocation = False
        self.state = SimpleNamespace(
            pack_stream=self.Stream(self, "pack"),
            copy_stream=self.Stream(self, "copy"),
            stream_ctx=self.stream_ctx,
            staging_buffer=SimpleNamespace(
                tensor=None,
                ready_event=self.Event(self, "ready"),
                free_event=self.Event(self, "free"),
                free_event_valid=False,
            ),
        )
        codec = SimpleNamespace(
            device=torch.device("cpu"),
            num_blocks=8,
            bytes_per_block=4,
            has_fused_chunk_major_staging=True,
            gpu_to_chunk_major_device_buffer=self.legacy_launch,
            chunk_major_device_buffer_to_gpu=self.legacy_launch,
        )
        if supported != "no_prepare":
            codec.prepare_block_id_groups = self.prepare
        if supported in ("both", "no_prepare", "d2h"):
            codec.gpu_to_chunk_major_device_buffer_prepared = self.prepared_launch
        if supported in ("both", "no_prepare", "h2d"):
            codec.chunk_major_device_buffer_to_gpu_prepared = self.prepared_launch
        self.connector = BlockGPUConnector(codec, block_size=1, chunk_size=1)
        monkeypatch.setattr(
            self.connector, "_assert_fused_chunk_major_available", lambda: None
        )
        monkeypatch.setattr(self.connector, "_thread_state", self.thread_state)
        monkeypatch.setattr(
            self.connector, "_ensure_staging_buffer", self.ensure_buffer
        )
        self.memory_objs = [
            SimpleNamespace(tensor=_Tensor(self, f"host{index}", "cpu", 4))
            for index in range(5)
        ]

    def thread_state(self):
        return self.state

    @contextmanager
    def stream_ctx(self, stream):
        self.trace.append(("enter", stream.name))
        yield
        self.trace.append(("exit", stream.name))

    class Stream:
        def __init__(self, case, name):
            self.case = case
            self.name = name

        def wait_event(self, event):
            self.case.trace.append(("wait", self.name, event.name))

        def synchronize(self):
            if self.case.owner_ref is not None:
                assert self.case.owner_ref() is not None, "owner released before fence"
            self.case.trace.append(("fence", self.name))
            if self.case.fail_stream == self.name:
                raise RuntimeError("fence failed")

    class Event:
        def __init__(self, case, name):
            self.case = case
            self.name = name

        def record(self, stream):
            self.case.trace.append(("record", self.name, stream.name))

    def ensure_buffer(self, staging_buffer, nbytes):
        self.trace.append(("allocate", nbytes))
        if self.fail_allocation:
            raise RuntimeError("allocation failed")
        if staging_buffer.tensor is None:
            staging_buffer.tensor = _Tensor(self, "staging", "cuda", 8)
        return staging_buffer.tensor[:nbytes]

    def prepare(self, groups, *, device, stream):
        assert device == self.connector.device
        assert stream is self.state.pack_stream
        self.trace.append(("prepare",))
        if self.prepare_error:
            raise ValueError("invalid later group")
        owner = _Owner(groups)
        self.owner_ref = weakref.ref(owner)
        return owner

    def legacy_launch(self, device_buf, groups, stream=None):
        assert stream is self.state.pack_stream
        self.trace.append(("legacy", tuple(tuple(chunk) for chunk in groups)))

    def prepared_launch(self, device_buf, owner, group_index, *, stream=None):
        assert owner is self.owner_ref()
        assert stream is self.state.pack_stream
        self.trace.append(("prepared", group_index, owner.groups[group_index]))
        if group_index == self.fail_group:
            raise RuntimeError("launch failed")

    def invoke(self, direction):
        method = (
            self.connector.batched_from_gpu
            if direction == "d2h"
            else self.connector.batched_to_gpu
        )
        method(
            self.memory_objs,
            list(range(5)),
            list(range(1, 6)),
            block_ids=[5, 3, 7, 1, 6],
        )


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_prepared_groups_keep_final_order_and_all_enqueue_before_fence(
    monkeypatch, direction
):
    case = _Case(monkeypatch)
    case.invoke(direction)

    launches = [item for item in case.trace if item[0] == "prepared"]
    expected = (
        [((6,), (1,)), ((7,), (3,)), ((5,),)]
        if direction == "d2h"
        else [((5,), (3,)), ((7,), (1,)), ((6,),)]
    )
    assert launches == [
        ("prepared", index, group) for index, group in enumerate(expected)
    ]
    assert case.trace[0] == ("prepare",)
    assert [item for item in case.trace if item[0] == "prepare"] == [("prepare",)]
    assert not any(item[0] == "legacy" for item in case.trace)
    copies = [item for item in case.trace if item[0] == "copy"]
    assert len(copies) == 5
    assert all(item[-1] is True for item in copies)
    last_stream = "copy" if direction == "d2h" else "pack"
    assert case.trace[-1] == ("fence", last_stream)
    assert [item for item in case.trace if item[0] == "fence"] == [case.trace[-1]]
    assert case.connector._quarantined_block_id_owners == []
    assert case.owner_ref() is None
    stats = case.connector.last_transfer_stats()
    assert stats["stats_available"] == stats["counts_available"] == 1
    assert stats["transfer_succeeded"] == 1
    assert stats["chunks"] == 5
    assert stats["groups"] == stats["batch_id_groups"] == 3
    assert stats["total_bytes"] == stats["completed_bytes"] == 20
    assert stats["batch_block_ids_enabled"] == 1
    assert stats["batch_id_uploads"] == 1
    assert stats["async_host_copy_enabled"] == 1
    assert stats["async_host_copy_chunks"] == 5
    assert stats["blocking_host_copy_chunks"] == 0


@pytest.mark.parametrize(
    ("direction", "supported"),
    [("d2h", "no_prepare"), ("h2d", "no_prepare"), ("d2h", "h2d"), ("h2d", "d2h")],
)
def test_falls_back_when_direction_has_no_prepared_api(
    monkeypatch, direction, supported
):
    case = _Case(monkeypatch, supported=supported)
    case.invoke(direction)
    assert len([item for item in case.trace if item[0] == "legacy"]) == 3
    assert case.owner_ref is None
    copies = [item for item in case.trace if item[0] == "copy"]
    assert len(copies) == 5
    assert all(item[-1] is False for item in copies)
    stats = case.connector.last_transfer_stats()
    assert stats["batch_block_ids_enabled"] == 0
    assert stats["async_host_copy_enabled"] == 0
    assert stats["batch_id_groups"] == stats["batch_id_uploads"] == 0
    assert stats["async_host_copy_chunks"] == 0
    assert stats["blocking_host_copy_chunks"] == 5


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_preparation_failure_starts_no_staging_work(monkeypatch, direction):
    case = _Case(monkeypatch)
    case.prepare_error = True
    with pytest.raises(ValueError, match="invalid later group"):
        case.invoke(direction)
    assert case.trace == [("prepare",)]
    assert case.state.staging_buffer.tensor is None
    stats = case.connector.last_transfer_stats()
    assert stats["stats_available"] == stats["counts_available"] == 1
    assert stats["transfer_succeeded"] == 0
    assert stats["completed_bytes"] == -1


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
@pytest.mark.parametrize("failure", ["launch", "allocation", "final_fence"])
@pytest.mark.parametrize("uncertain", [False, True])
def test_owner_lifetime_covers_errors_and_uncertain_fences(
    monkeypatch, direction, failure, uncertain
):
    case = _Case(monkeypatch)
    if failure == "launch":
        case.fail_group = 1
    elif failure == "allocation":
        case.fail_allocation = True
    else:
        # Fail the final fence once; recovery may then confirm completion.
        final_stream = (
            case.state.copy_stream if direction == "d2h" else case.state.pack_stream
        )
        original_sync = final_stream.synchronize
        first = True

        def fail_first_sync():
            nonlocal first
            if first:
                first = False
                assert case.owner_ref() is not None
                case.trace.append(("fence_failed", final_stream.name))
                raise RuntimeError("final fence failed")
            original_sync()

        monkeypatch.setattr(final_stream, "synchronize", fail_first_sync)
    if uncertain:
        case.fail_stream = "pack" if direction == "d2h" else "copy"

    with pytest.raises(RuntimeError, match="failed"):
        case.invoke(direction)
    gc.collect()
    assert [item for item in case.trace if item[0] == "fence"] == (
        [("fence", "pack"), ("fence", "copy")]
        if direction == "d2h"
        else [("fence", "copy"), ("fence", "pack")]
    )
    owners = case.connector._quarantined_block_id_owners
    if uncertain:
        assert len(owners) == 1
        assert owners[0] is case.owner_ref()
        assert case.state.staging_buffer.tensor is None
    else:
        assert owners == []
        assert case.owner_ref() is None


def test_empty_transfer_does_no_preparation(monkeypatch):
    case = _Case(monkeypatch)
    case.connector.reset_transfer_stats()
    assert case.connector.last_transfer_stats()["chunks"] == -1
    case.connector.batched_from_gpu([], [], [], block_ids=[])
    assert case.trace == []
    assert case.owner_ref is None
    stats = case.connector.last_transfer_stats()
    assert stats["stats_available"] == stats["counts_available"] == 1
    assert stats["transfer_succeeded"] == 1
    assert stats["chunks"] == stats["groups"] == stats["total_bytes"] == 0
    assert stats["completed_bytes"] == 0
