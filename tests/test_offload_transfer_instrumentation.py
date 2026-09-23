# SPDX-License-Identifier: MIT
"""CPU-only checks for offload transfer accounting and byte mapping."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector


class _PipelineHarness:
    def __init__(self):
        self.pack = self.Stream()
        self.copy = self.Stream()
        self.state = SimpleNamespace(
            pack_stream=self.pack,
            copy_stream=self.copy,
            stream_ctx=self.stream_ctx,
            staging_buffer=SimpleNamespace(
                tensor=None,
                ready_event=self.Event(),
                free_event=self.Event(),
                free_event_valid=False,
            ),
        )

    @contextmanager
    def stream_ctx(self, _stream):
        yield

    class Stream:
        def wait_event(self, _event):
            pass

        def synchronize(self):
            pass

    class Event:
        def record(self, _stream):
            pass


def _cpu_connector(monkeypatch):
    monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "2")
    monkeypatch.delenv("OFFLOAD_GPU_STAGING_MAX_BYTES", raising=False)
    monkeypatch.delenv("OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER", raising=False)
    codec = SimpleNamespace(device=torch.device("cpu"), bytes_per_block=3)
    return BlockGPUConnector(codec, block_size=4, chunk_size=8)


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_connector_maps_pack_copy_for_both_directions(monkeypatch, direction):
    connector = _cpu_connector(monkeypatch)
    harness = _PipelineHarness()
    # Real CPU tensor pack/copy, fake streams/events: no GPU runtime access.
    monkeypatch.setattr(connector, "_use_cuda", lambda: True)
    monkeypatch.setattr(connector, "_assert_fused_chunk_major_available", lambda: None)
    monkeypatch.setattr(connector, "_thread_state", lambda: harness.state)
    source = torch.arange(15, dtype=torch.uint8).reshape(5, 3)
    restored = torch.zeros_like(source)

    def pack(buf, block_groups, stream):
        block_ids = [block for group in block_groups for block in group]
        buf.copy_(source[block_ids].flatten())

    def unpack(buf, block_groups, stream):
        block_ids = [block for group in block_groups for block in group]
        restored[block_ids] = buf.reshape(-1, 3)

    connector.codec.gpu_to_chunk_major_device_buffer = pack
    connector.codec.chunk_major_device_buffer_to_gpu = unpack
    memory_objs = [
        SimpleNamespace(tensor=source[:2].flatten().clone()),
        SimpleNamespace(tensor=source[2:4].flatten().clone()),
        SimpleNamespace(
            tensor=torch.cat((source[4], torch.full((3,), 255, dtype=torch.uint8)))
        ),
    ]
    if direction == "d2h":
        memory_objs[0].tensor.zero_()
        memory_objs[1].tensor.zero_()
        memory_objs[2].tensor[:3].zero_()
        transfer = connector.batched_from_gpu
    else:
        transfer = connector.batched_to_gpu
    transfer(memory_objs, [0, 8, 16], [8, 16, 17], block_ids=list(range(5)))
    assert torch.equal(
        memory_objs[2].tensor[3:], torch.full((3,), 255, dtype=torch.uint8)
    )
    if direction == "d2h":
        assert torch.equal(
            torch.cat([obj.tensor[:size] for obj, size in zip(memory_objs, [6, 6, 3])]),
            source.flatten(),
        )
    else:
        assert torch.equal(restored, source)
    stats = connector.last_transfer_stats()
    assert stats["stats_available"] == stats["counts_available"] == 1
    assert stats["transfer_succeeded"] == 1
    assert stats["chunks"] == 3
    assert stats["groups"] == 2
    assert stats["max_chunk_bytes"] == 6
    assert stats["max_group_bytes"] == (9 if direction == "d2h" else 12)
    assert stats["total_bytes"] == stats["completed_bytes"] == 15
    assert stats["batch_block_ids_enabled"] == 0
    assert stats["async_host_copy_enabled"] == 0


def test_transfer_evidence_is_thread_local_and_returned_as_a_snapshot(monkeypatch):
    connector = _cpu_connector(monkeypatch)
    connector.batched_from_gpu([], [], [], block_ids=[])
    main_stats = connector.last_transfer_stats()
    main_stats["chunks"] = 99
    assert connector.last_transfer_stats()["chunks"] == 0

    results = []

    def worker():
        results.append(connector.last_transfer_stats()["stats_available"])
        connector.reset_transfer_stats()
        results.append(connector.last_transfer_stats()["chunks"])

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert results == [0, -1]
    assert connector.last_transfer_stats()["chunks"] == 0
    assert connector.last_transfer_stats()["stats_available"] == 1
