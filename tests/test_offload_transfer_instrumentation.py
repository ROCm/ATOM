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


@pytest.fixture
def startup_warmup(monkeypatch):
    """Real transfer pipeline/CPU tensors; fake CUDA streams and storage only."""
    from concurrent.futures import ThreadPoolExecutor

    from atom.kv_transfer.offload.connector import LMCacheOffloadConnector
    from atom.kv_transfer.offload.hybrid.kimi_k3.connector import KimiK3OffloadConnector

    gpu = _cpu_connector(monkeypatch)
    gpu.codec.num_blocks = 16
    cache = torch.arange(48, dtype=torch.uint8).reshape(16, 3)
    before = cache.clone()
    tls = threading.local()

    def thread_state():
        if not hasattr(tls, "harness"):
            tls.harness = _PipelineHarness()
        return tls.harness.state

    monkeypatch.setattr(gpu, "_use_cuda", lambda: True)
    monkeypatch.setattr(gpu, "_assert_fused_chunk_major_available", lambda: None)
    monkeypatch.setattr(gpu, "_thread_state", thread_state)

    def pack(buf, block_groups, stream):
        blocks = [b for group in block_groups for b in group]
        buf.copy_(cache[blocks].flatten())

    def unpack(buf, block_groups, stream):
        blocks = [b for group in block_groups for b in group]
        cache[blocks] = buf.reshape(-1, 3)

    gpu.codec.gpu_to_chunk_major_device_buffer = pack
    gpu.codec.chunk_major_device_buffer_to_gpu = unpack

    class Engine:
        storage_manager = SimpleNamespace(
            storage_backends={"LocalCPUBackend": object()}
        )
        async_loading = False
        save_only_first_rank = False
        gpu_connector = gpu

        def __init__(self):
            self.entries = {}
            self.calls = []
            self.cleared = []
            self.failure = None

        def store(self, tokens, **kwargs):
            chunks = len(tokens) // 8
            objects = [
                SimpleNamespace(tensor=torch.zeros(6, dtype=torch.uint8))
                for _ in range(chunks)
            ]
            gpu.batched_from_gpu(
                objects,
                list(range(0, len(tokens), 8)),
                list(range(8, len(tokens) + 1, 8)),
                **kwargs,
            )
            self.entries[tuple(tokens.tolist())] = objects
            self.calls.append(
                ("save", threading.get_ident(), dict(gpu.last_transfer_stats()))
            )
            if self.failure == "store":
                raise RuntimeError("store failed after partial publication")

        def lookup(self, tokens, **kwargs):
            return len(tokens) if tuple(tokens.tolist()) in self.entries else 0

        def retrieve(self, tokens, **kwargs):
            if self.failure == "load":
                raise RuntimeError("load failed")
            objects = self.entries[tuple(tokens.tolist())]
            if self.failure != "no_copy":
                gpu.batched_to_gpu(
                    objects,
                    list(range(0, len(tokens), 8)),
                    list(range(8, len(tokens) + 1, 8)),
                    **kwargs,
                )
            self.calls.append(
                ("load", threading.get_ident(), dict(gpu.last_transfer_stats()))
            )
            return torch.full((len(tokens),), self.failure != "miss", dtype=torch.bool)

        def clear(self, tokens, **kwargs):
            key = tuple(tokens.tolist())
            self.cleared.append(key)
            self.entries.pop(key, None)

    engine = Engine()
    worker = KimiK3OffloadConnector.__new__(KimiK3OffloadConnector)
    worker._engine, worker._codec = engine, gpu.codec
    worker._rank = 0
    worker._do_save = worker._do_load = True
    worker.save_workers = worker.load_workers = 1
    worker.chunk_size = 8
    # A virtual block can cover DCP token stripes; physical block size is 2.
    worker.block_size, worker.virtual_block_size = 2, 4
    shell = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    shell._impl = worker
    with ThreadPoolExecutor(1) as save_pool, ThreadPoolExecutor(1) as load_pool:
        worker._save_executor, worker._load_executor = save_pool, load_pool
        yield shell, worker, engine, cache, before


def test_startup_warmup_restores_cpu_bytes_on_serving_threads(startup_warmup):
    shell, worker, engine, cache, before = startup_warmup
    shell.warmup()  # Traverse the public shell and Kimi's actual inherited method.
    assert torch.equal(cache[7:13], before[1:7])
    assert torch.equal(cache[:7], before[:7])
    assert not engine.entries
    assert len(engine.cleared) == 2
    assert all(tokens[0] < 0 for tokens in engine.cleared)
    assert [op for op, _, _ in engine.calls] == ["save", "load", "save", "load"]
    loads = [call for call in engine.calls if call[0] == "load"]
    assert [stats["groups"] for _, _, stats in loads] == [1, 2]
    assert [stats["completed_bytes"] for _, _, stats in loads] == [6, 18]
    assert len({tid for _, tid, _ in loads}) == 1
    assert loads[0][1] == worker._load_executor.submit(threading.get_ident).result()
    assert (
        engine.calls[0][1] == worker._save_executor.submit(threading.get_ident).result()
    )
    assert len({tid for _, tid, _ in engine.calls}) == 2
    assert all(tid != threading.get_ident() for _, tid, _ in engine.calls)
    shell.warmup()  # No second write after startup.
    assert len(engine.calls) == 4


@pytest.mark.parametrize("failure", ["store", "load", "miss", "no_copy"])
def test_startup_warmup_fails_and_cleans_partial_cpu_entries(startup_warmup, failure):
    shell, worker, engine, _, _ = startup_warmup
    engine.failure = failure
    with pytest.raises(RuntimeError):
        shell.warmup()
    assert not engine.entries
    assert len(engine.cleared) == 1
    assert not getattr(worker, "_offload_warmed", False)


@pytest.mark.parametrize(
    "setting,value", [("load_workers", 2), ("save_workers", 2), ("_do_load", False)]
)
def test_startup_warmup_rejects_a_partially_warmed_configuration(
    startup_warmup, setting, value
):
    shell, worker, engine, _, _ = startup_warmup
    setattr(worker, setting, value)
    with pytest.raises(RuntimeError):
        shell.warmup()
    assert not engine.calls
