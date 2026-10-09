# SPDX-License-Identifier: MIT
"""CPU policy checks; real pinned GPU-copy correctness needs the separate probe."""

from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector


class _Tensor:
    def __init__(self, device, pinned, trace):
        self.device = SimpleNamespace(type=device)
        self.pinned = pinned
        self.trace = trace
        self.pin_queries = 0

    def is_pinned(self):
        self.pin_queries += 1
        return self.pinned

    def __getitem__(self, _slice):
        return self

    def copy_(self, source, *, non_blocking):
        self.trace.append((self.device.type, source.device.type, non_blocking))


def _connector(monkeypatch):
    monkeypatch.delenv("OFFLOAD_GPU_STAGING_MAX_BYTES", raising=False)
    return BlockGPUConnector(
        SimpleNamespace(device=torch.device("cpu"), bytes_per_block=4),
        block_size=4,
        chunk_size=4,
    )


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
@pytest.mark.parametrize("prepared_ids_active", [False, True])
@pytest.mark.parametrize("physically_pinned", [False, True])
def test_copy_requires_physical_pinning_and_prepared_ids(
    monkeypatch, direction, prepared_ids_active, physically_pinned
):
    connector = _connector(monkeypatch)
    trace = []
    host = _Tensor("cpu", physically_pinned, trace)
    device = _Tensor("cuda", False, trace)
    # Deliberately opposite: cache eviction pin must never select copy policy.
    memory_obj = SimpleNamespace(tensor=host, is_pinned=not physically_pinned)
    group = SimpleNamespace(
        chunks=[SimpleNamespace(memory_obj=memory_obj, tensor=host, nbytes=4)]
    )
    if direction == "d2h":
        connector._slice_to_memory_objs(
            group, device, prepared_ids_active=prepared_ids_active
        )
    else:
        connector._memory_objs_to_slice(
            group, device, prepared_ids_active=prepared_ids_active
        )

    asynchronous = prepared_ids_active and physically_pinned
    dst, src = ("cpu", "cuda") if direction == "d2h" else ("cuda", "cpu")
    assert trace == [(dst, src, asynchronous)]
    assert host.pin_queries == int(prepared_ids_active)


@pytest.mark.parametrize(
    ("memory_device", "staging_device", "expected"),
    [("cpu", "cpu", False), ("cuda", "cuda", True), ("cuda", "cpu", True)],
)
def test_non_cpu_cuda_pairs_do_not_query_pinning(
    monkeypatch, memory_device, staging_device, expected
):
    connector = _connector(monkeypatch)
    memory = _Tensor(memory_device, True, [])
    staging = _Tensor(staging_device, False, [])
    assert (
        connector._non_blocking_memory_copy(memory, staging, prepared_ids_active=True)
        is expected
    )
    assert memory.pin_queries == 0
