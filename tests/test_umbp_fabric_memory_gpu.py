# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Opt-in allocation accounting checks; no service or peer GPU is needed.

Run on an idle FABRIC-capable GPU with the matching MORI package/extension:
RUN_UMBP_FABRIC_GPU_TESTS=1 python -m pytest -s tests/test_umbp_fabric_memory_gpu.py
"""

import gc
import json
import os
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_UMBP_FABRIC_GPU_TESTS") != "1",
    reason="set RUN_UMBP_FABRIC_GPU_TESTS=1 on an idle FABRIC-capable GPU",
)

_ARENA_BYTES = 64 << 20
# Preserve owners and tensors if GPU completion/safe close cannot be confirmed.
_IN_DOUBT_ALLOCATIONS = []


@pytest.mark.parametrize("transport", ["ipc", "fabric"])
def test_pool_allocation_is_counted_by_torch(transport, record_property):
    # Keep all runtime imports/probes behind the explicit opt-in marker.
    import torch

    from atom.kv_transfer.pool.umbp.pool import UmbpPoolMemory

    assert torch.cuda.is_available(), "this opt-in test requires a GPU"
    assert (
        torch.cuda.get_allocator_backend() == "native"
    ), "allocated_bytes accounting requires the native caching allocator"
    device = torch.device("cuda", torch.cuda.current_device())

    # Initialize both zeroing paths before measuring, including their kernels.
    warmup = torch.zeros(_ARENA_BYTES, dtype=torch.uint8, device=device)
    torch.bitwise_and(warmup, 0, out=warmup)
    torch.cuda.synchronize(device)
    del warmup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    client = Mock()
    memory = UmbpPoolMemory(
        client, f"accounting-{transport}", "accounting", 1.0, transport=transport
    )
    arena = None
    try:
        pre_alloc = torch.cuda.memory_stats(device)["allocated_bytes.all.current"]
        arena = memory.allocate(_ARENA_BYTES, device)
        torch.cuda.synchronize(device)
        post_alloc = torch.cuda.memory_stats(device)["allocated_bytes.all.current"]
        actual_bytes = post_alloc - pre_alloc
        result = {
            "transport": transport,
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "hip": torch.version.hip,
            "allocator_backend": torch.cuda.get_allocator_backend(),
            "device": str(device),
            "expected_bytes": _ARENA_BYTES,
            "pre_alloc": pre_alloc,
            "post_alloc": post_alloc,
            "actual_bytes": actual_bytes,
        }
        print(json.dumps(result, sort_keys=True))
        record_property("allocation_accounting", json.dumps(result, sort_keys=True))
        assert memory.counted_by_torch
        assert arena.numel() == _ARENA_BYTES and arena.device == device
        # Match ModelRunner's allocation diagnostic, including its 3% tolerance.
        assert abs(actual_bytes - _ARENA_BYTES) / _ARENA_BYTES <= 0.03, result
        client.attach.assert_not_called()
        client.bind_layout.assert_not_called()
        client.detach.assert_not_called()
    finally:
        try:
            torch.cuda.synchronize(device)
            memory.close()
        except Exception:
            _IN_DOUBT_ALLOCATIONS.append((memory, arena))
            raise
        # NativeL1Memory.close retains its tensor/pool until the owner dies.
        # Drop both the external tensor and owner only after the safe close.
        arena = None
        memory = None
        gc.collect()
        torch.cuda.empty_cache()
