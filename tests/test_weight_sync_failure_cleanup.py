# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-sync scratch must not survive a failed update now that its worker does."""

from multiprocessing import shared_memory

import pytest
import torch

from atom.rollout.weight_updater import WeightUpdaterMixin


class _Runner(WeightUpdaterMixin):
    device = torch.device("cpu")
    label = "test"
    rank = 0
    world_size = 1

    def __init__(self):
        self._init_weight_updater()
        self.model = torch.nn.Module()
        self._ipc_buffer = torch.zeros(1, dtype=torch.uint8)
        self._packed_weight_accum = {"old": object()}
        self._expert_relayout_pending = {"old": object()}

    def _get_param_to_module_mapping(self):
        raise RuntimeError("bucket failed")


@pytest.mark.parametrize("path", ["direct", "shm", "ipc"])
def test_a_failed_update_leaves_nothing_for_a_later_sync(path):
    """IPC mappings and FP8 packed shards cannot cross syncs, nor can expert
    slices awaiting relayout: a later sync's shard would complete one whose
    other shard came from the failed sync. The engine stays fenced until a
    sync rewrites every parameter, and every expert with it."""
    runner = _Runner()
    shm = None
    try:
        with pytest.raises(RuntimeError, match="bucket failed"):
            if path == "direct":
                runner.update_weights([])
            elif path == "shm":
                shm = shared_memory.SharedMemory(create=True, size=1)
                runner.update_weights_from_shm(shm.name, {}, is_last=False)
            else:
                runner.update_weights_from_ipc(None, {}, is_last=False)
    finally:
        if shm is not None:
            shm.close()
            shm.unlink()

    assert runner._ipc_buffer is None
    assert runner._packed_weight_accum == {}
    assert runner._expert_relayout_pending == {}
    assert runner._sync is None
