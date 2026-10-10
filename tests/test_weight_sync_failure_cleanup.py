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
        self._ipc_buffer = torch.zeros(1, dtype=torch.uint8)
        self._packed_weight_accum = {"old": object()}
        self._expert_relayout_pending = {"old": object()}

    def _get_param_to_module_mapping(self):
        raise RuntimeError("bucket failed")


@pytest.mark.parametrize("path", ["direct", "shm", "ipc"])
def test_a_failed_update_discards_scratch_but_keeps_layout_recovery_state(path):
    """IPC mappings and FP8 packed shards cannot cross syncs. An expert already
    written row-major still needs its pending relayout entry to survive."""
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
    assert set(runner._expert_relayout_pending) == {"old"}
