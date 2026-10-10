# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A bucket sender owns the sync it starts and aborts it if iteration fails."""

import pytest
import torch

from atom.rollout.weight_sync import load_weights_via_shm


class _CoreManager:
    def __init__(self):
        self.calls = []

    def broadcast_utility_command_sync(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return [{"cmd": cmd, "result": True}]

    def broadcast_utility_command(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))


def test_a_sender_that_stops_after_a_nonfinal_bucket_aborts_the_sync():
    mgr = _CoreManager()

    def weights():
        # The second tensor flushes the first 1 MiB bucket; the failure then
        # abandons a sync workers have already started.
        yield "a", torch.zeros(700_000, dtype=torch.uint8)
        yield "b", torch.zeros(700_000, dtype=torch.uint8)
        raise RuntimeError("trainer stopped")

    with pytest.raises(RuntimeError, match="trainer stopped"):
        load_weights_via_shm(mgr, weights(), bucket_size_mb=1)

    assert [cmd for cmd, _ in mgr.calls] == [
        "update_weights_shm",
        "discard_failed_weight_sync",
    ]
    assert mgr.calls[0][1]["is_last"] is False
    assert "abandoned" in mgr.calls[1][1]["error"]


def test_a_sender_that_never_sent_a_bucket_has_nothing_to_abort():
    mgr = _CoreManager()

    def weights():
        raise RuntimeError("trainer stopped before first tensor")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="before first tensor"):
        load_weights_via_shm(mgr, weights(), bucket_size_mb=1)

    assert mgr.calls == []
