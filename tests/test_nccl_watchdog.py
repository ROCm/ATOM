# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`drain_nccl_works`: every NCCL group waits out its watchdog, no other does."""

from unittest.mock import MagicMock

import torch.distributed.distributed_c10d as c10d

from atom.distributed import nccl_watchdog


def _groups(monkeypatch, backends):
    groups = [MagicMock(name=b) for b in backends]
    pg_map = {g: (b, None) for g, b in zip(groups, backends)}
    # `_World.pg_map` reads this module global.
    monkeypatch.setattr(c10d, "_pg_map", pg_map)
    monkeypatch.setattr(nccl_watchdog.torch.cuda, "synchronize", MagicMock())
    return groups


def test_every_nccl_group_waits_and_gloo_is_skipped(monkeypatch):
    groups = _groups(monkeypatch, ["nccl", "gloo", "nccl"])
    monkeypatch.setattr(nccl_watchdog.dist, "is_initialized", lambda: True)

    nccl_watchdog.drain_nccl_works()

    assert groups[0]._wait_for_pending_works.call_count == 1
    assert groups[1]._wait_for_pending_works.call_count == 0
    assert groups[2]._wait_for_pending_works.call_count == 1
    nccl_watchdog.torch.cuda.synchronize.assert_called_once()


def test_nothing_runs_without_a_process_group(monkeypatch):
    groups = _groups(monkeypatch, ["nccl"])
    monkeypatch.setattr(nccl_watchdog.dist, "is_initialized", lambda: False)

    nccl_watchdog.drain_nccl_works()

    assert groups[0]._wait_for_pending_works.call_count == 0
    nccl_watchdog.torch.cuda.synchronize.assert_not_called()
