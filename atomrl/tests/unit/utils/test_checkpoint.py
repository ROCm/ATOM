# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.utils.checkpoint import (
    CheckpointManager,
    checkpoint_rank_phases,
    create_checkpoint_control_group,
    run_checkpoint_phase,
)


def test_save_and_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "run.pt"
    state = {"w": torch.tensor([1.0, 2.0]), "n": 3}
    CheckpointManager.save(state, path, step=7)
    payload = CheckpointManager.load(path)
    assert payload["step"] == 7
    assert torch.equal(payload["state_dict"]["w"], state["w"])
    assert payload["state_dict"]["n"] == 3


def test_get_latest(tmp_path) -> None:
    CheckpointManager.save({"a": 1}, tmp_path / "checkpoint_5.pt", step=5)
    CheckpointManager.save({"a": 2}, tmp_path / "checkpoint_20.pt", step=20)
    latest = CheckpointManager.get_latest(tmp_path)
    assert latest is not None
    assert "checkpoint_20.pt" in latest
    loaded = CheckpointManager.load(latest)
    assert loaded["state_dict"]["a"] == 2


def test_checkpoint_phase_collectives_use_gloo_control_group(monkeypatch) -> None:
    control_group = object()
    new_group_calls = []
    collective_groups = []

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "new_group",
        lambda *, ranks, backend: (
            new_group_calls.append((ranks, backend)) or control_group
        ),
    )

    def all_gather_object(output, value, *, group=None):
        collective_groups.append(group)
        if isinstance(value, str):
            output[:] = ["trainer-a", "trainer-b"]
        else:
            output[:] = [None, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        "atomrl.utils.checkpoint.socket.gethostname", lambda: "trainer-a"
    )

    group = create_checkpoint_control_group(2)
    phases = checkpoint_rank_phases(0, 2, group=group)
    run_checkpoint_phase(0, 2, lambda: None, group=group)

    assert new_group_calls == [([0, 1], "gloo")]
    assert phases == [[0, 1]]
    assert collective_groups == [control_group, control_group]
