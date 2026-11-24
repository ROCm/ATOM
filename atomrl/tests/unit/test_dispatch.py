# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    sys.modules["resource"] = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=0),
    )

from atomrl.controller.dispatch import (
    DISPATCH_MODE_FN_REGISTRY,
    DispatchMode,
    collect_proto,
    dispatch_proto,
)
from atomrl.controller.worker_group_factory import resolve_worker_class
from atomrl.core.protocol import DataProto
from atomrl.trainer.rl_trainer import RLTrainer


def test_dispatch_even_split() -> None:
    p = DataProto(tensors={"x": torch.arange(12).view(4, 3).float()})
    chunks = dispatch_proto(p, num_workers=2)
    assert len(chunks) == 2
    assert chunks[0].batch_size == 2
    assert chunks[1].batch_size == 2
    recomposed = torch.cat([c["x"] for c in chunks], dim=0)
    assert torch.allclose(recomposed, p["x"])


def test_collect_merges() -> None:
    parts = [
        DataProto(tensors={"x": torch.zeros(2, 1)}),
        DataProto(tensors={"x": torch.ones(3, 1)}),
    ]
    merged = collect_proto(parts)
    assert merged.batch_size == 5
    assert torch.allclose(merged["x"][:2], torch.zeros(2, 1))
    assert torch.allclose(merged["x"][2:], torch.ones(3, 1))


def test_dispatch_one_to_all_keeps_identical_batch_size() -> None:
    p = DataProto(tensors={"x": torch.arange(6).view(3, 2).float()})
    chunks = dispatch_proto(p, num_workers=3, mode=DispatchMode.ONE_TO_ALL)
    assert len(chunks) == 3
    assert all(c.batch_size == p.batch_size for c in chunks)


def test_dispatch_nd_mapping() -> None:
    p = DataProto(tensors={"x": torch.arange(20).view(10, 2).float()})
    # Workers [0,1] share group 0; [2,3] share group 1.
    chunks = dispatch_proto(
        p,
        num_workers=4,
        mode=DispatchMode.DP_COMPUTE_PROTO,
        mesh_mapping=[0, 0, 1, 1],
    )
    assert len(chunks) == 4
    assert chunks[0].batch_size == chunks[1].batch_size
    assert chunks[2].batch_size == chunks[3].batch_size


def test_dispatch_rank_zero_returns_single_chunk() -> None:
    p = DataProto(tensors={"x": torch.arange(6).view(3, 2).float()})
    chunks = dispatch_proto(p, num_workers=4, mode=DispatchMode.RANK_ZERO)
    assert len(chunks) == 1
    assert chunks[0].batch_size == p.batch_size


def test_collect_rank_zero_returns_first_result() -> None:
    first = DataProto(tensors={"x": torch.zeros(2, 1)})
    second = DataProto(tensors={"x": torch.ones(2, 1)})
    merged = collect_proto([first, second], mode=DispatchMode.RANK_ZERO)
    assert merged.batch_size == first.batch_size
    assert torch.allclose(merged["x"], first["x"])


def test_registry_contains_extended_dispatch_modes() -> None:
    expected_modes = {
        DispatchMode.RANK_ZERO,
        DispatchMode.ONE_TO_ALL,
        DispatchMode.ALL_TO_ALL,
        DispatchMode.DP_COMPUTE,
        DispatchMode.DP_COMPUTE_PROTO,
        DispatchMode.DP_COMPUTE_PROTO_WITH_FUNC,
        DispatchMode.DP_COMPUTE_METRIC,
        DispatchMode.DIRECT_ROLLOUT_METHOD,
    }
    assert expected_modes.issubset(set(DISPATCH_MODE_FN_REGISTRY))


def test_dispatch_broadcast_alias_maps_to_one_to_all() -> None:
    p = DataProto(tensors={"x": torch.arange(6).view(3, 2).float()})
    chunks = dispatch_proto(p, num_workers=3, mode="broadcast")
    assert len(chunks) == 3
    assert all(c.batch_size == p.batch_size for c in chunks)


def test_direct_rollout_mode_is_forbidden() -> None:
    p = DataProto(tensors={"x": torch.arange(6).view(3, 2).float()})
    with pytest.raises(RuntimeError, match="Direct rollout call is forbidden"):
        dispatch_proto(p, num_workers=2, mode=DispatchMode.DIRECT_ROLLOUT_METHOD)


def test_resolve_worker_class_from_role_key() -> None:
    cls = resolve_worker_class("actor.default")
    assert cls.__name__ == "AtomActorWorker"


def test_ray_actor_dispatch_preserves_full_global_batch_size(monkeypatch) -> None:
    received = []

    class FakeActorGroup:
        num_workers = 2
        _collect_mask = None

        @property
        def dp_rank_mapping(self):
            return [0, 1]

        @staticmethod
        def call_single_async(rank, method, batch):
            assert method == "update_policy"
            received.append((rank, batch))
            return {"loss": float(rank)}

    fake_ray = SimpleNamespace(put=lambda value: value, get=lambda refs: refs)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    trainer = RLTrainer.__new__(RLTrainer)
    trainer._actor_wg = FakeActorGroup()
    trainer._ray_dispatch_state = {}
    trainer.config = SimpleNamespace(
        controller=SimpleNamespace(
            ray=SimpleNamespace(
                actor=SimpleNamespace(
                    dispatch_mode="dp_compute_proto",
                    lazy_dispatch_key=None,
                )
            )
        )
    )
    batch = DataProto(tensors={"x": torch.arange(8).view(4, 2)})

    trainer._update_actor_with_ray(batch)

    assert batch.meta["global_batch_size"] == 4
    assert len(received) == 2
    assert all(shard.batch_size == 2 for _, shard in received)
    assert all(shard.meta["global_batch_size"] == 4 for _, shard in received)
