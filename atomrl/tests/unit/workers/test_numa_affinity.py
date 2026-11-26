# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import json
import logging
import sys
from types import SimpleNamespace

from atomrl.workers.numa_affinity import (
    bind_current_process_to_gpu_numa,
    current_physical_gpu_id,
    parse_linux_cpu_list,
    resolve_physical_gpu_id,
)


def test_parse_linux_cpu_list_expands_ranges() -> None:
    assert parse_linux_cpu_list("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}


def test_megatron_numa_affinity_is_opt_in() -> None:
    from atomrl.core.config import MegatronConfig

    assert MegatronConfig().numa_affinity is False


def test_resolve_physical_gpu_id_prefers_ray_assignment() -> None:
    assert resolve_physical_gpu_id(ray_gpu_ids=[5], cuda_visible_devices="2") == 5


def test_resolve_physical_gpu_id_falls_back_to_cuda_visibility() -> None:
    assert resolve_physical_gpu_id(ray_gpu_ids=[], cuda_visible_devices="6") == 6


def test_current_physical_gpu_id_falls_back_after_ray_error(monkeypatch) -> None:
    class FakeRayError(Exception):
        pass

    fake_ray = SimpleNamespace(
        exceptions=SimpleNamespace(RayError=FakeRayError),
        get_gpu_ids=lambda: (_ for _ in ()).throw(FakeRayError("not ready")),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    assert current_physical_gpu_id() == 3


def test_bind_current_process_uses_gpu_numa_node(tmp_path) -> None:
    topology = {
        "card0": {"(Topology) Numa Affinity": "0"},
        "card4": {"(Topology) Numa Affinity": "1"},
    }
    cpulist = tmp_path / "node1" / "cpulist"
    cpulist.parent.mkdir()
    cpulist.write_text("56-59,168-171\n", encoding="utf-8")
    calls = []

    binding = bind_current_process_to_gpu_numa(
        4,
        topology_json=json.dumps(topology),
        node_root=tmp_path,
        set_affinity=lambda pid, cpus: calls.append((pid, cpus)),
    )

    assert binding is not None
    assert binding.physical_gpu_id == 4
    assert binding.numa_node == 1
    assert binding.cpus == frozenset({56, 57, 58, 59, 168, 169, 170, 171})
    assert calls == [(0, set(binding.cpus))]


def test_bind_current_process_is_disabled_without_side_effects(tmp_path) -> None:
    calls = []

    binding = bind_current_process_to_gpu_numa(
        0,
        enabled=False,
        node_root=tmp_path,
        set_affinity=lambda pid, cpus: calls.append((pid, cpus)),
    )

    assert binding is None
    assert calls == []


def test_bind_current_process_fails_open_for_missing_topology(tmp_path) -> None:
    calls = []

    binding = bind_current_process_to_gpu_numa(
        7,
        topology_json="{}",
        node_root=tmp_path,
        set_affinity=lambda pid, cpus: calls.append((pid, cpus)),
    )

    assert binding is None
    assert calls == []


def test_actor_binds_numa_before_engine_initialization(monkeypatch) -> None:
    from atomrl.workers import actor_worker

    events = []

    class FakeEngine:
        def initialize(self) -> None:
            events.append("initialize")

        def get_data_parallel_rank(self) -> int:
            return 0

        def is_mp_src_rank_with_outputs(self) -> bool:
            return False

    worker = object.__new__(actor_worker.AtomActorWorker)
    worker.config = {
        "policy": {
            "model_name": "unused",
            "training_backend": "megatron_lumen_dsv4",
            "training": {"megatron_cfg": {"numa_affinity": True}},
        }
    }
    worker._log = logging.getLogger("test-numa-affinity")
    worker._build_engine_config = lambda *args: {}
    worker._build_optimizer_config = lambda *args: {}
    worker._build_model_config = lambda *args: {}

    monkeypatch.setattr(actor_worker, "current_physical_gpu_id", lambda: 4)
    monkeypatch.setattr(
        actor_worker,
        "bind_current_process_to_gpu_numa",
        lambda gpu_id, **kwargs: events.append(("bind", gpu_id)),
    )
    monkeypatch.setattr(
        actor_worker.EngineRegistry,
        "get_engine_cls",
        lambda **kwargs: lambda **engine_kwargs: FakeEngine(),
    )

    worker.init_model()

    assert events == [("bind", 4), "initialize"]
