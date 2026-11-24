# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from examples.GRPO.dsv4.cluster_gpu_guard import (
    ProcessInfo,
    find_target_tree,
    foreign_containers,
)


def test_foreign_containers_preserves_training_and_monitoring() -> None:
    assert foreign_containers(
        ["dsv4-rl", "node-exporter.service", "kv-indexer-scale", "benchmark"]
    ) == ["kv-indexer-scale", "benchmark"]


def test_find_target_tree_selects_benchmark_and_descendants() -> None:
    processes = {
        10: ProcessInfo(10, 1, "/opt/ROCmTest/runner"),
        11: ProcessInfo(11, 10, "python child.py"),
        12: ProcessInfo(12, 1, "ray::AtomActorWorker"),
    }

    assert find_target_tree(processes) == {10, 11}
