# SPDX-License-Identifier: MIT
# Unit tests for the inter-node DP+EP smoke runner's pure launch math.

import importlib.util
from pathlib import Path


def _mod():
    path = (
        Path(__file__).resolve().parents[1]
        / "atom"
        / "benchmarks"
        / "internode_dp_ep_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("internode_dp_ep_smoke_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_worker_specs_use_global_dp_rank_offset():
    m = _mod()

    specs = m.build_worker_specs(
        data_parallel_size=16,
        data_parallel_size_local=8,
        data_parallel_rank=8,
        tensor_parallel_size=1,
        prefill_context_parallel_size=1,
    )

    assert [
        (s.global_dp_rank, s.local_dp_rank, s.model_rank, s.local_rank) for s in specs
    ] == [
        (8, 0, 0, 0),
        (9, 1, 0, 1),
        (10, 2, 0, 2),
        (11, 3, 0, 3),
        (12, 4, 0, 4),
        (13, 5, 0, 5),
        (14, 6, 0, 6),
        (15, 7, 0, 7),
    ]


def test_worker_specs_include_tp_model_ranks_per_local_dp_rank():
    m = _mod()

    specs = m.build_worker_specs(
        data_parallel_size=4,
        data_parallel_size_local=2,
        data_parallel_rank=2,
        tensor_parallel_size=2,
        prefill_context_parallel_size=1,
    )

    assert [
        (s.global_dp_rank, s.local_dp_rank, s.model_rank, s.local_rank) for s in specs
    ] == [
        (2, 0, 0, 0),
        (2, 0, 1, 1),
        (3, 1, 0, 2),
        (3, 1, 1, 3),
    ]


def test_validate_topology_rejects_rank_range_overflow():
    m = _mod()

    try:
        m.validate_topology(
            data_parallel_size=8,
            data_parallel_size_local=4,
            data_parallel_rank=6,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            visible_gpu_count=8,
        )
    except ValueError as exc:
        assert "exceeds global data_parallel_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_topology_rejects_insufficient_visible_gpus():
    m = _mod()

    try:
        m.validate_topology(
            data_parallel_size=16,
            data_parallel_size_local=8,
            data_parallel_rank=0,
            tensor_parallel_size=2,
            prefill_context_parallel_size=1,
            visible_gpu_count=8,
        )
    except ValueError as exc:
        assert "requires 16 local GPU workers" in str(exc)
    else:
        raise AssertionError("expected ValueError")
