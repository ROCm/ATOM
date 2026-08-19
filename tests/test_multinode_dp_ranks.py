# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Global and node-local DP ranks are different numbers with different jobs.

The global rank identifies a rank within the whole DP group (process-group
formation, expert sharding). The local rank indexes hardware on this node (GPU
index, NUMA node). Using one where the other belongs is the multi-node bug
class these tests pin.
"""

import ast
import pathlib

import atom.model_engine.async_proc as async_proc_mod
import atom.model_engine.engine_core as engine_core_mod


def _source_of(module, func_name, class_name):
    """Source of `class_name.func_name`, read from disk rather than imported.

    engine_core pulls the aiter/GPU chain on import, so parsing the file keeps
    this test runnable on a CPU-only runner.
    """
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == func_name:
                    return ast.unparse(fn)
    raise AssertionError(f"{class_name}.{func_name} not found")


class TestEngineCoreAssertion:
    def test_local_rank_is_bounded_by_local_size_not_global_rank(self):
        """`local_dp_rank <= dp_rank` is a single-node coincidence.

        On the second node of a 2x4 run the local ranks are 0..3 while the
        global ranks are 4..7, and the relation only holds by accident of
        ordering. The local rank's real bound is the local size.
        """
        src = _source_of(engine_core_mod, "_init_data_parallel", "DPEngineCoreProc")
        assert "local_dp_rank <= dp_rank" not in src, (
            "the old chained assertion conflates the local rank with the "
            "global one and rejects valid multi-node topologies"
        )
        assert (
            "data_parallel_size_local" in src
        ), "the local rank must be bounded by the local size"


class TestNumaBinding:
    def test_numa_binds_by_local_rank(self):
        """NUMA binding indexes a GPU on THIS node, so it needs the local rank."""
        src = pathlib.Path(async_proc_mod.__file__).read_text()
        assert "data_parallel_rank_local" in src, (
            "NUMA bind computes a physical GPU index and must use the "
            "node-local DP rank; the global rank overruns this node's GPUs"
        )


class TestDistributedInitLocalRank:
    def test_init_dist_env_receives_the_local_device_rank(self):
        """Without it aiter falls back to the global rank as a device index.

        `init_distributed_environment` scales rank by data_parallel_rank, so on
        the second node of a 2x8 run the fallback yields device 8..15 on a box
        that only has 0..7.
        """
        import pathlib

        import atom.model_engine.model_runner as mr

        src = pathlib.Path(mr.__file__).read_text()
        setup = src.split("def _setup_device_and_distributed")[1].split("\n    def ")[0]
        assert (
            "local_rank=local_device_rank" in setup
        ), "init_dist_env must be told this node's device index explicitly"

    def test_aiter_accepts_the_kwarg(self):
        """Guards against a pinned aiter without the parameter.

        Skipped rather than failed where aiter is absent: this asserts a fact
        about the installed dependency, so on a CPU-only runner without it
        there is nothing to check.
        """
        import inspect

        import pytest

        aiter = pytest.importorskip("aiter")

        assert "local_rank" in inspect.signature(aiter.init_dist_env).parameters
