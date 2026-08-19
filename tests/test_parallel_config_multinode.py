# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ParallelConfig must distinguish the global DP topology from this node's slice.

`data_parallel_size_local` previously defaulted to 1, which made a single-node
`-dp 8` run report `gpu_per_node=tp_size` to MoRI instead of 8. It now defaults
to the global size, so the single-node case -- where local IS global -- is
correct without the operator setting anything.
"""

import pytest

from atom.config import ParallelConfig


class TestLocalSizeDefaulting:
    def test_local_size_defaults_to_global(self):
        pc = ParallelConfig(data_parallel_size=8)
        assert pc.data_parallel_size_local == 8

    def test_explicit_local_size_is_kept(self):
        pc = ParallelConfig(data_parallel_size=8, data_parallel_size_local=4)
        assert pc.data_parallel_size_local == 4


class TestIsMultinodeDP:
    def test_single_node_is_not_multinode(self):
        pc = ParallelConfig(data_parallel_size=8)
        assert pc.is_multinode_dp is False

    def test_partial_local_slice_is_multinode(self):
        pc = ParallelConfig(data_parallel_size=8, data_parallel_size_local=4)
        assert pc.is_multinode_dp is True

    def test_nonzero_rank_offset_is_multinode(self):
        pc = ParallelConfig(
            data_parallel_size=8, data_parallel_size_local=4, data_parallel_rank=4
        )
        assert pc.is_multinode_dp is True

    def test_default_single_rank_is_not_multinode(self):
        assert ParallelConfig().is_multinode_dp is False


class TestValidation:
    def test_rejects_zero_global_size(self):
        with pytest.raises(ValueError, match="data_parallel_size must be at least 1"):
            ParallelConfig(data_parallel_size=0)

    def test_rejects_zero_local_size(self):
        with pytest.raises(
            ValueError, match="data_parallel_size_local must be at least 1"
        ):
            ParallelConfig(data_parallel_size=4, data_parallel_size_local=0)

    def test_rejects_negative_rank(self):
        with pytest.raises(ValueError, match="data_parallel_rank must be non-negative"):
            ParallelConfig(data_parallel_size=4, data_parallel_rank=-1)

    def test_rejects_slice_overrunning_global_size(self):
        with pytest.raises(ValueError, match="must not exceed"):
            ParallelConfig(
                data_parallel_size=8,
                data_parallel_size_local=4,
                data_parallel_rank=6,
            )

    def test_accepts_exactly_fitting_last_node(self):
        pc = ParallelConfig(
            data_parallel_size=8, data_parallel_size_local=4, data_parallel_rank=4
        )
        assert pc.data_parallel_rank + pc.data_parallel_size_local == 8

    def test_zero_global_size_raises_size_error_not_local_error(self):
        """Validation ordering: size<1 must be caught before None-defaulting.

        With data_parallel_size=0, the None-defaulting would copy the invalid 0
        into data_parallel_size_local, which could then trigger the local error
        instead of the global one.  The correct error is "data_parallel_size must
        be at least 1" — if we get the local error, the ordering is wrong.
        """
        with pytest.raises(ValueError, match="data_parallel_size must be at least 1"):
            ParallelConfig(data_parallel_size=0)


class TestComputeHash:
    def test_local_size_changes_the_hash(self):
        """It feeds MoRI's gpu_per_node, so it must key the compile cache."""
        a = ParallelConfig(data_parallel_size=8, data_parallel_size_local=8)
        b = ParallelConfig(data_parallel_size=8, data_parallel_size_local=4)
        assert a.compute_hash() != b.compute_hash()


class TestEnvOverride:
    def test_env_sets_local_size(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SIZE_LOCAL", "2")
        pc = ParallelConfig(data_parallel_size=8)
        assert pc.data_parallel_size_local == 2

    def test_unset_env_leaves_default(self, monkeypatch):
        monkeypatch.delenv("ATOM_DP_SIZE_LOCAL", raising=False)
        pc = ParallelConfig(data_parallel_size=6)
        assert pc.data_parallel_size_local == 6
