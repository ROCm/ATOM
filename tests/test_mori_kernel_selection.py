# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoRI's kernel type must follow real topology, not a rank-count guess.

`world_size <= 8` conflates "8 ranks" with "one node". On 2 nodes x 4 GPUs with
EP=8 it selects IntraNode -- whose kernels rely on P2P/XGMI mappings that do
not exist across a node boundary -- while aiter's synchronous handle probes
shared memory and correctly selects InterNodeV1. Prefill and decode would then
run different kernels over one group.
"""

from atom.model_ops.fused_moe.mori_prepare_finalize import select_mori_kernel_params


class TestKernelSelection:
    def test_single_node_selects_intranode(self):
        name, _, _, rdma = select_mori_kernel_params(low_latency=False, internode=False)
        assert name == "IntraNode"
        assert rdma == 0

    def test_internode_selects_rdma_kernel(self):
        name, _, _, rdma = select_mori_kernel_params(low_latency=False, internode=True)
        assert name == "InterNodeV1"
        assert rdma > 0

    def test_low_latency_wins_over_topology(self):
        for internode in (False, True):
            name, _, _, _ = select_mori_kernel_params(
                low_latency=True, internode=internode
            )
            assert name == "AsyncLL"

    def test_geometry_matches_the_original_branches(self):
        """The constants aiter's sync handle uses; a drift here splits the paths."""
        assert select_mori_kernel_params(low_latency=True, internode=False) == (
            "AsyncLL",
            8,
            64,
            32,
        )
        assert select_mori_kernel_params(low_latency=False, internode=False) == (
            "IntraNode",
            16,
            80,
            0,
        )
        assert select_mori_kernel_params(low_latency=False, internode=True) == (
            "InterNodeV1",
            16,
            32,
            16,
        )


class TestNoRankCountHeuristic:
    def test_world_size_no_longer_picks_the_kernel(self):
        import inspect

        from atom.model_ops.fused_moe import mori_prepare_finalize

        src = inspect.getsource(mori_prepare_finalize.init_mori_op)
        assert (
            "world_size <= 8" not in src
        ), "kernel choice must come from the topology probe, not a rank count"

    def test_init_mori_op_takes_internode(self):
        import inspect

        from atom.model_ops.fused_moe.mori_prepare_finalize import init_mori_op

        assert "internode" in inspect.signature(init_mori_op).parameters


class TestCallerPassesRealTopology:
    def test_moe_forwards_the_all2all_managers_probe(self):
        import inspect

        from atom.model_ops import moe

        src = inspect.getsource(moe.FusedMoEMethodBase._maybe_make_prepare_finalize)
        assert "all2all_manager.internode" in src, (
            "the TBO path must reuse the same topology probe as the sync handle, "
            "otherwise prefill and decode can pick different kernels"
        )
