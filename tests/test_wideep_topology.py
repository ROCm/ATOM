# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for M-TOPO (WideEPTopology)."""

from types import SimpleNamespace

import pytest

from atom.model_engine.topology import (
    WideEPTopology,
    format_startup_summary,
    node_count,
    parse_dist_init_addr,
)


def _legacy_local_engine_count(
    *, dp_attention: bool, raw_tp: int, raw_dp: int, pp_size: int = 1
) -> int:
    """Mirror CoreManager's engine-count fold for Gate 0 regression.

    engine_core_mgr.py: iter_dp_rank_assignments + the enable_dp_attention
    branch. Excludes the --fake-eplb shrink, which the topology refuses to
    describe (see TestSimulatedDeployment).
    """
    if dp_attention:
        assert pp_size == 1
        return raw_tp * raw_dp
    return raw_dp * pp_size


class TestParseDistInitAddr:
    def test_ipv4(self):
        assert parse_dist_init_addr("10.0.0.5:29500") == ("10.0.0.5", 29500)

    def test_ipv6(self):
        assert parse_dist_init_addr("[::1]:30000") == ("::1", 30000)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid dist_init_addr"):
            parse_dist_init_addr("no-port")


class TestGate0Regression:
    """nnodes=1 values must match today's engine_core_mgr derivation."""

    @pytest.mark.parametrize(
        "raw_tp, raw_dp",
        [(8, 1), (4, 2), (2, 4)],
    )
    def test_dp_attention_matches_legacy(self, raw_tp, raw_dp):
        topo = WideEPTopology.create(
            dp_attention=True,
            raw_tp_size=raw_tp,
            raw_dp_size=raw_dp,
        )
        expected_count = _legacy_local_engine_count(
            dp_attention=True, raw_tp=raw_tp, raw_dp=raw_dp
        )
        assert topo.tp_size == 1
        assert topo.global_dp_size == raw_tp * raw_dp
        assert topo.local_engine_count == expected_count
        assert topo.ep_size == topo.global_dp_size
        assert topo.gpu_per_node == expected_count
        assert not topo.is_multinode
        assert topo.dist_init_host is None
        assert topo.dist_init_base_port is None

        for i in range(expected_count):
            assert topo.dp_rank(i) == i
            assert topo.dp_rank_local(i) == i
            assert topo.local_device_rank(i, tp_rank=0) == i

    @pytest.mark.parametrize("raw_dp", [1, 2, 4])
    def test_no_dp_attention_matches_legacy(self, raw_dp):
        topo = WideEPTopology.create(
            dp_attention=False,
            raw_tp_size=8,
            raw_dp_size=raw_dp,
        )
        expected_count = _legacy_local_engine_count(
            dp_attention=False, raw_tp=8, raw_dp=raw_dp
        )
        assert topo.tp_size == 8
        assert topo.global_dp_size == raw_dp
        assert topo.local_engine_count == expected_count
        # Without the DP-attention flatten, FusedMoEParallelConfig.make leaves
        # ep_size at tp_size: EP spans a TP group and there are raw_dp
        # independent groups. Asserting the multinode identity here instead
        # would only be checking the view against itself.
        assert topo.ep_size == 8
        assert topo.gpu_per_node == raw_dp * 8


class TestMultinodeLayout:
    def test_gate1_logical_node(self):
        topo0 = WideEPTopology.create(
            nnodes=2,
            node_rank=0,
            dp_attention=True,
            raw_tp_size=8,
            raw_dp_size=1,
            dist_init_addr="127.0.0.1:29500",
        )
        topo1 = WideEPTopology.create(
            nnodes=2,
            node_rank=1,
            dp_attention=True,
            raw_tp_size=8,
            raw_dp_size=1,
            dist_init_addr="127.0.0.1:29500",
        )
        assert topo0.ep_size == 8
        assert topo0.gpu_per_node == 4
        assert topo0.local_engine_count == 4
        assert topo0.rendezvous_port_world == 29500
        assert topo0.rendezvous_port_dp_gloo == 29501
        assert topo0.rendezvous_port_reserved(0) == 29502
        assert topo0.rendezvous_port_reserved(5) == 29507

        assert topo0.dp_rank(0) == 0
        assert topo0.dp_rank(3) == 3
        assert topo1.dp_rank(0) == 4
        assert topo1.dp_rank(3) == 7
        assert topo0.dp_rank_local(2) == 2
        assert topo1.dp_rank_local(2) == 2

    def test_ep16_two_nodes(self):
        topo = WideEPTopology.create(
            nnodes=2,
            node_rank=0,
            dp_attention=True,
            raw_tp_size=8,
            raw_dp_size=2,
            dist_init_addr="192.168.1.10:30000",
        )
        assert topo.ep_size == 16
        assert topo.gpu_per_node == 8
        assert topo.dist_init_host == "192.168.1.10"


class _FakeParallelConfig(SimpleNamespace):
    """Duck-typed stand-in: atom.config pulls torch, which unit tests avoid."""

    def __init__(self, **kw):
        kw.setdefault("data_parallel_master_ip", "10.0.0.1")
        kw.setdefault("data_parallel_master_port", 29500)
        super().__init__(**kw)


class TestFromParallelConfig:
    """Derivation from the DP fields CoreManager owns."""

    def test_fold_is_a_change_of_units(self):
        """Pre- and post-fold ParallelConfig must yield the same topology.

        CoreManager rewrites the DP fields into engine units (scaling them by
        tp_size and setting tp_size to 1). Every quantity this view exposes is
        a ratio or a product of those, so the rewrite must not move it. This
        test is what makes that invariant executable rather than assumed.
        """
        pre = _FakeParallelConfig(
            data_parallel_size=2, data_parallel_size_local=1, data_parallel_rank=1
        )
        post = _FakeParallelConfig(
            data_parallel_size=16, data_parallel_size_local=8, data_parallel_rank=8
        )
        a = WideEPTopology.from_parallel_config(
            pre, tensor_parallel_size=8, dp_attention=True
        )
        b = WideEPTopology.from_parallel_config(
            post, tensor_parallel_size=1, dp_attention=True
        )
        assert a == b
        assert (a.nnodes, a.node_rank, a.ep_size, a.gpu_per_node) == (2, 1, 16, 8)

    def test_single_node_defaults_local_to_global(self):
        pc = _FakeParallelConfig(
            data_parallel_size=1, data_parallel_size_local=None, data_parallel_rank=0
        )
        topo = WideEPTopology.from_parallel_config(
            pc, tensor_parallel_size=8, dp_attention=True
        )
        assert not topo.is_multinode
        assert (topo.nnodes, topo.ep_size, topo.gpu_per_node) == (1, 8, 8)

    def test_ragged_split_rejected(self):
        """EPLB's hierarchical placement asserts num_gpus % num_nodes == 0."""
        pc = _FakeParallelConfig(
            data_parallel_size=8, data_parallel_size_local=3, data_parallel_rank=0
        )
        with pytest.raises(ValueError, match="must be divisible by"):
            WideEPTopology.from_parallel_config(
                pc, tensor_parallel_size=1, dp_attention=True
            )

    def test_rank_offset_must_align_to_slice(self):
        pc = _FakeParallelConfig(
            data_parallel_size=8, data_parallel_size_local=4, data_parallel_rank=3
        )
        with pytest.raises(ValueError, match="must be a multiple of"):
            WideEPTopology.from_parallel_config(
                pc, tensor_parallel_size=1, dp_attention=True
            )


class TestSimulatedDeployment:
    """--fake-eplb makes the DP fields describe a width, not a node split."""

    def test_simulation_is_rejected_not_guessed(self):
        # -tp 8 on a 4-GPU box: post-fold the fields read exactly like node 0 of
        # a real 2-node deployment, so the ratio must not be taken as a node
        # count. Callers that support simulation handle it explicitly.
        pc = _FakeParallelConfig(
            data_parallel_size=8, data_parallel_size_local=4, data_parallel_rank=0
        )
        with pytest.raises(ValueError, match="--fake-eplb"):
            WideEPTopology.from_parallel_config(
                pc, tensor_parallel_size=1, dp_attention=True, dp_logical_size=8
            )

    def test_not_simulating_is_the_default(self):
        pc = _FakeParallelConfig(
            data_parallel_size=8, data_parallel_size_local=4, data_parallel_rank=0
        )
        topo = WideEPTopology.from_parallel_config(
            pc, tensor_parallel_size=1, dp_attention=True, dp_logical_size=0
        )
        assert topo.nnodes == 2


class TestStartupSummary:
    def _topo(self):
        return WideEPTopology.create(
            nnodes=2,
            node_rank=1,
            dp_attention=True,
            raw_tp_size=8,
            raw_dp_size=2,
            dist_init_addr="10.0.0.1:29500",
        )

    def test_carries_every_field_needed_to_compare_two_nodes(self):
        line = format_startup_summary(
            self._topo(),
            dp_rank=9,
            dp_rank_local=1,
            device_rank=1,
            visible_devices="0,1,2,3,4,5,6,7",
            mori_env={"MORI_SHMEM_HEAP_SIZE": "8G"},
        )
        assert line.startswith("[wideep] ")
        for field in (
            "nnodes=2",
            "node_rank=1",
            "ep=16",
            "gpu_per_node=8",
            "global=16",
            "local=8",
            "rank=9",
            "local_rank=1",
            "device=cuda:1",
            "visible=0,1,2,3,4,5,6,7",
            "MORI_SHMEM_HEAP_SIZE=8G",
        ):
            assert field in line, field

    def test_optional_parts_are_omitted_not_blank(self):
        line = format_startup_summary(
            self._topo(), dp_rank=9, dp_rank_local=1, device_rank=1
        )
        assert "visible=" not in line
        assert "mori:" not in line
        assert "device=cuda:1" in line

    def test_stays_on_one_line(self):
        line = format_startup_summary(
            self._topo(),
            dp_rank=9,
            dp_rank_local=1,
            device_rank=1,
            visible_devices="0,1",
            mori_env={"A": "1", "B": "2"},
        )
        assert "\n" not in line


class TestNodeCount:
    """The one place that answers "how many nodes" including the simulated case."""

    def _config(self, pc, *, tp, dp_attention=True, dp_logical_size=0):
        return SimpleNamespace(
            parallel_config=pc,
            tensor_parallel_size=tp,
            enable_dp_attention=dp_attention,
            dp_logical_size=dp_logical_size,
        )

    def test_reports_the_split(self):
        pc = _FakeParallelConfig(
            data_parallel_size=16, data_parallel_size_local=8, data_parallel_rank=8
        )
        assert node_count(self._config(pc, tp=1)) == 2

    def test_simulation_reports_one_box(self):
        # Same fields as node 0 of a real 2-node run; only dp_logical_size
        # distinguishes them, which is why this rule lives in one place.
        pc = _FakeParallelConfig(
            data_parallel_size=8, data_parallel_size_local=4, data_parallel_rank=0
        )
        assert node_count(self._config(pc, tp=1, dp_logical_size=8)) == 1

    def test_missing_field_is_not_simulating(self):
        # Older Config objects and test doubles have no dp_logical_size.
        pc = _FakeParallelConfig(
            data_parallel_size=8, data_parallel_size_local=4, data_parallel_rank=0
        )
        cfg = SimpleNamespace(
            parallel_config=pc, tensor_parallel_size=1, enable_dp_attention=True
        )
        assert node_count(cfg) == 2


class TestValidation:
    def test_global_dp_not_divisible_suggests_nnodes(self):
        with pytest.raises(ValueError, match="Valid nnodes values"):
            WideEPTopology.create(
                nnodes=3,
                dp_attention=True,
                raw_tp_size=8,
                raw_dp_size=1,
                dist_init_addr="10.0.0.1:29500",
            )

    def test_multinode_requires_dist_init_addr(self):
        with pytest.raises(ValueError, match="dist_init_addr"):
            WideEPTopology.create(
                nnodes=2,
                dp_attention=True,
                raw_tp_size=8,
                raw_dp_size=1,
            )

    def test_multinode_requires_dp_attention(self):
        with pytest.raises(ValueError, match="dp_attention"):
            WideEPTopology.create(
                nnodes=2,
                dp_attention=False,
                raw_tp_size=8,
                raw_dp_size=1,
                dist_init_addr="10.0.0.1:29500",
            )

    def test_invalid_node_rank(self):
        with pytest.raises(ValueError, match="node_rank"):
            WideEPTopology.create(
                nnodes=2,
                node_rank=2,
                dp_attention=True,
                raw_tp_size=8,
                raw_dp_size=1,
                dist_init_addr="10.0.0.1:29500",
            )

    def test_rendezvous_ports_require_multinode(self):
        topo = WideEPTopology.create(dp_attention=True, raw_tp_size=8, raw_dp_size=1)
        with pytest.raises(ValueError, match="rendezvous ports"):
            _ = topo.rendezvous_port_world

    def test_reserved_port_index_bounds(self):
        topo = WideEPTopology.create(
            nnodes=2,
            dp_attention=True,
            raw_tp_size=8,
            raw_dp_size=1,
            dist_init_addr="10.0.0.1:29500",
        )
        with pytest.raises(ValueError, match="reserved port index"):
            topo.rendezvous_port_reserved(6)


class TestLocalDeviceRank:
    def test_with_pcp(self):
        topo = WideEPTopology.create(
            dp_attention=True,
            raw_tp_size=4,
            raw_dp_size=2,
        )
        # tp_size=1, pcp=2 => stage_span=2
        assert topo.local_device_rank(1, tp_rank=0, pcp_size=2) == 2
        assert topo.local_device_rank(1, tp_rank=1, pcp_size=2) == 3


class TestDescribe:
    def test_one_line_summary(self):
        topo = WideEPTopology.create(
            nnodes=2,
            node_rank=1,
            dp_attention=True,
            raw_tp_size=8,
            raw_dp_size=2,
            dist_init_addr="10.0.0.1:29500",
        )
        assert topo.describe() == (
            "[wideep] nnodes=2 node_rank=1 | ep=16 gpu_per_node=8 | "
            "dp: global=16 local=8"
        )
