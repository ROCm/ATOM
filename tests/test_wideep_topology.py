# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for M-TOPO (WideEPTopology)."""

import pytest

from atom.model_engine.topology import WideEPTopology, parse_dist_init_addr


def _legacy_local_engine_count(
    *, dp_attention: bool, raw_tp: int, raw_dp: int, pp_size: int = 1
) -> int:
    """Mirror engine_core_mgr.py:113-130 for Gate 0 regression."""
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
        assert topo.ep_size == topo.gpu_per_node * topo.nnodes


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
        topo = WideEPTopology.create(
            dp_attention=True, raw_tp_size=8, raw_dp_size=1
        )
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
