# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic port assignment and node-slice rank mapping.

Multi-node engines cannot use IPC paths, and both ends must derive the same
TCP ports without negotiating. Three ports per rank, not two: main added a
control socket alongside input and output.
"""

from types import SimpleNamespace

from atom.model_engine.engine_core_mgr import (
    build_dp_socket_plan,
    iter_dp_rank_assignments,
)


def _config(*, dp_size, dp_size_local, dp_rank=0, tp_size=1, dp_attention=False):
    return SimpleNamespace(
        tensor_parallel_size=tp_size,
        enable_dp_attention=dp_attention,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_size_local=dp_size_local,
            data_parallel_rank=dp_rank,
            is_multinode_dp=(dp_size_local < dp_size or dp_rank > 0),
        ),
    )


class TestSocketPlan:
    def test_three_ports_per_rank_from_base(self):
        plan = build_dp_socket_plan(engine_count=2, master_port=29500)
        assert len(plan) == 2
        assert (plan[0].input_port, plan[0].output_port, plan[0].control_port) == (
            29600,
            29601,
            29602,
        )
        assert (plan[1].input_port, plan[1].output_port, plan[1].control_port) == (
            29603,
            29604,
            29605,
        )

    def test_no_port_is_reused(self):
        plan = build_dp_socket_plan(engine_count=8, master_port=29500)
        ports = [p for e in plan for p in (e.input_port, e.output_port, e.control_port)]
        assert len(ports) == len(set(ports)) == 24

    def test_same_inputs_yield_the_same_plan(self):
        """Each node derives the plan alone, so equal inputs must agree.

        This is within-process determinism only -- it cannot observe a real
        cross-node disagreement, which would come from the two nodes being
        given different master_port or engine_count values.
        """
        a = build_dp_socket_plan(engine_count=4, master_port=31000)
        b = build_dp_socket_plan(engine_count=4, master_port=31000)
        assert a == b

    def test_a_differing_master_port_shifts_every_port(self):
        """The real cross-node failure: mismatched master_port between nodes."""
        a = build_dp_socket_plan(engine_count=2, master_port=29500)
        b = build_dp_socket_plan(engine_count=2, master_port=29501)
        assert a != b
        assert b[0].input_port - a[0].input_port == 1

    def test_rank_field_matches_index(self):
        plan = build_dp_socket_plan(engine_count=3, master_port=29500)
        assert [p.rank for p in plan] == [0, 1, 2]


class TestRankAssignments:
    def test_single_node_local_equals_global(self):
        got = iter_dp_rank_assignments(_config(dp_size=4, dp_size_local=4))
        assert got == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_second_node_offsets_global_but_not_local(self):
        got = iter_dp_rank_assignments(_config(dp_size=8, dp_size_local=4, dp_rank=4))
        assert got == [(4, 0), (5, 1), (6, 2), (7, 3)]

    def test_first_node_of_multinode_run(self):
        got = iter_dp_rank_assignments(_config(dp_size=8, dp_size_local=4, dp_rank=0))
        assert got == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_four_nodes_of_two(self):
        got = iter_dp_rank_assignments(_config(dp_size=8, dp_size_local=2, dp_rank=6))
        assert got == [(6, 0), (7, 1)]

    def test_dp_attention_expands_by_tp_size(self):
        """Under DP-attention each TP rank becomes its own engine."""
        got = iter_dp_rank_assignments(
            _config(dp_size=2, dp_size_local=2, tp_size=2, dp_attention=True)
        )
        assert got == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_dp_attention_second_node_offsets_by_tp_size(self):
        got = iter_dp_rank_assignments(
            _config(dp_size=4, dp_size_local=2, dp_rank=2, tp_size=2, dp_attention=True)
        )
        assert got == [(4, 0), (5, 1), (6, 2), (7, 3)]
