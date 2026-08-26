# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounds on the barriers multi-node can hang at (M-DPSYNC).

The failure being designed against is not an exception, it is silence: a node
that never started leaves every other rank waiting on a barrier that logs
nothing. What is tested here is the message that replaces the silence, and the
port sequence that has to agree across ranks for a group to form at all.
"""

from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

try:
    from atom.config import ParallelConfig
    from atom.model_engine.engine_core_mgr import CoreManager
except ImportError as _e:  # transformers/zmq absent on a bare runner
    skip_if_dependency_missing(_e, "requires atom engine import env")


class TestReadyTimeoutMessage:
    """The coordinator is the only process that can say which node is missing."""

    def _message(self, ready, *, per_node=4, timeout_s=120.0):
        mgr = SimpleNamespace(label="Engine Core Mgr")
        return CoreManager._ready_timeout_message(mgr, ready, timeout_s, per_node)

    def test_groups_missing_ranks_by_node(self):
        # 8 engines over 2 nodes; the whole second node never reported.
        message = self._message([True] * 4 + [False] * 4)
        assert "node 1: dp ranks [4, 5, 6, 7]" in message
        assert "node 0" not in message

    def test_counts_against_the_total(self):
        message = self._message([True] * 4 + [False] * 4)
        assert "4 of 8 engines" in message

    def test_names_the_timeout(self):
        message = self._message([False] * 8, timeout_s=90.0)
        assert "90s" in message

    def test_partial_node_is_reported_as_such(self):
        # One engine down on an otherwise healthy node is a different problem
        # from a node that never launched, and the message has to distinguish.
        message = self._message([True] * 5 + [False] + [True] * 2)
        assert "node 1: dp ranks [5]" in message

    def test_falls_back_when_node_width_is_unknown(self):
        mgr = SimpleNamespace(label="Engine Core Mgr")
        message = CoreManager._ready_timeout_message(
            mgr, [True, False, False], 30.0, None
        )
        assert "dp ranks [1, 2]" in message

    def test_says_what_to_check(self):
        message = self._message([True] * 4 + [False] * 4)
        assert "Fix:" in message
        assert "--data-parallel-master-ip" in message


class TestInitPortSequence:
    """All ranks must walk the same port sequence or the group never forms."""

    def _pc(self, rank):
        return ParallelConfig(
            data_parallel_size=8,
            data_parallel_size_local=4,
            data_parallel_rank=rank,
            data_parallel_master_port=29500,
        )

    def test_every_rank_gets_the_same_first_port(self):
        assert {self._pc(r).get_next_dp_init_port() for r in (0, 4)} == {29500}

    def test_every_rank_gets_the_same_second_port(self):
        # The previous rule advanced by data_parallel_rank, so rank 0 stayed on
        # the base port while every other rank moved a different distance --
        # the second group was formed on a different port per rank.
        seconds = set()
        for rank in (0, 4):
            pc = self._pc(rank)
            pc.get_next_dp_init_port()
            seconds.add(pc.get_next_dp_init_port())
        assert seconds == {29501}

    def test_sequence_is_contiguous(self):
        pc = self._pc(0)
        assert [pc.get_next_dp_init_port() for _ in range(4)] == [
            29500,
            29501,
            29502,
            29503,
        ]


class TestDpSyncTimeout:
    def test_none_means_backend_default(self):
        pc = ParallelConfig(data_parallel_size=1)
        assert pc.dp_sync_timeout is None

    def test_seconds_become_a_timedelta(self):
        pc = ParallelConfig(data_parallel_size=1, dp_sync_timeout_s=90)
        assert pc.dp_sync_timeout.total_seconds() == pytest.approx(90.0)
