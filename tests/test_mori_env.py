# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoRI's multi-node environment (M-MOE).

Pure planning and validation; nothing here touches a process. The applier is
covered through the plan it applies.
"""

import pytest

from atom.model_ops.fused_moe.mori_env import check_mori_env, plan_mori_env


class TestPlan:
    def test_single_node_changes_nothing(self):
        # The single-node defaults are what production has been running on.
        assert plan_mori_env(nnodes=1, current={}) == {}

    def test_multinode_sets_heap_and_launch_mode(self):
        assert plan_mori_env(nnodes=2, current={}) == {
            "MORI_SHMEM_HEAP_SIZE": "8G",
            "MORI_EP_LAUNCH_CONFIG_MODE": "AUTO",
        }

    def test_operator_settings_win(self):
        # Someone who exported these did it against a specific fabric; this
        # has strictly less information than they do.
        current = {"MORI_SHMEM_HEAP_SIZE": "16G", "MORI_EP_LAUNCH_CONFIG_MODE": "MANUAL"}
        assert plan_mori_env(nnodes=4, current=current) == {}

    def test_partial_override_fills_only_the_gap(self):
        current = {"MORI_SHMEM_HEAP_SIZE": "16G"}
        assert plan_mori_env(nnodes=2, current=current) == {
            "MORI_EP_LAUNCH_CONFIG_MODE": "AUTO"
        }

    def test_empty_string_counts_as_unset(self):
        assert "MORI_SHMEM_HEAP_SIZE" in plan_mori_env(
            nnodes=2, current={"MORI_SHMEM_HEAP_SIZE": ""}
        )


class TestCheck:
    @pytest.mark.parametrize("ifname", ["lo", "LO", "lo0", "localhost"])
    def test_loopback_across_nodes_is_rejected(self, ifname):
        # A single-node run needs lo, and that value survives into the
        # multi-node launcher. Wrong, it hangs the first collective silently.
        with pytest.raises(ValueError, match="MORI_SOCKET_IFNAME"):
            check_mori_env(nnodes=2, current={"MORI_SOCKET_IFNAME": ifname})

    def test_loopback_on_one_node_is_correct(self):
        check_mori_env(nnodes=1, current={"MORI_SOCKET_IFNAME": "lo"})

    def test_real_interface_passes(self):
        check_mori_env(nnodes=2, current={"MORI_SOCKET_IFNAME": "ens3"})

    def test_unset_interface_passes(self):
        # MoRI has its own discovery; only an actively wrong value is rejected.
        check_mori_env(nnodes=2, current={})

    def test_message_says_how_to_find_the_interface(self):
        with pytest.raises(ValueError) as exc:
            check_mori_env(nnodes=2, current={"MORI_SOCKET_IFNAME": "lo"})
        assert "Fix:" in str(exc.value)
        assert "ip -o addr show" in str(exc.value)
