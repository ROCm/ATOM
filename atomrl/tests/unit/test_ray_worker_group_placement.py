# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import pytest
from atomrl.controller.ray_worker_group import _placement_node_ips


def test_placement_node_ips_assigns_one_ip_per_process_group() -> None:
    tags = {"node_ips": "10.0.0.1, 10.0.0.2"}

    assert _placement_node_ips(tags, 2) == ["10.0.0.1", "10.0.0.2"]


def test_placement_node_ips_preserves_single_node_compatibility() -> None:
    assert _placement_node_ips({"node_ip": "10.0.0.1"}, 1) == ["10.0.0.1"]
    assert _placement_node_ips({}, 2) == ["", ""]


def test_placement_node_ips_rejects_count_mismatch() -> None:
    with pytest.raises(
        ValueError, match="one entry per positive process_on_nodes group"
    ):
        _placement_node_ips({"node_ips": "10.0.0.1"}, 2)


def test_placement_node_ips_rejects_ambiguous_tags() -> None:
    with pytest.raises(ValueError, match="cannot define both"):
        _placement_node_ips(
            {"node_ip": "10.0.0.1", "node_ips": "10.0.0.1,10.0.0.2"},
            2,
        )
