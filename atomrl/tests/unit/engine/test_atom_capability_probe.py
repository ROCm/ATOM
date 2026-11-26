# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import inspect

from atomrl.engine.inference import atom_moe_weight_sync


def test_weight_sync_does_not_probe_atom_private_routes() -> None:
    source = inspect.getsource(atom_moe_weight_sync)
    assert "atom.rollout.weight_updater" not in source
    assert "atom_routes_fused_experts" not in source
