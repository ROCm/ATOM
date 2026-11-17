# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``recipe.sgpr_spill_ok`` under an explicit ``RESIDUAL`` (the schedule's ``DEFAULT`` or a (kind, rows) entry): a
sharded residual's SGPR rule applies as under "auto", and a replicated one never raises.
"""

import pytest


@pytest.mark.parametrize(
    "residual, spill", [("auto", True), ("sharded", True), ("replicated", False)]
)
def test_window_spill_follows_the_residual(monkeypatch, residual, spill):
    pytest.importorskip("flydsl")
    pytest.importorskip(
        "aiter"
    )  # (the hi device code re-exports atom.mono.device's primitives)
    from atom.models.deepseek_v41.mono.hi.v41 import recipe, schedules

    monkeypatch.setitem(schedules.DEFAULT, "RESIDUAL", residual)
    # a window layer at 6 rows, TP4: SPILL_OK "auto", SHARD_SPILL_OK on, "auto" shards from 6 rows
    assert recipe.sgpr_spill_ok(0, None, rows=6, tp=4) is spill
