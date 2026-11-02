# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU contracts for packed gfx1250 K2 tail LDS."""

import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import moe_tail_layout as tail


@pytest.mark.parametrize("tokens", (1, 16, 32, 48))
def test_tail_lds_plan_is_aligned_bounded_and_tiled(tokens):
    layout = tail.tail_lds_plan(tokens, waves=8)
    assert layout.required_bytes > 0
    assert layout.lds_rows == min(tokens, 16)
    assert sum(tile.rows for tile in layout.token_tiles) == tokens
    for regions in (layout.gemv_regions, layout.norm_regions):
        assert all(region.offset % tail.ALIGNMENT == 0 for region in regions)

    plan = Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=320 * 1024,
        tokens=tokens,
        nblocks=1,
    )
    assert layout.required_bytes <= plan.lds_plan().stage_budget_bytes


@pytest.mark.parametrize(("tokens", "waves"), ((0, 8), (49, 8), (1, 0)))
def test_tail_lds_plan_rejects_unsupported_geometry(tokens, waves):
    with pytest.raises(ValueError):
        tail.tail_lds_plan(tokens, waves)
