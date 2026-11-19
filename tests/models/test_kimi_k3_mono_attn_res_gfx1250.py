# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 AttnRes mix kernel."""

import pytest

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import attn_res_gfx1250 as mix


def test_a_cta_holds_a_token_seven_columns_a_thread():
    assert mix.THREADS * mix.ELEMENTS == HIDDEN
    # every candidate's two per-wave sums sit a wave a row
    assert 2 * mix.MAX_CANDIDATES <= mix.WAVES


@pytest.mark.parametrize("count", (1, 4, 8))
@pytest.mark.parametrize("addends", (0, 1, 2))
def test_builds_take_the_mix_abi(count, addends):
    kernel = mix.build_attn_res(count, addends, True, 1e-6, 1e-5)
    assert kernel.abi is mix.ATTN_RES_ABI
    assert kernel.name == f"k3_gfx1250_attn_res_b{count}_a{addends}_n"


@pytest.mark.parametrize(
    ("count", "addends"), ((0, 1), (mix.MAX_CANDIDATES, 1), (4, 3))
)
def test_refuses_what_it_cannot_hold(count, addends):
    with pytest.raises(ValueError):
        mix.build_attn_res(count, addends, True, 1e-6, 1e-5)
