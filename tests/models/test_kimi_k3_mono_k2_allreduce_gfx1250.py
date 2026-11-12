# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 in-kernel K2 all-reduce."""

import pytest

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar


@pytest.mark.parametrize("rows", (1, 5, 16, 40, 48))
def test_lines_cover_the_routed_and_shared_partials(rows):
    lines = ar.allreduce_lines(rows)
    assert lines * ar.LINE_ELEMENTS == rows * (LATENT + HIDDEN)
    ctas = ar.allreduce_ctas(rows)
    assert 1 <= ctas <= ar.MAX_CTAS
    # one line a thread, so no thread loads after its own remote stores
    assert ctas * ar.allreduce_threads(rows) >= lines
    # every epoch counter is in some thread's reach
    assert ctas * ar.allreduce_threads(rows) >= ar.MAX_CTAS


def test_peer_banks_hold_two_epochs_of_every_source():
    assert ar.allreduce_peer_bytes(8, 40) == 2 * 8 * ar.allreduce_lines(40) * 16
    assert ar.allreduce_peer_bytes(8, 48) > ar.allreduce_peer_bytes(8, 40)


def test_o_proj_all_reduce_gives_each_rank_whole_ctas_of_columns():
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    assert oar.CTAS * oar.CTA_COLUMNS == HIDDEN
    assert oar.CTAS <= ar.MAX_CTAS
    for tp in (2, 4, 8):
        assert HIDDEN % (tp * oar.CTA_COLUMNS) == 0
        assert oar.own_pairs(tp) * 2 * tp == HIDDEN
    # partials from every rank for its own columns, then every column's sum
    assert oar.bank_words(8, 40) == 2 * (8 * 40 * oar.own_pairs(8) + 40 * HIDDEN // 2)
    assert oar.oproj_peer_bytes(8, 48) == 2 * oar.bank_words(8, 48) * 4
