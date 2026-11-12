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
