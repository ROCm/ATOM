# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dsa_indexer.plan``: a rank's keys of a row, under DCP (interleave 1) and
replicated (128-key runs), against counting every position's owner."""

import pytest

from atom.mono.dsa_indexer.plan import REPLICA_INTERLEAVE, key_pos, local_len

CTXS = [0, 1, 7, 127, 128, 129, 1023, 1024, 1025, 2049, 5000, 33333, 131073]


@pytest.mark.parametrize("interleave", [1, REPLICA_INTERLEAVE])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_a_ranks_keys_are_the_positions_it_owns_in_order(world, interleave):
    for ctx in CTXS:
        owned = [[] for _ in range(world)]
        for pos in range(ctx):
            owned[pos // interleave % world].append(pos)
        for rank in range(world):
            n = local_len(ctx, rank, world, interleave)
            assert n == len(owned[rank]), (ctx, rank)
            got = [key_pos(j, rank, world, interleave) for j in range(n)]
            assert got == owned[rank], (ctx, rank)
