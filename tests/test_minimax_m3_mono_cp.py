# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Indexer context parallelism in the wide builds: the block-ownership arithmetic the scorer, the selection and
the exchange share (``index_score.own_first`` / the per-request task counts, ``select_long``'s shares), on the CPU.
"""

import pytest

from atom.models.minimax_m3.mono.config import LONG_FROM_BLOCKS, TOPK_BLOCKS, TP


def own_first(rank, init):  # index_score.own_first
    return init + (rank + TP - init % TP) % TP


def n_own(
    nb, rank, init, local
):  # the scorer's per-request task count (index_score, BIG plan)
    scored = nb - (init + local) if nb > TOPK_BLOCKS else 0
    return max(init + scored - own_first(rank, init) + TP - 1, 0) // TP


@pytest.mark.parametrize("init", [0, 1])
@pytest.mark.parametrize("nb", [LONG_FROM_BLOCKS + 1, 700, 1024, 2051, 8192])
def test_the_ranks_own_blocks_partition_the_scored_blocks(nb, init):
    local = 1
    seen = []
    for rank in range(TP):
        first, n = own_first(rank, init), n_own(nb, rank, init, local)
        blocks = [first + TP * i for i in range(n)]
        assert all(b % TP == rank for b in blocks)  # ATOM's owner(block) = block % TP
        seen += blocks
    assert sorted(seen) == list(
        range(init, nb - local)
    )  # every scored block once, none pinned


@pytest.mark.parametrize("nb", [LONG_FROM_BLOCKS + 1, 1024, 8192])
def test_a_share_holds_at_least_topk_blocks(nb):
    # select_long: half (share % 2) of a rank's n_own blocks; the merge needs TOPK_BLOCKS distinct ranks each
    for rank in range(TP):
        n = n_own(nb, rank, 1, 1)
        assert (n + 1) // 2 >= TOPK_BLOCKS and n - (n + 1) // 2 >= TOPK_BLOCKS


def test_cp_step_statistics_follow_the_kernel_gate():
    """MonoDecode._note_cp: the host-side image of the CP gate (statistics only)."""
    from types import SimpleNamespace as NS

    import numpy as np

    from atom.models.minimax_m3.mono.dispatch import MonoDecode
    from atom.models.minimax_m3.mono.policy import IndexerCp

    def run(n, lens, cp):
        me = NS(cp=cp, cp_steps={"taken": {}, "gate_closed": {}})
        MonoDecode._note_cp(me, n, np.asarray(lens, dtype=np.int32), 1)
        return me.cp_steps

    cp = IndexerCp(True, 16, 4096)
    big = 66000  # > 512 blocks
    assert run(16, [big] * 16, cp) == {"taken": {16: 1}, "gate_closed": {}}  # narrow
    assert run(32, [big] * 32, cp)["taken"] == {32: 1}  # 32 * 516 blocks >= 4096
    assert run(32, [big] * 4 + [100] * 28, cp)["gate_closed"] == {32: 1}  # 2064 < 4096
    assert run(32, [100] * 32, cp) == {
        "taken": {},
        "gate_closed": {},
    }  # no long request
    assert run(32, [big] * 32, IndexerCp(False)) == {"taken": {}, "gate_closed": {}}
