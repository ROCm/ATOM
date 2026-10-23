# SPDX-License-Identifier: MIT
"""Candidate addresses use the same pure function as the selection kernel.

FULL history rows can exceed the compact REINDEX domain. Even though FULL
discards the candidate value, its speculative load must stay in its token's
allocated row. All division in the shared function has nonnegative operands.
"""

import pytest

from atom.models.deepseek_v41.mono import index_plan as ip

TOKENS = [0, 1, 35]
COMPACT_ROWS = ip.TOPK_BLOCKS * ip.BLOCK_ROWS


@pytest.mark.parametrize("t", TOKENS)
@pytest.mark.parametrize(
    "ident",
    [
        -(1 << 31),
        -1,
        0,
        COMPACT_ROWS - 1,
        COMPACT_ROWS,
        316436,
        (1 << 20) - 1,
        (1 << 31) - 1,
    ],
)
def test_speculative_candidate_address_stays_in_token_row(t, ident):
    offset = ip.candidate_offset(t, ident)
    assert t * ip.TOPK_BLOCKS <= offset < (t + 1) * ip.TOPK_BLOCKS
    if ident < 0:
        assert offset == t * ip.TOPK_BLOCKS
    if ident >= COMPACT_ROWS:
        assert offset == (t + 1) * ip.TOPK_BLOCKS - 1


@pytest.mark.parametrize("t", TOKENS)
def test_every_live_reindex_column_keeps_its_candidate(t):
    for block in range(ip.TOPK_BLOCKS):
        for within in range(ip.BLOCK_ROWS):
            ident = block * ip.BLOCK_ROWS + within
            assert ip.candidate_offset(t, ident) == t * ip.TOPK_BLOCKS + block


def test_last_token_full_history_regression():
    # The failing FULL input read slot 111234 from a 73728-element allocation.
    assert 35 * ip.TOPK_BLOCKS + 316436 // ip.BLOCK_ROWS == 111234
    assert ip.candidate_offset(35, 316436) == 36 * ip.TOPK_BLOCKS - 1
