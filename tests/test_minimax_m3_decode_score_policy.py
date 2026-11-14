# SPDX-License-Identifier: MIT

import pytest

from atom.model_ops.minimax_m3.decode_score_policy import (
    decode_score_num_warps,
    use_transposed_decode_score,
)


@pytest.mark.parametrize(
    "arch,batch,max_block,max_q,want",
    [
        ("gfx950", 128, 2000, 1, 4),
        ("gfx1250", 1, 2000, 1, 4),
        ("gfx1250", 4, 512, 4, 4),
        ("gfx1250", 4, 513, 1, 2),
        ("gfx1250", 128, 2000, 4, 2),
        ("gfx1250", 128, 2000, 2, 4),
    ],
)
def test_decode_score_num_warps(arch, batch, max_block, max_q, want):
    assert decode_score_num_warps(arch, batch, max_block, max_q) == want


@pytest.mark.parametrize(
    "arch,batch,max_block,max_q,want",
    [
        ("gfx950", 128, 2000, 4, False),
        ("gfx1250", 0, 2000, 4, False),
        ("gfx1250", 7, 1500, 1, False),
        ("gfx1250", 8, 1500, 1, True),
        ("gfx1250", 4, 999, 4, False),
        ("gfx1250", 4, 1000, 4, True),
        ("gfx1250", 15, 500, 4, False),
        ("gfx1250", 16, 500, 4, True),
        ("gfx1250", 6, 2000, 2, False),
        ("gfx1250", 6, 2000, 3, False),
    ],
)
def test_transposed_decode_score_policy(arch, batch, max_block, max_q, want):
    assert use_transposed_decode_score(arch, batch, max_block, max_q) is want
