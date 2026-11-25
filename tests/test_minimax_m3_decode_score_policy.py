# SPDX-License-Identifier: MIT

import pytest

from atom.model_ops.minimax_m3.decode_score_policy import (
    decode_score_num_warps,
    use_transposed_decode_score,
)


@pytest.mark.parametrize(
    "arch,batch,max_block,max_q,warps,transposed",
    [
        ("gfx950", 128, 2000, 1, 4, False),  # other archs keep the old launch
        ("gfx1250", 8, 1500, 1, 2, True),
        ("gfx1250", 7, 1500, 1, 2, False),  # below the transposed work bound
        ("gfx1250", 4, 512, 4, 4, False),  # narrow grid
        ("gfx1250", 4, 1000, 4, 2, True),
        ("gfx1250", 16, 8192, 8, 2, False),  # TP4 q8
        ("gfx1250", 32, 8192, 8, 4, False),
        ("gfx1250", 6, 2000, 2, 4, False),
    ],
)
def test_decode_score_launch_policy(arch, batch, max_block, max_q, warps, transposed):
    assert decode_score_num_warps(arch, batch, max_block, max_q) == warps
    assert use_transposed_decode_score(arch, batch, max_block, max_q) is transposed
