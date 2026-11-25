# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The unfused MXFP8 shared expert's weight addressing: ``mxfp8_tile_dword`` against
aiter's (16, 16) preshuffle of a random fp8-sized matrix (CPU, no kernel)."""

import pytest
import torch

pytest.importorskip("aiter")
from aiter.ops.shuffle import shuffle_weight

from atom.models.minimax_m3.mono.layout import mxfp8_tile_dword


@pytest.mark.parametrize("rows,cols", [(64, 256), (48, 768)])
def test_tile_dword_addresses_the_operand_pieces(rows, cols):
    w = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
    flat = shuffle_weight(w, layout=(16, 16)).contiguous().view(-1)
    for row in range(rows):
        for kc in range(cols // 128):
            for g in range(4):
                dw = mxfp8_tile_dword(row, cols, kc, g)
                for half, k0 in enumerate((16 * g, 64 + 16 * g)):
                    at = (dw + 256 * half) * 4
                    want = w[row, kc * 128 + k0 : kc * 128 + k0 + 16]
                    assert torch.equal(flat[at : at + 16], want), (row, kc, g, half)
