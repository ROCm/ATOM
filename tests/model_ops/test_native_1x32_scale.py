# SPDX-License-Identifier: MIT
"""A preshuffled native 1x32 FP8 weight's E8M0 scale is laid out as the arch's GEMM reads it."""

import pytest
import torch

pytest.importorskip("aiter", reason="the scale layouts under test are AITER's")

from aiter.jit.utils import chip_info

from atom.model_ops.linear import native_1x32_scale_for_preshuffle


def _scale(n, k_blocks):
    torch.manual_seed(0)
    return torch.randint(0, 255, (n, k_blocks), dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )


def test_gfx950_keeps_the_row_major_scale(monkeypatch):
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx950")
    scale = _scale(64, 8)
    assert native_1x32_scale_for_preshuffle(scale) is scale


@pytest.mark.parametrize("n, k_blocks", [(32, 4), (64, 8), (128, 192)])
def test_gfx1250_scale_is_n32k4_under_the_row_major_shape(monkeypatch, n, k_blocks):
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx1250")
    scale = _scale(n, k_blocks)
    out = native_1x32_scale_for_preshuffle(scale)
    assert out.shape == scale.shape and out.dtype == scale.dtype
    # n32k4: each 32-row group holds one 128-byte run per 4 scale blocks, 4 bytes a row.
    row = torch.arange(n).view(n, 1)
    kb = torch.arange(k_blocks).view(1, k_blocks)
    offset = (row // 32) * 32 * k_blocks + (kb // 4) * 128 + (row % 32) * 4 + kb % 4
    flat = out.view(torch.uint8).flatten()
    assert torch.equal(flat[offset], scale.view(torch.uint8))


@pytest.mark.parametrize("n, k_blocks", [(48, 8), (64, 6)])
def test_gfx1250_rejects_shapes_its_gemm_cannot_read(monkeypatch, n, k_blocks):
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx1250")
    with pytest.raises(ValueError, match="N % 32 == 0 and K % 128 == 0"):
        native_1x32_scale_for_preshuffle(_scale(n, k_blocks))
