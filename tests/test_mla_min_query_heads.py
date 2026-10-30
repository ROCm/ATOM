# SPDX-License-Identifier: MIT
# Query-head padding for a non-causal MLA block decode (mla_min_query_heads), per GPU.
#
# mla_query_heads imports neither triton nor aiter, and get_gfx comes from a stub
# of aiter's chip_info, so these run on the CPU-only CI.

import sys
from types import ModuleType

import pytest

from atom.model_ops.mla_query_heads import mla_min_query_heads


@pytest.fixture
def gfx(monkeypatch):
    def on(name):
        aiter = ModuleType("aiter")
        aiter.__path__ = []
        jit = ModuleType("aiter.jit")
        jit.__path__ = []
        utils = ModuleType("aiter.jit.utils")
        utils.__path__ = []
        chip_info = ModuleType("aiter.jit.utils.chip_info")
        chip_info.get_gfx = lambda: name
        aiter.jit, jit.utils, utils.chip_info = jit, utils, chip_info
        for mod in (aiter, jit, utils, chip_info):
            monkeypatch.setitem(sys.modules, mod.__name__, mod)

    return on


@pytest.mark.parametrize(
    "kv_cache_dtype, width, num_heads, expected",
    [
        ("fp8", 2, 64, 32),
        ("fp8", 7, 64, 16),
        ("fp8", 1, 64, 16),
        ("auto", 2, 64, 16),
    ],
)
def test_gfx950(gfx, kv_cache_dtype, width, num_heads, expected):
    gfx("gfx950")
    assert mla_min_query_heads(kv_cache_dtype, width, num_heads) == expected


@pytest.mark.parametrize(
    "kv_cache_dtype, width, num_heads, expected",
    [
        # Kimi-K3 DSpark at TP 1: 64 heads, a 7-wide block.
        ("fp8", 7, 64, 96),
        ("auto", 7, 64, 96),
        # Padded for aiter before ROCm/aiter#6563, which takes these unpadded.
        ("fp8", 2, 64, 96),
        ("fp8", 3, 32, 96),
        # 16 heads take the PS1 kernel up to 4 wide on every aiter.
        ("fp8", 4, 16, 16),
        ("fp8", 5, 16, 96),
        ("fp8", 4, 8, 16),
        ("fp8", 1, 64, 16),
        ("auto", 1, 64, 16),
    ],
)
def test_gfx1250(gfx, kv_cache_dtype, width, num_heads, expected):
    gfx("gfx1250")
    assert mla_min_query_heads(kv_cache_dtype, width, num_heads) == expected
