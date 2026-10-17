# SPDX-License-Identifier: MIT
# Query-head padding for a non-causal MLA block decode (mla_min_query_heads), per GPU.

import pytest


@pytest.fixture
def min_heads_on(monkeypatch):
    pytest.importorskip("triton", reason="attention_mla defines @triton.jit kernels")
    pytest.importorskip("aiter", reason="attention_mla imports the AITER runtime")
    from atom.model_ops import attention_mla

    def on(gfx):
        monkeypatch.setattr(attention_mla, "get_gfx", lambda: gfx)
        return attention_mla.mla_min_query_heads

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
def test_gfx950(min_heads_on, kv_cache_dtype, width, num_heads, expected):
    assert min_heads_on("gfx950")(kv_cache_dtype, width, num_heads) == expected


@pytest.mark.parametrize(
    "kv_cache_dtype, width, num_heads, expected",
    [
        # Kimi-K3 DSpark at TP 1: 64 heads, a 7-wide block.
        ("fp8", 7, 64, 96),
        ("fp8", 2, 64, 96),
        ("fp8", 3, 32, 96),
        # 16 heads take the 16-head kernel up to 4 wide.
        ("fp8", 4, 16, 16),
        ("fp8", 5, 16, 96),
        ("fp8", 4, 8, 16),
        ("fp8", 1, 64, 16),
        ("auto", 7, 64, 16),
    ],
)
def test_gfx1250(min_heads_on, kv_cache_dtype, width, num_heads, expected):
    assert min_heads_on("gfx1250")(kv_cache_dtype, width, num_heads) == expected
