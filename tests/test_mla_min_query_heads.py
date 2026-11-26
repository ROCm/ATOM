# SPDX-License-Identifier: MIT
# Query-head padding for a non-causal MLA block decode (mla_min_query_heads), per GPU.
#
# mla_query_heads imports aiter only inside its functions, and aiter here is a stub
# (chip_info.get_gfx, dtypes, optionally ops.attention.get_mla_decode_num_heads),
# so these run on the CPU-only CI.

import sys
from types import ModuleType

import pytest

from atom.model_ops.mla_query_heads import mla_min_query_heads


@pytest.fixture
def gfx(monkeypatch):
    monkeypatch.delenv("ATOM_MLA_PAGE_SIZE", raising=False)

    def on(name, num_heads_query=None):
        aiter = ModuleType("aiter")
        aiter.__path__ = []
        jit = ModuleType("aiter.jit")
        jit.__path__ = []
        utils = ModuleType("aiter.jit.utils")
        utils.__path__ = []
        chip_info = ModuleType("aiter.jit.utils.chip_info")
        chip_info.get_gfx = lambda: name
        dtypes = ModuleType("aiter.dtypes")
        dtypes.fp8 = "fp8"
        aiter.jit, jit.utils, utils.chip_info = jit, utils, chip_info
        aiter.dtypes = dtypes
        mods = [aiter, jit, utils, chip_info, dtypes]
        if num_heads_query is not None:
            ops = ModuleType("aiter.ops")
            ops.__path__ = []
            attention = ModuleType("aiter.ops.attention")
            attention.get_mla_decode_num_heads = num_heads_query
            aiter.ops, ops.attention = ops, attention
            mods += [ops, attention]
        for mod in mods:
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
def test_gfx1250_without_aiter_query(gfx, kv_cache_dtype, width, num_heads, expected):
    """aiter without get_mla_decode_num_heads: the fixed rule."""
    gfx("gfx1250")
    assert mla_min_query_heads(kv_cache_dtype, width, num_heads) == expected


@pytest.mark.parametrize(
    "width, num_heads, answer, expected",
    [
        (4, 64, 64, 64),  # Kimi-K3 DSpark at TP 1 with ROCm/aiter#7913
        (4, 64, 96, 96),  # an aiter that pads 64 heads at this width
        (7, 64, 64, 64),
        (4, 8, 16, 16),  # Kimi-K3 DSpark at TP 8
        (4, 8, 8, 16),  # never below the 16-head floor
        (4, 64, None, 96),  # aiter names no count: the fixed rule
    ],
)
def test_gfx1250_asks_aiter(gfx, width, num_heads, answer, expected):
    calls = []

    def query(heads, max_seqlen_qo, q_dtype, kv_dtype, **kwargs):
        calls.append((heads, max_seqlen_qo, q_dtype, kv_dtype, kwargs))
        return answer

    gfx("gfx1250", query)
    assert mla_min_query_heads("fp8", width, num_heads) == expected
    # The block decode as _forward_decode runs it without DCP.
    assert calls == [
        (
            num_heads,
            width,
            "fp8",
            "fp8",
            {
                "persistent": True,
                "page_size": 1,
                "return_lse": False,
                "causal": False,
                "cp_world_size": 1,
                "cp_round_robin": False,
            },
        )
    ]


def test_gfx1250_one_row_skips_aiter(gfx):
    def query(*args, **kwargs):
        raise AssertionError("a one-row decode needs no query")

    gfx("gfx1250", query)
    assert mla_min_query_heads("fp8", 1, 64) == 16
    assert mla_min_query_heads("fp8", 0, 64) == 16


def test_gfx1250_bf16_cache_asks_in_bf16(gfx):
    torch = pytest.importorskip("torch")
    seen = []

    def query(heads, max_seqlen_qo, q_dtype, kv_dtype, **kwargs):
        seen.append((q_dtype, kv_dtype))
        return 64

    gfx("gfx1250", query)
    assert mla_min_query_heads("auto", 4, 64) == 64
    assert seen == [(torch.bfloat16, torch.bfloat16)]
