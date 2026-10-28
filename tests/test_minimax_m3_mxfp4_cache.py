# SPDX-License-Identifier: MIT
"""MiniMax-M3's MXFP4 sparse KV pool and its startup refusals.

The pool lays one (page, kv head) of K, of V and of the index key out as one
8704 B field -- [128 x 64 B e2m1][128 x 4 B e8m0] -- which is what the aiter
writer produces and `pa_decode_mxfp4` / `score_flydsl_fp4` read. Pinned here:
the bytes per token the capacity arithmetic rests on, the views the kernels are
handed, and the combinations that must fail at startup and not in a forward.
"""

import pytest
import torch

from atom.config import _check_m3_mxfp4
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.minimax_m3.kv_mxfp4 import PAGE_BYTES, page_views

M3 = ["MiniMaxM3SparseForConditionalGeneration"]


def pool(layers=3, heads=1, **kw):
    return MhaKvPool(
        layers=layers,
        block_size=128,
        num_kv_heads=heads,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fnuz,
        kv_format="mxfp4",
        **kw,
    )


def test_a_token_costs_136_bytes_of_k_and_v():
    p = pool(layers=60, heads=1)
    # 60 layers x (K + V) x 8704 B a 128-token block, 256 B field alignment holds.
    assert p.entry_bytes == 60 * 2 * PAGE_BYTES
    assert p.entry_bytes / (60 * 128) == 136
    assert p.scale_fields == []


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_views_are_pages_of_heads_and_alias_the_pool(heads):
    p = pool(layers=2, heads=heads)
    p.allocate(5, "cpu")
    k, v = p.kv_views(1)
    assert k.shape == (5, heads, PAGE_BYTES) == v.shape
    assert k.dtype == torch.uint8
    assert k.stride() == (heads * PAGE_BYTES, PAGE_BYTES, 1)
    k.fill_(7)
    assert not (v == 7).any()
    assert not (p.kv_views(0)[0] == 7).any()


def test_page_views_split_data_from_scale():
    pages = torch.arange(3 * 2 * PAGE_BYTES, dtype=torch.int64).remainder(251)
    pages = pages.to(torch.uint8).view(3, 2, PAGE_BYTES)
    data, scale = page_views(pages)
    assert data.shape == (3, 2, 128, 64) and scale.shape == (3, 2, 128, 4)
    assert data.stride() == (2 * PAGE_BYTES, PAGE_BYTES, 64, 1)
    assert scale.stride() == (2 * PAGE_BYTES, PAGE_BYTES, 4, 1)
    assert torch.equal(data[2, 1].flatten(), pages[2, 1, :8192])
    assert torch.equal(scale[2, 1].flatten(), pages[2, 1, 8192:])
    # an index cache is [pages, 8704] with no head axis
    idata, iscale = page_views(pages.view(6, PAGE_BYTES)[1:5])
    assert idata.shape == (4, 128, 64) and iscale.shape == (4, 128, 4)
    assert torch.equal(iscale[0].flatten(), pages.view(6, PAGE_BYTES)[1, 8192:])


@pytest.mark.parametrize(
    "kw", [{"blocks_per_entry": 2}, {"num_kv_heads": 1, "head_dim": 64}]
)
def test_only_the_128_page_of_head_dim_128_is_a_pool(kw):
    base = {
        "layers": 1,
        "block_size": 128,
        "num_kv_heads": 1,
        "head_dim": 128,
        "kv_dtype": torch.float8_e4m3fnuz,
        "kv_format": "mxfp4",
    }
    with pytest.raises(ValueError):
        MhaKvPool(**{**base, **kw})


def check(**kw):
    args = {
        "arches": M3,
        "index_cache_dtype": "mxfp4",
        "kv_mxfp4": True,
        "kv_nvfp4": False,
        "block_size": 128,
        "plugin_mode": False,
    }
    _check_m3_mxfp4(**{**args, **kw})


def test_the_two_fp4_formats_never_mix():
    with pytest.raises(ValueError, match="exclusive"):
        check(kv_nvfp4=True)
    with pytest.raises(ValueError, match="do not mix"):
        check(index_cache_dtype="fp4")
    with pytest.raises(ValueError, match="do not mix"):
        check(kv_mxfp4=False, kv_nvfp4=True)


def test_refusals_name_what_is_wrong():
    with pytest.raises(ValueError, match="MiniMax-M3 only"):
        check(arches=["LlamaForCausalLM"])
    with pytest.raises(ValueError, match="bridges"):
        check(plugin_mode=True)
    with pytest.raises(ValueError, match="128 tokens"):
        check(block_size=16)


def test_the_baselines_are_never_refused():
    # Neither MXFP4 switch on: nothing to check, whatever else is set.
    check(arches=["LlamaForCausalLM"], index_cache_dtype="fp8", kv_mxfp4=False)
    check(index_cache_dtype="fp4", kv_mxfp4=False, kv_nvfp4=True, block_size=16)
