# SPDX-License-Identifier: MIT
"""CPU checks of the production pool-to-attention adapter; no GPU allocation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from atom.model_ops.paged_prefill import run_paged_prefill


@pytest.mark.parametrize("layout", ["shuffle", "nhd", "flash"])
@pytest.mark.parametrize("mode", ["bf16", "scalar", "token"])
def test_pool_aliases_scales_and_real_request_metadata(layout, mode):
    blocks, heads, page, dim = 11, 2, 16, 32
    dtype = torch.bfloat16 if mode == "bf16" else torch.float8_e4m3fn
    pack = 16 // dtype.itemsize
    if layout == "flash":
        kc = torch.empty(blocks, page, heads, dim, dtype=dtype)
        vc = torch.empty_like(kc)
    else:
        kc = torch.empty(blocks, heads, dim // pack, page, pack, dtype=dtype)
        vc = torch.empty(
            (
                (blocks, heads, page // pack, dim, pack)
                if layout == "shuffle"
                else (blocks, heads, dim, page)
            ),
            dtype=dtype,
        )
    q = torch.randn(7, 4, dim, dtype=torch.bfloat16)
    ks = vs = None
    if mode == "scalar":
        ks, vs = torch.tensor([0.7]), torch.tensor([1.3])
    if mode == "token":
        ks = torch.rand(blocks, heads, page * 2)[..., ::2]
        vs = torch.rand(blocks, heads, page * 2)[..., ::2]
    # Padded metadata deliberately has an invalid tail. Only two requests run.
    md = SimpleNamespace(
        has_cached=True,
        total_kv=66,
        block_tables=torch.tensor(
            [[8, 2, 9], [8, 1, 0], [-1, -1, -1]], dtype=torch.int32
        ),
        cu_seqlens_q=torch.tensor([0, 2, 7, 999], dtype=torch.int32),
        context_lens=torch.tensor([35, 31, 999], dtype=torch.int32),
        max_seqlen_q=5,
        max_seqlen_k=35,
    )
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, scheduled_bs=2), attn_metadata=md
    )
    sinks = torch.ones(4)
    calls = []

    def attention(query, keys, values, output, **kwargs):
        assert query is q and keys is kc and values is vc
        assert kwargs["cu_seqlens_q"].tolist() == [0, 2, 7]
        assert kwargs["seqused_k"].tolist() == [35, 31]
        assert kwargs["block_table"].tolist() == [[8, 2, 9], [8, 1, 0]]
        assert kwargs["block_table"].data_ptr() == md.block_tables.data_ptr()
        assert kwargs["shuffled_kv_cache"] == (layout != "flash")
        assert kwargs["sinks"] is sinks
        assert kwargs["window_size"] == (31, 0)
        assert kwargs["softmax_scale"] == 0.125 and kwargs["causal"]
        assert kwargs["q_descale"] is None
        if mode == "token":
            assert kwargs["k_scale_cache"] is ks and kwargs["v_scale_cache"] is vs
            assert kwargs["k_descale"] is None and kwargs["v_descale"] is None
        else:
            assert kwargs["k_descale"] is ks and kwargs["v_descale"] is vs
            assert "k_scale_cache" not in kwargs
        output.copy_(query)
        calls.append(kwargs)

    with patch("atom.model_ops.paged_prefill.torch.empty", wraps=torch.empty) as alloc:
        out = run_paged_prefill(
            attention,
            q,
            kc,
            vc,
            ks,
            vs,
            ctx,
            softmax_scale=0.125,
            sliding_window=32,
            sinks=sinks,
        )
        assert alloc.call_count == 1
        assert alloc.call_args.args[0] == q.shape
    assert len(calls) == 1
    torch.testing.assert_close(out, q)
    assert md.cu_seqlens_q[-1] == 999


def test_mismatched_scale_geometry_is_rejected_before_attention():
    q = torch.empty(2, 1, 32, dtype=torch.bfloat16)
    kc = torch.empty(3, 1, 2, 16, 16, dtype=torch.float8_e4m3fn)
    vc = torch.empty(3, 1, 1, 32, 16, dtype=kc.dtype)
    md = SimpleNamespace(
        has_cached=True,
        block_tables=torch.zeros(1, 3),
        cu_seqlens_q=torch.tensor([0, 2]),
        context_lens=torch.tensor([17]),
    )
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, scheduled_bs=1), attn_metadata=md
    )
    with pytest.raises(ValueError, match="Per-token FP8 scales"):
        run_paged_prefill(
            None,
            q,
            kc,
            vc,
            torch.ones(3, 2, 16),
            torch.ones(3, 2, 16),
            ctx,
            softmax_scale=1,
        )
