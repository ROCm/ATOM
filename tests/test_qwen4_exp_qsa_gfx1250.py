# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 TDM-gather QSA kernel must match the portable Triton kernel."""

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from atom.model_ops.qwen4_exp.ops.qsa import (
    _table_width_bucket,
    qsa_sparse_paged_gqa,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250",
    reason="gfx1250 GPU required",
)


def _case(tokens, requests, context, topk, invalid_frac, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    page, kv_heads, dim = 16, 2, 256
    pages_per_req = (context + page - 1) // page
    pages = requests * pages_per_req + 5
    k = (torch.randn(pages, page, kv_heads, dim, generator=g) * 0.5).to(torch.bfloat16)
    v = (torch.randn(pages, page, kv_heads, dim, generator=g) * 0.5).to(torch.bfloat16)
    q = torch.randn(tokens, 24, dim, generator=g).to(torch.bfloat16)
    table = torch.randperm(pages, generator=g)[: requests * pages_per_req]
    table = table.view(requests, pages_per_req).to(torch.int32)
    token_to_req = (torch.arange(tokens) * requests // tokens).to(torch.int32)
    rows = []
    for _ in range(tokens):
        row = torch.full((topk,), -1, dtype=torch.int32)
        n = min(topk, context)
        row[:n] = torch.randperm(context, generator=g)[:n].to(torch.int32)
        row[torch.rand(topk, generator=g) < invalid_frac] = -1
        rows.append(row)
    idx = torch.stack(rows)
    return [x.to("cuda") for x in (q, k, v, idx, table, token_to_req)]


@pytest.mark.parametrize(
    "tokens,requests,context,topk,invalid_frac",
    [
        (64, 16, 8192, 2051, 0.02),  # MTP3 decode at CONC16, split-K
        (8, 2, 100, 103, 0.1),  # short context, partial tiles
        (4, 1, 30, 31, 0.0),  # selection narrower than one tile
        (2100, 1, 4096, 2051, 0.0),  # prefill chunk, no split
    ],
)
def test_gfx1250_qsa_matches_portable_kernel(
    tokens, requests, context, topk, invalid_frac
):
    q, k, v, idx, table, token_to_req = _case(
        tokens, requests, context, topk, invalid_frac, seed=tokens + topk
    )
    token_to_req[min(1, tokens - 1)] = -1  # an unmapped row must come out zero
    scale = 256**-0.5

    portable = qsa_sparse_paged_gqa(
        q, k, v, idx, table, token_to_req, softmax_scale=scale, kv_splits=1
    )
    fast = qsa_sparse_paged_gqa(q, k, v, idx, table, token_to_req, softmax_scale=scale)

    from atom.model_ops.qwen4_exp.ops.qsa_gfx1250 import qsa_sparse_paged_gqa_gfx1250

    direct = qsa_sparse_paged_gqa_gfx1250(
        q,
        k,
        v,
        idx,
        table,
        token_to_req,
        scale,
        _table_width_bucket(table.shape[1]),
        get_cu_num(),
    )
    torch.testing.assert_close(fast, direct, rtol=0, atol=0)
    torch.testing.assert_close(fast, portable, rtol=1e-2, atol=4e-3)
    assert not fast[min(1, tokens - 1)].any()


def _fp8_rows(cache):
    from atom.model_ops.qwen4_exp.ops.qsa import FP8_DTYPE, FP8_MAX

    scale = cache.float().abs().amax(-1).clamp_min(1e-12) / FP8_MAX
    rows = (cache.float() / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return rows, scale.contiguous()


@pytest.mark.parametrize(
    "tokens,requests,context,topk,invalid_frac",
    [
        (64, 16, 8192, 2051, 0.02),
        (8, 2, 100, 103, 0.1),
        (2100, 1, 4096, 2051, 0.0),
    ],
)
def test_gfx1250_qsa_fp8_matches_portable_kernel(
    tokens, requests, context, topk, invalid_frac
):
    q, k, v, idx, table, token_to_req = _case(
        tokens, requests, context, topk, invalid_frac, seed=tokens + topk + 1
    )
    token_to_req[min(1, tokens - 1)] = -1
    k8, k_scale = _fp8_rows(k)
    v8, v_scale = _fp8_rows(v)
    scale = 256**-0.5
    kwargs = {
        "softmax_scale": scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }

    portable = qsa_sparse_paged_gqa(
        q, k8, v8, idx, table, token_to_req, kv_splits=1, **kwargs
    )
    fast = qsa_sparse_paged_gqa(q, k8, v8, idx, table, token_to_req, **kwargs)

    from atom.model_ops.qwen4_exp.ops.qsa_gfx1250 import qsa_sparse_paged_gqa_gfx1250

    direct = qsa_sparse_paged_gqa_gfx1250(
        q,
        k8,
        v8,
        idx,
        table,
        token_to_req,
        scale,
        _table_width_bucket(table.shape[1]),
        get_cu_num(),
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(fast, direct, rtol=0, atol=0)
    torch.testing.assert_close(fast, portable, rtol=1e-2, atol=4e-3)
    assert not fast[min(1, tokens - 1)].any()


def test_unsupported_layout_falls_back():
    """A cache that is not one row table keeps the portable kernel."""
    q, k, v, idx, table, token_to_req = _case(8, 2, 256, 259, 0.0, seed=3)
    k_t = k.transpose(1, 2).contiguous().transpose(1, 2)  # heads no longer packed
    v_t = v.transpose(1, 2).contiguous().transpose(1, 2)
    out = qsa_sparse_paged_gqa(q, k_t, v_t, idx, table, token_to_req)
    ref = qsa_sparse_paged_gqa(q, k, v, idx, table, token_to_req, kv_splits=1)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=4e-3)
