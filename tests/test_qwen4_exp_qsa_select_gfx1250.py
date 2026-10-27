# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 WMMA index scorer must preserve QSA's selected token ids."""

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from aiter.jit.utils.chip_info import get_gfx_runtime  # noqa: E402

from atom.model_ops.qwen4_exp.ops.qsa import qsa_select_paged_tokens  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250",
    reason="gfx1250 GPU required",
)


def _case(tokens_per_request: int, seed: int = 1):
    requests, page, heads, dim = 16, 4, 4, 128
    context = 4096
    compressed_groups = context // 4
    pages_per_request = compressed_groups // page
    pages = requests * pages_per_request
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(
        requests * tokens_per_request, heads, dim, generator=g, dtype=torch.bfloat16
    )
    cache = torch.randn(
        pages, page, 1, dim, generator=g, dtype=torch.bfloat16
    )
    table = torch.arange(pages, dtype=torch.int32).view(requests, pages_per_request)
    token_to_request = torch.arange(requests, dtype=torch.int32).repeat_interleave(
        tokens_per_request
    )
    positions = (
        torch.full((requests,), context - tokens_per_request, dtype=torch.int64)
        .repeat_interleave(tokens_per_request)
        + torch.arange(requests * tokens_per_request).remainder(tokens_per_request)
    )
    context_lens = torch.full((requests,), context, dtype=torch.int32)
    return tuple(
        x.cuda() for x in (q, cache, table, token_to_request, positions, context_lens)
    )


def _select(case):
    q, cache, table, token_to_request, positions, context_lens = case
    return qsa_select_paged_tokens(
        q,
        cache,
        table,
        token_to_request,
        positions,
        context_lens,
        token_topk=2048,
        compress_ratio=4,
        max_seq_len=4096,
        allow_flydsl=False,
    )


def test_gfx1250_scorer_matches_portable_selection(monkeypatch):
    case = _case(tokens_per_request=4)
    monkeypatch.setenv("ATOM_QSA_GFX1250_SCORER", "0")
    expected = _select(case)
    monkeypatch.setenv("ATOM_QSA_GFX1250_SCORER", "1")
    actual = _select(case)
    assert torch.equal(actual, expected)


def test_gfx1250_scorer_falls_back_for_draft_rows(monkeypatch):
    import atom.model_ops.qwen4_exp.ops.qsa_select_gfx1250 as gfx_select

    case = _case(tokens_per_request=1)
    monkeypatch.setenv("ATOM_QSA_GFX1250_SCORER", "1")

    def unexpected(*args, **kwargs):
        raise AssertionError("gfx1250 scorer must not run for one row per request")

    monkeypatch.setattr(gfx_select, "qsa_index_scores", unexpected)
    out = _select(case)
    assert out.shape == (16, 2051)


def test_gfx1250_scorer_falls_back_for_strided_page_table(monkeypatch):
    import atom.model_ops.qwen4_exp.ops.qsa_select_gfx1250 as gfx_select

    case = list(_case(tokens_per_request=4))
    table = case[2]
    case[2] = table.T.contiguous().T
    assert case[2].stride(1) != 1
    monkeypatch.setenv("ATOM_QSA_GFX1250_SCORER", "1")

    def unexpected(*args, **kwargs):
        raise AssertionError("gfx1250 scorer requires contiguous page-table rows")

    monkeypatch.setattr(gfx_select, "qsa_index_scores", unexpected)
    out = _select(tuple(case))
    assert out.shape == (64, 2051)
