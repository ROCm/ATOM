# SPDX-License-Identifier: MIT
"""NVFP4 sparse attention glue: what is checkable without a GPU. The row layout the
prefill kernel is handed (request and causal key count of each row) is pinned against
a loop; the kernels themselves are covered by aiter's op tests and the GPU tests of
the writer / reader in this directory."""

import pytest
import torch

from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import prefill_row_positions


@pytest.mark.parametrize(
    "qlens, prefix",
    [
        ([5], [0]),
        ([3, 1, 7], [0, 250, 128]),
        ([1, 1, 1], [10, 0, 5]),
        ([4, 2], [128, 0]),
    ],
)
def test_prefill_rows_are_request_and_position_plus_one(qlens, prefix):
    cu = torch.tensor([0] + qlens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    total = sum(qlens)
    req, kv = prefill_row_positions(cu, torch.tensor(prefix, dtype=torch.int32), total)
    want_req = [b for b, n in enumerate(qlens) for _ in range(n)]
    want_kv = [prefix[b] + i + 1 for b, n in enumerate(qlens) for i in range(n)]
    assert req.dtype == kv.dtype == torch.int32
    assert req.tolist() == want_req and kv.tolist() == want_kv


@pytest.mark.parametrize("nvfp4", [False, True])
def test_the_index_scorer_record_picks_its_fp4_page_views_and_scorers(
    monkeypatch, nvfp4
):
    index_topk = pytest.importorskip("atom.model_ops.minimax_m3.index_topk")
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers

    names = (
        "minimax_m3_index_score_prefill_nvfp4_flydsl",
        "minimax_m3_index_score_nvfp4_flydsl",
        "minimax_m3_index_score_prefill_fp4_flydsl",
        "minimax_m3_index_score_fp4_flydsl",
    )
    ops = type("Ops", (), {n: n for n in names})
    monkeypatch.setattr(index_topk, "flydsl_ops", lambda: ops)
    views, prefill, decode = index_topk._fp4_index_ops(
        IndexScoreBuffers(scores=None, nvfp4=nvfp4)
    )
    kind = "nvfp4" if nvfp4 else "fp4"
    assert (prefill, decode) == (
        f"minimax_m3_index_score_prefill_{kind}_flydsl",
        f"minimax_m3_index_score_{kind}_flydsl",
    )
    assert (views is index_topk.nvfp4_page_views) == nvfp4
