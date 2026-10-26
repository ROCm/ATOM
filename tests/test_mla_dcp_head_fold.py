# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A head-folded DCP decode is the same attention over fold*B pseudo-requests.

Pseudo-request ``b * fold + f`` holds query heads ``[f*h, (f+1)*h)`` of
request ``b``'s draft-token rows and reads request ``b``'s KV, so its CSR
repeats each request's lengths ``fold`` times and the padded tail stays empty.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.attention_mla import (
    fold_decode_heads,
    mla_dcp_head_fold,
    unfold_decode_heads,
)
from atom.model_ops.attentions.aiter_mla import fold_decode_csr


def test_csr_repeats_each_request_and_keeps_the_padding_empty():
    kv_lens = np.array([3, 5], dtype=np.int32)
    g_kv_lens = np.array([20, 37], dtype=np.int32)
    last = np.array([1, 1], dtype=np.int32)
    cu_q, kv_indptr, g_kv_indptr, last_out = fold_decode_csr(
        kv_lens, g_kv_lens, last, qlen=4, fold=3, padded_bs=3
    )
    np.testing.assert_array_equal(cu_q, [0, 4, 8, 12, 16, 20, 24, 24, 24, 24])
    np.testing.assert_array_equal(kv_indptr, [0, 3, 6, 9, 14, 19, 24, 24, 24, 24])
    np.testing.assert_array_equal(
        g_kv_indptr, [0, 20, 40, 60, 97, 134, 171, 171, 171, 171]
    )
    np.testing.assert_array_equal(last_out, [1, 1, 1, 1, 1, 1, 0, 0, 0])
    for a in (cu_q, kv_indptr, g_kv_indptr, last_out):
        assert a.dtype == np.int32


def test_fold_moves_head_groups_into_pseudo_requests():
    bs, qlen, fold, h, d = 2, 4, 3, 32, 5
    x = torch.randn(bs * qlen, fold * h, d)
    folded = fold_decode_heads(x, fold, qlen)
    assert folded.shape == (bs * fold * qlen, h, d)
    by_request = folded.view(bs, fold, qlen, h, d)
    for b in range(bs):
        rows = x[b * qlen : (b + 1) * qlen]
        for f in range(fold):
            torch.testing.assert_close(
                by_request[b, f], rows[:, f * h : (f + 1) * h], rtol=0, atol=0
            )
    torch.testing.assert_close(
        unfold_decode_heads(folded, fold, qlen), x, rtol=0, atol=0
    )


def test_unfold_handles_the_lse_layout():
    bs, qlen, fold, h = 3, 4, 3, 32
    lse = torch.randn(bs * qlen, fold * h)
    round_trip = unfold_decode_heads(fold_decode_heads(lse, fold, qlen), fold, qlen)
    torch.testing.assert_close(round_trip, lse, rtol=0, atol=0)


@pytest.mark.parametrize(
    "enabled, heads, dcp, persistent, fp8, expected",
    [
        ("1", 96, 8, True, True, 3),
        ("0", 96, 8, True, True, 1),
        ("1", 96, 1, True, True, 1),
        ("1", 96, 8, False, True, 1),
        ("1", 96, 8, True, False, 1),
        ("1", 128, 8, True, True, 1),
    ],
)
def test_fold_is_opt_in_and_limited_to_measured_shapes(
    monkeypatch, enabled, heads, dcp, persistent, fp8, expected
):
    monkeypatch.setenv("ATOM_MLA_DCP_HEAD_FOLD", enabled)
    assert mla_dcp_head_fold(heads, dcp, persistent=persistent, fp8=fp8) == expected
