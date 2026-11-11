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
