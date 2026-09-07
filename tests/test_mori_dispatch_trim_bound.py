# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mori dispatch trim bound must not multiply by top-k.

mori's IntraNode/AsyncLL dispatch kernel deduplicates per destination rank: a
source token whose several top-k experts all resolve to the same rank is written
into that rank's receive buffer exactly once (see the "Deduplicate" block in
mori intranode.hpp, which skips any top-k slot whose destPe already appeared in
an earlier slot of the same token). So a single rank's receive buffer holds at
most `running_tokens` tokens per source rank -> `running_tokens * dp_size` total, never
`running_tokens * topk * dp_size`. The topk-inflated bound sat above the real
valid-token count, making the trim a no-op and leaving fused_moe reading
uninitialized tail rows.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

import torch

import atom.model_ops.fused_moe.modular_kernel as mk


def _trim(monkeypatch, *, running_tokens, dp_size, topk, recv_rows):
    context = SimpleNamespace(
        running_tokens=running_tokens, is_prefill=False, running_tokens_are_unified=True
    )
    monkeypatch.setattr(
        mk, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(mk, "get_dp_group", lambda: SimpleNamespace(world_size=dp_size))
    kernel = mk.FusedMoEModularKernel.__new__(mk.FusedMoEModularKernel)
    hidden = 8
    a1 = torch.arange(recv_rows * hidden, dtype=torch.float32).reshape(
        recv_rows, hidden
    )
    ids = torch.zeros(recv_rows, topk, dtype=torch.int32)
    weights = torch.ones(recv_rows, topk, dtype=torch.float32)
    scale = torch.ones(recv_rows, 4, dtype=torch.float32)
    topk_ids = torch.zeros(3, topk, dtype=torch.int32)  # only its .shape[1] matters
    return kernel._maybe_trim_dispatch_output(
        a1, scale, ids, weights, topk_ids, expert_tokens_meta=None
    )


def test_trims_to_graph_bs_times_dp_size_not_topk(monkeypatch):
    running_tokens, dp_size, topk = 4, 8, 6
    a1, scale, ids, weights = _trim(
        monkeypatch,
        running_tokens=running_tokens,
        dp_size=dp_size,
        topk=topk,
        recv_rows=512,
    )
    expected = running_tokens * dp_size  # 32, NOT 32*topk=192
    assert a1.shape[0] == expected
    assert ids.shape[0] == expected
    assert weights.shape[0] == expected
    assert scale.shape[0] == expected


def test_topk_does_not_change_the_bound(monkeypatch):
    running_tokens, dp_size = 4, 8
    rows = {
        topk: _trim(
            monkeypatch,
            running_tokens=running_tokens,
            dp_size=dp_size,
            topk=topk,
            recv_rows=512,
        )[0].shape[0]
        for topk in (1, 2, 6, 9)
    }
    assert set(rows.values()) == {running_tokens * dp_size}


def test_no_trim_when_bound_exceeds_buffer(monkeypatch):
    # fused_moe is driven by num_local_tokens; the buffer must never be cut
    # below the bound.
    a1, _, _, _ = _trim(monkeypatch, running_tokens=64, dp_size=8, topk=4, recv_rows=16)
    assert a1.shape[0] == 16


def test_exact_dispatch_output_bypasses_mori_trim(monkeypatch):
    """An EP-wide exact RCCL buffer must not be cut to the smaller DP bound."""
    running_tokens, dp_size, ep_size, topk = 32, 2, 8, 6
    context = SimpleNamespace(
        running_tokens=running_tokens,
        is_prefill=False,
        running_tokens_are_unified=True,
    )
    monkeypatch.setattr(
        mk, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(mk, "get_dp_group", lambda: SimpleNamespace(world_size=dp_size))

    kernel = mk.FusedMoEModularKernel.__new__(mk.FusedMoEModularKernel)
    kernel.prepare_finalize = SimpleNamespace(needs_dispatch_output_trim=lambda: False)
    recv_rows = ep_size * running_tokens
    hidden = 8
    a1 = torch.arange(recv_rows * hidden, dtype=torch.float32).reshape(
        recv_rows, hidden
    )
    ids = torch.zeros(recv_rows, topk, dtype=torch.int32)
    weights = torch.ones(recv_rows, topk, dtype=torch.float32)
    scale = torch.ones(recv_rows, 4, dtype=torch.float32)
    topk_ids = torch.zeros(running_tokens, topk, dtype=torch.int32)

    trimmed = kernel._trim_dispatch_output_if_needed(
        a1, scale, ids, weights, topk_ids, expert_tokens_meta=None
    )

    assert [tensor.shape[0] for tensor in trimmed] == [recv_rows] * 4
