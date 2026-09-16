# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mori dispatch trim bound: the group's SUM, and not multiplied by top-k.

Two properties, and they fail in opposite directions.

mori's IntraNode/AsyncLL dispatch kernel deduplicates per destination rank: a
source token whose several top-k experts all resolve to the same rank is written
into that rank's receive buffer exactly once (see the "Deduplicate" block in
mori intranode.hpp, which skips any top-k slot whose destPe already appeared in
an earlier slot of the same token). So a rank's receive buffer holds at most
each source rank's own token count, never that times `topk`. A topk-inflated
bound sits ABOVE the real valid-token count, making the trim a no-op and leaving
fused_moe reading uninitialized tail rows.

The counts summed are every rank's, because that is what a receive buffer holds.
`this_rank * dp_size` is the same number only while the group is uniform; a TBO
ubatch splits per-rank, and on the smaller rank the product falls BELOW the rows
mori delivered. That bound trims the buffer under the `expert_num_tokens`
fused_moe is driven by, and moe_sorting's zero-fill -- sized on the host from
the trimmed width but bounded on the device by that count -- runs off the end of
its workspace.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

import torch

import atom.model_ops.fused_moe.modular_kernel as mk


def _context(across_dp, *, running_tokens=None, dp_rank=0):
    return SimpleNamespace(
        running_tokens=(
            across_dp[dp_rank] if running_tokens is None else running_tokens
        ),
        running_tokens_across_dp=tuple(across_dp),
        is_prefill=False,
        running_tokens_are_unified=True,
    )


def _trim(monkeypatch, *, across_dp, topk, recv_rows, dp_rank=0):
    context = _context(across_dp, dp_rank=dp_rank)
    monkeypatch.setattr(
        mk, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
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


def test_trims_to_the_group_total_not_topk(monkeypatch):
    across_dp = (4,) * 8
    topk = 6
    a1, scale, ids, weights = _trim(
        monkeypatch, across_dp=across_dp, topk=topk, recv_rows=512
    )
    expected = sum(across_dp)  # 32, NOT 32*topk=192
    assert a1.shape[0] == expected
    assert ids.shape[0] == expected
    assert weights.shape[0] == expected
    assert scale.shape[0] == expected


def test_topk_does_not_change_the_bound(monkeypatch):
    across_dp = (4,) * 8
    rows = {
        topk: _trim(monkeypatch, across_dp=across_dp, topk=topk, recv_rows=512)[
            0
        ].shape[0]
        for topk in (1, 2, 6, 9)
    }
    assert set(rows.values()) == {sum(across_dp)}


def test_the_smaller_rank_of_an_uneven_split_keeps_the_peer_s_rows(monkeypatch):
    """The regression: a TBO ubatch leaves the two ranks at different counts.

    `this_rank * dp_size` is 12426 on the smaller rank while mori delivers rows
    for both -- 13918 at worst. Trimming to the product cuts the buffer below
    what arrived; the sum does not.
    """
    across_dp = (6213, 7346)
    a1, _, _, _ = _trim(monkeypatch, across_dp=across_dp, topk=4, recv_rows=32768)
    assert a1.shape[0] == sum(across_dp) == 13559
    assert a1.shape[0] > across_dp[0] * len(across_dp)  # 12426, the old bound


def test_a_ragged_step_trims_too(monkeypatch):
    """A prefill used to be excluded, and went back with the per-rank bound.

    The exclusion was never about prefill: it was that one rank's count is the
    group's only when the group is uniform. A sum is the group's count however
    unevenly the group reached it, so there is nothing left to exclude -- and
    the arena a prefill used to carry whole is the larger part of this shrink.
    """
    across_dp = (900, 1100)
    context = SimpleNamespace(
        running_tokens=across_dp[0],
        running_tokens_across_dp=across_dp,
        is_prefill=True,
        running_tokens_are_unified=False,
    )
    monkeypatch.setattr(
        mk, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    kernel = mk.FusedMoEModularKernel.__new__(mk.FusedMoEModularKernel)
    trimmed = kernel._maybe_trim_dispatch_output(
        torch.zeros(32768, 8),
        None,
        torch.zeros(32768, 4, dtype=torch.int32),
        torch.ones(32768, 4),
        torch.zeros(3, 4, dtype=torch.int32),
        expert_tokens_meta=None,
    )
    assert trimmed[0].shape[0] == sum(across_dp) == 2000


def test_no_trim_when_bound_exceeds_buffer(monkeypatch):
    # fused_moe is driven by num_local_tokens; the buffer must never be cut
    # below the bound.
    a1, _, _, _ = _trim(monkeypatch, across_dp=(64,) * 8, topk=4, recv_rows=16)
    assert a1.shape[0] == 16


def test_exact_dispatch_output_bypasses_mori_trim(monkeypatch):
    """An EP-wide exact RCCL buffer must not be cut to the smaller DP bound."""
    running_tokens, dp_size, ep_size, topk = 32, 2, 8, 6
    context = _context((running_tokens,) * dp_size)
    monkeypatch.setattr(
        mk, "get_forward_context", lambda: SimpleNamespace(context=context)
    )

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


def test_a_step_with_no_reduced_counts_is_an_error(monkeypatch):
    """Silence here is what the bug looked like: no counts, so no bound, so a
    width taken from whatever single number was to hand."""
    context = SimpleNamespace(
        running_tokens=64,
        running_tokens_across_dp=None,
        is_prefill=False,
        running_tokens_are_unified=True,
    )
    monkeypatch.setattr(
        mk, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    kernel = mk.FusedMoEModularKernel.__new__(mk.FusedMoEModularKernel)
    a1 = torch.zeros(512, 8)
    with pytest.raises(AssertionError, match="per-rank counts"):
        kernel._maybe_trim_dispatch_output(
            a1,
            None,
            torch.zeros(512, 4, dtype=torch.int32),
            torch.ones(512, 4),
            torch.zeros(3, 4, dtype=torch.int32),
            expert_tokens_meta=None,
        )
