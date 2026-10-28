# SPDX-License-Identifier: MIT
"""Ragged row ownership preserves the delayed-post state at native seams."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.deepseek_v41 import row_sharded_mhc as owner
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState


@pytest.mark.parametrize("rows", [5, 8, 4097])
def test_ragged_capacity_and_zero_padding(monkeypatch, rows):
    full = torch.arange(rows * 2).reshape(1, rows, 2)
    parts = []
    for rank in range(4):
        monkeypatch.setattr(
            owner,
            "get_tp_group",
            lambda rank=rank: SimpleNamespace(world_size=4, rank_in_group=rank),
        )
        parts.append(owner.v41_mhc_local_rows(full, -2))
    capacity = (rows + 3) // 4
    assert all(part.shape == (1, capacity, 2) for part in parts)
    reconstructed = torch.cat(parts, dim=1)
    torch.testing.assert_close(reconstructed[:, :rows], full, rtol=0, atol=0)
    assert torch.count_nonzero(reconstructed[:, rows:]) == 0


def test_settle_preserves_owner_metadata_and_pays_post_once(monkeypatch):
    from atom.model_ops.deepseek_v41 import mhc

    residual = torch.ones(1, 2, 4, 8)
    pre = torch.ones(1, 2, 4)
    pending = torch.full((1, 2, 8), 3.0)
    post, comb = torch.zeros_like(pre), torch.zeros(1, 2, 4, 4)
    calls = []

    def pay(output, streams, post_mix, combination):
        calls.append((output, post_mix, combination))
        return streams + output.unsqueeze(-2)

    monkeypatch.setattr(mhc, "expand_residual", pay)
    state = SinglePassHCState(residual, pre, pending, post, comb, True, 5)
    settled = state.settle()
    assert settled.row_sharded and settled.original_rows == 5
    assert settled.pending is None and settled.post_mix is None
    torch.testing.assert_close(settled.residual, torch.full_like(residual, 4))
    assert settled.settle() is settled and len(calls) == 1


def test_native_restore_trims_padding_after_settlement(monkeypatch):
    from atom.model_ops.deepseek_v41 import mhc

    residual, pre = torch.ones(1, 2, 4, 8), torch.ones(1, 2, 4)
    state = SinglePassHCState(
        residual, pre, torch.ones(1, 2, 8), pre, torch.zeros(1, 2, 4, 4), True, 5
    )
    monkeypatch.setattr(
        mhc,
        "expand_residual",
        lambda output, streams, *args: streams + output.unsqueeze(-2),
    )
    monkeypatch.setattr(owner, "get_tp_group", lambda: SimpleNamespace(world_size=4))
    monkeypatch.setattr(
        owner,
        "v41_mhc_gather_rows",
        lambda value: value.repeat(4, *([1] * (value.ndim - 1))),
    )
    restored = owner.replicate(state)
    assert not restored.row_sharded and restored.pending is None
    assert restored.residual.shape == (1, 5, 4, 8)
    assert restored.pre_mix.shape == (1, 5, 4)
    torch.testing.assert_close(restored.residual, torch.full_like(restored.residual, 2))
