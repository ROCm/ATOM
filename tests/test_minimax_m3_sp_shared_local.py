# SPDX-License-Identifier: MIT
"""MiniMax-M3 SP prefill, owner-local shared expert (ATOM_M3_SP_SHARED_LOCAL): which
rows the owner adds its shared output to, and when the path is taken."""

from types import SimpleNamespace

import pytest
import torch

sp = pytest.importorskip("atom.models.minimax_m3.sp_prefill")

H = 8


def _moe(value: float):
    calls = []

    def shared_full(codes, scales):
        calls.append(codes.shape[0])
        return torch.full((codes.shape[0], H), value, dtype=torch.bfloat16)

    return SimpleNamespace(shared_full=shared_full), calls


@pytest.mark.parametrize(
    "tokens,rows,rank,start,n",
    [
        (12, 3, 0, 0, 3),  # every rank full
        (12, 3, 3, 9, 3),
        (10, 3, 3, 9, 1),  # last rank: one real row, two padding
        (8, 3, 3, 9, 0),  # last rank: padding only
    ],
)
def test_owner_rows(tokens, rows, rank, start, n):
    shard = sp.SeqShard(num_tokens=tokens, rows=rows, world=4, rank=rank)
    assert sp._owner_rows(shard) == (start, n)


def test_shared_local_gating(monkeypatch):
    from atom.utils import envs
    from atom.utils.tbo import ubatching

    moe, _ = _moe(0.0)
    monkeypatch.setattr(ubatching, "tbo_active", lambda: False)
    monkeypatch.setattr(envs, "ATOM_M3_SP_MOE_CHECK", False, raising=False)
    assert sp._shared_local(moe, True)
    assert not sp._shared_local(moe, False)  # no FlyDSL chain: bf16 rows path
    assert not sp._shared_local(SimpleNamespace(shared_full=None), True)
    monkeypatch.setattr(ubatching, "tbo_active", lambda: True)
    assert not sp._shared_local(moe, True)  # TBO micro-batches keep the gather
    monkeypatch.setattr(ubatching, "tbo_active", lambda: False)
    monkeypatch.setattr(envs, "ATOM_M3_SP_MOE_CHECK", True, raising=False)
    assert not sp._shared_local(moe, True)  # the check compares against the gather
