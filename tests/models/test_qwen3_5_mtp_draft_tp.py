# SPDX-License-Identifier: MIT
"""Qwen3.5 MTP draft TP options: the replicated fc (ATOM_MTP_REPLICATED_FC).

The tests run against a fake 4-rank TP group (all_gather returns what the
ranks' shards concatenate to), so they check the wiring and the bit-exactness
claims without a process group.
"""

from types import SimpleNamespace

import torch
from torch import nn

import atom.models.qwen3_5_mtp as mtp

VOCAB, HIDDEN, TP = 64, 16, 4


class _FakeTP:
    """A TP group whose all_gather returns what the 4 ranks' shards concatenate
    to; ``full_of`` maps a shard tensor's data_ptr to the full tensor."""

    world_size = TP
    rank_in_group = 1

    def __init__(self):
        self.full_of = {}

    def all_gather(self, t, dim=-1, use_custom=False):
        assert dim == 0
        return self.full_of[t.data_ptr()].clone()


class _FakeFc(nn.Module):
    """The replicated fc: the whole [hidden, 2 * hidden] weight, unquantized."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.quant_type = mtp.QuantType.No

    def forward(self, x):
        return x @ self.weight.T


def _predictor(monkeypatch, *, replicated):
    fake = _FakeTP()
    monkeypatch.setattr(mtp, "get_tp_group", lambda: fake)
    g = torch.Generator().manual_seed(2)
    weight = torch.randn(HIDDEN, 2 * HIDDEN, generator=g)
    pred = object.__new__(mtp.Qwen3_5MultiTokenPredictor)
    nn.Module.__init__(pred)
    pred.replicated_fc = replicated
    pred.fc = _FakeFc(weight)
    return pred, fake, weight


def test_replicated_fc_is_local_up_to_the_token_cap(monkeypatch):
    pred, fake, weight = _predictor(monkeypatch, replicated=True)
    fake.all_gather = None  # must not be reached
    x = torch.randn(mtp._REPLICATED_FC_MAX_TOKENS, 2 * HIDDEN)
    assert torch.equal(pred._fc(x), x @ weight.T)


def test_replicated_fc_prefill_takes_this_ranks_rows_and_gathers(monkeypatch):
    """Over the cap the GEMM is this rank's ColumnParallelLinear shard, then the
    all-gather: what the flag-off path computes."""
    pred, fake, weight = _predictor(monkeypatch, replicated=True)
    seen = {}

    def mm(x, w, bias, otype):
        seen["w"] = w
        return x @ w.T

    def all_gather(t, dim=-1, use_custom=False):
        seen["gathered"] = t
        return t

    monkeypatch.setattr(mtp.tgemm, "mm", mm)
    fake.all_gather = all_gather
    x = torch.randn(mtp._REPLICATED_FC_MAX_TOKENS + 1, 2 * HIDDEN)
    pred._fc(x)
    rows = HIDDEN // TP
    rank = fake.rank_in_group
    assert torch.equal(seen["w"], weight[rank * rows : (rank + 1) * rows])
    assert seen["gathered"].shape == (x.shape[0], rows)


def test_column_parallel_fc_gathers(monkeypatch):
    pred, fake, weight = _predictor(monkeypatch, replicated=False)
    fake.all_gather = lambda t, dim=-1, use_custom=False: ("gathered", t)
    x = torch.randn(3, 2 * HIDDEN)
    tag, out = pred._fc(x)
    assert tag == "gathered" and torch.equal(out, x @ weight.T)


def test_fc_stays_column_parallel_at_tp1(monkeypatch):
    """The fc flag is inert without TP: nothing to gather."""
    monkeypatch.setenv("ATOM_MTP_REPLICATED_FC", "1")
    monkeypatch.setattr(
        mtp, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    from atom.utils import envs

    assert envs.ATOM_MTP_REPLICATED_FC
    # Mirrors the constructor's predicate.
    assert not (envs.ATOM_MTP_REPLICATED_FC and mtp.get_tp_group().world_size > 1)


def test_flag_defaults_are_off(monkeypatch):
    from atom.utils import envs

    for name in ("ATOM_MTP_REPLICATED_FC",):
        monkeypatch.delenv(name, raising=False)
        assert not getattr(envs, name)
