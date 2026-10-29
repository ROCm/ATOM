# SPDX-License-Identifier: MIT
"""Qwen3.5 MTP draft options: the replicated fc (ATOM_MTP_REPLICATED_FC) and
embedding (ATOM_QWEN35_MTP_REPLICATED_EMBED), and both pre-fc norms in one
launch (ATOM_MTP_DUAL_PRE_FC_NORM).

The TP tests run against a fake 4-rank TP group (all_gather returns what the
ranks' shards concatenate to), so they check the wiring and the bit-exactness
claims without a process group.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import atom.models.qwen3_5_mtp as mtp
from atom.model_ops.embed_head import ReplicatedEmbedding

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


def _sharded(full, rank=1):
    rows = full.shape[0] // TP
    shard = nn.Module()
    shard.weight = nn.Parameter(full[rank * rows : (rank + 1) * rows].clone())
    shard.num_embeddings = full.shape[0]
    return shard


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


def _draft(monkeypatch, *, embed):
    fake = _FakeTP()
    monkeypatch.setattr(mtp, "get_tp_group", lambda: fake)
    monkeypatch.setenv("ATOM_QWEN35_MTP_REPLICATED_EMBED", "1" if embed else "0")
    # VocabParallelEmbedding is asserted on; skip its TP-reading __init__.
    monkeypatch.setattr(mtp, "VocabParallelEmbedding", nn.Module)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    g = torch.Generator().manual_seed(0)
    embed_full = torch.randn(VOCAB, HIDDEN, generator=g).to(torch.bfloat16)
    draft = object.__new__(mtp.Qwen3_5MTP)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.embed_tokens = _sharded(embed_full)
    fake.full_of[draft.model.embed_tokens.weight.data.data_ptr()] = embed_full
    return draft, embed_full


def test_embed_flag_off_is_a_noop(monkeypatch):
    draft, _ = _draft(monkeypatch, embed=False)
    before = draft.model.embed_tokens
    draft.post_share_with_target(target_base=None)
    assert draft.model.embed_tokens is before


def test_replicated_embed_is_the_gathered_table(monkeypatch):
    draft, embed_full = _draft(monkeypatch, embed=True)
    draft.post_share_with_target(target_base=None)
    emb = draft.model.embed_tokens
    assert isinstance(emb, ReplicatedEmbedding)
    assert torch.equal(emb.weight.data, embed_full)


def test_masked_shards_summed_equal_full_lookup():
    """Why the replicated lookup is bit-identical to masked-embedding +
    all-reduce: every rank but the owner contributes exact zeros, and the
    rank-ordered fp32 sum of x and zeros, rounded to bf16, is x."""
    g = torch.Generator().manual_seed(1)
    table = torch.randn(VOCAB, HIDDEN, generator=g).to(torch.bfloat16)
    ids = torch.tensor([0, 5, 17, 63, 31])
    rows = VOCAB // TP
    partials = []
    for r in range(TP):
        own = (ids >= r * rows) & (ids < (r + 1) * rows)
        part = torch.zeros(len(ids), HIDDEN, dtype=torch.bfloat16)
        part[own] = table[ids[own]]
        partials.append(part.float())
    summed = (((partials[0] + partials[1]) + partials[2]) + partials[3]).bfloat16()
    assert torch.equal(summed, table[ids])


def test_flag_defaults_are_off(monkeypatch):
    from atom.utils import envs

    for name in (
        "ATOM_QWEN35_MTP_REPLICATED_EMBED",
        "ATOM_MTP_REPLICATED_FC",
        "ATOM_MTP_DRAFT_RANK_CHECK",
    ):
        monkeypatch.delenv(name, raising=False)
        assert not getattr(envs, name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [1, 5, 40])
def test_dual_pre_fc_norm_is_the_two_norms_and_cat(rows):
    """ATOM_MTP_DUAL_PRE_FC_NORM: one launch, bitwise the separate norms' cat."""
    from atom.model_ops.layernorm import GemmaRMSNorm

    hidden = 2048
    g = torch.Generator(device="cuda").manual_seed(rows)
    pred = object.__new__(mtp.Qwen3_5MultiTokenPredictor)
    nn.Module.__init__(pred)
    pred.pre_fc_norm_embedding = GemmaRMSNorm(hidden, eps=1e-6).cuda()
    pred.pre_fc_norm_hidden = GemmaRMSNorm(hidden, eps=1e-5).cuda()
    for norm in (pred.pre_fc_norm_embedding, pred.pre_fc_norm_hidden):
        norm.weight.data = torch.randn(hidden, device="cuda", generator=g).bfloat16()
    e = torch.randn(rows, hidden, device="cuda", generator=g).bfloat16()
    h = torch.randn(rows, hidden, device="cuda", generator=g).bfloat16()
    assert mtp._dual_norm_inputs_ok(e, h)
    ref = torch.cat(
        [
            pred.pre_fc_norm_embedding.forward_cuda(e),
            pred.pre_fc_norm_hidden.forward_cuda(h),
        ],
        dim=-1,
    )
    out = pred._pre_fc_norms(e, h)
    assert out.shape == (rows, 2 * hidden) and out.is_contiguous()
    assert torch.equal(out, ref)


def test_dual_pre_fc_norm_inputs():
    e = torch.zeros(3, HIDDEN, dtype=torch.bfloat16)
    assert mtp._dual_norm_inputs_ok(e, e.clone())
    assert not mtp._dual_norm_inputs_ok(e, e.float())
    assert not mtp._dual_norm_inputs_ok(e, e[:2])
    assert not mtp._dual_norm_inputs_ok(e, e.view(3, 1, HIDDEN))
    assert not mtp._dual_norm_inputs_ok(e, torch.zeros(HIDDEN, 3).bfloat16().t())
