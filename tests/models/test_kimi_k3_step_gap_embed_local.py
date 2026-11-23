# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The local full-table embedding's host side (``step_gap.embed_local``) on a
CPU: the canonical form, the lookup's dispatch, the bind and its refusals with
the collectives played by the test. (Production's bits are compared on the GPU
by ``tools/kimi_k3_step_gap_embed_check.py``.)"""

from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("aiter", reason="the embedding module imports aiter")
pytest.importorskip("triton", reason="the embedding module imports Triton")

import torch.distributed as dist
from torch import nn

from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.models.kimi_k3.step_gap import embed_local as el

TP, ROWS, DIM = 4, 6, 16


def _bits(t):
    return [int(v) & 0xFFFF for v in t.view(torch.int16).flatten().tolist()]


def _embedding(shard, rank=1, cls=el.K3LocalEmbedding):
    e = cls.__new__(cls)
    nn.Module.__init__(e)
    e.tp_rank, e.tp_size = rank, TP
    e.num_embeddings, e.num_embeddings_per_partition = TP * ROWS, ROWS
    e.vocab_start_idx, e.vocab_end_idx = rank * ROWS, (rank + 1) * ROWS
    e.weight = nn.Parameter(shard, requires_grad=False)
    e.local_table = None
    return e


def test_an_unbound_embedding_is_productions(monkeypatch):
    seen = []
    monkeypatch.setattr(
        VocabParallelEmbedding,
        "forward",
        lambda self, x: seen.append(x) or "production",
    )
    emb = _embedding(torch.zeros(ROWS, DIM, dtype=torch.bfloat16))
    assert emb("ids") == "production" and seen == ["ids"]


def test_a_bound_embedding_looks_the_whole_table_up(monkeypatch):
    calls = []
    monkeypatch.setattr(el, "masked_embedding", lambda *a: calls.append(a) or "rows")
    monkeypatch.setattr(
        VocabParallelEmbedding, "forward", lambda self, x: pytest.fail("production ran")
    )
    emb = _embedding(torch.zeros(ROWS, DIM, dtype=torch.bfloat16))
    emb.local_table = table = torch.zeros(TP * ROWS, DIM, dtype=torch.bfloat16)
    assert emb("ids") == "rows"
    assert calls == [("ids", table, 0, TP * ROWS)]
    assert (
        "local_table" not in dict(emb.named_parameters())
        and "local_table" not in emb.state_dict()
    )


@pytest.fixture
def world(monkeypatch):
    """The TP group played by the test: every rank's shard, the gather, and
    the verdicts and digests the other ranks would send."""
    torch.manual_seed(3)
    table = torch.randn(TP * ROWS, DIM).to(torch.bfloat16)
    table.view(torch.int16)[2, :4] = torch.tensor(
        [-32768, 1, 127, 0x7FA0], dtype=torch.int16
    )
    state = NS(
        table=table,
        shards=[table[r * ROWS : (r + 1) * ROWS].clone() for r in range(TP)],
        gathers=0,
        others=None,
        peer_refuses=False,
    )

    def gather(out, piece, group=None):
        state.gathers += 1
        for r in range(TP):
            out[r].copy_(state.shards[r][: piece.shape[0]])

    def min_each(values, group):
        mine = tuple(int(v) for v in values)
        if len(mine) == 1:  # the vote before the gather
            return (int(mine[0] and not state.peer_refuses),)
        if state.others is None:
            return mine
        return tuple(min(a, b) for a, b in zip(mine, state.others(mine)))

    monkeypatch.setattr(dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(el, "group_min", min_each)
    monkeypatch.setattr(el, "local_refusal", lambda embedding: None)
    monkeypatch.setattr(el, "GATHER_ROWS", ROWS)
    return state


GROUP = NS(cpu_group=None, device_group=None)


def test_bind_assembles_the_canonical_table(world):
    emb = _embedding(world.shards[1])
    assert el.bind(emb, GROUP) is True and world.gathers == 1
    want = el.canonicalize_(world.table.clone())
    assert torch.equal(emb.local_table.view(torch.int16), want.view(torch.int16))
    assert tuple(emb.local_table.shape) == (TP * ROWS, DIM)
    assert _bits(emb.local_table[2, :4]) == [0x0000, 0x0000, 0x0000, 0x7FE0]


def test_a_rank_that_refuses_before_the_gather_stops_every_rank(world):
    world.peer_refuses = True
    emb = _embedding(world.shards[1])
    assert el.bind(emb, GROUP) is False and emb.local_table is None
    assert world.gathers == 0


def test_this_ranks_own_refusal_is_voted_too(world, monkeypatch):
    monkeypatch.setattr(el, "local_refusal", lambda embedding: "no room")
    emb = _embedding(world.shards[1])
    assert el.bind(emb, GROUP) is False and world.gathers == 0


def test_a_gather_that_loses_this_ranks_shard_is_refused(world):
    emb = _embedding(world.shards[2].clone(), rank=1)  # not what rank 1 gathers
    assert el.bind(emb, GROUP) is False and emb.local_table is None


def test_a_gather_that_raises_is_this_ranks_refusal(world, monkeypatch):
    def broken(out, piece, group=None):
        raise RuntimeError("the collective failed")

    monkeypatch.setattr(dist, "all_gather_into_tensor", broken)
    emb = _embedding(world.shards[1])
    assert el.bind(emb, GROUP) is False and emb.local_table is None


def test_another_ranks_failure_or_another_table_stops_this_rank(world):
    emb = _embedding(world.shards[1])
    world.others = lambda mine: (0, mine[1], mine[2])  # a peer's table failed
    assert el.bind(emb, GROUP) is False and emb.local_table is None
    world.others = lambda mine: (1, mine[1] + 1, mine[2] - 1)  # another table
    assert el.bind(emb, GROUP) is False and emb.local_table is None
    world.others = lambda mine: mine  # the peers hold this table
    assert el.bind(emb, GROUP) is True


def test_what_a_rank_refuses_locally(monkeypatch):
    emb = _embedding(torch.zeros(ROWS, DIM))
    assert "float32" in el.local_refusal(emb)
    emb = _embedding(torch.zeros(ROWS - 1, DIM, dtype=torch.bfloat16))
    assert "shard" in el.local_refusal(emb)
    emb = _embedding(torch.zeros(ROWS, DIM, dtype=torch.bfloat16))
    emb.num_embeddings += 1
    assert "evenly" in el.local_refusal(emb)
    emb = _embedding(torch.zeros(ROWS, DIM, dtype=torch.bfloat16))
    assert "cpu" in el.local_refusal(emb)  # a shard that is not on a GPU


def test_install_finds_the_embedding_and_takes_uniform_refusals(world, monkeypatch):
    from aiter.dist import parallel_state as ps

    monkeypatch.setattr(ps, "get_tp_group", lambda: GROUP)
    monkeypatch.setattr(ps, "get_pp_group", lambda: NS(world_size=1))
    emb = _embedding(world.shards[1])
    model = NS(language_model=NS(model=NS(embed_tokens=emb)))
    assert el.install(model) is True and emb.local_table is not None
    gathers = world.gathers
    assert el.install(model) is True and world.gathers == gathers  # bound once
    assert el.install(NS()) is False
    assert el.install(NS(model=NS(embed_tokens=NS()))) is False  # not ours
    one = _embedding(world.shards[1])
    one.tp_size = 1
    assert (
        el.install(NS(model=NS(embed_tokens=one))) is False and one.local_table is None
    )
    monkeypatch.setattr(ps, "get_pp_group", lambda: NS(world_size=2))
    staged = _embedding(world.shards[1])
    assert el.install(NS(model=NS(embed_tokens=staged))) is False
    assert staged.local_table is None


def _plain(shard):
    """The model's embedding as its constructor leaves it."""
    plain = _embedding(shard, cls=VocabParallelEmbedding)
    del plain.local_table
    return plain


def test_install_makes_the_models_own_embedding_local_in_place(world, monkeypatch):
    from aiter.dist import parallel_state as ps

    monkeypatch.setattr(ps, "get_tp_group", lambda: GROUP)
    monkeypatch.setattr(ps, "get_pp_group", lambda: NS(world_size=1))
    plain = _plain(world.shards[1])
    weight = plain.weight
    assert el.install(NS(language_model=NS(model=NS(embed_tokens=plain)))) is True
    assert type(plain) is el.K3LocalEmbedding and plain.weight is weight
    assert plain.local_table is not None and "local_table" not in plain.state_dict()


def test_an_embedding_install_leaves_alone_keeps_its_class(world, monkeypatch):
    from aiter.dist import parallel_state as ps

    monkeypatch.setattr(ps, "get_tp_group", lambda: GROUP)
    monkeypatch.setattr(ps, "get_pp_group", lambda: NS(world_size=2))
    staged = _plain(world.shards[1])
    assert el.install(NS(model=NS(embed_tokens=staged))) is False
    assert type(staged) is VocabParallelEmbedding and not hasattr(staged, "local_table")
    monkeypatch.setattr(ps, "get_pp_group", lambda: NS(world_size=1))
    world.peer_refuses = True  # the bind is refused: production's lookup stays
    refused = _plain(world.shards[1])
    assert el.install(NS(model=NS(embed_tokens=refused))) is False
    assert refused.local_table is None


def test_a_local_check_that_raises_is_this_ranks_refusal(world, monkeypatch):
    votes = []

    def broken(embedding):
        raise RuntimeError("no memory report")

    def recorded(values, group):
        votes.append(tuple(values))
        return tuple(values)

    monkeypatch.setattr(el, "local_refusal", broken)
    monkeypatch.setattr(el, "group_min", recorded)
    emb = _embedding(world.shards[1])
    assert el.bind(emb, GROUP) is False and emb.local_table is None
    assert votes == [(0,)] and world.gathers == 0
