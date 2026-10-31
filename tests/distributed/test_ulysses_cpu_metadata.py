# SPDX-License-Identifier: MIT
"""SP token sharding, row gathers and the vocab-sharded LM head, on CPU."""

from types import SimpleNamespace

import pytest
import torch

from atom.distributed import ulysses_sp as sp


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("total", [1, 3, 5, 17])
def test_shards_keep_token_order_and_pad_the_tail(monkeypatch, world, total):
    monkeypatch.setattr(sp, "_SP_WORLD_SIZE", world)
    tokens = torch.arange(1, total + 1)
    shards = []
    for rank in range(world):
        monkeypatch.setattr(sp, "get_sp_rank", lambda rank=rank: rank)
        shards.append(sp.sp_split_tokens(tokens))
    padded = torch.cat(shards)
    assert padded.numel() == sp.sp_pad_len(total)
    assert torch.equal(padded[:total], tokens) and not padded[total:].any()


@pytest.mark.parametrize("world", [2, 4])
def test_gather_rows_matches_full_gather(monkeypatch, world):
    # Last rows of prompts of 1, 3, 3, 1, 2 and 3 tokens: they land on every
    # rank, on a shard's first and last row, and next to the padded tail.
    hidden, rows = torch.randn(13, 8), torch.tensor([0, 3, 6, 7, 9, 12])
    monkeypatch.setattr(sp, "_SP_WORLD_SIZE", world)

    def on_each_rank(fn):
        out = []
        for rank in range(world):
            monkeypatch.setattr(sp, "get_sp_rank", lambda rank=rank: rank)
            out.append(fn(rank))
        return out

    shards = on_each_rank(lambda _: sp.sp_split_tokens(hidden))
    sent = []
    monkeypatch.setattr(sp, "get_sp_group", lambda: None)
    monkeypatch.setattr(
        sp, "gather_tokens", lambda x, group: sent.append(x) or x.repeat(world, 1)
    )
    on_each_rank(lambda rank: sp.sp_gather_rows(shards[rank], rows))
    monkeypatch.setattr(sp, "gather_tokens", lambda x, group: torch.cat(sent))
    for out in on_each_rank(lambda rank: sp.sp_gather_rows(shards[rank], rows)):
        assert torch.equal(out, hidden[rows])


@pytest.mark.parametrize("world", [2, 4])
def test_vocab_sharded_lm_head_matches_the_full_head(monkeypatch, world):
    embed_head = pytest.importorskip("atom.model_ops.embed_head", exc_type=ImportError)
    x, weight = torch.randn(3, 8), torch.randn(4 * world, 8)
    head = embed_head.ParallelLMHead.__new__(embed_head.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.num_embeddings, head.weight, head.bias = weight.shape[0], weight, None
    monkeypatch.setattr(
        embed_head, "tgemm", SimpleNamespace(mm=lambda x, w, b: x @ w.t())
    )

    def on_each_rank():
        out = []
        for rank in range(world):
            group = SimpleNamespace(world_size=world, rank_in_group=rank)
            monkeypatch.setattr(embed_head, "get_sp_group", lambda g=group: g)
            out.append(head._sp_sharded_logits(x))
        return out

    sent = []
    monkeypatch.setattr(
        embed_head, "gather_tokens", lambda s, g: sent.append(s) or s.repeat(world, 1)
    )
    on_each_rank()
    monkeypatch.setattr(embed_head, "gather_tokens", lambda s, g: torch.cat(sent))
    for logits in on_each_rank():
        torch.testing.assert_close(logits, x @ weight.t())
