# SPDX-License-Identifier: MIT
"""SP token sharding, row gathers and the vocab-sharded LM head, on CPU."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from atom.distributed import ulysses_sp as sp


def _on_each_rank(monkeypatch, world, fn):
    parallel_state = ModuleType("aiter.dist.parallel_state")
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)
    out = []
    for rank in range(world):
        group = SimpleNamespace(rank_in_group=rank)
        parallel_state.get_pcp_group = lambda g=group: g
        out.append(fn(rank))
    return out


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("total", [1, 3, 5, 17])
def test_rank_ranges_tile_the_sequence_in_order(monkeypatch, world, total):
    monkeypatch.setattr(sp, "_SP_WORLD_SIZE", world)
    ranges = _on_each_rank(monkeypatch, world, lambda _: sp.sp_local_slice(total))
    rows = [row for r in ranges for row in range(r.start, r.stop)]
    assert rows == list(range(len(rows))) and total <= len(rows) < total + world


@pytest.mark.parametrize("world", [2, 4])
def test_gather_rows_matches_full_gather(monkeypatch, world):
    # Last rows of prompts of 1, 3, 3, 1, 2 and 3 tokens: they land on every
    # rank, on a range's first and last row, and next to the rows past the end.
    buffer, rows = torch.randn(16, 8), torch.tensor([0, 3, 6, 7, 9, 12])
    hidden = buffer[:13]
    monkeypatch.setattr(sp, "_SP_WORLD_SIZE", world)

    def on_each_rank(fn):
        return _on_each_rank(monkeypatch, world, fn)

    shards = on_each_rank(lambda _: buffer[sp.sp_local_slice(13)])
    sent = []
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

    def on_each_rank(all_gather):
        out = []
        for rank in range(world):
            group = SimpleNamespace(
                world_size=world, rank_in_group=rank, all_gather=all_gather
            )
            monkeypatch.setattr(embed_head, "get_pcp_group", lambda g=group: g)
            out.append(head._sp_sharded_logits(x))
        return out

    sent = []
    on_each_rank(lambda s, dim, use_custom: sent.append(s) or s.repeat(1, world))
    for logits in on_each_rank(lambda s, dim, use_custom: torch.cat(sent, dim)):
        torch.testing.assert_close(logits, x @ weight.t())
