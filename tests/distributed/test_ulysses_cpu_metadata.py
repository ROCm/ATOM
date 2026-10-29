# SPDX-License-Identifier: MIT
"""SP token sharding and the per-rank counts published to MoE, on CPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.distributed import ulysses_sp as sp
from atom.utils.forward_context import (
    get_forward_context,
    reset_forward_context,
    set_forward_context,
)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("total", [1, 3, 5, 17])
def test_shards_keep_token_order_and_publish_padded_counts(monkeypatch, world, total):
    monkeypatch.setattr(sp, "_SP_WORLD_SIZE", world)
    tokens = torch.arange(1, total + 1)
    shards = []
    for rank in range(world):
        monkeypatch.setattr(sp, "get_sp_rank", lambda rank=rank: rank)
        shards.append(sp.sp_split_tokens(tokens))
    padded = torch.cat(shards)
    assert padded.numel() == sp.sp_pad_len(total)
    assert torch.equal(padded[:total], tokens) and not padded[total:].any()

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    try:
        context = SimpleNamespace(forward_mode=None)
        set_forward_context(None, config, context, np.array([total]))
        counts = get_forward_context().context.running_tokens_across_dp
        assert counts == (shards[0].numel(),) * world
    finally:
        reset_forward_context()


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
    monkeypatch.setattr(
        sp, "_all_gather_tokens", lambda x: sent.append(x) or x.repeat(world, 1)
    )
    on_each_rank(lambda rank: sp.sp_gather_rows(shards[rank], rows))
    monkeypatch.setattr(sp, "_all_gather_tokens", lambda x: torch.cat(sent))
    for out in on_each_rank(lambda rank: sp.sp_gather_rows(shards[rank], rows)):
        assert torch.equal(out, hidden[rows])
