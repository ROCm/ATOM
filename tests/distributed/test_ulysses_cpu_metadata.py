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
