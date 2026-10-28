# SPDX-License-Identifier: MIT
"""TP replicated o_proj: attention is reduced once, the residual never."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops import tp_replicated_o_proj as ops


@pytest.mark.parametrize("tokens", [0, 1, 2, 3, 4, 7, 32, 129])
def test_attention_is_added_once_before_replicated_residual(monkeypatch, tokens):
    world, width = 4, 8
    generator = torch.Generator().manual_seed(12)
    residual = torch.randn(tokens, width, generator=generator)
    attention = torch.randn(tokens, width, generator=generator)
    padded_attention = ops._pad_tokens(attention, world)
    partials = torch.randn(world, padded_attention.shape[0], width, generator=generator)
    expected = residual + attention + partials.sum(dim=0)[:tokens]
    contributions = []
    for rank in range(world):

        def reduce(x, **kwargs):
            assert kwargs == {"ca_fp8_quant": False}
            contributions.append(x.clone())
            return x.clone()

        group = SimpleNamespace(world_size=world, rank_in_group=rank, all_reduce=reduce)
        monkeypatch.setattr(ops, "get_tp_group", lambda group=group: group)
        owned = ops.replicated_o_proj_local_residual(attention, world)
        result = ops.replicated_o_proj_complete(partials[rank], owned, tokens, world)
        assert result.shape == (tokens, width)
    if tokens:
        # The accumulated replicated residual is added after the collective.
        torch.testing.assert_close(
            residual + torch.stack(contributions).sum(0), expected
        )
    else:
        assert contributions == []
