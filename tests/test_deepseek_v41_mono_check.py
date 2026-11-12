# SPDX-License-Identifier: MIT
"""The debug scorer observes the same weight ABI as native indexing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("aiter")

from atom.models.deepseek_v41.mono.check import check_index_score


@pytest.mark.parametrize("fp4", [False, True])
def test_reference_score_weight_contract(fp4):
    original = torch.tensor([[0.125, -2.5, 0.0, 17.0]], dtype=torch.bfloat16)
    scale = 1.0 / 64
    folded = original.float() * scale
    query, cache, step = object(), object(), object()
    runner = SimpleNamespace(index_fp4=fp4, iw=folded, index_query=lambda: query)
    indexer = SimpleNamespace(weights_scale=scale, score_quantized=Mock())
    check_index_score(runner, indexer, cache, step)
    args, kwargs = indexer.score_quantized.call_args
    assert args[0] is query and args[2] is cache and args[3] is step
    if fp4:
        assert args[1].dtype == torch.bfloat16
        torch.testing.assert_close(args[1], original, rtol=0, atol=0)
        assert kwargs == {"weight_scale": scale}
    else:
        assert args[1] is folded
        assert kwargs == {"weight_scale": 1.0}
