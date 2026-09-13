# SPDX-License-Identifier: MIT
"""Production CSA2 selection sizes, including actual candidate pruning."""

import pytest
import torch
import torch.nn.functional as F

from atom.model_ops.deepseek_v41.indexer import select_indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("tie_break", ["small_position", "large_position"])
def test_full_and_reindex_beyond_candidate_capacity(tie_break):
    torch.manual_seed(98431)
    # 2048 candidate blocks cover only 16384 keys: both selection levels
    # really discard keys, including a partial newest block at the end.
    width = 32771
    q = torch.randn(2, 7, 32, 128, dtype=torch.bfloat16, device="cuda")
    keys = torch.randn(2, width, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(2, 7, 32, dtype=torch.bfloat16, device="cuda")
    visible = torch.tensor([0, 511, 512, 513, 16384, 16385, width], device="cuda")
    scores = (torch.einsum("bqhd,bkd->bqhk", q, keys).relu() * weights[..., None]).sum(
        2
    )
    scores.masked_fill_(
        torch.arange(width, device="cuda") >= visible[:, None], -torch.inf
    )

    def expected_ids(values, count):
        ids = torch.arange(values.shape[-1], device="cuda").expand_as(values)
        if tie_break == "large_position":
            values, ids = values.flip(-1), ids.flip(-1)
        order = values.argsort(dim=-1, descending=True, stable=True)[..., :count]
        chosen = ids.gather(-1, order)
        valid = values.gather(-1, order) > -torch.inf
        result = chosen.masked_fill(~valid, width).sort(-1).values
        return result.masked_fill(result == width, -1).int()

    maxima = (
        F.pad(scores, (0, -width % 8), value=-torch.inf).unflatten(-1, (-1, 8)).amax(-1)
    )
    block_ids = torch.arange(maxima.shape[-1], device="cuda")
    maxima.masked_fill_(
        (visible[:, None] > 0) & (block_ids == (visible[:, None] - 1) // 8), torch.inf
    )
    expected_blocks = expected_ids(maxima, 2048)
    selected, blocks = select_indices(
        q, weights, keys, visible, topk=512, make_candidates=True, tie_break=tie_break
    )
    assert torch.equal(selected, expected_ids(scores, 512))
    assert torch.equal(blocks, expected_blocks)
    keep = torch.zeros_like(maxima, dtype=torch.int32)
    keep.scatter_add_(-1, blocks.long().clamp_min(0), (blocks >= 0).int())
    restricted = scores.masked_fill(
        keep.repeat_interleave(8, -1)[..., :width] == 0, -torch.inf
    )
    reindex, _ = select_indices(
        q,
        weights,
        keys,
        visible,
        topk=512,
        candidate_blocks=blocks,
        tie_break=tie_break,
    )
    assert torch.equal(reindex, expected_ids(restricted, 512))
    assert (blocks[:, -1] >= 0).sum().item() == 2 * 2048
    assert (blocks[:, -1] == (width - 1) // 8).any(-1).all()
