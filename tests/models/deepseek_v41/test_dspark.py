# SPDX-License-Identifier: MIT
"""Independent draft masks, ragged positions and the draft heads' checkpoint names."""

import pytest
import torch
from atom.model_ops.deepseek_v41.dspark import draft_attention, draft_step, rotate_rows
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.models.deepseek_v41.dspark import DeepseekV41DSpark

from atom.models.deepseek_v4_dspark import DSparkConfidenceHead, DSparkMarkovHead


def test_block_mask_keeps_all_draft_rows_and_only_the_visible_window():
    positions = torch.tensor([[-1, 0, 1, 2, 3, 4], [125, 126, 127, 128, 129, 130]])
    step = draft_step(positions, torch.tensor([2, 129]), 5, 4)
    expected = torch.tensor(
        [
            [-1, 1, 2, 3, -1, -1, 6, 7, 8, 9, 10],
            [-1, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(step.indices, expected[:, None].expand(-1, 5, -1))
    assert torch.equal(
        step.positions, torch.tensor([[3, 4, 5, 6, 7], [130, 131, 132, 133, 134]])
    )


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rope_ragged_request_positions(device, inverse):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    torch.manual_seed(711)
    rope = RotaryEmbedding(64, 256, base=10000).to(device)
    positions = torch.tensor(
        [[2, 3, 4, 5, 6], [126, 127, 128, 129, 130]], device=device
    )
    hidden = torch.randn(2, 5, 8, 512, dtype=torch.bfloat16, device=device)
    expected = torch.cat(
        [
            rope(hidden[i : i + 1].clone(), positions[i], inverse=inverse)
            for i in range(2)
        ]
    )
    actual = rotate_rows(rope, hidden.clone(), positions, inverse=inverse)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_draft_attention_against_dense_sink_oracle():
    device = "cuda"
    torch.manual_seed(901)
    query = torch.randn(2, 5, 8, 512, dtype=torch.bfloat16, device=device)
    context = torch.randn(2, 8, 512, dtype=torch.bfloat16, device=device)
    keys = torch.randn(2, 5, 512, dtype=torch.bfloat16, device=device)
    sink = torch.randn(8, device=device)
    context_positions = torch.arange(8, device=device).expand(2, -1)
    anchors = torch.tensor([3, 6], device=device)
    step = draft_step(context_positions, anchors, 5, 4)
    all_keys = torch.cat((context, keys), dim=1).float()
    scores = torch.einsum("bthd,bsd->bhts", query.float(), all_keys) * 512**-0.5
    mask = (context_positions <= anchors[:, None]) & (
        context_positions > anchors[:, None] - 4
    )
    mask = torch.cat((mask, torch.ones(2, 5, dtype=torch.bool, device=device)), dim=1)
    scores.masked_fill_(~mask[:, None, None], -torch.inf)
    scores = torch.cat((scores, sink[None, :, None, None].expand(2, -1, 5, -1)), dim=-1)
    probabilities = scores.softmax(-1)[..., :-1]
    expected = torch.einsum("bhts,bsd->bthd", probabilities, all_keys).bfloat16()
    original = context.clone()
    actual = draft_attention(query, context, keys, sink, step, 512**-0.5)
    assert torch.equal(context, original)
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < 0.004


def test_draft_head_names_and_width_match_the_v41_checkpoint():
    """What belongs to V4.1 here is the naming, not the arithmetic.

    The Markov tables ship under this checkpoint's own names while the shared
    V4 head calls them `markov_w1` / `markov_w2`, so the rename is checked
    against the destination's real parameters rather than against a copy of
    itself. The confidence projection consumes the concatenation, which is what
    makes the checkpoint's row `[1, hidden + rank]` wide. Both heads' math is
    V4's and is covered by `tests/test_dspark.py`.
    """
    hidden, rank, vocab = 64, 32, 128
    renamed = DeepseekV41DSpark.weights_mapper.apply_list(
        [f"mtp.2.markov_head.{name}.weight" for name in ("embed", "head")]
    )
    assert renamed == [
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
    ]
    markov = DSparkMarkovHead(vocab, rank)
    assert {f"mtp.2.markov_head.{name}" for name in markov.state_dict()} == set(renamed)
    assert DSparkConfidenceHead(hidden, rank).proj.weight.shape == (1, hidden + rank)
