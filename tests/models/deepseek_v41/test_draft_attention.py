# SPDX-License-Identifier: MIT
"""The draft block's bidirectional attention, against a dense sink oracle.

Its own file because `draft_attention` reaches Triton through `sparse_attn`,
and the rest of `test_dspark.py` is host arithmetic a CPU runner should still
check. A module-level guard is the only kind that helps: an import that raises
during collection takes the whole session down rather than one file.
"""

from dataclasses import replace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("runs a Triton attention kernel", allow_module_level=True)

from atom.model_ops.deepseek_v41 import dspark
from atom.model_ops.deepseek_v41.draft_block import draft_step
from atom.model_ops.deepseek_v41.dspark import (
    DraftSrcBuffers,
    draft_attention,
    draft_src_indices,
    draft_step_indices,
)


@pytest.fixture
def two_source(monkeypatch):
    """Take the two-source path whatever the arch: the op only picks it on
    gfx1250, and the index math and the attention run anywhere."""
    monkeypatch.setattr(dspark, "two_source_draft", lambda: True)


@pytest.mark.parametrize("sources", [False, True])
def test_draft_attention_against_dense_sink_oracle(sources, monkeypatch):
    device = "cuda"
    torch.manual_seed(901)
    query = torch.randn(2, 5, 8, 512, dtype=torch.bfloat16, device=device)
    context = torch.randn(2, 8, 512, dtype=torch.bfloat16, device=device)
    keys = torch.randn(2, 5, 512, dtype=torch.bfloat16, device=device)
    sink = torch.randn(8, device=device)
    context_positions = torch.arange(8, device=device).expand(2, -1)
    anchors = torch.tensor([3, 6], device=device)
    step = draft_step(context_positions, anchors, 5, 4)
    if sources:
        # The window and the block as the two sources of the sparse MLA kernel.
        monkeypatch.setattr(dspark, "two_source_draft", lambda: True)
        src = draft_src_indices(anchors, 8, 4, 5, DraftSrcBuffers(2, 5))
        step = replace(step, src_indices=src)
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


def test_two_source_draft_attention_with_no_rows(two_source):
    # No rows: the builder launches nothing and the kernel reads no indptr.
    device = "cuda"
    query = torch.randn(0, 5, 8, 512, dtype=torch.bfloat16, device=device)
    context = torch.randn(0, 8, 512, dtype=torch.bfloat16, device=device)
    keys = torch.randn(0, 5, 512, dtype=torch.bfloat16, device=device)
    sink = torch.randn(8, device=device)
    context_positions = torch.arange(8, device=device).expand(0, -1)
    anchors = torch.zeros(0, dtype=torch.long, device=device)
    step = draft_step(context_positions, anchors, 5, 4)
    src = draft_src_indices(anchors, 8, 4, 5, DraftSrcBuffers(4, 5))
    actual = draft_attention(
        query, context, keys, sink, replace(step, src_indices=src), 512**-0.5
    )
    assert actual.shape == (0, 5, 8, 512)


def test_two_source_draft_attention_rejects_a_strided_input(two_source):
    device = "cuda"
    anchors = torch.tensor([3, 6], device=device)
    step = draft_step(torch.arange(8, device=device).expand(2, -1), anchors, 5, 4)
    src = draft_src_indices(anchors, 8, 4, 5, DraftSrcBuffers(2, 5))
    query = torch.randn(2, 5, 8, 1024, dtype=torch.bfloat16, device=device)[..., ::2]
    context = torch.randn(2, 8, 512, dtype=torch.bfloat16, device=device)
    keys = torch.randn(2, 5, 512, dtype=torch.bfloat16, device=device)
    with pytest.raises(AssertionError, match="contiguous"):
        draft_attention(
            query,
            context,
            keys,
            torch.randn(8, device=device),
            replace(step, src_indices=src),
            512**-0.5,
        )


def test_draft_src_indices_follows_the_arch_gate(monkeypatch):
    anchors = torch.tensor([3, 6], device="cuda")
    buffers = DraftSrcBuffers(2, 5)
    monkeypatch.setattr(dspark, "two_source_draft", lambda: False)
    assert draft_src_indices(anchors, 8, 4, 5, buffers) is None
    assert buffers.context is None  # nothing allocated off the gated arch
    monkeypatch.setattr(dspark, "two_source_draft", lambda: True)
    assert draft_src_indices(anchors, 8, 4, 5, None) is None
    assert draft_src_indices(anchors, 8, 4, 5, buffers) is not None


def _expected_sources(indices, ring):
    """The index list with its -1 entries dropped, as flat window / block rows
    and their indptrs: what the two sources must say."""
    batch, width, _ = indices.shape
    context, block, context_ptr, block_ptr = [], [], [0], [0]
    for b in range(batch):
        for t in range(width):
            row = indices[b, t].tolist()
            context += [b * ring + i for i in row[:ring] if i >= 0]
            block += [b * width + i - ring for i in row[ring:] if i >= 0]
            context_ptr.append(len(context))
            block_ptr.append(len(block))
    return context, context_ptr, block, block_ptr


@pytest.mark.parametrize(
    "batch,width,ring,window,low",
    [
        (1, 1, 4, 4, 0),
        (2, 5, 8, 4, 0),  # anchors inside a partly filled ring
        (3, 4, 16, 8, 0),
        (5, 5, 128, 64, 0),  # production ring and window
        (5, 5, 128, 64, 1000),  # full rings
        (7, 8, 64, 128, 0),  # ring narrower than the window
        (33, 3, 128, 128, 0),  # more requests than one power of two
    ],
)
def test_draft_src_indices_match_the_index_list(
    batch, width, ring, window, low, two_source
):
    torch.manual_seed(batch * 100 + width + ring)
    anchors = torch.randint(low, low + 2 * ring, (batch,), device="cuda")
    _, indices, _ = draft_step_indices(anchors, ring, window, width)
    # Buffers wider than the batch: the slices must be the batch's answer.
    src = draft_src_indices(
        anchors, ring, window, width, DraftSrcBuffers(batch + 3, width + 1)
    )
    context, context_ptr, block, block_ptr = _expected_sources(indices, ring)
    assert src.context_ptr.tolist() == context_ptr
    assert src.block_ptr.tolist() == block_ptr
    assert src.context[: len(context)].tolist() == context
    assert src.block[: len(block)].tolist() == block


def test_draft_src_buffers_are_reused_across_batches(two_source):
    buffers = DraftSrcBuffers(4, 5)
    first = draft_src_indices(
        torch.tensor([40, 3, 90, 7], device="cuda"), 64, 32, 5, buffers
    )
    anchors = torch.tensor([2, 70], device="cuda")
    second = draft_src_indices(anchors, 64, 32, 3, buffers)
    assert second.context.data_ptr() == first.context.data_ptr()
    _, indices, _ = draft_step_indices(anchors, 64, 32, 3)
    context, context_ptr, block, block_ptr = _expected_sources(indices, 64)
    assert second.context_ptr.tolist() == context_ptr
    assert second.block_ptr.tolist() == block_ptr
    assert second.context[: len(context)].tolist() == context
    assert second.block[: len(block)].tolist() == block


def test_draft_src_buffers_refuse_what_they_were_not_sized_for(two_source):
    buffers = DraftSrcBuffers(2, 5)
    with pytest.raises(ValueError, match="max_num_seqs"):
        draft_src_indices(
            torch.zeros(3, dtype=torch.long, device="cuda"), 8, 4, 5, buffers
        )
    with pytest.raises(ValueError, match="max_num_seqs"):
        draft_src_indices(
            torch.zeros(2, dtype=torch.long, device="cuda"), 8, 4, 6, buffers
        )
    draft_src_indices(torch.zeros(2, dtype=torch.long, device="cuda"), 8, 4, 5, buffers)
    with pytest.raises(ValueError, match="ring"):
        draft_src_indices(
            torch.zeros(2, dtype=torch.long, device="cuda"), 16, 4, 5, buffers
        )
