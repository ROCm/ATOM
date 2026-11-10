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

from atom.model_ops.deepseek_v41.draft_block import draft_step
from atom.model_ops.deepseek_v41.dspark import draft_attention, draft_csr_indices


@pytest.mark.parametrize("two_source", [False, True])
def test_draft_attention_against_dense_sink_oracle(two_source):
    device = "cuda"
    torch.manual_seed(901)
    query = torch.randn(2, 5, 8, 512, dtype=torch.bfloat16, device=device)
    context = torch.randn(2, 8, 512, dtype=torch.bfloat16, device=device)
    keys = torch.randn(2, 5, 512, dtype=torch.bfloat16, device=device)
    sink = torch.randn(8, device=device)
    context_positions = torch.arange(8, device=device).expand(2, -1)
    anchors = torch.tensor([3, 6], device=device)
    step = draft_step(context_positions, anchors, 5, 4)
    if two_source:
        # The window and the block as the two sources of the sparse MLA kernel.
        step = replace(step, csr_indices=draft_csr_indices(step.indices, 8))
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


@pytest.mark.parametrize("batch,width,ring", [(1, 1, 4), (2, 5, 8), (3, 4, 16)])
def test_draft_csr_indices_drops_the_unused_entries_in_order(batch, width, ring):
    torch.manual_seed(batch * 100 + width)
    device = "cuda"
    history = torch.randint(0, ring, (batch, width, ring), device=device)
    history[torch.rand(batch, width, ring, device=device) < 0.4] = -1
    block = ring + torch.arange(width, device=device).expand(batch, width, -1)
    block = torch.where(torch.rand(batch, width, width, device=device) < 0.2, -1, block)
    indices = torch.cat((history, block), dim=-1).int().contiguous()
    context, context_ptr, draft, draft_ptr = draft_csr_indices(indices, ring)
    context_rows, draft_rows = [0], [0]
    for b in range(batch):
        for t in range(width):
            row = indices[b, t].tolist()
            want_context = [b * ring + i for i in row[:ring] if i >= 0]
            want_draft = [b * width + i - ring for i in row[ring:] if i >= 0]
            lo, hi = (
                context_ptr[b * width + t].item(),
                context_ptr[b * width + t + 1].item(),
            )
            assert context[lo:hi].tolist() == want_context
            lo, hi = (
                draft_ptr[b * width + t].item(),
                draft_ptr[b * width + t + 1].item(),
            )
            assert draft[lo:hi].tolist() == want_draft
            context_rows.append(context_rows[-1] + len(want_context))
            draft_rows.append(draft_rows[-1] + len(want_draft))
    assert context_ptr.tolist() == context_rows
    assert draft_ptr.tolist() == draft_rows
