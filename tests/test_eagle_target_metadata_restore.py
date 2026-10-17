from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.spec_decode.eagle_proposer import (
    _restore_target_attention_state,
    _save_target_attention_state,
)


def test_eagle_restores_target_attention_metadata_in_place():
    context_lens = torch.tensor([101, 202], dtype=torch.int32)
    slot_mapping = torch.tensor([3, 4, 5, 6], dtype=torch.int64)
    cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    block_tables = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    sparse = object()
    plan = object()
    metadata = SimpleNamespace(
        context_lens=context_lens,
        slot_mapping=slot_mapping,
        cu_seqlens_q=cu_seqlens_q,
        block_tables=block_tables,
        sparse_attention_metadata=sparse,
        flydsl_work_plan=plan,
        max_seqlen_q=2,
        max_seqlen_k=202,
    )
    context = SimpleNamespace(is_prefill=True, is_draft=False)
    state = _save_target_attention_state(metadata, context)

    context_lens.add_(2)
    slot_mapping.fill_(-1)
    cu_seqlens_q.copy_(torch.tensor([0, 1, 2], dtype=torch.int32))
    metadata.context_lens = context_lens[:1]
    metadata.slot_mapping = slot_mapping[:1]
    metadata.cu_seqlens_q = cu_seqlens_q[:2]
    metadata.block_tables = block_tables[:1]
    metadata.sparse_attention_metadata = object()
    metadata.flydsl_work_plan = object()
    metadata.max_seqlen_q = 1
    metadata.max_seqlen_k = 204
    context.is_prefill = False
    context.is_draft = True

    _restore_target_attention_state(metadata, context, state)

    assert metadata.context_lens is context_lens
    assert metadata.slot_mapping is slot_mapping
    assert metadata.cu_seqlens_q is cu_seqlens_q
    assert torch.equal(context_lens, torch.tensor([101, 202], dtype=torch.int32))
    assert torch.equal(slot_mapping, torch.tensor([3, 4, 5, 6], dtype=torch.int64))
    assert torch.equal(cu_seqlens_q, torch.tensor([0, 2, 4], dtype=torch.int32))
    assert metadata.block_tables is block_tables
    assert metadata.sparse_attention_metadata is sparse
    assert metadata.flydsl_work_plan is plan
    assert metadata.max_seqlen_q == 2
    assert metadata.max_seqlen_k == 202
    assert context.is_prefill is True
    assert context.is_draft is False
