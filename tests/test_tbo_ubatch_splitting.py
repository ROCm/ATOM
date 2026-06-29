import torch

from atom.utils.forward_context import AttentionMetaData
from atom.utils.tbo.ubatch_splitting import (
    UBatchSlice,
    derive_prefill_lens_from_positions,
    split_attn_metadata,
)


def test_split_attn_metadata_clamps_context_lens_for_token_split_prefill():
    attn_metadata = AttentionMetaData(
        cu_seqlens_q=torch.tensor([0, 10, 20], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 10, 20], dtype=torch.int32),
        context_lens=torch.tensor([10, 10], dtype=torch.int32),
        slot_mapping=torch.arange(20, dtype=torch.int64),
        has_cached=False,
    )

    ub_attn = split_attn_metadata(
        attn_metadata,
        UBatchSlice(request_slice=slice(0, 2), token_slice=slice(5, 20)),
        padded_bs=2,
    )

    assert ub_attn.cu_seqlens_q.tolist() == [0, 5, 15]
    assert ub_attn.cu_seqlens_k.tolist() == [0, 5, 15]
    assert ub_attn.context_lens.tolist() == [5, 10]
    assert ub_attn.max_seqlen_k == 10


def test_split_attn_metadata_preserves_context_lens_for_cached_prefill():
    attn_metadata = AttentionMetaData(
        cu_seqlens_q=torch.tensor([0, 5, 15], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 10, 25], dtype=torch.int32),
        context_lens=torch.tensor([10, 15], dtype=torch.int32),
        num_cached_tokens=torch.tensor([5, 5], dtype=torch.int32),
        slot_mapping=torch.arange(15, dtype=torch.int64),
        has_cached=True,
    )

    ub_attn = split_attn_metadata(
        attn_metadata,
        UBatchSlice(request_slice=slice(0, 2), token_slice=slice(0, 15)),
        padded_bs=2,
    )

    assert ub_attn.context_lens.tolist() == [10, 15]
    assert ub_attn.num_cached_tokens.tolist() == [5, 5]
    assert ub_attn.max_seqlen_k == 15


def test_derive_prefill_lens_from_positions_handles_straddled_request():
    extend_lens, context_lens = derive_prefill_lens_from_positions(
        positions=[5, 6, 7, 8, 9, 0, 1, 2],
        full_cu_seqlens_q=[0, 10, 13],
        ub_slice=UBatchSlice(request_slice=slice(0, 2), token_slice=slice(5, 13)),
    )

    assert extend_lens.tolist() == [5, 3]
    assert context_lens.tolist() == [10, 3]


def test_derive_prefill_lens_from_positions_handles_cached_prefix():
    extend_lens, context_lens = derive_prefill_lens_from_positions(
        positions=[8, 9, 10, 11],
        full_cu_seqlens_q=[0, 4],
        ub_slice=UBatchSlice(request_slice=slice(0, 1), token_slice=slice(0, 4)),
    )

    assert extend_lens.tolist() == [4]
    assert context_lens.tolist() == [12]
