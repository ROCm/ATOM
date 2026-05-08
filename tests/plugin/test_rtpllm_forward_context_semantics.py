"""Semantic checks for rtpllm forward-context bridge."""

from types import SimpleNamespace

import torch

from atom.plugin.rtpllm.utils.forward_context import RTPForwardContext


def _make_attn_inputs(
    *,
    input_lengths,
    prefix_lengths=None,
    sequence_lengths=None,
    cu_seqlens=None,
    kv_cache_kernel_block_id_device=None,
    is_prefill=False,
):
    return SimpleNamespace(
        input_lengths=input_lengths,
        prefix_lengths=prefix_lengths,
        sequence_lengths=sequence_lengths,
        cu_seqlens=cu_seqlens,
        kv_cache_kernel_block_id_device=kv_cache_kernel_block_id_device,
        is_prefill=is_prefill,
    )


def test_rtpllm_forward_context_prefill_metadata_uses_real_inputs():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([3, 2], dtype=torch.int32),
        prefix_lengths=torch.tensor([5, 0], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor(
            [[10, 11, 12], [20, 21, 22]], dtype=torch.int32
        ),
        is_prefill=True,
    )

    md = RTPForwardContext._build_gdn_metadata(
        attn_inputs, seq_size_per_block=4
    )

    assert md.num_prefills == 2
    assert md.num_prefill_tokens == 5
    assert md.num_decodes == 0
    assert md.num_decode_tokens == 0
    assert tuple(md.non_spec_query_start_loc.shape) == (3,)
    assert tuple(md.non_spec_state_indices_tensor.shape) == (2,)
    assert torch.equal(
        md.non_spec_query_start_loc.cpu(), torch.tensor([0, 3, 5], dtype=torch.int32)
    )
    assert md.has_initial_state is not None
    assert md.has_initial_state.dtype == torch.bool
    assert md.has_initial_state.cpu().tolist() == [True, False]
    # last token idx = [5+3-1, 0+2-1] = [7, 1], block ids at col [1, 0].
    assert md.non_spec_state_indices_tensor.cpu().tolist() == [11, 20]


def test_rtpllm_forward_context_decode_metadata_state_indices_shape():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([1], dtype=torch.int32),
        sequence_lengths=torch.tensor([35], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor([[123, 124, 125]], dtype=torch.int32),
        is_prefill=False,
    )

    md = RTPForwardContext._build_gdn_metadata(
        attn_inputs, seq_size_per_block=16
    )

    assert md.num_prefills == 0
    assert md.num_decodes == 1
    assert md.num_decode_tokens == 1
    assert tuple(md.non_spec_query_start_loc.shape) == (2,)
    assert tuple(md.non_spec_state_indices_tensor.shape) == (1,)
    assert md.non_spec_state_indices_tensor.dtype == torch.int32
    # Ensure indices are valid int32 ids from RTP block table (no synthetic values).
    assert int(md.non_spec_state_indices_tensor.min().item()) >= 0
    # last token idx = 35 -> block col 2 under seq_size_per_block=16.
    assert md.non_spec_state_indices_tensor.cpu().tolist() == [125]
