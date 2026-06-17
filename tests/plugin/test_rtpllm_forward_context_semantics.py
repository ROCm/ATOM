"""Semantic checks for rtpllm forward-context bridge."""

import sys
import types
from types import SimpleNamespace

import torch


class _KwargsObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_forward_context_stubs():
    sys.modules["atom.config"].get_current_atom_config = lambda: sys.modules[
        "atom.config"
    ].Config()

    attention_gdn = types.ModuleType("atom.model_ops.attention_gdn")
    attention_gdn.GatedDeltaNet = type("GatedDeltaNet", (), {})
    sys.modules["atom.model_ops.attention_gdn"] = attention_gdn

    paged_attention = types.ModuleType("atom.model_ops.paged_attention")
    paged_attention.PagedAttention = type("PagedAttention", (), {})
    sys.modules["atom.model_ops.paged_attention"] = paged_attention

    gdn_attn = types.ModuleType("atom.model_ops.attentions.gdn_attn")
    gdn_attn.GDNAttentionMetadata = _KwargsObject
    gdn_attn.compute_causal_conv1d_metadata = lambda query_start_loc: (None, None, None)
    sys.modules["atom.model_ops.attentions.gdn_attn"] = gdn_attn

    plugin_attention = types.ModuleType("atom.plugin.attention")
    plugin_attention.AiterFlashAttentionDecodeMetadata = _KwargsObject
    plugin_attention.AiterFlashAttentionMetadataForPluginMode = _KwargsObject
    plugin_attention.AiterFlashAttentionPrefillMetadata = _KwargsObject
    sys.modules["atom.plugin.attention"] = plugin_attention

    utils_forward_context = types.ModuleType("atom.utils.forward_context")
    utils_forward_context.AttentionMetaData = _KwargsObject
    utils_forward_context.Context = _KwargsObject
    utils_forward_context._forward_kv_cache_context = {}
    utils_forward_context.reset_forward_context = lambda *args, **kwargs: None
    utils_forward_context.set_forward_context = lambda *args, **kwargs: None
    utils_forward_context.set_kv_cache_data = lambda *args, **kwargs: None
    sys.modules["atom.utils.forward_context"] = utils_forward_context


_install_forward_context_stubs()

from atom.plugin.rtpllm.utils.forward_context import RTPForwardContext  # noqa: E402


def _make_attn_inputs(
    *,
    input_lengths,
    prefix_lengths=None,
    sequence_lengths=None,
    sequence_lengths_plus_1_d=None,
    cu_seqlens=None,
    kv_cache_kernel_block_id_device=None,
    is_prefill=False,
    is_cuda_graph=False,
):
    return SimpleNamespace(
        input_lengths=input_lengths,
        prefix_lengths=prefix_lengths,
        sequence_lengths=sequence_lengths,
        sequence_lengths_plus_1_d=sequence_lengths_plus_1_d,
        cu_seqlens=cu_seqlens,
        kv_cache_kernel_block_id_device=kv_cache_kernel_block_id_device,
        is_prefill=is_prefill,
        is_cuda_graph=is_cuda_graph,
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
        attn_inputs, seq_size_per_block=4, num_tokens=5
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
        kv_cache_kernel_block_id_device=torch.tensor(
            [[123, 124, 125]], dtype=torch.int32
        ),
        is_prefill=False,
    )

    md = RTPForwardContext._build_gdn_metadata(
        attn_inputs, seq_size_per_block=16, num_tokens=1
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


def test_rtpllm_decode_seq_lens_priority_splits_graph_and_eager_modes():
    input_lengths = torch.tensor([1], dtype=torch.int32)
    sequence_lengths = torch.tensor([35], dtype=torch.int32)
    sequence_lengths_plus_1 = torch.tensor([35], dtype=torch.int32)

    eager_inputs = _make_attn_inputs(
        input_lengths=input_lengths,
        sequence_lengths=sequence_lengths,
        sequence_lengths_plus_1_d=sequence_lengths_plus_1,
        is_prefill=False,
    )
    eager_seq_lens = RTPForwardContext._build_seq_lens(
        eager_inputs, device=input_lengths.device
    )
    assert eager_seq_lens.cpu().tolist() == [35]

    graph_inputs = _make_attn_inputs(
        input_lengths=input_lengths,
        sequence_lengths=sequence_lengths,
        sequence_lengths_plus_1_d=sequence_lengths_plus_1,
        is_prefill=False,
        is_cuda_graph=True,
    )
    graph_seq_lens = RTPForwardContext._build_seq_lens(
        graph_inputs, device=input_lengths.device
    )
    assert graph_seq_lens.cpu().tolist() == [36]
