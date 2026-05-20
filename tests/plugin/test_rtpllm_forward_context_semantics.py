"""Semantic checks for rtpllm forward-context bridge."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from aiter import dtypes


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
    utils_forward_context._forward_kv_cache_context = SimpleNamespace(kv_cache_data={})
    utils_forward_context.reset_forward_context = lambda *args, **kwargs: None
    utils_forward_context.set_forward_context = lambda *args, **kwargs: None
    utils_forward_context.get_forward_context = lambda *args, **kwargs: SimpleNamespace()

    def _set_kv_cache_data(value):
        utils_forward_context._forward_kv_cache_context.kv_cache_data = value

    utils_forward_context.set_kv_cache_data = _set_kv_cache_data
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
    kv_cache_block_id_device=None,
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
        kv_cache_block_id_device=kv_cache_block_id_device,
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


def test_plugin_attention_metadata_slot_mapping_uses_physical_block_table():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([1], dtype=torch.int32),
        sequence_lengths=torch.tensor([1030], dtype=torch.int32),
        kv_cache_block_id_device=torch.tensor([[7, 8]], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor(
            [[700, 701, 702]], dtype=torch.int32
        ),
        is_prefill=False,
    )

    md = RTPForwardContext._build_plugin_attention_metadata(
        attn_inputs=attn_inputs,
        positions=torch.tensor([1029], dtype=torch.int32),
        seq_size_per_block=1024,
    )

    assert md.plugin_metadata.block_table.cpu().tolist() == [[7, 8]]
    assert md.plugin_metadata.slot_mapping.cpu().tolist() == [8 * 1024 + 5]


def test_plugin_attention_metadata_builds_req_id_per_token():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([2, 1], dtype=torch.int32),
        prefix_lengths=torch.tensor([0, 0], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 2, 3], dtype=torch.int32),
        kv_cache_block_id_device=torch.tensor([[3], [4]], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor([[30], [40]], dtype=torch.int32),
        is_prefill=True,
    )

    md = RTPForwardContext._build_plugin_attention_metadata(
        attn_inputs=attn_inputs,
        positions=torch.tensor([0, 1, 0], dtype=torch.int32),
        seq_size_per_block=1024,
    )

    assert md.plugin_metadata.req_id_per_token.cpu().tolist() == [0, 0, 1]
    assert md.plugin_metadata.sparse_block_size == 1024
    assert md.cu_seqlens_q.cpu().tolist() == [0, 2, 3]
    assert md.cu_seqlens_k.cpu().tolist() == [0, 2, 3]
    assert md.cu_seqlen_ks.cpu().tolist() == [0, 0, 2]
    assert md.cu_seqlen_ke.cpu().tolist() == [1, 2, 3]
    assert md.total_kv == 3


def test_rtp_indexer_cache_accepts_byte_packed_kv_scale_base():
    kv_scale_base = torch.empty((2, 1024, 132), dtype=torch.uint8)
    layer_cache = SimpleNamespace(kv_scale_base=kv_scale_base)
    indexer = SimpleNamespace(head_dim=128)

    cache = RTPForwardContext._resolve_rtp_indexer_cache(
        layer_num=0,
        layer_cache=layer_cache,
        indexer=indexer,
        block_size=1024,
    )

    assert tuple(cache.shape) == (2, 1024, 132)
    assert cache.dtype == dtypes.fp8


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


def test_collect_layer_maps_keeps_mla_layers_separate():
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    mla_layer = RTPMLAAttention(dense_backend=object(), layer_num=7)
    model = SimpleNamespace(modules=lambda: [mla_layer])

    gdn_map, full_attn_map, mla_map = RTPForwardContext.collect_layer_maps(model)

    assert gdn_map == {}
    assert full_attn_map == {}
    assert mla_map == {7: mla_layer}


def test_collect_layer_maps_keeps_sparse_mla_owner_for_indexer_cache():
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    mla_layer = RTPMLAAttention(dense_backend=object(), layer_num=7)
    sparse_owner = SimpleNamespace(
        layer_num=7,
        indexer=SimpleNamespace(),
        mla_attn=mla_layer,
    )
    model = SimpleNamespace(modules=lambda: [sparse_owner, mla_layer])

    gdn_map, full_attn_map, mla_map = RTPForwardContext.collect_layer_maps(model)

    assert gdn_map == {}
    assert full_attn_map == {}
    assert mla_map == {7: sparse_owner}


def test_collect_layer_maps_recognizes_atom_mla_wrapper_by_indexer_and_mla_attn():
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    inner_mla = RTPMLAAttention(dense_backend=object(), layer_num=9)
    atom_wrapper = SimpleNamespace(
        layer_num=9,
        indexer=SimpleNamespace(),
        mla_attn=inner_mla,
    )
    model = SimpleNamespace(modules=lambda: [atom_wrapper])

    _, _, mla_map = RTPForwardContext.collect_layer_maps(model)

    assert mla_map == {9: atom_wrapper}


def test_build_kv_cache_tensors_threads_raw_layer_cache_for_mla():
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    layer_cache = SimpleNamespace(kv_cache_base=torch.empty(2, 3))
    runtime = SimpleNamespace(
        kv_cache=SimpleNamespace(get_layer_cache=lambda layer_num: layer_cache)
    )
    mla_layer = RTPMLAAttention(dense_backend=object(), layer_num=7)

    cache_tensors = RTPForwardContext._build_kv_cache_tensors(
        runtime=runtime,
        layer_maps=({}, {}, {7: mla_layer}),
    )

    assert cache_tensors["layer_7"].layer_num == 7
    assert cache_tensors["layer_7"].k_cache is layer_cache


def test_bind_temporarily_attaches_mla_layer_cache(monkeypatch):
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    old_cache = SimpleNamespace(name="old-cache")
    new_cache = SimpleNamespace(name="new-cache")
    mla_layer = RTPMLAAttention(dense_backend=object(), layer_num=7, kv_cache=old_cache)
    forward_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(),
        gdn_metadata=SimpleNamespace(),
        rtp_attn_inputs=SimpleNamespace(),
        rtp_kernel_seq_size_per_block=16,
        layer_group_map={},
        kv_cache_data={"layer_7": SimpleNamespace(k_cache=new_cache)},
        context=SimpleNamespace(),
        num_tokens=1,
        mla_layer_map={7: mla_layer},
        use_rtp_indexer_cache=False,
    )

    monkeypatch.setattr(
        RTPForwardContext,
        "build",
        classmethod(lambda cls, **kwargs: forward_context),
    )
    monkeypatch.setattr(
        "atom.plugin.rtpllm.utils.forward_context.get_current_atom_config",
        lambda: SimpleNamespace(kv_cache_block_size=99),
    )

    with RTPForwardContext.bind(
        model=SimpleNamespace(),
        runtime=SimpleNamespace(),
        inputs=SimpleNamespace(),
        positions=torch.tensor([0], dtype=torch.int32),
    ):
        assert mla_layer.kv_cache is new_cache

    assert mla_layer.kv_cache is old_cache


def test_bind_writes_kv_cache_to_mla_attn_owner_not_outer_wrapper(monkeypatch):
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    outer_cache = SimpleNamespace(name="outer-cache")
    old_inner_cache = SimpleNamespace(name="old-inner-cache")
    new_cache = SimpleNamespace(kv_cache_base=torch.empty(2, 3))
    indexer = SimpleNamespace(
        head_dim=128,
        k_cache=SimpleNamespace(kv_cache=[torch.empty(0)]),
    )
    mla_layer = RTPMLAAttention(
        dense_backend=object(),
        layer_num=7,
        kv_cache=old_inner_cache,
    )
    outer = SimpleNamespace(
        layer_num=7,
        indexer=indexer,
        mla_attn=mla_layer,
        kv_cache=outer_cache,
    )
    forward_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(),
        gdn_metadata=SimpleNamespace(),
        rtp_attn_inputs=SimpleNamespace(),
        rtp_kernel_seq_size_per_block=16,
        layer_group_map={},
        kv_cache_data={"layer_7": SimpleNamespace(k_cache=new_cache)},
        context=SimpleNamespace(),
        num_tokens=1,
        mla_layer_map={7: outer},
        use_rtp_indexer_cache=False,
    )

    monkeypatch.setattr(
        RTPForwardContext,
        "build",
        classmethod(lambda cls, **kwargs: forward_context),
    )

    with RTPForwardContext.bind(
        model=SimpleNamespace(),
        runtime=SimpleNamespace(),
        inputs=SimpleNamespace(),
        positions=torch.tensor([0], dtype=torch.int32),
    ):
        assert outer.kv_cache is outer_cache
        assert mla_layer.kv_cache is new_cache

    assert outer.kv_cache is outer_cache
    assert mla_layer.kv_cache is old_inner_cache


def test_bind_temporarily_attaches_sparse_mla_indexer_cache(monkeypatch):
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    old_cache = SimpleNamespace(name="old-cache")
    layer_cache = SimpleNamespace(kv_cache_base=torch.empty(2, 3))
    old_index_cache = torch.empty(0)
    indexer = SimpleNamespace(
        head_dim=128,
        k_cache=SimpleNamespace(kv_cache=[old_index_cache]),
    )
    mla_layer = RTPMLAAttention(
        dense_backend=object(),
        layer_num=7,
        kv_cache=old_cache,
        mla_modules=SimpleNamespace(indexer=indexer),
    )
    forward_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(),
        gdn_metadata=SimpleNamespace(),
        rtp_attn_inputs=SimpleNamespace(),
        rtp_kernel_seq_size_per_block=16,
        layer_group_map={},
        kv_cache_data={"layer_7": SimpleNamespace(k_cache=layer_cache)},
        context=SimpleNamespace(),
        num_tokens=1,
        mla_layer_map={7: mla_layer},
        use_rtp_indexer_cache=False,
    )

    monkeypatch.setattr(
        RTPForwardContext,
        "build",
        classmethod(lambda cls, **kwargs: forward_context),
    )
    monkeypatch.setattr(
        "atom.plugin.rtpllm.utils.forward_context.get_current_atom_config",
        lambda: SimpleNamespace(kv_cache_block_size=16),
    )

    with RTPForwardContext.bind(
        model=SimpleNamespace(),
        runtime=SimpleNamespace(),
        inputs=SimpleNamespace(),
        positions=torch.tensor([0], dtype=torch.int32),
    ):
        assert mla_layer.kv_cache is layer_cache
        assert indexer.k_cache.kv_cache[0] is not old_index_cache
        assert indexer.k_cache.kv_cache[0].shape == (32, 1, 144)

    assert mla_layer.kv_cache is old_cache
    assert indexer.k_cache.kv_cache[0] is old_index_cache


def test_bind_uses_rtp_kv_scale_base_when_enabled(monkeypatch):
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    old_cache = SimpleNamespace(name="old-cache")
    old_index_cache = torch.empty(0)
    kv_scale_base = torch.empty(2, 16, 132, dtype=dtypes.fp8)
    layer_cache = SimpleNamespace(
        kv_cache_base=torch.empty(2, 3),
        kv_scale_base=kv_scale_base,
    )
    indexer = SimpleNamespace(
        head_dim=128,
        k_cache=SimpleNamespace(kv_cache=[old_index_cache]),
    )
    mla_layer = RTPMLAAttention(
        dense_backend=object(),
        layer_num=7,
        kv_cache=old_cache,
        mla_modules=SimpleNamespace(indexer=indexer),
    )
    forward_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(),
        gdn_metadata=SimpleNamespace(),
        rtp_attn_inputs=SimpleNamespace(),
        rtp_kernel_seq_size_per_block=16,
        layer_group_map={},
        kv_cache_data={"layer_7": SimpleNamespace(k_cache=layer_cache)},
        context=SimpleNamespace(),
        num_tokens=1,
        mla_layer_map={7: mla_layer},
        use_rtp_indexer_cache=True,
    )

    monkeypatch.setattr(
        RTPForwardContext,
        "build",
        classmethod(lambda cls, **kwargs: forward_context),
    )

    with RTPForwardContext.bind(
        model=SimpleNamespace(),
        runtime=SimpleNamespace(),
        inputs=SimpleNamespace(),
        positions=torch.tensor([0], dtype=torch.int32),
    ):
        assert indexer.k_cache.kv_cache[0] is kv_scale_base

    assert indexer.k_cache.kv_cache[0] is old_index_cache


def test_bind_accepts_flattened_rtp_kv_scale_base_when_enabled(monkeypatch):
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    old_index_cache = torch.empty(0)
    flat_kv_scale_base = torch.empty(2, 16 * 132, dtype=dtypes.fp8)
    layer_cache = SimpleNamespace(
        kv_cache_base=torch.empty(2, 3),
        kv_scale_base=flat_kv_scale_base,
    )
    indexer = SimpleNamespace(
        head_dim=128,
        k_cache=SimpleNamespace(kv_cache=[old_index_cache]),
    )
    mla_layer = RTPMLAAttention(
        dense_backend=object(),
        layer_num=7,
        mla_modules=SimpleNamespace(indexer=indexer),
    )
    forward_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(),
        gdn_metadata=SimpleNamespace(),
        rtp_attn_inputs=SimpleNamespace(),
        rtp_kernel_seq_size_per_block=16,
        layer_group_map={},
        kv_cache_data={"layer_7": SimpleNamespace(k_cache=layer_cache)},
        context=SimpleNamespace(),
        num_tokens=1,
        mla_layer_map={7: mla_layer},
        use_rtp_indexer_cache=True,
    )

    monkeypatch.setattr(
        RTPForwardContext,
        "build",
        classmethod(lambda cls, **kwargs: forward_context),
    )

    with RTPForwardContext.bind(
        model=SimpleNamespace(),
        runtime=SimpleNamespace(),
        inputs=SimpleNamespace(),
        positions=torch.tensor([0], dtype=torch.int32),
    ):
        assert indexer.k_cache.kv_cache[0].data_ptr() == flat_kv_scale_base.data_ptr()
        assert indexer.k_cache.kv_cache[0].shape == (2, 16, 132)

    assert indexer.k_cache.kv_cache[0] is old_index_cache


def test_bind_rejects_incompatible_indexer_cache_layout(monkeypatch):
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    layer_cache = SimpleNamespace(
        kv_cache_base=torch.empty(2, 3),
        kv_scale_base=torch.empty(2, 16, 64, dtype=dtypes.fp8),
    )
    indexer = SimpleNamespace(
        head_dim=128,
        k_cache=SimpleNamespace(kv_cache=[torch.empty(0)]),
    )
    mla_layer = RTPMLAAttention(
        dense_backend=object(),
        layer_num=7,
        mla_modules=SimpleNamespace(indexer=indexer),
    )
    forward_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(),
        gdn_metadata=SimpleNamespace(),
        rtp_attn_inputs=SimpleNamespace(),
        rtp_kernel_seq_size_per_block=16,
        layer_group_map={},
        kv_cache_data={"layer_7": SimpleNamespace(k_cache=layer_cache)},
        context=SimpleNamespace(),
        num_tokens=1,
        mla_layer_map={7: mla_layer},
        use_rtp_indexer_cache=True,
    )

    monkeypatch.setattr(
        RTPForwardContext,
        "build",
        classmethod(lambda cls, **kwargs: forward_context),
    )

    with pytest.raises(ValueError, match="layout mismatch"):
        with RTPForwardContext.bind(
            model=SimpleNamespace(),
            runtime=SimpleNamespace(),
            inputs=SimpleNamespace(),
            positions=torch.tensor([0], dtype=torch.int32),
        ):
            pass
