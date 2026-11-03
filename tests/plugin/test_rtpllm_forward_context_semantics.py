"""Semantic checks for rtpllm forward-context bridge."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import atom.config


class _KwargsObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_forward_context_stubs():
    """Stub the heavy deps `atom.plugin.rtpllm...forward_context` imports.

    Returns the `sys.modules` entries it displaced so the caller can put them
    back. Restoring matters: these are real modules other test files import,
    and a stub left in `sys.modules` makes the genuine one unreachable for the
    rest of the session. `from atom.model_ops.attentions.gdn_attn import
    GDNStateMixin` then fails with "unknown location" -- an import error whose
    text points at the imported module and says nothing about this file.
    """
    displaced = {}

    def stub(name, module):
        displaced[name] = sys.modules.get(name)
        sys.modules[name] = module
        return module

    config = atom.config
    config_attr = getattr(config, "get_current_atom_config", None)
    config.get_current_atom_config = lambda: SimpleNamespace(
        kv_cache_block_size=16, parallel_config=None
    )

    attention_gdn = stub(
        "atom.model_ops.attention_gdn",
        types.ModuleType("atom.model_ops.attention_gdn"),
    )
    attention_gdn.GatedDeltaNet = type("GatedDeltaNet", (), {})

    paged_attention = stub(
        "atom.model_ops.paged_attention",
        types.ModuleType("atom.model_ops.paged_attention"),
    )
    paged_attention.PagedAttention = type("PagedAttention", (), {})

    gdn_attn = stub(
        "atom.model_ops.attentions.gdn_attn",
        types.ModuleType("atom.model_ops.attentions.gdn_attn"),
    )
    gdn_attn.GDNAttentionMetadata = _KwargsObject
    gdn_attn.compute_causal_conv1d_metadata = lambda seqlens, device: (None, None, None)

    plugin_attention = stub(
        "atom.plugin.attention", types.ModuleType("atom.plugin.attention")
    )
    plugin_attention.AiterFlashAttentionDecodeMetadata = _KwargsObject
    plugin_attention.AiterFlashAttentionMetadataForPluginMode = _KwargsObject
    plugin_attention.AiterFlashAttentionPrefillMetadata = _KwargsObject

    # The one real symbol: `running_tokens_from_bs` is pure arithmetic and
    # imports standalone, and it is exactly the rule under test here -- a
    # hand-copied stub would pass while the bridge converted rows wrongly.
    # Imported before `stub` displaces the module, or this would find the stub.
    from atom.utils.forward_context import running_tokens_from_bs

    utils_forward_context = stub(
        "atom.utils.forward_context",
        types.ModuleType("atom.utils.forward_context"),
    )
    utils_forward_context.running_tokens_from_bs = running_tokens_from_bs
    utils_forward_context.DPMetadata = SimpleNamespace(
        gather_running_tokens=lambda *args: None
    )
    utils_forward_context.AttentionMetaData = _KwargsObject
    utils_forward_context.Context = _KwargsObject
    utils_forward_context._forward_kv_cache_context = SimpleNamespace(kv_cache_data={})
    utils_forward_context.reset_forward_context = lambda *args, **kwargs: None
    utils_forward_context.set_forward_context = lambda *args, **kwargs: None
    utils_forward_context.get_forward_context = lambda *args, **kwargs: (
        SimpleNamespace()
    )

    def _set_kv_cache_data(value):
        utils_forward_context._forward_kv_cache_context.kv_cache_data = value

    utils_forward_context.set_kv_cache_data = _set_kv_cache_data

    def restore():
        for name, previous in displaced.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if config_attr is None:
            delattr(config, "get_current_atom_config")
        else:
            config.get_current_atom_config = config_attr

    return restore


# Import a private copy so this test also works when another test has already
# imported the production module. Its dependencies are stubbed only while the
# private copy is loaded; the public module remains untouched.
_MODULE_NAME = "atom.plugin.rtpllm.utils._forward_context_semantics_test"
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "atom/plugin/rtpllm/utils/forward_context.py"
)
_restore_forward_context_stubs = _install_forward_context_stubs()
try:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    assert _spec is not None and _spec.loader is not None
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _module
    _spec.loader.exec_module(_module)
finally:
    _restore_forward_context_stubs()

RTPForwardContext = _module.RTPForwardContext
RTPForwardMLAContext = _module.RTPForwardMLAContext
RTPForwardQwen35HybridContext = _module.RTPForwardQwen35HybridContext


def _make_attn_inputs(
    *,
    input_lengths,
    prefix_lengths=None,
    sequence_lengths=None,
    sequence_lengths_plus_1_d=None,
    sequence_lengths_plus_1_device=None,
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
        sequence_lengths_plus_1_device=sequence_lengths_plus_1_device,
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


def test_qwen35_prefill_reads_prefix_state_and_writes_last_chunk_block():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([5, 3], dtype=torch.int32),
        prefix_lengths=torch.tensor([4, 0], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 5, 8], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor(
            [[10, 11, 12], [20, 21, 22]], dtype=torch.int32
        ),
        is_prefill=True,
    )
    md = RTPForwardQwen35HybridContext._build_gdn_metadata(
        attn_inputs, seq_size_per_block=4, num_tokens=8
    )
    assert md.non_spec_state_indices_in_tensor.tolist() == [10, 20]
    assert md.non_spec_state_indices_tensor.tolist() == [12, 20]
    assert md.has_initial_state.tolist() == [True, False]
    assert md.non_spec_state_indices_i64.tolist() == [12, 20]
    assert md.non_spec_state_indices_in_i64.tolist() == [10, 20]
    assert md.no_initial_state.tolist() == [False, True]


def test_qwen35_decode_mixed_boundary_reads_previous_and_writes_current_block():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([1, 1], dtype=torch.int32),
        sequence_lengths=torch.tensor([0, 0], dtype=torch.int32),
        sequence_lengths_plus_1_device=torch.tensor([5, 7], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor(
            [[10, 11, 12], [20, 21, 22]], dtype=torch.int32
        ),
        is_prefill=False,
    )
    md = RTPForwardQwen35HybridContext._build_gdn_metadata(
        attn_inputs, seq_size_per_block=4, num_tokens=2
    )
    # The device lengths, not the deliberately stale host lengths, drive replay.
    assert md.non_spec_state_indices_in_tensor.tolist() == [10, 21]
    assert md.non_spec_state_indices_tensor.tolist() == [11, 21]


def test_qwen35_decode_context_length_is_not_query_length():
    # RTP input_lengths is the accumulated context, not the number of tokens
    # scheduled in this decode step. Only the last table column is populated.
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([1274], dtype=torch.int32),
        sequence_lengths=torch.tensor([1274], dtype=torch.int32),
        sequence_lengths_plus_1_device=torch.tensor([1275], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor(
            [[-1] * 19 + [7]], dtype=torch.int32
        ),
        is_prefill=False,
    )
    md = RTPForwardQwen35HybridContext._build_gdn_metadata(
        attn_inputs, seq_size_per_block=64, num_tokens=1
    )
    assert md.non_spec_state_indices_in_tensor.tolist() == [7]
    assert md.non_spec_state_indices_tensor.tolist() == [7]


def test_qwen35_decode_context_length_crosses_block_boundary():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([1280], dtype=torch.int32),
        sequence_lengths=torch.tensor([1280], dtype=torch.int32),
        sequence_lengths_plus_1_device=torch.tensor([1281], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor(
            [[-1] * 19 + [7, 8]], dtype=torch.int32
        ),
        is_prefill=False,
    )
    md = RTPForwardQwen35HybridContext._build_gdn_metadata(
        attn_inputs, seq_size_per_block=64, num_tokens=1
    )
    assert md.non_spec_state_indices_in_tensor.tolist() == [7]
    assert md.non_spec_state_indices_tensor.tolist() == [8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm/CUDA")
def test_qwen35_graph_replay_updates_previous_state_at_block_boundary():
    device = torch.device("cuda")
    # Graph block-table views can have a wider backing row stride.
    table = torch.tensor(
        [[10, 11, -1], [20, 21, -1]], dtype=torch.int32, device=device
    )[:, :2]
    next_lengths = torch.tensor([5, 7], dtype=torch.int32, device=device)
    query_lengths = torch.ones(2, dtype=torch.int32, device=device)
    outputs = torch.tensor([11, 21], dtype=torch.int32, device=device)
    previous = torch.empty_like(outputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        RTPForwardQwen35HybridContext._previous_state_slots(
            table, next_lengths, outputs, 4, previous, query_lengths
        )
    graph.replay()
    assert previous.cpu().tolist() == [10, 21]
    next_lengths.copy_(torch.tensor([6, 5], dtype=torch.int32, device=device))
    graph.replay()
    assert previous.cpu().tolist() == [11, 20]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm/CUDA")
def test_qwen35_graph_replay_uses_one_query_token_per_request():
    device = torch.device("cuda")
    table = torch.tensor([[-1, -1, 7, 8]], dtype=torch.int32, device=device)
    next_lengths = torch.tensor([12], dtype=torch.int32, device=device)
    outputs = torch.tensor([7], dtype=torch.int32, device=device)
    previous = torch.empty_like(outputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        RTPForwardQwen35HybridContext._previous_state_slots(
            table, next_lengths, outputs, 4, previous, unit_query_length=True
        )
    graph.replay()
    assert previous.cpu().tolist() == [7]
    next_lengths.copy_(torch.tensor([13], dtype=torch.int32, device=device))
    outputs.copy_(torch.tensor([8], dtype=torch.int32, device=device))
    graph.replay()
    assert previous.cpu().tolist() == [7]


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


def test_recover_physical_block_table_accepts_expanded_kernel_layout():
    expanded = torch.tensor(
        [[448, 449, 450, 451, 452, 453, 454, 455]], dtype=torch.int32
    )

    recovered = RTPForwardContext._recover_physical_block_table_from_kernel(
        expanded,
        seq_size_per_block=1024,
        kernel_seq_size_per_block=128,
    )

    assert recovered.cpu().tolist() == [[56]]


def test_recover_physical_block_table_discards_incomplete_reserved_columns():
    compact = torch.tensor([[7, 8, 9]], dtype=torch.int32)

    # Only a complete ratio-sized group describes one physical page.
    with pytest.raises(ValueError, match="no complete physical page"):
        RTPForwardContext._recover_physical_block_table_from_kernel(
            compact,
            seq_size_per_block=1024,
            kernel_seq_size_per_block=16,
        )


def test_recover_physical_block_table_trims_mtp_reserved_columns():
    kernel = torch.tensor([[28, 29, 30, 31, 32, 33, 34, 35, 999]], dtype=torch.int32)
    recovered = RTPForwardContext._recover_physical_block_table_from_kernel(
        kernel,
        seq_size_per_block=64,
        kernel_seq_size_per_block=16,
    )
    assert recovered.tolist() == [[7, 8]]


def test_plugin_attention_metadata_keeps_indexer_block_table_expanded():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([1030], dtype=torch.int32),
        prefix_lengths=torch.tensor([0], dtype=torch.int32),
        kv_cache_block_id_device=torch.tensor([[7, 8]], dtype=torch.int32),
        is_prefill=True,
    )

    md = RTPForwardMLAContext._build_plugin_attention_metadata(
        attn_inputs=attn_inputs,
        positions=torch.arange(1030, dtype=torch.int32),
        seq_size_per_block=1024,
        kernel_seq_size_per_block=16,
    )

    assert md.plugin_metadata.block_table.cpu().tolist() == [[7, 8]]
    assert md.block_tables.shape == (1, 128)
    assert md.block_tables[0, :4].cpu().tolist() == [448, 449, 450, 451]
    assert md.block_tables[0, 64:68].cpu().tolist() == [512, 513, 514, 515]


def test_qwen35_context_does_not_use_glm5_indexer_block_expansion():
    block_table = torch.tensor([[7, 8]], dtype=torch.int32)

    qwen_block_tables = RTPForwardQwen35HybridContext._build_indexer_block_tables(
        block_table_i32=block_table,
        seq_size_per_block=1024,
        kernel_seq_size_per_block=16,
        cg_max_seq_len=0,
        in_capture=False,
        cg_bufs=None,
    )
    glm5_block_tables = RTPForwardMLAContext._build_indexer_block_tables(
        block_table_i32=block_table,
        seq_size_per_block=1024,
        kernel_seq_size_per_block=16,
        cg_max_seq_len=0,
        in_capture=False,
        cg_bufs=None,
    )

    assert qwen_block_tables.shape == (1, 2)
    assert qwen_block_tables.cpu().tolist() == [[7, 8]]
    assert glm5_block_tables.shape[1] > qwen_block_tables.shape[1]


def test_plugin_attention_metadata_keeps_physical_block_table_for_base_context():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([1030], dtype=torch.int32),
        prefix_lengths=torch.tensor([0], dtype=torch.int32),
        kv_cache_block_id_device=torch.tensor([[7, 8]], dtype=torch.int32),
        is_prefill=True,
    )

    md = RTPForwardContext._build_plugin_attention_metadata(
        attn_inputs=attn_inputs,
        positions=torch.arange(1030, dtype=torch.int32),
        seq_size_per_block=1024,
        kernel_seq_size_per_block=16,
    )

    assert md.plugin_metadata.block_table.cpu().tolist() == [[7, 8]]
    assert md.block_tables.shape == (1, 2)
    assert md.block_tables.cpu().tolist() == [[7, 8]]


def test_base_context_capture_recovers_physical_table_with_prewarmed_buffer():
    attn_inputs = _make_attn_inputs(
        input_lengths=torch.tensor([1], dtype=torch.int32),
        sequence_lengths=torch.tensor([35], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor(
            [[448, 449, 450, 451, 452, 453, 454, 455]], dtype=torch.int32
        ),
        is_prefill=False,
        is_cuda_graph=True,
    )

    cg_bufs = {"physical_block_table_i32": torch.empty((1, 1), dtype=torch.int32)}
    block_table = RTPForwardContext._resolve_plugin_block_table(
        attn_inputs=attn_inputs,
        seq_size_per_block=1024,
        kernel_seq_size_per_block=128,
        cg_bufs=cg_bufs,
        in_capture=True,
    )

    assert block_table is not None
    assert block_table.cpu().tolist() == [[56]]


def test_plugin_attention_metadata_builds_batch_id_per_q_token():
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

    assert md.plugin_metadata.batch_id_per_q_token.cpu().tolist() == [0, 0, 1]
    assert md.plugin_metadata.sparse_block_size == 1024
    assert md.cu_seqlens_q.cpu().tolist() == [0, 2, 3]
    assert md.cu_seqlens_k.cpu().tolist() == [0, 2, 3]
    assert md.cu_seqlen_ks.cpu().tolist() == [0, 0, 2]
    assert md.cu_seqlen_ke.cpu().tolist() == [1, 2, 3]
    assert md.total_kv == 3


def test_build_batch_id_per_q_token_prefers_prewarmed_i32_buffer(monkeypatch):
    query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    seq_id_i32 = torch.arange(8, dtype=torch.int32)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    req_id = RTPForwardContext._build_batch_id_per_q_token(
        query_start_loc=query_start_loc,
        num_tokens=3,
        device=query_start_loc.device,
        cg_bufs={
            "seq_id": torch.arange(8, dtype=torch.int64),
            "seq_id_i32": seq_id_i32,
        },
    )

    assert req_id.dtype == torch.int32
    assert req_id.data_ptr() == seq_id_i32.data_ptr()
    assert req_id.cpu().tolist() == [0, 1, 2]


def test_build_batch_id_per_q_token_requires_prewarmed_i32_buffer_in_capture(
    monkeypatch,
):
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    try:
        RTPForwardContext._build_batch_id_per_q_token(
            query_start_loc=query_start_loc,
            num_tokens=1,
            device=query_start_loc.device,
            cg_bufs={"seq_id": torch.arange(1, dtype=torch.int64)},
        )
    except RuntimeError as exc:
        assert "prewarmed seq_id_i32" in str(exc)
    else:
        raise AssertionError("expected missing seq_id_i32 to fail during capture")


def test_rtpllm_decode_seq_lens_uses_rtp_plus_one_in_graph_and_eager_modes():
    input_lengths = torch.tensor([1], dtype=torch.int32)
    sequence_lengths = torch.tensor([35], dtype=torch.int32)
    sequence_lengths_plus_1 = torch.tensor([35], dtype=torch.int32)

    eager_inputs = _make_attn_inputs(
        input_lengths=input_lengths,
        sequence_lengths=sequence_lengths,
        sequence_lengths_plus_1_device=sequence_lengths_plus_1,
        is_prefill=False,
    )
    eager_seq_lens = RTPForwardContext._build_seq_lens(
        eager_inputs, device=input_lengths.device
    )
    assert eager_seq_lens.cpu().tolist() == [35]

    graph_inputs = _make_attn_inputs(
        input_lengths=input_lengths,
        sequence_lengths=sequence_lengths,
        sequence_lengths_plus_1_device=sequence_lengths_plus_1,
        is_prefill=False,
        is_cuda_graph=True,
    )
    graph_seq_lens = RTPForwardContext._build_seq_lens(
        graph_inputs, device=input_lengths.device
    )
    assert graph_seq_lens.cpu().tolist() == [35]


def test_collect_layer_maps_keeps_mla_layers_separate():
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    mla_layer = RTPMLAAttention(sparse_backend=object(), layer_num=7)
    model = SimpleNamespace(modules=lambda: [mla_layer])

    gdn_map, full_attn_map, mla_map = RTPForwardContext.collect_layer_maps(model)

    assert gdn_map == {}
    assert full_attn_map == {}
    assert mla_map == {7: mla_layer}


def test_collect_layer_maps_keeps_sparse_mla_owner_for_indexer_cache():
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    mla_layer = RTPMLAAttention(sparse_backend=object(), layer_num=7)
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

    inner_mla = RTPMLAAttention(sparse_backend=object(), layer_num=9)
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
    mla_layer = RTPMLAAttention(sparse_backend=object(), layer_num=7)

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
    mla_layer = RTPMLAAttention(
        sparse_backend=object(), layer_num=7, kv_cache=old_cache
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
        mla_layer_map={7: mla_layer},
    )

    monkeypatch.setattr(
        RTPForwardContext,
        "build",
        classmethod(lambda cls, **kwargs: forward_context),
    )
    monkeypatch.setattr(
        "atom.plugin.rtpllm.utils.forward_context.get_current_atom_config",
        lambda: SimpleNamespace(kv_cache_block_size=99, parallel_config=None),
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
        sparse_backend=object(),
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
        sparse_backend=object(),
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
    )

    monkeypatch.setattr(
        RTPForwardContext,
        "build",
        classmethod(lambda cls, **kwargs: forward_context),
    )
    monkeypatch.setattr(
        "atom.plugin.rtpllm.utils.forward_context.get_current_atom_config",
        lambda: SimpleNamespace(kv_cache_block_size=16, parallel_config=None),
    )

    with RTPForwardContext.bind(
        model=SimpleNamespace(),
        runtime=SimpleNamespace(),
        inputs=SimpleNamespace(),
        positions=torch.tensor([0], dtype=torch.int32),
    ):
        assert mla_layer.kv_cache is layer_cache
        assert indexer.k_cache.kv_cache[0] is not old_index_cache
        # The mocked KV base has 2 x 3 physical token slots.
        assert indexer.k_cache.kv_cache[0].shape == (6, 1, 144)

    assert mla_layer.kv_cache is old_cache
    assert indexer.k_cache.kv_cache[0] is old_index_cache
