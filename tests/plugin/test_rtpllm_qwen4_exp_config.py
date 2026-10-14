"""Qwen4Exp RTP plugin registration and hybrid-cache configuration."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import triton
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.ops import DataType, HybridAttentionType

from atom.models.qwen4_exp import qwen4_exp_packed_modules_mapping
from atom.plugin.rtpllm.models.qwen3_5 import _ATOMQwen35MoeRuntime
from atom.plugin.rtpllm.models.qwen4_exp import (
    ATOMQwen4Exp,
    _ATOMQwen4ExpRuntime,
)
from atom.plugin.rtpllm.utils import qwen4_exp_context
from atom.plugin.rtpllm.utils.qwen4_exp_context import (
    RTPQwen4ExpContext,
    _qsa_graph_slots_kernel,
)


def test_qwen4_exp_ngram_mapping_is_shared_across_plugin_paths():
    mapping = qwen4_exp_packed_modules_mapping({"base": ("target", 0)}, 3)
    assert mapping["base"] == ("target", 0)
    assert mapping[".ngram_embedding.shard_2."] == (".ngram_embedding.", 2)
    assert ".ngram_embedding.shard_3." not in mapping


def test_qwen4_exp_atom_registry_is_independent_of_import_order():
    from atom.plugin.prepare import _set_framework_backbone
    from atom.plugin.register import _ATOM_SUPPORTED_MODELS

    _set_framework_backbone("atom")
    try:
        assert (
            _ATOM_SUPPORTED_MODELS["Qwen4ExpForConditionalGeneration"].__name__
            == "Qwen4ExpForConditionalGeneration"
        )
    finally:
        _set_framework_backbone("atom")


def test_qwen4_exp_model_creation_survives_early_rtp_package_import():
    # A fresh interpreter matters: the test module has already imported both
    # the package and the registry by the time this test runs.
    script = """
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from atom.plugin.prepare import _set_framework_backbone, is_rtpllm, prepare_model
_set_framework_backbone("atom")
assert not is_rtpllm()
import atom.plugin.rtpllm.models
from atom.plugin import config as plugin_config, register
from atom.models.qwen4_exp import Qwen4ExpForConditionalGeneration
assert not is_rtpllm()

quant_config = SimpleNamespace(exclude_layers=[], remap_layer_name=Mock())
atom_config = SimpleNamespace(
    hf_config=SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        split_ngram_parts=1,
    ),
    quant_config=quant_config,
    multimodal_config=object(),
)
created = []

def lightweight_init(self, atom_config, prefix=""):
    torch.nn.Module.__init__(self)
    self.atom_config = atom_config
    created.append(self)

with (
    patch.object(plugin_config, "generate_atom_config_for_plugin_mode", return_value=atom_config),
    patch.object(register, "set_attn_cls"),
    patch.object(register, "init_aiter_dist"),
    patch.object(Qwen4ExpForConditionalGeneration, "__init__", lightweight_init),
):
    model = prepare_model(SimpleNamespace(), engine="rtpllm")

assert type(model) is Qwen4ExpForConditionalGeneration
assert created == [model]
assert model.atom_config is atom_config
quant_config.remap_layer_name.assert_called_once()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_lazy_model_registry_mapping_accessors_resolve_classes():
    from atom.plugin.register import _LazyModelRegistry

    registry = _LazyModelRegistry({"thing": "collections:Counter"})
    from collections import Counter

    assert registry.get("thing") is Counter
    assert registry.get("missing") is None
    assert list(registry.values()) == [Counter]
    assert list(registry.items()) == [("thing", Counter)]


def test_qwen4_exp_plugin_config_and_registration(tmp_path):
    text_config = {
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_hidden_layers": 4,
        "hidden_size": 2560,
        "vocab_size": 1000,
        "max_position_embeddings": 8192,
        "rope_parameters": {"rope_theta": 1000000, "partial_rotary_factor": 0.25},
        "rms_norm_eps": 1e-6,
        "num_experts_per_tok": 10,
        "num_experts": 512,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 2560,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "indexer_head_dim": 128,
        "indexer_compress_ratio": 4,
        "ple_layer_ids": [2],
        "ngram_size": 3,
        "ple_conv_kernel_size": 4,
        "hc_count": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text_config}))
    config = ATOMQwen4Exp._create_config(str(tmp_path))

    assert _model_factory["atom_qwen4_exp"] is ATOMQwen4Exp
    assert config.hybrid_attention_config.hybrid_attention_types == [
        HybridAttentionType.LINEAR,
        HybridAttentionType.LINEAR,
        HybridAttentionType.LINEAR,
        HybridAttentionType.NONE,
    ]
    assert config.linear_attention_config.ssm_state_dtype == DataType.TYPE_FP32
    assert not config.has_post_decoder_layernorm
    assert config.compute_dtype == torch.bfloat16
    assert config.config_dtype == "bf16"
    config.init_precision_config(None, None)
    assert config.compute_dtype == torch.bfloat16
    config.attn_config.tokens_per_block = 64
    kv_cache_config = SimpleNamespace(seq_size_per_block=64, reuse_cache=True)
    ATOMQwen4Exp._apply_kv_cache_config(config, kv_cache_config)
    qsa_bytes = (64 + 16) * 128 * 2
    ple_bytes = (3 - 1) * 8 + (2560 * 4) * ((4 - 1) * 3) * 2
    assert config.auxiliary_paged_cache_bytes_per_block == qsa_bytes + ple_bytes
    assert kv_cache_config.reuse_cache is False

    with pytest.raises(ValueError, match="cached task prompts"):
        ATOMQwen4Exp._apply_kv_cache_config(
            config,
            SimpleNamespace(seq_size_per_block=64, multi_task_prompt="tasks.json"),
        )


def test_qwen4_exp_cuda_graph_switch(monkeypatch):
    monkeypatch.setenv("ENABLE_CUDA_GRAPH", "1")
    assert ATOMQwen4Exp.support_cuda_graph(None)
    monkeypatch.setenv("ENABLE_CUDA_GRAPH", "0")
    assert not ATOMQwen4Exp.support_cuda_graph(None)


def test_qwen4_exp_graph_uses_refreshed_device_sequence_lengths(monkeypatch):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    lengths = torch.tensor([65, 130], dtype=torch.int32)
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=False,
            is_cuda_graph=True,
            sequence_lengths_plus_1_device=lengths,
        )
    )
    assert runtime._extract_positions(inputs, torch.device("cpu"), 2).tolist() == [
        64,
        129,
    ]


def test_qwen4_exp_graph_marks_both_cache_tags(monkeypatch):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    seen = {}

    def parent_prepare(self, inputs, is_cuda_graph=False):
        seen["inputs"] = inputs.attention_inputs
        return is_cuda_graph

    monkeypatch.setattr(_ATOMQwen35MoeRuntime, "prepare_fmha_impl", parent_prepare)
    linear = SimpleNamespace(is_cuda_graph=False)
    full = SimpleNamespace(is_cuda_graph=False)
    inputs = SimpleNamespace(attention_inputs={"linear": linear, "full": full})
    assert runtime.prepare_fmha_impl(inputs, True)
    assert seen["inputs"] is linear
    assert linear.is_cuda_graph and full.is_cuda_graph
    assert inputs.attention_inputs == {"linear": linear, "full": full}


def test_qwen4_exp_graph_prewarms_qsa_caches_before_capture(monkeypatch):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    monkeypatch.setattr(
        _ATOMQwen35MoeRuntime,
        "_ensure_cuda_graph_prewarmed",
        lambda self: setattr(self, "_cg_meta_bufs", {}),
    )
    calls = []
    qsa = SimpleNamespace(is_qsa_attention=True)
    runtime._rtp_layer_maps = ({}, {7: qsa}, {})
    runtime._rtp_forward_context_cls = SimpleNamespace(
        _bind_qsa_caches=lambda *args: calls.append(args)
    )
    runtime.kv_cache = SimpleNamespace(get_seq_size_per_block=lambda _: 64)
    runtime._cg_max_num_tokens = 2
    runtime._cg_max_seq_len = 128
    runtime._get_model_device = lambda: torch.device("cpu")

    runtime._ensure_cuda_graph_prewarmed()

    assert calls == [(runtime, [qsa], 64)]
    assert runtime._cg_meta_bufs["physical_block_table_i32"].shape == (2, 3)


def test_qwen4_exp_decode_state_reads_previous_block_at_boundary():
    block_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    positions = torch.tensor([64, 69], dtype=torch.int32)
    output_slots = torch.tensor([11, 21], dtype=torch.int32)

    input_slots = RTPQwen4ExpContext._previous_state_slots(
        block_table, positions, output_slots, 64
    )

    assert input_slots.tolist() == [10, 21]
    assert output_slots.tolist() == [11, 21]


def test_qwen4_exp_prefill_reads_prefix_block_when_continuing_across_boundary():
    # A cached prefix ends exactly at the block boundary. One new token must
    # read block 10's state and write block 11, including for PLE consumers.
    attn_inputs = SimpleNamespace(
        is_prefill=True,
        is_cuda_graph=False,
        input_lengths=torch.tensor([1], dtype=torch.int32),
        prefix_lengths=torch.tensor([64], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor([[10, 11]], dtype=torch.int32),
    )
    metadata = RTPQwen4ExpContext._build_gdn_metadata(
        attn_inputs, seq_size_per_block=64, num_tokens=1
    )
    assert metadata.non_spec_state_indices_in_tensor.tolist() == [10]
    assert metadata.non_spec_state_indices_tensor.tolist() == [11]
    assert metadata.has_initial_state.tolist() == [True]


@pytest.mark.parametrize(
    ("block_table", "positions", "output_slots", "message"),
    [
        ([[10]], [129], [11], "out of range"),
        ([[10, -1]], [65], [11], "invalid"),
        ([[10, 11]], [64, 65], [11], "one starting position"),
    ],
)
def test_qwen4_exp_decode_rejects_invalid_state_mapping(
    block_table, positions, output_slots, message
):
    with pytest.raises(ValueError, match=message):
        RTPQwen4ExpContext._previous_state_slots(
            torch.tensor(block_table, dtype=torch.int32),
            torch.tensor(positions, dtype=torch.int32),
            torch.tensor(output_slots, dtype=torch.int32),
            64,
        )


def test_qwen4_exp_qsa_cache_binding_reuses_paged_kv_storage(monkeypatch):
    class FakeQSA:
        layer_num = 3
        num_kv_heads = 2
        head_dim = 4
        indexer = SimpleNamespace(index_head_dim=8, compress_ratio=4)

        def bind_caches(self, *caches):
            self.caches = caches

    module = FakeQSA()
    raw = torch.arange(2 * 2 * 2 * 64 * 4, dtype=torch.bfloat16).reshape(2, 2, 2, 64, 4)
    runtime = SimpleNamespace(
        kv_cache=SimpleNamespace(
            get_layer_cache=lambda layer_num: SimpleNamespace(kv_cache_base=raw)
        )
    )
    monkeypatch.setattr(qwen4_exp_context, "Qwen4ExpAttention", FakeQSA)
    from rtp_llm.models_py.modules.factory.attention import common

    monkeypatch.setattr(common, "reshape_paged_kv_cache", lambda *args: raw)

    RTPQwen4ExpContext._bind_qsa_caches(runtime, [module], block_size=64)
    key, value, index_cache, compressed_cache = module.caches
    assert key.data_ptr() == raw[:, 0].data_ptr()
    assert value.data_ptr() == raw[:, 1].data_ptr()
    assert key.shape == value.shape == (2, 64, 2, 4)
    assert key.stride(1) == value.stride(1) == 2 * 4
    assert key.stride(2) == value.stride(2) == 4
    assert index_cache.shape == (2, 64, 1, 8)
    assert compressed_cache.shape == (2, 16, 1, 8)
    assert not index_cache.any()
    assert not compressed_cache.any()

    caches = module.caches
    RTPQwen4ExpContext._bind_qsa_caches(runtime, [module], block_size=64)
    assert module.caches is caches


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm/CUDA")
def test_qwen4_exp_qsa_two_kv_heads_cache_write_and_read():
    import aiter

    from atom.model_ops.qwen4_exp.ops.qsa import qsa_sparse_paged_gqa

    class FakeQSA:
        layer_num = 3
        num_kv_heads = 2
        head_dim = 16
        indexer = SimpleNamespace(index_head_dim=8, compress_ratio=4)

        def bind_caches(self, *caches):
            self.caches = caches

    module = FakeQSA()
    block_size = 4
    # The extra elements model RTP's padded hybrid per-block stride.
    raw = torch.zeros(
        (1, 2 * module.num_kv_heads * block_size * module.head_dim + 16),
        device="cuda",
        dtype=torch.bfloat16,
    )
    runtime = SimpleNamespace(
        kv_cache=SimpleNamespace(
            get_layer_cache=lambda _: SimpleNamespace(kv_cache_base=raw)
        )
    )
    RTPQwen4ExpContext._bind_qsa_caches(runtime, [module], block_size)
    key_cache, value_cache, _, _ = module.caches
    keys = (
        torch.arange(2 * 2 * 16, device="cuda", dtype=torch.float32)
        .reshape(2, 2, 16)
        .to(torch.bfloat16)
        / 16
    )
    values = (keys + 1).contiguous()
    slots = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    aiter.reshape_and_cache_flash(
        keys, values, key_cache, value_cache, slots, "auto", scale, scale
    )
    torch.testing.assert_close(key_cache[0, :2], keys, rtol=0, atol=0)
    torch.testing.assert_close(value_cache[0, :2], values, rtol=0, atol=0)

    query = torch.ones((1, 2, 16), device="cuda", dtype=torch.bfloat16)
    selected = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    token_to_request = torch.tensor([0], device="cuda", dtype=torch.int32)
    actual = qsa_sparse_paged_gqa(
        query,
        key_cache,
        value_cache,
        selected,
        block_table,
        token_to_request,
        kv_splits=1,
    )
    scores = torch.einsum("bhd,thd->bht", query.float(), keys.float()) / 4
    expected = torch.einsum("bht,thd->bhd", scores.softmax(dim=-1), values.float())
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm/CUDA")
def test_qwen4_exp_qsa_reused_page_hides_stale_groups_until_overwrite():
    from atom.model_ops.qwen4_exp.ops.qsa import (
        qsa_compress_groups,
        qsa_paged_mqa_logits,
    )

    device = torch.device("cuda")
    query = torch.ones((1, 2, 16), device=device, dtype=torch.bfloat16)
    stale = torch.full((1, 16, 1, 16), 42, device=device, dtype=query.dtype)
    fresh = torch.zeros_like(stale)
    table = torch.zeros((1, 1), device=device, dtype=torch.int32)
    request = torch.zeros((1,), device=device, dtype=torch.int32)

    # No compressed group is visible until all four tokens have been written.
    for position in range(3):
        pos = torch.tensor([position], device=device, dtype=torch.int32)
        length = pos + 1
        _, visible = qsa_paged_mqa_logits(
            query, stale, table, request, pos, length, compress_ratio=4
        )
        assert visible.item() == 0

    raw = torch.zeros((1, 64, 1, 16), device=device, dtype=query.dtype)
    raw[0, :4] = torch.arange(1, 5, device=device, dtype=query.dtype).view(4, 1, 1)
    pos = torch.tensor([3], device=device, dtype=torch.int32)
    pooled, _ = qsa_compress_groups(
        raw, table, request, pos, torch.tensor([0], device=device), 4
    )
    stale[0, 0] = pooled[0]
    fresh[0, 0] = pooled[0]
    stale_logits, stale_visible = qsa_paged_mqa_logits(
        query, stale, table, request, pos, pos + 1, compress_ratio=4
    )
    fresh_logits, fresh_visible = qsa_paged_mqa_logits(
        query, fresh, table, request, pos, pos + 1, compress_ratio=4
    )
    assert stale_visible.item() == fresh_visible.item() == 1
    torch.testing.assert_close(stale_logits[0, 0], fresh_logits[0, 0], rtol=0, atol=0)


def test_qwen4_exp_qsa_rejects_invalid_ratio_and_dtype():
    class FakeQSA:
        layer_num = 7
        indexer = SimpleNamespace(compress_ratio=8)

    module = FakeQSA()
    raw = torch.empty(2, dtype=torch.bfloat16)
    runtime = SimpleNamespace(
        kv_cache=SimpleNamespace(
            get_layer_cache=lambda _: SimpleNamespace(kv_cache_base=raw)
        )
    )
    with pytest.raises(ValueError, match="compression ratio"):
        RTPQwen4ExpContext._bind_qsa_caches(runtime, [module], block_size=36)
    raw = torch.empty(2, dtype=torch.float16)
    with pytest.raises(ValueError, match="BF16"):
        RTPQwen4ExpContext._bind_qsa_caches(runtime, [module], block_size=64)


def test_qwen4_exp_qsa_uses_physical_table_when_kernel_pages_are_smaller():
    physical = torch.tensor([[7, 8]], dtype=torch.int32)
    kernel = torch.tensor([[28, 29, 30, 31, 32, 33, 34, 35]], dtype=torch.int32)
    full_inputs = SimpleNamespace(
        kv_cache_block_id_device=physical,
        kv_cache_kernel_block_id_device=kernel,
    )
    table = RTPQwen4ExpContext._resolve_qsa_block_table(full_inputs, 64, 16, None)
    assert table.data_ptr() == physical.data_ptr()
    slots = RTPQwen4ExpContext._build_slot_mapping(
        positions=torch.tensor([64], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        block_table=table,
        seq_size_per_block=64,
    )
    assert slots.tolist() == [8 * 64]

    full_inputs.kv_cache_block_id_device = None
    recovered = RTPQwen4ExpContext._resolve_qsa_block_table(full_inputs, 64, 16, None)
    assert recovered.tolist() == physical.tolist()


def test_qwen4_exp_ple_uses_actual_gdn_cache_layer():
    raw = torch.empty(3, 1, dtype=torch.bfloat16)
    seen = []

    def get_layer_cache(layer):
        seen.append(layer)
        assert layer == 5
        return SimpleNamespace(kv_cache_base=raw)

    runtime = SimpleNamespace(kv_cache=SimpleNamespace(get_layer_cache=get_layer_cache))
    ple = SimpleNamespace(
        hc_hidden_size=4,
        conv_state_len=3,
        ple_embedding=SimpleNamespace(ngram_size=2),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(ple=ple)]),
        config=SimpleNamespace(eos_token_id=2),
        atom_config=SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    ids = torch.tensor([1], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        gdn_metadata=SimpleNamespace(
            non_spec_state_indices_tensor=ids,
            non_spec_state_indices_in_tensor=ids,
            has_initial_state=torch.tensor([True]),
        ),
        plugin_metadata=SimpleNamespace(rtp_cu_seqlens_q=torch.tensor([0, 1])),
    )
    md = RTPQwen4ExpContext._ple_metadata(runtime, model, attn_metadata, {5: object()})
    assert seen == [5]
    assert md.ngram_state.shape == (3, 1)
    assert md.conv_state.shape == (3, 4, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_qwen4_exp_recovered_physical_table_respects_prewarmed_buffer_stride(
    monkeypatch,
):
    device = torch.device("cuda")
    kernel = torch.tensor(
        [[28, 29, 30, 31, 32, 33, 34, 35], [40, 41, 42, 43, 44, 45, 46, 47]],
        dtype=torch.int32,
        device=device,
    )
    # The graph buffer is larger than the active table, so its row stride
    # differs from the returned view's logical column count.
    buffer = torch.full((2, 5), -99, dtype=torch.int32, device=device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    table = RTPQwen4ExpContext._recover_physical_block_table_from_kernel(
        kernel,
        seq_size_per_block=64,
        kernel_seq_size_per_block=16,
        cg_bufs={"physical_block_table_i32": buffer},
    )
    assert table.cpu().tolist() == [[7, 8], [10, 11]]
    assert buffer[:, 2:].cpu().tolist() == [[-99] * 3, [-99] * 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_qwen4_exp_graph_state_and_qsa_slots_handle_boundary_and_padding():
    device = torch.device("cuda")
    table = torch.tensor(
        [[5, 6], [10, 11], [20, 21], [30, 31]],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.tensor([0, 63, 64, -1], dtype=torch.int32, device=device)
    outputs = torch.tensor([5, 10, 21, -1], dtype=torch.int32, device=device)
    state_inputs = torch.empty_like(outputs)
    RTPQwen4ExpContext._previous_state_slots(
        table, positions, outputs, 64, graph_output=state_inputs
    )
    slots = torch.empty(4, dtype=torch.int64, device=device)
    compressed = torch.empty_like(slots)
    _qsa_graph_slots_kernel[(triton.cdiv(4, 128),)](
        table, positions, slots, compressed, 4, *table.stride(), 2, 64, 4, 128
    )
    assert state_inputs.cpu().tolist() == [5, 10, 20, -1]
    assert slots.cpu().tolist() == [320, 703, 1344, -1]
    assert compressed.cpu().tolist() == [-1, 175, -1, -1]
