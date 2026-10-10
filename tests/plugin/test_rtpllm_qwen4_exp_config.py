"""Qwen4Exp RTP plugin registration and hybrid-cache configuration."""

import json
from types import SimpleNamespace

import pytest
import torch
import triton
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.ops import DataType, HybridAttentionType

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
    assert config.compute_dtype == torch.bfloat16
    assert config.config_dtype == "bf16"
    config.init_precision_config(None, None)
    assert config.compute_dtype == torch.bfloat16


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


def test_qwen4_exp_decode_state_reads_previous_block_at_boundary():
    block_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    positions = torch.tensor([64, 69], dtype=torch.int32)
    output_slots = torch.tensor([11, 21], dtype=torch.int32)

    input_slots = RTPQwen4ExpContext._previous_state_slots(
        block_table, positions, output_slots, 64
    )

    assert input_slots.tolist() == [10, 21]
    assert output_slots.tolist() == [11, 21]


@pytest.mark.parametrize(
    ("block_table", "positions", "output_slots", "message"),
    [
        ([[10]], [129], [11], "out of range"),
        ([[10, -1]], [65], [11], "invalid"),
        ([[10, 11]], [64, 65], [11], "one token per request"),
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
    model = SimpleNamespace(modules=lambda: [module])
    monkeypatch.setattr(qwen4_exp_context, "Qwen4ExpAttention", FakeQSA)
    from rtp_llm.models_py.modules.factory.attention import common

    monkeypatch.setattr(common, "reshape_paged_kv_cache", lambda *args: raw)

    RTPQwen4ExpContext._bind_qsa_caches(runtime, model, block_size=64)
    key, value, index_cache, compressed_cache = module.caches
    assert key.data_ptr() == raw[:, 0].data_ptr()
    assert value.data_ptr() == raw[:, 1].data_ptr()
    assert key.shape == value.shape == (2, 64, 2, 4)
    assert index_cache.shape == (2, 64, 1, 8)
    assert compressed_cache.shape == (2, 16, 1, 8)

    caches = module.caches
    RTPQwen4ExpContext._bind_qsa_caches(runtime, model, block_size=64)
    assert module.caches is caches


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
