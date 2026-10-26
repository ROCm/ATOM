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

from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpPLEMetadata
from atom.models.qwen4_exp import qwen4_exp_packed_modules_mapping
from atom.plugin.rtpllm.models import qwen4_exp_mtp_hooks
from atom.plugin.rtpllm.models.qwen3_5 import _ATOMQwen35MoeRuntime
from atom.plugin.rtpllm.models.qwen4_exp import (
    ATOMQwen4Exp,
    ATOMQwen4ExpMTP,
    _ATOMQwen4ExpMTPRuntime,
    _ATOMQwen4ExpRuntime,
    _Qwen4ExpAttnPyObj,
)
from atom.plugin.rtpllm.models.qwen4_exp_mtp_hooks import (
    _rtp_apply_ple,
    _rtp_spec_conv1d_update,
)
from atom.plugin.rtpllm.utils import qwen4_exp_context
from atom.plugin.rtpllm.utils.forward_context import (
    RTPForwardContext,
    _target_verify_state_slots_graph_kernel,
)
from atom.plugin.rtpllm.utils.qwen4_exp_context import (
    RTPQwen4ExpContext,
    _qsa_graph_slots_kernel,
)


@pytest.mark.parametrize("prefill_enabled", [True, False])
@pytest.mark.parametrize("vk_layout", [True, False])
@pytest.mark.parametrize("spec_decodes", [0, 1])
def test_qwen4_exp_rtp_gdn_policy_requires_vk_decode_state(
    monkeypatch, prefill_enabled, vk_layout, spec_decodes
):
    from atom.model_ops.fla_ops.gdn_flydsl import GDNFlyDSLPolicy

    layer = SimpleNamespace(
        allow_aiter_flydsl=True,
        dt_bias=torch.empty((), dtype=torch.bfloat16),
    )
    metadata = SimpleNamespace(
        num_prefills=2,
        num_prefill_tokens=5,
        num_spec_decodes=spec_decodes,
        non_spec_query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        flydsl_prefill_metadata=None,
    )
    state = torch.empty((4, 2, 128, 128), dtype=torch.float32)
    if vk_layout:
        state = state.transpose(-1, -2)
    calls = []

    def select(**kwargs):
        calls.append(kwargs)
        return GDNFlyDSLPolicy(prefill=prefill_enabled, decode=True)

    monkeypatch.setattr(qwen4_exp_context, "select_policy", select)
    monkeypatch.setattr(
        qwen4_exp_context,
        "build_prefill_metadata",
        lambda lengths, cu: (tuple(lengths), cu),
    )
    RTPQwen4ExpContext._configure_gdn_backends(
        metadata,
        {3: layer},
        {"layer_3": SimpleNamespace(v_cache=state)},
        SimpleNamespace(input_lengths=torch.tensor([2, 3], dtype=torch.int32)),
    )
    assert calls[0]["state"] is state
    assert calls[0]["lossy_decode"] is False
    assert layer.gdn_flydsl_policy.prefill is prefill_enabled
    assert layer.gdn_flydsl_policy.decode is (vk_layout and spec_decodes == 0)
    assert (metadata.flydsl_prefill_metadata is not None) is prefill_enabled


def test_qwen4_exp_rtp_gdn_decode_does_not_build_prefill_schedule(monkeypatch):
    from atom.model_ops.fla_ops.gdn_flydsl import GDNFlyDSLPolicy

    layer = SimpleNamespace(
        allow_aiter_flydsl=True,
        dt_bias=torch.empty((), dtype=torch.bfloat16),
    )
    metadata = SimpleNamespace(num_prefills=0, flydsl_prefill_metadata=None)
    monkeypatch.setattr(
        qwen4_exp_context,
        "select_policy",
        lambda **kwargs: GDNFlyDSLPolicy(prefill=True, decode=True),
    )
    monkeypatch.setattr(
        qwen4_exp_context,
        "build_prefill_metadata",
        lambda *args: pytest.fail("decode must not prepare prefill metadata"),
    )
    RTPQwen4ExpContext._configure_gdn_backends(
        metadata,
        {3: layer},
        {"layer_3": SimpleNamespace(v_cache=torch.empty((4, 2, 128, 128)))},
        SimpleNamespace(input_lengths=torch.tensor([1], dtype=torch.int32)),
    )
    assert layer.gdn_flydsl_policy.decode is False
    assert metadata.flydsl_prefill_metadata is None


def test_qwen4_exp_rtp_gdn_decode_supports_non_128_head_dimensions(monkeypatch):
    from atom.model_ops.fla_ops.gdn_flydsl import GDNFlyDSLPolicy

    layer = SimpleNamespace(
        allow_aiter_flydsl=True,
        dt_bias=torch.empty((), dtype=torch.bfloat16),
    )
    metadata = SimpleNamespace(num_prefills=0, num_spec_decodes=0)
    monkeypatch.setattr(
        qwen4_exp_context,
        "select_policy",
        lambda **kwargs: GDNFlyDSLPolicy(prefill=False, decode=True),
    )
    state = torch.empty((4, 2, 64, 128)).transpose(-1, -2)
    RTPQwen4ExpContext._configure_gdn_backends(
        metadata,
        {3: layer},
        {"layer_3": SimpleNamespace(v_cache=state)},
        SimpleNamespace(input_lengths=torch.tensor([1], dtype=torch.int32)),
    )
    assert layer.gdn_flydsl_policy.decode is True


def test_qwen4_exp_mtp_graph_rejects_verify_batch_above_buffer_capacity(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    buffers = {
        "query_start_loc": torch.empty(2, dtype=torch.int32),
        "target_read_slots_i32": torch.empty((1, 3), dtype=torch.int32),
        "target_write_slots_i32": torch.empty((1, 3), dtype=torch.int32),
        "target_invalid_live_rows": torch.empty(1, dtype=torch.int32),
    }
    inputs = SimpleNamespace(
        is_target_verify=True,
        input_lengths=torch.tensor([3, 3], dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.empty((2, 4), dtype=torch.int32),
        sequence_lengths_plus_1_device=torch.tensor([8, 8], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="buffers are too small"):
        RTPQwen4ExpContext._build_gdn_metadata(
            inputs, seq_size_per_block=16, num_tokens=6, cg_bufs=buffers
        )


@pytest.mark.parametrize(
    ("decode_enabled", "mtp_target", "mtp_draft"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_qwen4_exp_rtp_gdn_fp32_state_uses_full_byte_region(
    monkeypatch, decode_enabled, mtp_target, mtp_draft
):
    from atom.model_ops.fla_ops.gdn_flydsl import GDNFlyDSLPolicy

    monkeypatch.setattr(
        qwen4_exp_context,
        "select_policy",
        lambda **kwargs: GDNFlyDSLPolicy(decode=decode_enabled),
    )
    heads, k_dim, v_dim, conv_dim, conv_length = 2, 4, 4, 6, 2
    state_elems = heads * k_dim * v_dim
    raw = torch.zeros(
        (3, 2 * state_elems + conv_dim * conv_length), dtype=torch.bfloat16
    )
    layer = SimpleNamespace(
        allow_aiter_flydsl=True,
        dt_bias=torch.empty(heads),
        head_k_dim=k_dim,
        head_v_dim=v_dim,
        conv1d=SimpleNamespace(weight=torch.empty(conv_dim, 1, conv_length + 1)),
    )
    runtime = SimpleNamespace(
        _rtp_model_config=SimpleNamespace(
            linear_attention_config=SimpleNamespace(ssm_state_dtype=DataType.TYPE_FP32),
            atom_qwen4_exp_mtp_target=mtp_target,
            is_mtp=mtp_draft,
        ),
        kv_cache=SimpleNamespace(
            get_layer_cache=lambda _: SimpleNamespace(kv_cache_base=raw)
        ),
    )
    cache = RTPQwen4ExpContext._build_kv_cache_tensors(runtime, ({3: layer}, {}, {}))[
        "layer_3"
    ]
    assert cache.v_cache.dtype == torch.float32
    assert cache.k_cache.dtype == torch.bfloat16
    vk_layout = decode_enabled and not (mtp_target or mtp_draft)
    assert cache.v_cache.stride()[1:] == (
        (k_dim * v_dim, 1, k_dim) if vk_layout else (k_dim * v_dim, v_dim, 1)
    )
    cache.v_cache[1, 0, 1, 2] = 1.25
    cache.k_cache[1, 0, 0] = 2
    state_offset = 2 * k_dim + 1 if vk_layout else v_dim + 2
    assert raw.view(torch.float32)[1, state_offset].item() == 1.25
    assert raw[1, 2 * state_elems].item() == 2
    assert cache.v_cache[0].abs().sum().item() == 0


def test_qwen4_exp_graph_uses_device_block_tables_only():
    assert _Qwen4ExpAttnPyObj.requires_host_block_table_mirror is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("draft_steps", [1, 2, 3])
@pytest.mark.parametrize("second_has_initial", [False, True])
def test_qwen4_exp_ple_candidate_snapshots_match_prefix_reruns(
    monkeypatch, draft_steps, second_has_initial
):
    from dataclasses import replace

    from atom.model_ops.qwen4_exp.ops.ple import (
        advance_ngram_state,
        dilated_causal_conv1d,
    )

    device = torch.device("cuda")
    width = draft_steps + 1
    write_slots = torch.tensor(
        [[9] + list(range(10, 9 + width)), [19] + list(range(20, 19 + width))],
        device=device,
        dtype=torch.int32,
    )
    write_slots[:, 0] = torch.tensor([9, 19], device=device, dtype=torch.int32)
    ngram_initial = torch.arange(32 * 3, device=device, dtype=torch.int64).reshape(
        32, 3
    )
    conv_initial = torch.arange(32 * 4, device=device, dtype=torch.float32).reshape(
        32, 4, 1
    )

    def make_metadata():
        result = Qwen4ExpPLEMetadata(
            state_indices_out=write_slots[:, -1],
            query_start_loc=torch.arange(
                0, 2 * width + 1, width, device=device, dtype=torch.int32
            ),
            ngram_state=ngram_initial.clone(),
            state_indices_in=torch.tensor([9, 19], device=device, dtype=torch.int32),
            has_initial_state=torch.tensor(
                [True, second_has_initial], device=device, dtype=torch.bool
            ),
            conv_state=conv_initial.clone(),
        )
        result.rtp_spec_state_indices = write_slots
        return result

    class PLE:
        ple_embedding = SimpleNamespace(ngram_size=4, eos_token_id=0)
        norm_conv = None
        conv_state_len = 1

        def gated_memory(self, hidden, ids, starts, context):
            return hidden.unsqueeze(1)

        def _apply_norm(self, norm, value):
            return value

        def _short_conv(self, normalized, starts, state, src, dst, has, accepted):
            return dilated_causal_conv1d(
                normalized,
                torch.ones((4, 2), device=device),
                state,
                starts,
                src,
                dst,
                has,
                1,
                accepted,
            )

        def forward_with_state(self, hidden, ids, meta):
            advance_ngram_state(
                ids,
                meta.query_start_loc,
                meta.ngram_state,
                meta.state_indices_in,
                meta.state_indices_out,
                meta.has_initial_state,
                0,
                history_width=3,
            )
            return hidden + self._short_conv(
                hidden,
                meta.query_start_loc,
                meta.conv_state,
                meta.state_indices_in,
                meta.state_indices_out,
                meta.has_initial_state,
                None,
            )

    layer = SimpleNamespace(ple=PLE())
    ids = torch.arange(2 * width, device=device, dtype=torch.int64) + 101
    hidden = torch.arange(2 * width * 4, device=device, dtype=torch.float32).reshape(
        2 * width, 4
    )
    reference = make_metadata()
    expected = hidden + layer.ple.forward_with_state(hidden, ids, reference)
    for prefix_len in range(width - 1, 0, -1):
        prefix_metadata = replace(
            reference,
            query_start_loc=torch.arange(
                0, (2 + 1) * prefix_len, prefix_len, device=device, dtype=torch.int32
            ),
            state_indices_out=write_slots[:, prefix_len - 1],
        )
        layer.ple.forward_with_state(
            hidden.reshape(2, width, 4)[:, :prefix_len]
            .reshape(2 * prefix_len, 4)
            .contiguous(),
            ids.reshape(2, width)[:, :prefix_len].reshape(-1).contiguous(),
            prefix_metadata,
        )

    actual = make_metadata()
    monkeypatch.setattr(
        qwen4_exp_mtp_hooks,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=SimpleNamespace(ple_metadata=actual)),
    )
    output = _rtp_apply_ple(layer, hidden, ids)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.ngram_state, reference.ngram_state, rtol=0, atol=0
    )
    torch.testing.assert_close(actual.conv_state, reference.conv_state, rtol=0, atol=0)


def test_qwen4_exp_mtp_spec_conv_uses_rtp_block_map(monkeypatch):
    from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d

    mixed = torch.arange(2 * 3 * 8).reshape(6, 8)
    block_map = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    lengths = torch.tensor([4, 5], dtype=torch.int32)
    metadata = SimpleNamespace(
        rtp_target_verify_block_map=block_map,
        rtp_target_verify_sequence_lengths=lengths,
        rtp_seq_size_per_block=64,
        num_spec_decodes=2,
        spec_state_indices_tensor=torch.empty((2, 3), dtype=torch.int32),
    )
    gdn = SimpleNamespace(
        num_k_heads=1,
        head_k_dim=2,
        num_v_heads=1,
        head_v_dim=4,
        tp_size=1,
        conv1d=SimpleNamespace(bias=None),
        activation="silu",
    )
    seen = {}

    def fake_update(inputs, states, weights, **kwargs):
        seen.update(inputs=inputs, states=states, weights=weights, **kwargs)
        return inputs

    monkeypatch.setattr(causal_conv1d, "causal_conv1d_update", fake_update)
    states = torch.empty(1)
    weights = torch.empty(1)
    q, k, v = _rtp_spec_conv1d_update(
        gdn, mixed, states, weights, metadata, None, False
    )

    assert seen["inputs"].shape == (2, 8, 3)
    assert seen["inputs"].stride(1) == 1
    assert not seen["inputs"].is_contiguous()
    assert seen["block_map"] is block_map
    assert seen["sequence_lengths"] is lengths
    assert seen["seq_size_per_block"] == 64
    assert seen["states"] is states and seen["weights"] is weights
    assert torch.equal(torch.cat((q, k, v), dim=-1), mixed)


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


def test_prepared_qwen4_exp_mtp_receives_atom_config(monkeypatch):
    from atom.models.qwen4_exp_mtp import Qwen4ExpMTP
    from atom.plugin.prepare import _instantiate_prepared_model

    atom_config = SimpleNamespace(hf_config=SimpleNamespace())

    def lightweight_init(self, atom_config):
        torch.nn.Module.__init__(self)
        self.config = atom_config.hf_config

    monkeypatch.setattr(Qwen4ExpMTP, "__init__", lightweight_init)
    draft = _instantiate_prepared_model(None, atom_config, Qwen4ExpMTP)

    assert draft.atom_config is atom_config


def test_qwen4_exp_mtp_fp8_exclusion_keeps_draft_layer_name():
    from atom.config import LayerQuantConfig, QuantizationConfig
    from atom.models.qwen4_exp import _Qwen4ExpQuantizationConfig
    from atom.models.qwen4_exp_mtp import Qwen4ExpMTP

    quant = QuantizationConfig()
    quant.quant_method = "compressed-tensors"
    quant.global_spec = LayerQuantConfig(quant_dtype=torch.float8_e4m3fn)
    quant.exclude_layers = ["mtp.layers.0.self_attn.q_proj", "mtp.fc_embedding"]
    # RTP's draft prepare path must not map mtp.* exclusions to model.*.
    quant.remap_layer_name(
        SimpleNamespace(model_type="qwen4_exp"),
        packed_modules_mapping=Qwen4ExpMTP.packed_modules_mapping,
    )
    draft_quant = _Qwen4ExpQuantizationConfig(quant)
    assert (
        draft_quant.get_layer_quant_config(
            "mtp.layers.0.self_attn.qkv_proj"
        ).quant_dtype
        is torch.bfloat16
    )
    assert (
        draft_quant.get_layer_quant_config("mtp.fc_embedding").quant_dtype
        is torch.bfloat16
    )
    assert (
        draft_quant.get_layer_quant_config("mtp.layers.0.self_attn.o_proj").quant_dtype
        is torch.float8_e4m3fn
    )


@pytest.mark.parametrize("fixed_inputs", [False, True])
@pytest.mark.parametrize("multimodal", [False, True])
def test_qwen4_exp_mtp_draft_marks_forward_context(
    monkeypatch, fixed_inputs, multimodal
):
    from contextlib import contextmanager

    import atom.plugin.rtpllm.models.qwen4_exp as qwen4_module
    from atom.utils import forward_context as forward_context_module

    runtime = object.__new__(_ATOMQwen4ExpMTPRuntime)
    runtime._rtp_model_config = SimpleNamespace(
        hidden_size=2,
        hc_mult=1,
        vocab_size=100,
        mm_model_config=SimpleNamespace(is_multimodal=multimodal),
    )
    runtime._image_token_id = 42
    runtime._rtp_layer_maps = ({}, {}, {})
    runtime._cg_max_seq_len = 32
    runtime._get_model_device = lambda: torch.device("cpu")
    runtime._extract_positions = lambda *args: torch.tensor([3], dtype=torch.int32)
    runtime._graph_buffers_for_inputs = lambda *_: None
    context = SimpleNamespace(is_draft=False)
    monkeypatch.setattr(
        forward_context_module,
        "get_forward_context",
        lambda: SimpleNamespace(context=context),
    )

    @contextmanager
    def bind(**kwargs):
        assert kwargs["inputs"].attention_inputs_by_tag == {
            "full": kwargs["inputs"].attention_inputs
        }
        yield

    runtime._rtp_forward_context_cls = SimpleNamespace(bind=bind)

    class Draft:
        atom_config = SimpleNamespace()
        model = SimpleNamespace(embed_tokens=SimpleNamespace(tp_size=2))

        def __call__(self, ids, positions, hc_input, inputs_embeds=None):
            assert context.is_draft
            assert inputs_embeds is None
            assert ids.tolist() == ([42] if multimodal else [1])
            assert positions.tolist() == ([[9], [10], [11]] if multimodal else [4])
            return hc_input

        def _head_input(self, hc):
            return hc

    runtime.model = Draft()
    monkeypatch.setattr(qwen4_module, "PyModelOutputs", lambda mixed, hc: (mixed, hc))
    fields = {
        "input_ids": torch.tensor([999 if multimodal else 1], dtype=torch.int32),
        "input_hiddens": torch.ones(1, 2),
        "attention_inputs": SimpleNamespace(is_cuda_graph=False),
        "combo_position_ids": (
            torch.tensor([[8], [9], [10]], dtype=torch.int32) if multimodal else None
        ),
        "embedding_inputs": None,
        "multimodal_inputs": None,
        "bert_embedding_inputs": None,
    }
    if fixed_inputs:

        class FixedInputs:
            __slots__ = tuple(fields)

        inputs = FixedInputs()
        for name, value in fields.items():
            setattr(inputs, name, value)
    else:
        inputs = SimpleNamespace(**fields)
    runtime.forward(inputs)
    assert context.is_draft


def test_qwen4_exp_target_forward_restores_its_atom_config(monkeypatch):
    from atom import config as atom_config_module
    from atom.config import get_current_atom_config, set_current_atom_config

    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    target_config = SimpleNamespace(name="target")
    draft_config = SimpleNamespace(name="draft")
    runtime.model = SimpleNamespace(atom_config=target_config)
    runtime._rtp_model_config = SimpleNamespace(atom_qwen4_exp_mtp_target=False)
    expected = object()

    def parent_forward(_self, _inputs, _fmha_impl):
        assert get_current_atom_config() is target_config
        return expected

    monkeypatch.setattr(_ATOMQwen35MoeRuntime, "forward", parent_forward)
    previous = atom_config_module._current_atom_config
    set_current_atom_config(draft_config)
    try:
        result = _ATOMQwen4ExpRuntime.forward(
            runtime, SimpleNamespace(attention_inputs=None), None
        )
        assert result is expected
        assert get_current_atom_config() is draft_config
    finally:
        set_current_atom_config(previous)


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

    text_config["mtp_num_hidden_layers"] = 1
    text_config["mtp"] = {"layer_types": ["full_attention"]}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text_config}))
    draft = ATOMQwen4ExpMTP._create_config(str(tmp_path))
    assert draft.is_mtp and draft.num_layers == 1
    draft.init_precision_config(None, None)
    draft.attn_config.tokens_per_block = 64
    ATOMQwen4ExpMTP._apply_kv_cache_config(
        draft, SimpleNamespace(seq_size_per_block=64, reuse_cache=True)
    )
    assert draft.auxiliary_paged_cache_bytes_per_block == qsa_bytes

    text_config["rope_parameters"].update(
        {"mrope_section": [11, 11, 10], "mrope_interleaved": True}
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": text_config,
                "language_model_only": False,
                "vision_start_token_id": 248053,
                "vision_end_token_id": 248054,
            }
        )
    )
    vision = ATOMQwen4Exp._create_config(str(tmp_path))
    assert vision.mm_model_config.is_multimodal
    assert vision.attn_config.rope_config.index_factor == 3
    assert vision.mm_model_config.mm_position_ids_style == 2


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


@pytest.mark.parametrize("is_draft", [False, True])
def test_qwen4_exp_mtp_graph_keeps_capture_cap_separate_from_request_capacity(
    monkeypatch, is_draft
):
    def init_with_narrow_decode_bucket(
        self,
        model_config,
        parallelism_config=None,
        weights=None,
        max_generate_batch_size=None,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        self.model = torch.nn.Module()
        self._cg_max_num_tokens = 4

    monkeypatch.setattr(
        _ATOMQwen35MoeRuntime, "__init__", init_with_narrow_decode_bucket
    )
    runtime = _ATOMQwen4ExpRuntime(
        SimpleNamespace(
            atom_qwen4_exp_mtp_target=not is_draft,
            is_mtp=is_draft,
            mm_model_config=SimpleNamespace(is_multimodal=False),
        ),
        max_generate_batch_size=8,
    )
    assert runtime._cg_max_num_tokens == 4
    assert runtime._rtp_mtp_request_capacity == 8


def test_qwen4_exp_mtp_graph_capacity_uses_bound_positional_argument(monkeypatch):
    def init_with_narrow_decode_bucket(
        self, model_config, parallelism, weights, max_generate_batch_size, **kwargs
    ):
        torch.nn.Module.__init__(self)
        self.model = torch.nn.Module()
        self.max_generate_batch_size = max_generate_batch_size
        self._cg_max_num_tokens = 4

    monkeypatch.setattr(
        _ATOMQwen35MoeRuntime, "__init__", init_with_narrow_decode_bucket
    )
    runtime = _ATOMQwen4ExpRuntime(
        SimpleNamespace(
            is_mtp=True, mm_model_config=SimpleNamespace(is_multimodal=False)
        ),
        None,
        None,
        8,
    )
    assert runtime._cg_max_num_tokens == 4
    assert runtime._rtp_mtp_request_capacity == 8


def test_qwen4_exp_mtp_graph_rejects_undersized_position_buffer(monkeypatch):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    runtime._rtp_model_config = SimpleNamespace(gen_num_per_cycle=2)
    runtime._cg_mtp_meta_bufs = {
        "qwen4_positions_i32": torch.empty(12, dtype=torch.int32)
    }
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            is_cuda_graph=True,
            sequence_lengths_plus_1_device=torch.ones(8, dtype=torch.int32),
        )
    )
    with pytest.raises(ValueError, match="capacity=12, tokens=24"):
        runtime._extract_positions(inputs, torch.device("cpu"), 24)


@pytest.mark.parametrize("draft_steps", [1, 2, 3])
def test_qwen4_exp_mtp_graph_builds_positions_per_request(monkeypatch, draft_steps):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    width = draft_steps + 1
    runtime._rtp_model_config = SimpleNamespace(gen_num_per_cycle=draft_steps)
    runtime._cg_mtp_meta_bufs = {
        "qwen4_positions_i32": torch.empty(2 * width, dtype=torch.int32)
    }
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            is_cuda_graph=True,
            sequence_lengths_plus_1_device=torch.tensor([17, 130], dtype=torch.int32),
        )
    )
    assert runtime._extract_positions(
        inputs, torch.device("cpu"), 2 * width
    ).tolist() == list(range(16, 16 + width)) + list(range(129, 129 + width))


@pytest.mark.parametrize("draft_steps", [1, 2, 3])
def test_qwen4_exp_mtp_draft_graph_uses_prefix_lengths(monkeypatch, draft_steps):
    runtime = object.__new__(_ATOMQwen4ExpMTPRuntime)
    width = draft_steps + 1
    runtime._rtp_model_config = SimpleNamespace(
        is_mtp=True, gen_num_per_cycle=draft_steps
    )
    runtime._cg_mtp_meta_bufs = {
        "qwen4_positions_i32": torch.empty(2 * width, dtype=torch.int32)
    }
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=True,
            prefix_lengths_device=torch.tensor([16, 129], dtype=torch.int32),
        )
    )
    assert runtime._extract_positions(
        inputs, torch.device("cpu"), 2 * width
    ).tolist() == list(range(16, 16 + width)) + list(range(129, 129 + width))


@pytest.mark.parametrize("draft_steps", [1, 2, 3])
def test_qwen4_exp_mtp_verify_keeps_per_candidate_state(draft_steps):
    width = draft_steps + 1
    block_table = torch.tensor(
        [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int32
    )
    linear = SimpleNamespace(
        is_prefill=True,
        is_target_verify=True,
        input_lengths=torch.tensor([width, width], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, width, 2 * width], dtype=torch.int32),
        kv_cache_kernel_block_id_device=block_table,
        sequence_lengths_plus_1_device=torch.tensor([2, 17], dtype=torch.int32),
    )
    metadata = RTPQwen4ExpContext._build_gdn_metadata(
        linear, seq_size_per_block=16, num_tokens=2 * width
    )
    assert metadata.num_spec_decodes == 2
    assert metadata.spec_state_indices_in_tensor.tolist() == [
        [10] * width,
        [20] * width,
    ]
    assert metadata.spec_state_indices_tensor.tolist() == [
        list(range(10, 10 + width)),
        list(range(21, 21 + width)),
    ]
    assert (
        metadata.spec_state_indices_in_tensor.stride()
        == metadata.spec_state_indices_tensor.stride()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("input_length", [3, 2])
def test_qwen4_exp_mtp_verify_device_input_lengths(input_length):
    inputs = SimpleNamespace(
        is_target_verify=True,
        input_lengths=torch.tensor([input_length], device="cuda", dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.tensor(
            [[5, 6, 7, 8]], device="cuda", dtype=torch.int32
        ),
        sequence_lengths_plus_1_device=torch.tensor(
            [17], device="cuda", dtype=torch.int32
        ),
    )
    if input_length == 3:
        RTPQwen4ExpContext._build_gdn_metadata(
            inputs, seq_size_per_block=16, num_tokens=3
        )
    else:
        with pytest.raises(ValueError, match="state block is invalid"):
            RTPQwen4ExpContext._build_gdn_metadata(
                inputs, seq_size_per_block=16, num_tokens=3
            )


def test_non_qwen4_exp_context_rejects_mtp_target_verify():
    inputs = SimpleNamespace(
        is_target_verify=True,
        kv_cache_kernel_block_id_device=torch.tensor([[1]], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="requires RTPQwen4ExpContext"):
        RTPForwardContext._build_gdn_metadata(
            inputs, seq_size_per_block=16, num_tokens=2
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("draft_steps", [1, 2, 3])
def test_qwen4_exp_mtp_graph_target_slots(draft_steps):
    width = draft_steps + 1
    device = torch.device("cuda")
    table = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.tensor([17, 32, 2, 1, 97], dtype=torch.int32, device=device)
    reads = torch.empty((5, width), dtype=torch.int32, device=device)
    writes = torch.empty_like(reads)
    invalid_live_rows = torch.zeros((), dtype=torch.int32, device=device)
    _target_verify_state_slots_graph_kernel[(1,)](
        table,
        lengths,
        torch.tensor([16, 31, 0, 0, 0], dtype=torch.int32, device=device),
        reads,
        writes,
        invalid_live_rows,
        5,
        *table.stride(),
        5,
        16,
        width,
        128,
    )
    assert reads.cpu().tolist() == [
        [10] * width,
        [21] * width,
        [0] * width,
        [0] * width,
        [0] * width,
    ]
    assert writes.cpu().tolist() == [
        [11 + i for i in range(width)],
        [21 + i for i in range(width)],
        [0] * width,
        [0] * width,
        [0] * width,
    ]
    assert invalid_live_rows.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_qwen4_exp_mtp_graph_counts_invalid_real_rows_not_padding():
    device = torch.device("cuda")
    table = torch.tensor([[5, 6], [0, 0]], device=device, dtype=torch.int32)
    lengths = torch.tensor([32, 32], device=device, dtype=torch.int32)
    reads = torch.empty((2, 3), device=device, dtype=torch.int32)
    writes = torch.empty_like(reads)
    counter = torch.zeros((), device=device, dtype=torch.int32)
    _target_verify_state_slots_graph_kernel[(1,)](
        table,
        lengths,
        torch.tensor([31, 0], dtype=torch.int32, device=device),
        reads,
        writes,
        counter,
        2,
        *table.stride(),
        2,
        16,
        3,
        128,
    )
    assert counter.item() == 1
    assert writes.cpu().tolist() == [[6, 0, 0], [0, 0, 0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_qwen4_exp_mtp_graph_checks_long_live_row_after_first_page_is_freed():
    device = torch.device("cuda")
    # RTP releases old linear-cache blocks, including column zero. The first
    # row is still live; the second row is Graph padding.
    table = torch.tensor(
        [[-1, -1, 7, -1], [0, 0, 0, 0]], device=device, dtype=torch.int32
    )
    lengths = torch.tensor([49, 3], device=device, dtype=torch.int32)
    prefix_lengths = torch.tensor([48, 0], device=device, dtype=torch.int32)
    reads = torch.empty((2, 3), device=device, dtype=torch.int32)
    writes = torch.empty_like(reads)
    invalid_live_rows = torch.empty(1, device=device, dtype=torch.int32)
    _target_verify_state_slots_graph_kernel[(1,)](
        table,
        lengths,
        prefix_lengths,
        reads,
        writes,
        invalid_live_rows,
        2,
        *table.stride(),
        4,
        16,
        3,
        128,
    )
    assert invalid_live_rows.item() == 1
    assert reads.cpu().tolist() == [[7, 7, 7], [0, 0, 0]]
    assert writes.cpu().tolist() == [[0, 0, 0], [0, 0, 0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_qwen4_exp_mtp_graph_replay_refreshes_live_prefix_lengths():
    device = torch.device("cuda")
    table = torch.tensor(
        [[5, 6, 7, 8], [-1, -1, 9, -1]], device=device, dtype=torch.int32
    )
    lengths = torch.tensor([17, 49], device=device, dtype=torch.int32)
    prefix_lengths = torch.tensor([16, 0], device=device, dtype=torch.int32)
    reads = torch.empty((2, 3), device=device, dtype=torch.int32)
    writes = torch.empty_like(reads)
    invalid_live_rows = torch.empty(1, device=device, dtype=torch.int32)

    def launch():
        _target_verify_state_slots_graph_kernel[(1,)](
            table,
            lengths,
            prefix_lengths,
            reads,
            writes,
            invalid_live_rows,
            2,
            *table.stride(),
            4,
            16,
            3,
            128,
        )

    launch()  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    assert invalid_live_rows.item() == 0

    prefix_lengths[1] = 48
    graph.replay()
    torch.cuda.synchronize()
    assert invalid_live_rows.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_qwen4_exp_mtp_graph_overwrites_each_tile_error_flag():
    device = torch.device("cuda")
    table = torch.tensor([[5, 6, 7, 8]] * 129, device=device, dtype=torch.int32)
    table[128, 1] = -1
    lengths = torch.full((129,), 17, device=device, dtype=torch.int32)
    reads = torch.empty((129, 3), device=device, dtype=torch.int32)
    writes = torch.empty_like(reads)
    flags = torch.full((2,), 9, device=device, dtype=torch.int32)

    def launch():
        _target_verify_state_slots_graph_kernel[(2,)](
            table,
            lengths,
            torch.full((129,), 16, dtype=torch.int32, device=device),
            reads,
            writes,
            flags,
            129,
            *table.stride(),
            4,
            16,
            3,
            128,
        )

    launch()
    assert flags.cpu().tolist() == [0, 1]
    table[128, 1] = 8
    launch()
    assert flags.cpu().tolist() == [0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("table_width, expected_invalid", [(4, False), (2, True)])
def test_qwen4_exp_mtp_graph_metadata_checks_invalid_live_rows(
    monkeypatch, table_width, expected_invalid
):
    device = torch.device("cuda")
    table = torch.arange(5, 5 + table_width, device=device, dtype=torch.int32)[None, :]
    buffers = {
        "query_start_loc": torch.tensor([0, 3], device=device, dtype=torch.int32),
        "target_read_slots_i32": torch.empty((1, 3), device=device, dtype=torch.int32),
        "target_write_slots_i32": torch.empty((1, 3), device=device, dtype=torch.int32),
        "target_invalid_live_rows": torch.full(
            (1,), 17, device=device, dtype=torch.int32
        ),
        "target_sequence_masks": torch.ones(1, device=device, dtype=torch.bool),
        "target_accepted_tokens": torch.ones(1, device=device, dtype=torch.int32),
    }
    inputs = SimpleNamespace(
        is_target_verify=True,
        input_lengths=torch.tensor([3], dtype=torch.int32),
        prefix_lengths_device=torch.tensor([16], device=device, dtype=torch.int32),
        kv_cache_kernel_block_id_device=table,
        sequence_lengths_plus_1_device=torch.tensor(
            [32], device=device, dtype=torch.int32
        ),
    )
    checked = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        torch,
        "_assert_async",
        lambda condition, message: checked.append((bool(condition.item()), message)),
    )
    RTPQwen4ExpContext._build_gdn_metadata(
        inputs, seq_size_per_block=16, num_tokens=3, cg_bufs=buffers
    )
    assert buffers["target_invalid_live_rows"].item() == int(expected_invalid)
    assert checked == [
        (
            not expected_invalid,
            "Qwen4Exp MTP target-verify state block is invalid",
        )
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("device_lengths", [False, True])
def test_qwen4_exp_mtp_target_metadata_captures_and_resets_counter(device_lengths):
    device = torch.device("cuda")
    table = torch.tensor([[5, 6, 7, 8]], device=device, dtype=torch.int32)
    lengths = torch.tensor([17], device=device, dtype=torch.int32)
    buffers = {
        "query_start_loc": torch.tensor([0, 3], device=device, dtype=torch.int32),
        "target_read_slots_i32": torch.empty((1, 3), device=device, dtype=torch.int32),
        "target_write_slots_i32": torch.empty((1, 3), device=device, dtype=torch.int32),
        "target_invalid_live_rows": torch.zeros(1, device=device, dtype=torch.int32),
        "target_sequence_masks": torch.ones(1, device=device, dtype=torch.bool),
        "target_accepted_tokens": torch.ones(1, device=device, dtype=torch.int32),
    }
    inputs = SimpleNamespace(
        is_target_verify=True,
        input_lengths=torch.tensor(
            [3], device=device if device_lengths else "cpu", dtype=torch.int32
        ),
        prefix_lengths_device=torch.tensor([16], device=device, dtype=torch.int32),
        kv_cache_kernel_block_id_device=table,
        sequence_lengths_plus_1_device=lengths,
    )
    _target_verify_state_slots_graph_kernel[(1,)](
        table,
        lengths,
        inputs.prefix_lengths_device,
        buffers["target_read_slots_i32"],
        buffers["target_write_slots_i32"],
        buffers["target_invalid_live_rows"],
        1,
        *table.stride(),
        4,
        16,
        3,
        128,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        RTPQwen4ExpContext._build_gdn_metadata(
            inputs, seq_size_per_block=16, num_tokens=3, cg_bufs=buffers
        )
    buffers["target_invalid_live_rows"].fill_(9)
    graph.replay()
    torch.cuda.synchronize()
    assert buffers["target_invalid_live_rows"].item() == 0


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
        lambda self: setattr(
            self,
            "_cg_meta_bufs",
            {
                "query_start_loc": torch.arange(
                    self._cg_max_num_tokens + 1, dtype=torch.int32
                ),
                "seq_id": torch.arange(self._cg_max_num_tokens, dtype=torch.int64),
                "slot_mapping": torch.empty(self._cg_max_num_tokens, dtype=torch.int64),
                "block_table_i32": torch.empty(
                    self._cg_max_num_tokens, 3, dtype=torch.int32
                ),
            },
        ),
    )
    calls = []
    qsa = SimpleNamespace(is_qsa_attention=True)
    runtime.__dict__["model"] = object()
    runtime._rtp_layer_maps = ({}, {7: qsa}, {})
    runtime._rtp_forward_context_cls = SimpleNamespace(
        _bind_qsa_caches=lambda *args: calls.append(("qsa", args)),
        _bind_ple_caches=lambda *args: calls.append(("ple", args)),
    )
    runtime.kv_cache = SimpleNamespace(get_seq_size_per_block=lambda _: 64)
    runtime._cg_max_num_tokens = 2
    runtime._cg_max_seq_len = 128
    runtime._get_model_device = lambda: torch.device("cpu")

    runtime._ensure_cuda_graph_prewarmed()

    assert calls == [
        ("ple", (runtime, runtime.model, {})),
        ("qsa", (runtime, [qsa], 64)),
    ]
    assert runtime._cg_meta_bufs["physical_block_table_i32"].shape == (2, 3)


@pytest.mark.parametrize("draft_steps", [1, 2, 3])
def test_qwen4_exp_mtp_graph_prewarms_verify_window(monkeypatch, draft_steps):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    monkeypatch.setattr(
        _ATOMQwen35MoeRuntime,
        "_ensure_cuda_graph_prewarmed",
        lambda self: setattr(
            self,
            "_cg_meta_bufs",
            {
                "query_start_loc": torch.arange(
                    self._cg_max_num_tokens + 1, dtype=torch.int32
                ),
                "seq_id": torch.arange(self._cg_max_num_tokens, dtype=torch.int64),
                "slot_mapping": torch.empty(self._cg_max_num_tokens, dtype=torch.int64),
                "block_table_i32": torch.empty(
                    self._cg_max_num_tokens, 3, dtype=torch.int32
                ),
            },
        ),
    )
    width = draft_steps + 1
    runtime._rtp_model_config = SimpleNamespace(
        atom_qwen4_exp_mtp_target=True, gen_num_per_cycle=draft_steps
    )
    runtime._rtp_layer_maps = ({}, {}, {})
    runtime.kv_cache = SimpleNamespace(get_seq_size_per_block=lambda _: 64)
    runtime._cg_max_num_tokens = 3
    runtime._rtp_mtp_request_capacity = 8
    runtime._cg_max_seq_len = 128
    runtime._get_model_device = lambda: torch.device("cpu")

    runtime._ensure_cuda_graph_prewarmed()

    decode_buffers = runtime._cg_meta_bufs
    buffers = runtime._cg_mtp_meta_bufs
    assert decode_buffers["query_start_loc"].tolist() == [0, 1, 2, 3]
    assert decode_buffers["seq_id"].tolist() == [0, 1, 2]
    assert decode_buffers["seq_id_i32"].tolist() == [0, 1, 2]
    assert decode_buffers["slot_mapping"].shape == (3,)
    assert decode_buffers["qwen4_positions_i32"].shape == (8,)
    assert decode_buffers["block_table_i32"].shape == (3, 3 + draft_steps)
    assert (
        decode_buffers["block_table_i32"].data_ptr()
        != buffers["block_table_i32"].data_ptr()
    )
    assert (
        runtime._graph_buffers_for_inputs(
            SimpleNamespace(is_prefill=False, is_target_verify=False)
        )
        is decode_buffers
    )
    assert (
        runtime._graph_buffers_for_inputs(
            SimpleNamespace(is_prefill=True, is_target_verify=True)
        )
        is buffers
    )
    assert buffers["query_start_loc"].tolist() == [i * width for i in range(4)]
    assert buffers["seq_id"].tolist() == [i for i in range(3) for _ in range(width)]
    assert buffers["seq_id_i32"].tolist() == buffers["seq_id"].tolist()
    assert buffers["slot_mapping"].shape == (3 * width,)
    assert buffers["qwen4_positions_i32"].shape == (8 * width,)
    assert buffers["target_read_slots_i32"].shape == (3, width)
    assert buffers["target_write_slots_i32"].shape == (3, width)
    assert buffers["block_table_i32"].shape == (3, 3 + draft_steps)
    assert buffers["physical_block_table_i32"].shape == (3, 2 + width)


@pytest.mark.parametrize("draft_steps", [1, 2, 3])
def test_qwen4_exp_mtp_draft_graph_prewarms_verify_window(monkeypatch, draft_steps):
    runtime = object.__new__(_ATOMQwen4ExpMTPRuntime)
    monkeypatch.setattr(
        _ATOMQwen35MoeRuntime,
        "_ensure_cuda_graph_prewarmed",
        lambda self: setattr(
            self,
            "_cg_meta_bufs",
            {
                "query_start_loc": torch.arange(
                    self._cg_max_num_tokens + 1, dtype=torch.int32
                ),
                "seq_id": torch.arange(self._cg_max_num_tokens, dtype=torch.int64),
                "slot_mapping": torch.empty(self._cg_max_num_tokens, dtype=torch.int64),
                "block_table_i32": torch.empty(
                    self._cg_max_num_tokens, 3, dtype=torch.int32
                ),
            },
        ),
    )
    width = draft_steps + 1
    runtime._rtp_model_config = SimpleNamespace(
        is_mtp=True, gen_num_per_cycle=draft_steps
    )
    runtime._rtp_layer_maps = ({}, {}, {})
    runtime.kv_cache = SimpleNamespace(get_seq_size_per_block=lambda _: 64)
    runtime._cg_max_num_tokens = 3
    runtime._cg_max_seq_len = 128
    runtime._get_model_device = lambda: torch.device("cpu")

    runtime._ensure_cuda_graph_prewarmed()

    decode_buffers = runtime._cg_meta_bufs
    buffers = runtime._cg_mtp_meta_bufs
    assert decode_buffers["query_start_loc"].tolist() == [0, 1, 2, 3]
    assert decode_buffers["seq_id_i32"].tolist() == [0, 1, 2]
    assert decode_buffers["block_table_i32"].shape == (3, 3 + draft_steps)
    assert (
        runtime._graph_buffers_for_inputs(
            SimpleNamespace(is_prefill=False, is_target_verify=False)
        )
        is decode_buffers
    )
    assert (
        runtime._graph_buffers_for_inputs(
            SimpleNamespace(is_prefill=True, is_target_verify=False)
        )
        is buffers
    )
    assert buffers["query_start_loc"].tolist() == [i * width for i in range(4)]
    assert buffers["seq_id"].tolist() == [i for i in range(3) for _ in range(width)]
    assert buffers["qwen4_positions_i32"].shape == (3 * width,)
    assert buffers["slot_mapping"].shape == (3 * width,)
    assert buffers["block_table_i32"].shape == (3, 3 + draft_steps)


@pytest.mark.parametrize("draft_steps", [1, 2, 3])
def test_qwen4_exp_mtp_draft_graph_warmup_keeps_padding_rows(monkeypatch, draft_steps):
    width = draft_steps + 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn_inputs = SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        prefill_cuda_graph_copy_params=SimpleNamespace(),
        input_lengths=torch.tensor([width, width - 1, 0], dtype=torch.int32),
        cu_seqlens_device=torch.tensor(
            [0, width, 2 * width - 1, 2 * width - 1], dtype=torch.int32
        ),
    )
    query_start_loc = RTPQwen4ExpContext._build_query_start_loc_for_plugin(
        attn_inputs=attn_inputs,
        seq_lens=torch.tensor([18, 18, 16], dtype=torch.int32),
        num_tokens=3 * width,
        device=torch.device("cpu"),
        cg_bufs={"mtp_window_width": width},
    )
    assert query_start_loc.tolist() == [i * width for i in range(4)]


@pytest.mark.parametrize("capturing", [False, True])
def test_qwen4_exp_mtp_draft_graph_rejects_uneven_live_lengths(monkeypatch, capturing):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    attn_inputs = SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        prefill_cuda_graph_copy_params=SimpleNamespace(),
        input_lengths_device=torch.tensor([1, 5], dtype=torch.int32),
    )
    with pytest.raises(
        (ValueError, RuntimeError),
        match="length exceeds capture window",
    ):
        RTPQwen4ExpContext._build_query_start_loc_for_plugin(
            attn_inputs=attn_inputs,
            seq_lens=torch.tensor([18, 18], dtype=torch.int32),
            num_tokens=6,
            device=torch.device("cpu"),
            cg_bufs={
                "mtp_window_width": 3,
                "query_start_loc": torch.tensor([0, 3, 6], dtype=torch.int32),
            },
        )


def test_qwen4_exp_mtp_draft_capture_keeps_padding_rows(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    attn_inputs = SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        prefill_cuda_graph_copy_params=SimpleNamespace(),
        input_lengths_device=torch.tensor([3, 3, 0], dtype=torch.int32),
    )
    query_start_loc = RTPQwen4ExpContext._build_query_start_loc_for_plugin(
        attn_inputs=attn_inputs,
        seq_lens=torch.tensor([18, 18, 16], dtype=torch.int32),
        num_tokens=9,
        device=torch.device("cpu"),
        cg_bufs={
            "mtp_window_width": 3,
            "query_start_loc": torch.tensor([0, 3, 6, 9], dtype=torch.int32),
        },
    )
    assert query_start_loc.tolist() == [0, 3, 6, 9]


def test_qwen4_exp_generation_prefill_is_not_mtp_window(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn_inputs = SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        prefill_cuda_graph_copy_params=SimpleNamespace(),
        input_lengths=torch.tensor([2, 4], dtype=torch.int32),
        cu_seqlens_device=torch.tensor([0, 2, 6], dtype=torch.int32),
    )
    query_start_loc = RTPQwen4ExpContext._build_query_start_loc_for_plugin(
        attn_inputs=attn_inputs,
        seq_lens=torch.tensor([4, 6], dtype=torch.int32),
        num_tokens=6,
        device=torch.device("cpu"),
        cg_bufs={},
    )
    assert query_start_loc.tolist() == [0, 2, 6]


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


def test_qwen4_exp_ple_graph_uses_selected_forward_buffers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    raw = torch.empty(3, 1, dtype=torch.bfloat16)
    runtime = SimpleNamespace(
        kv_cache=SimpleNamespace(
            get_layer_cache=lambda _: SimpleNamespace(kv_cache_base=raw)
        ),
        _cg_meta_bufs={"qwen4_has_initial": torch.empty(1, dtype=torch.bool)},
    )
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
    RTPQwen4ExpContext._bind_ple_caches(runtime, model, {5: object()})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    selected = {"qwen4_has_initial": torch.empty(2, dtype=torch.bool)}
    ids = torch.tensor([1, 2], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        gdn_metadata=SimpleNamespace(
            non_spec_state_indices_tensor=ids,
            non_spec_state_indices_in_tensor=ids,
            has_initial_state=None,
        ),
        qsa_metadata=SimpleNamespace(
            logical_positions=torch.tensor([0, -1], dtype=torch.int32)
        ),
        plugin_metadata=SimpleNamespace(
            rtp_cu_seqlens_q=torch.tensor([0, 1, 2]), cg_bufs=selected
        ),
    )
    md = RTPQwen4ExpContext._ple_metadata(runtime, model, attn_metadata, {5: object()})
    assert md.has_initial_state.data_ptr() == selected["qwen4_has_initial"].data_ptr()
    assert md.has_initial_state.tolist() == [True, False]


def test_qwen4_exp_ple_rebind_requires_prewarm_before_graph_capture(monkeypatch):
    raw = torch.empty(3, 1, dtype=torch.bfloat16)
    layer_cache = SimpleNamespace(kv_cache_base=raw)
    runtime = SimpleNamespace(
        kv_cache=SimpleNamespace(get_layer_cache=lambda _: layer_cache)
    )
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
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    RTPQwen4ExpContext._bind_ple_caches(runtime, model, {5: object()})
    ngram, conv = runtime._rtp_ple_ngram, runtime._rtp_ple_conv
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    RTPQwen4ExpContext._bind_ple_caches(runtime, model, {5: object()})
    assert runtime._rtp_ple_ngram is ngram
    assert runtime._rtp_ple_conv is conv

    layer_cache.kv_cache_base = torch.empty_like(raw)
    with pytest.raises(RuntimeError, match="bound before Graph capture"):
        RTPQwen4ExpContext._bind_ple_caches(runtime, model, {5: object()})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("reserved_tail", [0, 1, 2, 3])
def test_qwen4_exp_recovered_physical_table_respects_prewarmed_buffer_stride(
    monkeypatch, reserved_tail
):
    device = torch.device("cuda")
    kernel = torch.tensor(
        [[28, 29, 30, 31, 32, 33, 34, 35], [40, 41, 42, 43, 44, 45, 46, 47]],
        dtype=torch.int32,
        device=device,
    )
    if reserved_tail:
        kernel = torch.cat(
            [
                kernel,
                torch.full((2, reserved_tail), 999, dtype=torch.int32, device=device),
            ],
            dim=1,
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
def test_qwen4_exp_graph_qsa_ignores_stale_physical_block_table(monkeypatch):
    device = torch.device("cuda")
    kernel = torch.tensor([[28, 29, 30, 31, 32, 33, 34, 35]], device=device)
    stale_physical = torch.tensor([[100, 101]], device=device)
    full_inputs = SimpleNamespace(
        kv_cache_block_id_device=stale_physical,
        kv_cache_kernel_block_id_device=kernel,
    )
    buffer = torch.empty((1, 2), dtype=torch.int32, device=device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    table = RTPQwen4ExpContext._resolve_qsa_block_table(
        full_inputs, 64, 16, {"physical_block_table_i32": buffer}
    )
    assert table.cpu().tolist() == [[7, 8]]
    assert table.data_ptr() != stale_physical.data_ptr()


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
        table,
        positions,
        torch.arange(4, device=device, dtype=torch.int32),
        slots,
        compressed,
        4,
        *table.stride(),
        2,
        64,
        4,
        128,
    )
    assert state_inputs.cpu().tolist() == [5, 10, 20, -1]
    assert slots.cpu().tolist() == [320, 703, 1344, -1]
    assert compressed.cpu().tolist() == [-1, 175, -1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_qwen4_exp_mtp_graph_qsa_slots_use_request_rows():
    device = torch.device("cuda")
    table = torch.tensor([[5, 6, 7], [10, 11, 12]], device=device, dtype=torch.int32)
    positions = torch.tensor([15, 16, 31, 32], device=device, dtype=torch.int32)
    request_ids = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32)
    slots = torch.empty(4, device=device, dtype=torch.int64)
    compressed = torch.empty_like(slots)
    _qsa_graph_slots_kernel[(1,)](
        table,
        positions,
        request_ids,
        slots,
        compressed,
        4,
        *table.stride(),
        3,
        16,
        4,
        128,
    )
    assert slots.cpu().tolist() == [95, 96, 191, 192]
    assert compressed.cpu().tolist() == [23, -1, 47, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("draft_steps", [2, 3])
def test_qwen4_exp_mtp_draft_decode_graph_uses_one_row_per_request(
    monkeypatch, draft_steps
):
    # The same draft runtime captures Q-token prefill and one-token decode.
    # At concurrency four, the latter must not reuse prefill's repeated IDs.
    runtime = object.__new__(_ATOMQwen4ExpMTPRuntime)
    device = torch.device("cuda")
    monkeypatch.setattr(
        _ATOMQwen35MoeRuntime,
        "_ensure_cuda_graph_prewarmed",
        lambda self: setattr(
            self,
            "_cg_meta_bufs",
            {
                "query_start_loc": torch.arange(5, device=device, dtype=torch.int32),
                "seq_id": torch.arange(4, device=device, dtype=torch.int64),
                "slot_mapping": torch.empty(4, device=device, dtype=torch.int64),
                "block_table_i32": torch.empty(
                    (4, 3), device=device, dtype=torch.int32
                ),
            },
        ),
    )
    runtime._rtp_model_config = SimpleNamespace(
        is_mtp=True, gen_num_per_cycle=draft_steps
    )
    runtime._rtp_layer_maps = ({}, {}, {})
    runtime.kv_cache = SimpleNamespace(get_seq_size_per_block=lambda _: 16)
    runtime._cg_max_num_tokens = 4
    runtime._cg_max_seq_len = 32
    runtime._get_model_device = lambda: device
    runtime._ensure_cuda_graph_prewarmed()

    decode = runtime._graph_buffers_for_inputs(
        SimpleNamespace(is_prefill=False, is_target_verify=False)
    )
    prefill = runtime._graph_buffers_for_inputs(
        SimpleNamespace(is_prefill=True, is_target_verify=False)
    )
    assert decode["query_start_loc"].cpu().tolist() == [0, 1, 2, 3, 4]
    assert decode["seq_id_i32"].cpu().tolist() == [0, 1, 2, 3]
    assert prefill["seq_id_i32"].cpu().tolist() == [
        row for row in range(4) for _ in range(draft_steps + 1)
    ]

    table = torch.tensor(
        [[10, 11], [20, 21], [30, 31], [40, 41]],
        device=device,
        dtype=torch.int32,
    )
    positions = torch.full((4,), 1, device=device, dtype=torch.int32)
    slots = torch.empty(4, device=device, dtype=torch.int64)
    compressed = torch.empty_like(slots)
    _qsa_graph_slots_kernel[(1,)](
        table,
        positions,
        decode["seq_id_i32"],
        slots,
        compressed,
        4,
        *table.stride(),
        2,
        16,
        4,
        128,
    )
    assert slots.cpu().tolist() == [161, 321, 481, 641]
