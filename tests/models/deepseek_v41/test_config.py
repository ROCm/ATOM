# SPDX-License-Identifier: MIT
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from atom.config import SpeculativeConfig, get_hf_config
from atom.models.deepseek_v41.config import (
    AttentionMode,
    build_attention_topology,
    normalize_hf_config,
    validate_native_quantization,
)
from atom.quant_spec import get_quant_parser
from atom.utils.selector import Family, attn_family, get_attn_backend_cls

from .reference import FIXTURES


@pytest.fixture
def raw_config():
    return json.loads((FIXTURES / "config.json").read_text())


def test_hf_config_preserves_text_vision_quantization_and_root_tokens(raw_config):
    config = get_hf_config(str(FIXTURES))
    assert config.model_type == "deepseek_v41_text"
    assert config.hidden_size == 5120 and config.rms_norm_eps == 1e-20
    assert config.architectures == ["DeepseekV41ForCausalLM"]
    assert (config.bos_token_id, config.eos_token_id, config.pad_token_id) == (0, 1, 2)
    assert config.image_token_id == 129264
    assert config.quantization_config == raw_config["quantization_config"]
    full = config._multimodal_config
    assert full.model_type == "deepseek_v41"
    assert full.vision_config.patch_size == 14
    assert full.text_config.hidden_size == 5120
    assert full.text_config is not config
    serialized = json.loads(config.to_json_string())
    assert serialized["dtype"] == raw_config["dtype"]
    assert serialized["hidden_size"] == 5120
    assert serialized["_multimodal_config"]["model_type"] == "deepseek_v41"
    assert (
        serialized["_multimodal_config"]["vision_config"]["model_type"]
        == "deepseek_v41_vision"
    )
    config.validate_parallelism(8, 8)


def test_the_published_quantization_block_is_accepted(raw_config):
    validate_native_quantization(raw_config["quantization_config"])


def test_the_published_block_parses_to_the_group32_spec(raw_config):
    """The parser resolves AITER quant types, so it needs the runtime."""
    pytest.importorskip("aiter", reason="the FP8 parser resolves AITER quant types")
    parsed = get_quant_parser("fp8").parse(raw_config["quantization_config"])
    assert parsed.global_spec.quant_type.name == "per_1x32"
    assert parsed.global_spec.weight_block_size == (32, 32)
    for shape, expected in [
        ([1, 32], "per_1x32"),
        ([1, 128], "per_1x128"),
        ([128, 128], "per_1x128"),
    ]:
        cfg = {**raw_config["quantization_config"], "weight_block_size": shape}
        assert (
            get_quant_parser("fp8").parse(cfg).global_spec.quant_type.name == expected
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("quant_method", "awq", "native FP8 checkpoint format"),
        ("weight_block_size", [128, 128], "weight_block_size must be"),
        ("scale_fmt", "e8m0", "dense scales must be ue8m0"),
        ("activation_scheme", "static", "dynamic per-32 FP8 activations"),
        ("expert_dtype", "fp8", "routed experts must use native fp4"),
    ],
)
def test_every_unsupported_quantization_field_is_refused(
    raw_config, field, value, message
):
    """Each of the five is a separate refusal, not one collapsed check.

    The kernels read exactly one format, so a checkpoint declaring another has
    no slower path to fall back to -- and a refusal that named only the first
    wrong field would send the next one back for a second round trip.
    """
    cfg = {**raw_config["quantization_config"], field: value}
    with pytest.raises(ValueError, match=message):
        validate_native_quantization(cfg)


def test_topology_resolves_four_physical_owners(raw_config):
    topology = build_attention_topology(normalize_hf_config(raw_config))
    main = topology[:40]
    assert [layer.layer_id for layer in main if layer.mode == AttentionMode.FULL] == [
        2,
        8,
        14,
        20,
    ]
    assert [
        layer.layer_id for layer in main if layer.mode == AttentionMode.REINDEX
    ] == [24, 28, 32, 36]
    assert sum(layer.mode == AttentionMode.REUSE for layer in main) == 30
    assert topology[19].kv_owner == 14 and topology[19].topk_owner == 14
    assert topology[39].kv_owner == 20 and topology[39].topk_owner == 36
    assert topology[39].candidate_owner == 20
    assert all(layer.mode == AttentionMode.WINDOW for layer in topology[40:])
    with pytest.raises(FrozenInstanceError):
        topology[20].kv_owner = 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("compress_ratios", [0] * 40, "cover 43"),
        ("kv_source_layer_ids", [2, 2, 14, 20], "duplicates"),
        ("index_source_layer_ids", [2, 8, 20], "also be an index source"),
        ("candidate_source_layer_id", 21, "must be an index source"),
        ("engram_num_embeddings", [384006168], "equal lengths"),
        ("qk_rope_head_dim", 129, "must be even"),
    ],
)
def test_malformed_model_config_is_rejected(raw_config, field, value, message):
    raw_config["text_config"][field] = value
    with pytest.raises(ValueError, match=message):
        normalize_hf_config(raw_config)


def test_ratio_transition_requires_a_compatible_owner(raw_config):
    raw_config["text_config"]["compress_ratios"][7] = 1
    with pytest.raises(ValueError, match="preceding KV owner"):
        normalize_hf_config(raw_config)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_valid_parallel_shards_are_accepted(raw_config, tp_size):
    config = normalize_hf_config(raw_config)
    config.validate_parallelism(tp_size)


def test_invalid_parallel_shards_are_rejected(raw_config):
    config = normalize_hf_config(raw_config)
    with pytest.raises(ValueError, match="tensor parallel"):
        config.validate_parallelism(3)
    with pytest.raises(ValueError, match="expert parallel"):
        config.validate_parallelism(8, 5)


def test_v41_never_falls_through_to_v4_backend(raw_config):
    config = normalize_hf_config(raw_config)
    assert attn_family(config) == Family.CSA2
    by_arch = deepcopy(config)
    by_arch.model_type = "deepseek_v3"
    assert attn_family(by_arch) == Family.CSA2
    assert get_attn_backend_cls(Family.CSA2, False, False) == (
        "atom.model_ops.attentions.deepseek_v41.backend.DeepseekV41Backend"
    )
    with pytest.raises(NotImplementedError, match="native ATOM"):
        get_attn_backend_cls(Family.CSA2, True, False)


def test_native_draft_config_preserves_v41_architecture():
    config = get_hf_config(str(FIXTURES))
    SpeculativeConfig.hf_config_override(config, model_path=None)
    assert config.architectures == ["DeepseekV41DSparkModel"]
    assert config.model_type == "deepseek_v41_dspark"
    assert config.num_nextn_predict_layers == 3
    assert config.head_dim == 512 and config.qk_rope_head_dim == 64
    assert config.dspark_n_routed_experts == 128
    assert config.n_routed_experts == 384
