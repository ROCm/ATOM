# SPDX-License-Identifier: MIT
# Tests for LayerQuantConfig, QuantizationConfig, and the
# parser registry (atom/config.py + atom/quant_spec.py).
#
# Covers: per-layer quant config dispatch, quark config parsing,
# layer name matching (exact / regex / fnmatch), packed-module remapping,
# typed LayerQuantConfig API, and backward compatibility.
#
# atom.config depends on torch, aiter, and transformers.  We load the source
# files under temporary sys.modules mocks so the tests run in any environment.

import contextlib
import enum
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)

# -------------------------------------------------------------------------
# Mock primitives
# -------------------------------------------------------------------------


class QuantType(enum.IntEnum):
    No = 0
    per_Token = 1
    per_Tensor = 2
    per_1x32 = 3
    per_1x128 = 4


BF16 = "torch.bfloat16"
FP8 = "mock_fp8"
FP4X2 = "mock_fp4x2"
INT8 = "mock_int8"

D_DTYPES = {
    "fp8": FP8,
    "fp4x2": FP4X2,
    "int8": INT8,
    "int4x2": "mock_int4x2",
    "i8": INT8,
    "i4x2": "mock_int4x2",
}


class FakeHFConfig:
    """Lightweight stand-in for transformers.PretrainedConfig."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def get_config_dict(model):
        return {}, {}


# -------------------------------------------------------------------------
# Module loader — patch sys.modules only while exec-ing config.py
# -------------------------------------------------------------------------


@contextlib.contextmanager
def _temporary_mocks():
    mock_torch = MagicMock()
    mock_torch.bfloat16 = BF16

    mock_aiter = types.ModuleType("aiter")
    mock_aiter.QuantType = QuantType

    mock_aiter_dtypes = types.ModuleType("aiter.utility.dtypes")
    mock_aiter_dtypes.d_dtypes = D_DTYPES

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.PretrainedConfig = FakeHFConfig
    mock_transformers.AutoConfig = MagicMock()
    mock_transformers.GenerationConfig = MagicMock()

    mock_atom_utils = types.ModuleType("atom.utils")
    mock_atom_utils.envs = MagicMock()
    mock_atom_utils.get_open_port = MagicMock(return_value=8000)

    mock_dist_utils = types.ModuleType("atom.utils.distributed.utils")
    mock_dist_utils.stateless_init_torch_distributed_process_group = MagicMock()

    mock_aiter.__path__ = []

    mock_plugin = types.ModuleType("atom.plugin")
    mock_plugin.is_plugin_mode = MagicMock(return_value=False)
    mock_plugin.is_vllm = MagicMock(return_value=False)
    mock_plugin_config = types.ModuleType("atom.plugin.config")
    mock_plugin_config.PluginConfig = MagicMock()

    patches = {
        "torch": mock_torch,
        "torch.distributed": MagicMock(),
        "aiter": mock_aiter,
        "aiter.utility": types.ModuleType("aiter.utility"),
        "aiter.utility.dtypes": mock_aiter_dtypes,
        "transformers": mock_transformers,
        "atom.utils": mock_atom_utils,
        "atom.utils.distributed": types.ModuleType("atom.utils.distributed"),
        "atom.utils.distributed.utils": mock_dist_utils,
        "atom.plugin": mock_plugin,
        "atom.plugin.config": mock_plugin_config,
    }

    saved = {}
    for name, mock in patches.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock
    try:
        yield
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def _load_module(filename: str, module_name: str):
    path = os.path.join(ATOM_ROOT, "atom", filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass etc. can resolve the module
    sys.modules[module_name] = mod
    with _temporary_mocks():
        spec.loader.exec_module(mod)
    return mod


# Load quant_spec first, then inject it so config.py can import it.
_qs = _load_module("quant_spec.py", "atom.quant_spec")
sys.modules["atom.quant_spec"] = _qs

_m = _load_module("config.py", "_atom_config_test")

QuantizationConfig = _m.QuantizationConfig
LayerQuantConfig = _qs.LayerQuantConfig
QuarkParser = _qs.QuarkParser
GenericParser = _qs.GenericParser
get_quant_parser = _qs.get_quant_parser


# =========================================================================
# Tests — LayerQuantConfig
# =========================================================================


class TestLayerQuantConfig:
    def test_defaults(self):
        spec = LayerQuantConfig()
        assert spec.quant_type == QuantType.No
        assert spec.quant_dtype == BF16
        assert spec.is_dynamic is True
        assert spec.quant_method is None
        assert spec.is_quantized is False

    def test_no_quant_factory(self):
        spec = LayerQuantConfig.no_quant(FP8)
        assert spec.quant_type == QuantType.No
        assert spec.quant_dtype == FP8
        assert spec.is_quantized is False

    def test_is_quantized(self):
        spec = LayerQuantConfig(quant_type=QuantType.per_Token, quant_dtype=FP8)
        assert spec.is_quantized is True

    def test_frozen(self):
        spec = LayerQuantConfig()
        with pytest.raises(AttributeError):
            spec.quant_type = QuantType.per_Token  # type: ignore[misc]


# =========================================================================
# Tests — Parser Registry
# =========================================================================


class TestParserRegistry:
    def test_quark_registered(self):
        parser = get_quant_parser("quark")
        assert isinstance(parser, QuarkParser)

    def test_generic_fallback(self):
        parser = get_quant_parser("compressed-tensors")
        assert isinstance(parser, GenericParser)

    def test_unknown_falls_to_generic(self):
        parser = get_quant_parser("some_unknown_method")
        assert isinstance(parser, GenericParser)


# =========================================================================
# Tests — QuarkParser
# =========================================================================


class TestQuarkParser:
    def test_per_channel_fp8(self):
        parser = QuarkParser()
        result = parser.parse(
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_channel", "dtype": "fp8_e4m3"},
                    "input_tensors": {"is_dynamic": True},
                },
            }
        )
        assert result.global_spec.quant_type == QuantType.per_Token
        assert result.global_spec.quant_dtype == FP8
        assert result.global_spec.is_dynamic is True

    def test_per_group_fp4(self):
        parser = QuarkParser()
        result = parser.parse(
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_group", "dtype": "fp4_e2m1"},
                    "input_tensors": {"is_dynamic": False},
                },
            }
        )
        assert result.global_spec.quant_type == QuantType.per_1x32
        assert result.global_spec.quant_dtype == FP4X2
        assert result.global_spec.is_dynamic is False

    def test_no_input_tensors_defaults_dynamic(self):
        parser = QuarkParser()
        result = parser.parse(
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_tensor", "dtype": "int8"},
                    "input_tensors": None,
                },
            }
        )
        assert result.global_spec.quant_type == QuantType.per_Tensor
        assert result.global_spec.is_dynamic is True

    def test_layer_config_parsed(self):
        parser = QuarkParser()
        result = parser.parse(
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_channel", "dtype": "fp8_e4m3"},
                    "input_tensors": {"is_dynamic": True},
                },
                "layer_quant_config": {
                    "*.mlp.*": {
                        "weight": {"qscheme": "per_group", "dtype": "fp4_e2m1"},
                        "input_tensors": {"is_dynamic": False},
                    },
                },
                "exclude": ["lm_head"],
            }
        )
        assert len(result.layer_pattern_specs) == 1
        pattern, spec = result.layer_pattern_specs[0]
        assert pattern == "*.mlp.*"
        assert spec.quant_type == QuantType.per_1x32
        assert spec.quant_dtype == FP4X2
        assert result.exclude_layers == ["lm_head"]


# =========================================================================
# Tests — QuantizationConfig init
# =========================================================================


class TestQuantizationConfigInit:
    def test_none_config(self):
        qcfg = QuantizationConfig(config=None)
        assert qcfg.quant_method == ""
        assert qcfg.exclude_layers == []
        assert qcfg.global_quant_config.quant_type == QuantType.No
        assert qcfg.global_quant_config.is_quantized is False

    def test_config_without_quantization(self):
        hf = FakeHFConfig(torch_dtype=BF16)
        qcfg = QuantizationConfig(hf)
        assert qcfg.quant_method == ""
        assert qcfg.global_quant_config.quant_type == QuantType.No
        assert qcfg.global_quant_config.quant_dtype == BF16

    def test_quark_config_parses_global_and_layer(self):
        hf = FakeHFConfig(
            torch_dtype=BF16,
            quantization_config={
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_channel", "dtype": "fp8_e4m3"},
                    "input_tensors": {"is_dynamic": True},
                },
                "layer_quant_config": {
                    "*.mlp.*": {
                        "weight": {"qscheme": "per_group", "dtype": "fp4_e2m1"},
                        "input_tensors": {"is_dynamic": False},
                    }
                },
                "exclude": ["lm_head"],
            },
        )
        qcfg = QuantizationConfig(hf)
        assert qcfg.quant_method == "quark"
        assert qcfg.global_quant_config.quant_type == QuantType.per_Token
        assert qcfg.global_quant_config.quant_dtype == FP8
        # layer pattern specs
        assert len(qcfg.layer_pattern_specs) == 1
        mlp_pattern, mlp_spec = qcfg.layer_pattern_specs[0]
        assert mlp_pattern == "*.mlp.*"
        assert mlp_spec.quant_type == QuantType.per_1x32
        assert mlp_spec.quant_dtype == FP4X2
        assert mlp_spec.is_dynamic is False

        assert qcfg.exclude_layers == ["lm_head"]


# =========================================================================
# Tests — get_layer_quant_config resolution
# =========================================================================


class TestGetLayerQuantConfig:
    def test_falls_back_to_global(self):
        qcfg = QuantizationConfig(config=None)
        qcfg.global_spec = LayerQuantConfig(
            quant_type=QuantType.per_Token, quant_dtype=FP8
        )
        result = qcfg.get_layer_quant_config("model.layers.0.self_attn.q_proj")
        assert result.quant_type == QuantType.per_Token
        assert result.quant_dtype == FP8

    def test_layer_specific_overrides_global(self):
        qcfg = QuantizationConfig(config=None)
        qcfg.global_spec = LayerQuantConfig(quant_dtype=FP8)
        qcfg.layer_pattern_specs = [
            (
                "*.mlp.*",
                LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=FP4X2),
            ),
        ]
        result = qcfg.get_layer_quant_config("model.layers.0.mlp.gate_proj")
        assert result.quant_dtype == FP4X2
        assert result.quant_type == QuantType.per_1x32

    def test_excluded_layer_returns_unquantized(self):
        qcfg = QuantizationConfig(config=None)
        qcfg.torch_dtype = BF16
        qcfg.global_spec = LayerQuantConfig(
            quant_type=QuantType.per_Token, quant_dtype=FP8
        )
        qcfg.exclude_layers = ["lm_head"]

        result = qcfg.get_layer_quant_config("lm_head")
        assert result.quant_type == QuantType.No
        assert result.quant_dtype == BF16


# =========================================================================
# Tests — Exclude layer matching
# =========================================================================


class TestExcludeMatching:
    def _make(self, exclude_layers):
        qcfg = QuantizationConfig(config=None)
        qcfg.exclude_layers = exclude_layers
        return qcfg

    def test_empty_exclude(self):
        qcfg = self._make([])
        assert not qcfg._is_excluded("any_layer")

    def test_none_layer_name(self):
        qcfg = self._make(["lm_head"])
        assert not qcfg._is_excluded(None)

    def test_exact_match(self):
        qcfg = self._make(["lm_head"])
        assert qcfg._is_excluded("lm_head")

    def test_regex_match(self):
        qcfg = self._make(["re:model\\.layers\\..*shared_expert.*"])
        assert qcfg._is_excluded("model.layers.3.shared_expert.gate_proj")

    def test_no_match(self):
        qcfg = self._make(["lm_head"])
        assert not qcfg._is_excluded("self_attn.q_proj")


class TestMatchesExclude:
    def test_exact(self):
        assert QuantizationConfig._matches_exclude("lm_head", "lm_head") is True
        assert QuantizationConfig._matches_exclude("lm_head", "other") is False

    def test_regex(self):
        assert (
            QuantizationConfig._matches_exclude(
                "model.layers.5.self_attn.q_proj",
                "re:model\\.layers\\..*self_attn.*",
            )
            is True
        )
        assert (
            QuantizationConfig._matches_exclude(
                "model.layers.5.mlp.gate_proj",
                "re:model\\.layers\\..*self_attn.*",
            )
            is False
        )

    def test_contains_mode(self):
        assert (
            QuantizationConfig._matches_exclude(
                "self_attn",
                "model.layers.0.self_attn.q_a_proj",
                check_contains=True,
            )
            is True
        )
        assert (
            QuantizationConfig._matches_exclude(
                "mlp", "self_attn.q_proj", check_contains=True
            )
            is False
        )


# =========================================================================
# Tests — remap_layer_name
# =========================================================================


class TestRemapLayerName:
    @staticmethod
    def _pattern_dict(qcfg):
        """Helper: return pattern->spec dict from layer_pattern_specs."""
        return dict(qcfg.layer_pattern_specs)

    def test_deepseek_v3_with_q_lora_rank(self):
        """Individual proj names -> fused names for deepseek_v3."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_pattern_specs = [
            ("*.q_a_proj", LayerQuantConfig(quant_type=QuantType.per_Token)),
            ("*.gate_proj", LayerQuantConfig(quant_type=QuantType.per_1x32)),
        ]
        qcfg.exclude_layers = ["model.layers.0.q_a_proj"]

        hf = FakeHFConfig(model_type="deepseek_v3", q_lora_rank=512)
        qcfg.remap_layer_name(hf)

        pats = self._pattern_dict(qcfg)
        assert "*.fused_qkv_a_proj" in pats
        assert "*.gate_up_proj" in pats
        assert "*.q_a_proj" not in pats
        assert "model.layers.0.fused_qkv_a_proj" in qcfg.exclude_layers

    def test_qwen3_moe_splits_fused(self):
        """Fused gate_up_proj -> [gate_proj, up_proj] for qwen3_moe."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_pattern_specs = [
            ("*.gate_up_proj", LayerQuantConfig(quant_type=QuantType.per_Token)),
        ]
        qcfg.exclude_layers = []

        hf = FakeHFConfig(model_type="qwen3_moe", mlp_only_layers=[1])
        qcfg.remap_layer_name(hf, packed_modules_mapping={})

        pats = self._pattern_dict(qcfg)
        assert "*.gate_proj" in pats
        assert "*.up_proj" in pats
        assert "*.gate_up_proj" not in pats

    def test_exclude_layers_deduplication(self):
        """gate_proj and up_proj both map to gate_up_proj -- only one remains."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_pattern_specs = []
        qcfg.exclude_layers = [
            "model.layers.0.gate_proj",
            "model.layers.0.up_proj",
        ]

        hf = FakeHFConfig(model_type="deepseek_v3", q_lora_rank=512)
        qcfg.remap_layer_name(hf)

        assert qcfg.exclude_layers.count("model.layers.0.gate_up_proj") == 1

    def test_glm_moe_dsa_remaps_like_deepseek_v3(self):
        """GLM-5 (glm_moe_dsa) uses same packed fusing as deepseek_v3."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_pattern_specs = []
        qcfg.exclude_layers = [
            "model.layers.0.self_attn.q_a_proj",
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
        ]

        hf = FakeHFConfig(model_type="glm_moe_dsa", q_lora_rank=2048)
        qcfg.remap_layer_name(hf)

        assert "model.layers.0.self_attn.fused_qkv_a_proj" in qcfg.exclude_layers
        assert "model.layers.0.mlp.gate_up_proj" in qcfg.exclude_layers
        assert "model.layers.0.self_attn.q_a_proj" not in qcfg.exclude_layers
        assert "model.layers.0.mlp.gate_proj" not in qcfg.exclude_layers


class TestComputeHash:
    def test_hash_is_deterministic(self):
        qcfg = QuantizationConfig(config=None)
        h1 = qcfg.compute_hash()
        h2 = qcfg.compute_hash()
        assert h1 == h2
        assert isinstance(h1, str) and len(h1) == 64

    def test_different_configs_produce_different_hashes(self):
        qcfg1 = QuantizationConfig(config=None)
        qcfg2 = QuantizationConfig(config=None)
        qcfg2.global_spec = LayerQuantConfig(
            quant_type=QuantType.per_Token, quant_dtype=FP8
        )
        assert qcfg1.compute_hash() != qcfg2.compute_hash()

    def test_exclude_layers_affect_hash(self):
        qcfg1 = QuantizationConfig(config=None)
        qcfg2 = QuantizationConfig(config=None)
        qcfg2.exclude_layers = ["lm_head"]
        assert qcfg1.compute_hash() != qcfg2.compute_hash()

    def test_layer_pattern_specs_affect_hash(self):
        qcfg1 = QuantizationConfig(config=None)
        qcfg2 = QuantizationConfig(config=None)
        qcfg2.layer_pattern_specs = [
            ("*.mlp.*", LayerQuantConfig(quant_type=QuantType.per_1x32)),
        ]
        assert qcfg1.compute_hash() != qcfg2.compute_hash()


# =========================================================================
# Tests — Convenience properties
# =========================================================================


class TestConvenienceProperties:
    def test_quant_type_property(self):
        hf = FakeHFConfig(
            torch_dtype=BF16,
            quantization_config={
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_channel", "dtype": "fp8_e4m3"},
                    "input_tensors": {"is_dynamic": True},
                },
            },
        )
        qcfg = QuantizationConfig(hf)
        assert qcfg.quant_type == QuantType.per_Token
        assert qcfg.quant_dtype == FP8
        assert qcfg.is_dynamic is True
