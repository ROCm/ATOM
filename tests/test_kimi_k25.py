# SPDX-License-Identifier: MIT
# Tests for Kimi-K2.5 model support (kimi_k25.py, config.py, loader.py,
# model_runner.py).
#
# Covers: multimodal text_config extraction, config registry entries,
# skip_weight_prefixes filtering, model arch registration, remap_layer_name
# for kimi_k2, and KimiK25ForCausalLM wrapper class attributes.

import contextlib
import enum
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)

# ---------------------------------------------------------------------------
# Mock primitives (same pattern as test_quant_config.py)
# ---------------------------------------------------------------------------


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
    def get_config_dict(model, **kwargs):
        return {}, {}


class FakeAutoConfig:
    """Mock for transformers.AutoConfig."""

    _registry: dict = {}

    @classmethod
    def for_model(cls, model_type):
        return cls

    @classmethod
    def from_dict(cls, d):
        return FakeHFConfig(**d)

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        return FakeHFConfig(model_type=model)


# ---------------------------------------------------------------------------
# Module loader — same approach as test_quant_config.py
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _temporary_mocks():
    mock_torch = MagicMock()
    mock_torch.bfloat16 = BF16

    mock_aiter = types.ModuleType("aiter")
    mock_aiter.QuantType = QuantType
    mock_aiter.__path__ = []

    mock_aiter_dtypes = types.ModuleType("aiter.utility.dtypes")
    mock_aiter_dtypes.d_dtypes = D_DTYPES

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.PretrainedConfig = FakeHFConfig
    mock_transformers.AutoConfig = FakeAutoConfig
    mock_transformers.GenerationConfig = MagicMock()

    mock_atom_utils = types.ModuleType("atom.utils")
    mock_atom_utils.envs = MagicMock()
    mock_atom_utils.get_open_port = MagicMock(return_value=8000)

    mock_dist_utils = types.ModuleType("atom.utils.distributed.utils")
    mock_dist_utils.stateless_init_torch_distributed_process_group = MagicMock()

    mock_plugin = types.ModuleType("atom.plugin")
    mock_plugin.is_plugin_mode = MagicMock(return_value=False)
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


def _load_config():
    path = os.path.join(ATOM_ROOT, "atom", "config.py")
    spec = importlib.util.spec_from_file_location("_atom_config_kimi_test", path)
    mod = importlib.util.module_from_spec(spec)
    with _temporary_mocks():
        spec.loader.exec_module(mod)
    return mod


_m = _load_config()
QuantizationConfig = _m.QuantizationConfig
LayerQuantConfig = _m.LayerQuantConfig
_CONFIG_REGISTRY = _m._CONFIG_REGISTRY
_MULTIMODAL_MODEL_TYPES = _m._MULTIMODAL_MODEL_TYPES
get_hf_config = _m.get_hf_config


# ===========================================================================
# Tests
# ===========================================================================


class TestConfigRegistry:
    """Verify Kimi K2 / K2.5 entries in config registries."""

    def test_kimi_k2_in_config_registry(self):
        assert "kimi_k2" in _CONFIG_REGISTRY
        assert _CONFIG_REGISTRY["kimi_k2"] == "deepseek_v3"

    def test_kimi_k25_in_multimodal_registry(self):
        assert "kimi_k25" in _MULTIMODAL_MODEL_TYPES
        assert _MULTIMODAL_MODEL_TYPES["kimi_k25"] == "text_config"


class TestGetHfConfigMultimodal:
    """Test multimodal text_config extraction in get_hf_config."""

    def _make_config_dict(self, **overrides):
        """Build a realistic kimi_k25 config_dict."""
        base = {
            "model_type": "kimi_k25",
            "architectures": ["KimiK25ForConditionalGeneration"],
            "bos_token_id": 100000,
            "eos_token_id": 100001,
            "pad_token_id": 100002,
            "quantization_config": {
                "quant_method": "mxfp",
                "quant_type": "mxfp4",
            },
            "text_config": {
                "model_type": "kimi_k2",
                "hidden_size": 7168,
                "num_hidden_layers": 61,
                "auto_map": {"AutoModel": "modeling.KimiK25Model"},
            },
        }
        base.update(overrides)
        return base

    @contextlib.contextmanager
    def _patch_get_config_dict(self, config_dict):
        """Temporarily patch FakeHFConfig.get_config_dict."""
        original = FakeHFConfig.get_config_dict

        @staticmethod
        def patched(model, **kwargs):
            return config_dict, {}

        FakeHFConfig.get_config_dict = patched
        try:
            yield
        finally:
            FakeHFConfig.get_config_dict = original

    def test_extracts_text_config(self):
        config_dict = self._make_config_dict()
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("amd/Kimi-K2.5-MXFP4")

        assert hf_config.hidden_size == 7168
        assert hf_config.num_hidden_layers == 61

    def test_propagates_quantization_config(self):
        config_dict = self._make_config_dict()
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("amd/Kimi-K2.5-MXFP4")

        assert hasattr(hf_config, "quantization_config")
        assert hf_config.quantization_config["quant_method"] == "mxfp"

    def test_does_not_overwrite_existing_quant_config(self):
        """If text_config already has quantization_config, don't override it."""
        config_dict = self._make_config_dict()
        config_dict["text_config"]["quantization_config"] = {
            "quant_method": "inner",
        }
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("amd/Kimi-K2.5-MXFP4")

        assert hf_config.quantization_config["quant_method"] == "inner"

    def test_preserves_original_architectures(self):
        config_dict = self._make_config_dict()
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("amd/Kimi-K2.5-MXFP4")

        assert hf_config.architectures == ["KimiK25ForConditionalGeneration"]

    def test_propagates_token_ids(self):
        config_dict = self._make_config_dict()
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("amd/Kimi-K2.5-MXFP4")

        assert hf_config.bos_token_id == 100000
        assert hf_config.eos_token_id == 100001
        assert hf_config.pad_token_id == 100002

    def test_removes_auto_map_from_text_config(self):
        """auto_map should be stripped to avoid trust_remote_code issues."""
        config_dict = self._make_config_dict()
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("amd/Kimi-K2.5-MXFP4")

        assert not hasattr(hf_config, "auto_map")

    def test_missing_text_config_uses_empty_dict(self):
        """If text_config key is absent, should use empty dict."""
        config_dict = self._make_config_dict()
        del config_dict["text_config"]
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("amd/Kimi-K2.5-MXFP4")

        # Should still return a config (from empty dict), not crash
        assert hf_config is not None


class TestRemapLayerNameKimiK2:
    """Verify remap_layer_name handles kimi_k2 model_type correctly."""

    def test_kimi_k2_with_q_lora_rank(self):
        """kimi_k2 should get deepseek-v3-style fused packed_modules_mapping."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_quant_config = {
            "*.q_a_proj": LayerQuantConfig(quant_type=QuantType.per_Token),
            "*.gate_proj": LayerQuantConfig(quant_type=QuantType.per_1x32),
        }
        qcfg.exclude_layers = ["model.layers.0.q_a_proj"]

        hf = FakeHFConfig(model_type="kimi_k2", q_lora_rank=512)
        qcfg.remap_layer_name(hf)

        # Should fuse q_a_proj -> fused_qkv_a_proj
        assert "*.fused_qkv_a_proj" in qcfg.layer_quant_config
        assert "*.q_a_proj" not in qcfg.layer_quant_config
        # Should fuse gate_proj -> gate_up_proj
        assert "*.gate_up_proj" in qcfg.layer_quant_config
        assert "*.gate_proj" not in qcfg.layer_quant_config
        # Exclude layers should also be remapped
        assert "model.layers.0.fused_qkv_a_proj" in qcfg.exclude_layers

    def test_kimi_k2_without_q_lora_rank(self):
        """kimi_k2 without q_lora_rank should still fuse gate/up proj."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_quant_config = {
            "*.gate_proj": LayerQuantConfig(quant_type=QuantType.per_Token),
        }
        qcfg.exclude_layers = []

        hf = FakeHFConfig(model_type="kimi_k2")
        qcfg.remap_layer_name(hf)

        assert "*.gate_up_proj" in qcfg.layer_quant_config
        assert "*.gate_proj" not in qcfg.layer_quant_config


class TestModelArchRegistration:
    """Verify KimiK25ForConditionalGeneration is in model_runner."""

    def test_kimi_k25_in_support_model_arch_dict(self):
        path = os.path.join(ATOM_ROOT, "atom", "model_engine", "model_runner.py")
        with open(path) as f:
            content = f.read()
        assert "KimiK25ForConditionalGeneration" in content
        assert "atom.models.kimi_k25.KimiK25ForCausalLM" in content

    def test_kimi_k2_in_is_deepseek_mla(self):
        """kimi_k2 model_type should be recognized as MLA architecture."""
        path = os.path.join(ATOM_ROOT, "atom", "model_engine", "model_runner.py")
        with open(path) as f:
            content = f.read()
        # kimi_k2 should be in the is_deepseek_mla model_type tuple
        assert '"kimi_k2"' in content


class TestKimiK25ModelClass:
    """Test the KimiK25ForCausalLM class attributes without instantiation."""

    def test_skip_weight_prefixes_defined(self):
        """Model class should define prefixes for vision/projector weights."""
        path = os.path.join(ATOM_ROOT, "atom", "models", "kimi_k25.py")
        with open(path) as f:
            content = f.read()
        assert "vision_tower." in content
        assert "mm_projector." in content

    def test_skip_weight_prefixes_values(self):
        """Import the class attributes without full init to check values."""
        path = os.path.join(ATOM_ROOT, "atom", "models", "kimi_k25.py")
        with open(path) as f:
            src = f.read()
        # Execute in a sandbox with mocked imports
        ns = {}
        import torch as real_torch

        mock_config = types.ModuleType("atom.config")
        mock_config.Config = MagicMock()

        mock_deepseek = types.ModuleType("atom.models.deepseek_v2")
        mock_deepseek.DeepseekV2ForCausalLM = MagicMock()

        mock_utils = types.ModuleType("atom.models.utils")
        mock_utils.IntermediateTensors = MagicMock()

        saved = {}
        patches = {
            "torch": real_torch,
            "torch.nn": real_torch.nn,
            "atom.config": mock_config,
            "atom.models.deepseek_v2": mock_deepseek,
            "atom.models.utils": mock_utils,
        }
        for name, mock in patches.items():
            saved[name] = sys.modules.get(name)
            sys.modules[name] = mock
        try:
            exec(compile(src, path, "exec"), ns)
        finally:
            for name, orig in saved.items():
                if orig is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = orig

        cls = ns["KimiK25ForCausalLM"]
        assert cls.skip_weight_prefixes == ["vision_tower.", "mm_projector."]


class TestSkipWeightPrefixesLogic:
    """Test the skip_weight_prefixes filtering logic from the loader."""

    @pytest.fixture
    def skip_prefixes(self):
        return ["vision_tower.", "mm_projector."]

    def _should_skip(self, name, prefixes):
        """Replicate the exact logic from loader.py."""
        return prefixes and any(name.startswith(p) for p in prefixes)

    def test_skips_vision_tower_weights(self, skip_prefixes):
        assert self._should_skip("vision_tower.encoder.layer.0.weight", skip_prefixes)

    def test_skips_mm_projector_weights(self, skip_prefixes):
        assert self._should_skip("mm_projector.linear.weight", skip_prefixes)

    def test_does_not_skip_language_model_weights(self, skip_prefixes):
        assert not self._should_skip(
            "language_model.model.layers.0.self_attn.q_proj.weight",
            skip_prefixes,
        )

    def test_does_not_skip_lm_head(self, skip_prefixes):
        assert not self._should_skip("language_model.lm_head.weight", skip_prefixes)

    def test_does_not_skip_embed_tokens(self, skip_prefixes):
        assert not self._should_skip(
            "language_model.model.embed_tokens.weight", skip_prefixes
        )

    def test_empty_prefixes_skip_nothing(self):
        assert not self._should_skip("vision_tower.weight", [])

    def test_partial_name_match_is_prefix_only(self, skip_prefixes):
        """'vision_tower' without the dot should not match 'vision_tower.'."""
        assert not self._should_skip("vision_tower_extra", skip_prefixes)

    def test_mm_projector_partial_no_match(self, skip_prefixes):
        assert not self._should_skip("mm_projector_v2.weight", skip_prefixes)


class TestLoaderSkipWeightPrefixesIntegration:
    """Verify the loader.py actually reads skip_weight_prefixes from model."""

    def test_loader_reads_skip_weight_prefixes(self):
        """The loader should call getattr(model, 'skip_weight_prefixes', [])."""
        path = os.path.join(ATOM_ROOT, "atom", "model_loader", "loader.py")
        with open(path) as f:
            content = f.read()
        assert 'getattr(model, "skip_weight_prefixes", [])' in content

    def test_loader_uses_startswith_for_filtering(self):
        """The loader should use name.startswith(p) to match prefixes."""
        path = os.path.join(ATOM_ROOT, "atom", "model_loader", "loader.py")
        with open(path) as f:
            content = f.read()
        assert "name.startswith(p) for p in skip_weight_prefixes" in content
