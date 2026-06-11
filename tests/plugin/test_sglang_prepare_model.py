"""Tests for prepare_model orchestration in sglang plugin mode.

Verifies that prepare_model correctly validates engine/arch, selects the
right model dict, and calls register_ops → set_attn_cls → init_aiter_dist
in the correct order.

Because importing atom.plugin.register triggers the full ATOM model import
chain, we inject a fake register module into sys.modules before calling
prepare_model.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch

from atom.plugin import prepare as plugin_runtime
from atom.plugin.sglang import prepare as sglang_prepare


class _Obj:
    """Minimal attribute bag for faking nested configs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_runtime._set_framework_backbone("atom")
    yield
    plugin_runtime._set_framework_backbone("atom")


def _make_fake_register_module(model_dict=None):
    """Create a fake atom.plugin.register module with controllable model dicts."""
    mod = MagicMock()
    mod._ATOM_SUPPORTED_MODELS = model_dict or {}
    mod.register_ops_to_sglang = MagicMock()
    mod.init_aiter_dist = MagicMock()
    mod.set_attn_cls = MagicMock()
    return mod


def _make_fake_runtime_module():
    mod = MagicMock()
    mod.get_model_arch_spec = MagicMock(return_value=_Obj(prepare_config=None))
    return mod


# ---------------------------------------------------------------------------
# Engine / architecture validation
# ---------------------------------------------------------------------------


def test_prepare_model_rejects_unsupported_architecture():
    """Unknown architecture should raise ValueError from the SGLang prepare path."""
    fake_register = _make_fake_register_module(
        model_dict={"DeepseekV3ForCausalLM": MagicMock()}
    )
    fake_runtime = _make_fake_runtime_module()

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.sglang.runtime": fake_runtime,
            "atom.plugin.sglang.graph_capture_patch": MagicMock(
                apply_graph_capture_patch=MagicMock()
            ),
        },
    ):
        config = _Obj(architectures=["TotallyFakeModelArch"])
        with pytest.raises(ValueError, match="does not support"):
            sglang_prepare.prepare_model(config=config)


# ---------------------------------------------------------------------------
# Happy path — sglang orchestration
# ---------------------------------------------------------------------------


def test_prepare_model_sglang_happy_path():
    """Verify sglang path calls register → set_attn → init_dist and returns model."""
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_model = MagicMock(name="FakeDeepseekModel")
    fake_model_cls = MagicMock(return_value=fake_model)

    fake_register = _make_fake_register_module(
        model_dict={"DeepseekV3ForCausalLM": fake_model_cls}
    )
    fake_runtime = _make_fake_runtime_module()

    mock_gen_config = MagicMock(return_value=fake_atom_config)
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = mock_gen_config

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
            "atom.plugin.sglang.runtime": fake_runtime,
            "atom.plugin.sglang.graph_capture_patch": MagicMock(
                apply_graph_capture_patch=MagicMock()
            ),
        },
    ):
        config = _Obj(architectures=["DeepseekV3ForCausalLM"])
        result = sglang_prepare.prepare_model(config=config)

    # Config generation called
    mock_gen_config.assert_called_once_with(config)

    # Registration sequence called
    fake_register.register_ops_to_sglang.assert_called_once_with(
        atom_config=fake_atom_config
    )
    fake_register.set_attn_cls.assert_called_once()
    fake_register.init_aiter_dist.assert_called_once_with(config=fake_atom_config)

    # Model class instantiated with atom_config and returned
    fake_model_cls.assert_called_once_with(atom_config=fake_atom_config)
    assert result is fake_model


def test_prepare_model_remaps_quant_config_for_generic_wrapper():
    fake_quant_config = MagicMock()
    fake_atom_config = _Obj(
        hf_config=_Obj(model_type="glm_moe_dsa"),
        plugin_config=_Obj(is_plugin_mode=True),
        quant_config=fake_quant_config,
    )
    fake_model = MagicMock(name="FakeGlmModel")
    fake_model_cls = MagicMock(return_value=fake_model)
    fake_model_cls.packed_modules_mapping = {"gate_proj": ("gate_up_proj", 0)}
    fake_model_cls.hf_to_atom_mapper = object()
    fake_model_cls.quant_exclude_name_mapping = {
        "indexers_proj": "indexer.weights_proj",
    }
    fake_model_cls.quant_default_exclude_layers = ["*.indexer.weights_proj"]

    fake_register = _make_fake_register_module(
        model_dict={"GlmMoeDsaForCausalLM": fake_model_cls}
    )
    fake_runtime = _make_fake_runtime_module()
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=fake_atom_config
    )

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
            "atom.plugin.sglang.runtime": fake_runtime,
            "atom.plugin.sglang.graph_capture_patch": MagicMock(
                apply_graph_capture_patch=MagicMock()
            ),
        },
    ):
        config = _Obj(architectures=["GlmMoeDsaForCausalLM"])
        result = sglang_prepare.prepare_model(config=config)

    fake_quant_config.remap_layer_name.assert_called_once_with(
        fake_atom_config.hf_config,
        packed_modules_mapping=fake_model_cls.packed_modules_mapping,
        weights_mapper=fake_model_cls.hf_to_atom_mapper,
        quant_exclude_name_mapping=fake_model_cls.quant_exclude_name_mapping,
    )
    fake_quant_config.apply_default_exclude_layers.assert_called_once_with(
        fake_model_cls.quant_default_exclude_layers
    )
    fake_model_cls.assert_called_once_with(atom_config=fake_atom_config)
    assert result is fake_model


def test_prepare_model_selects_sglang_dict_for_deepseek_v2():
    """Verify that sglang engine uses _ATOM_SUPPORTED_MODELS (has DeepSeekV2)."""
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_model = MagicMock()
    fake_model_cls = MagicMock(return_value=fake_model)

    # DeepseekV2 is in SGLANG dict but not VLLM dict
    fake_register = _make_fake_register_module(
        model_dict={"DeepseekV2ForCausalLM": fake_model_cls}
    )
    fake_runtime = _make_fake_runtime_module()
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=fake_atom_config
    )

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
            "atom.plugin.sglang.runtime": fake_runtime,
            "atom.plugin.sglang.graph_capture_patch": MagicMock(
                apply_graph_capture_patch=MagicMock()
            ),
        },
    ):
        config = _Obj(architectures=["DeepseekV2ForCausalLM"])
        result = sglang_prepare.prepare_model(config=config)

    assert result is fake_model


def test_prepare_model_sets_framework_to_sglang():
    """Verify prepare_model sets the framework backbone to sglang."""
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_model_cls = MagicMock(return_value=MagicMock())

    fake_register = _make_fake_register_module(
        model_dict={"DeepseekV3ForCausalLM": fake_model_cls}
    )
    fake_runtime = _make_fake_runtime_module()
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=fake_atom_config
    )

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
            "atom.plugin.sglang.runtime": fake_runtime,
            "atom.plugin.sglang.graph_capture_patch": MagicMock(
                apply_graph_capture_patch=MagicMock()
            ),
        },
    ):
        config = _Obj(architectures=["DeepseekV3ForCausalLM"])
        sglang_prepare.prepare_model(config=config)

    assert plugin_runtime.is_sglang() is True
    assert plugin_runtime.is_plugin_mode() is True
