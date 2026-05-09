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

from atom.plugin import prepare as plugin_prepare


class _Obj:
    """Minimal attribute bag for faking nested configs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


def _make_fake_register_module(model_dict=None):
    """Create a fake atom.plugin.register module with controllable model dicts."""
    mod = MagicMock()
    mod._ATOM_SUPPORTED_MODELS = model_dict or {}
    mod.register_ops_to_sglang = MagicMock()
    mod.init_aiter_dist = MagicMock()
    mod.set_attn_cls = MagicMock()
    return mod


# ---------------------------------------------------------------------------
# Engine / architecture validation
# ---------------------------------------------------------------------------


def test_prepare_model_rejects_unsupported_engine():
    """Unsupported engine should raise ValueError from _set_framework_backbone."""
    config = _Obj(architectures=["SomeModel"])
    with pytest.raises(ValueError, match="Unsupported framework"):
        plugin_prepare.prepare_model(config=config, engine="tensorflow")


def test_prepare_model_rejects_non_sglang_engine_gracefully():
    """vllm engine currently not supported in prepare_model (only sglang path)."""
    config = _Obj(architectures=["Qwen3ForCausalLM"])
    with pytest.raises(ValueError, match="does not support engine"):
        plugin_prepare.prepare_model(config=config, engine="vllm")


def test_prepare_model_rejects_unsupported_architecture():
    """Known engine but unknown arch should raise ValueError."""
    fake_register = _make_fake_register_module(
        model_dict={"DeepseekV3ForCausalLM": MagicMock()}
    )

    with patch.dict(sys.modules, {"atom.plugin.register": fake_register}):
        config = _Obj(architectures=["TotallyFakeModelArch"])
        with pytest.raises(ValueError, match="does not support"):
            plugin_prepare.prepare_model(config=config, engine="sglang")


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

    mock_gen_config = MagicMock(return_value=fake_atom_config)
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = mock_gen_config

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
        },
    ):
        config = _Obj(architectures=["DeepseekV3ForCausalLM"])
        result = plugin_prepare.prepare_model(config=config, engine="sglang")

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


def test_prepare_model_selects_sglang_dict_for_deepseek_v2():
    """Verify that sglang engine uses _ATOM_SUPPORTED_MODELS (has DeepSeekV2)."""
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_model = MagicMock()
    fake_model_cls = MagicMock(return_value=fake_model)

    # DeepseekV2 is in SGLANG dict but not VLLM dict
    fake_register = _make_fake_register_module(
        model_dict={"DeepseekV2ForCausalLM": fake_model_cls}
    )
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=fake_atom_config
    )

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
        },
    ):
        config = _Obj(architectures=["DeepseekV2ForCausalLM"])
        result = plugin_prepare.prepare_model(config=config, engine="sglang")

    assert result is fake_model


def test_prepare_model_sets_framework_to_sglang():
    """Verify prepare_model sets the framework backbone to sglang."""
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_model_cls = MagicMock(return_value=MagicMock())

    fake_register = _make_fake_register_module(
        model_dict={"DeepseekV3ForCausalLM": fake_model_cls}
    )
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=fake_atom_config
    )

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
        },
    ):
        config = _Obj(architectures=["DeepseekV3ForCausalLM"])
        plugin_prepare.prepare_model(config=config, engine="sglang")

    assert plugin_prepare.is_sglang() is True
    assert plugin_prepare.is_plugin_mode() is True
