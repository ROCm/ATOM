"""Tests for prepare_model orchestration in rtpllm plugin mode."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from atom.plugin import prepare as plugin_prepare


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


def test_prepare_model_rtpllm_happy_path():
    fake_quant_config = _Obj(
        exclude_layers=[],
        remap_layer_name=MagicMock(),
    )
    fake_atom_config = _Obj(
        hf_config=_Obj(architectures=["Qwen3_5MoeForConditionalGeneration"]),
        plugin_config=_Obj(is_plugin_mode=True),
        quant_config=fake_quant_config,
    )
    fake_model = MagicMock(name="FakeQwen35Moe")
    fake_model_cls = MagicMock(return_value=fake_model)

    fake_register = MagicMock()
    fake_register._ATOM_SUPPORTED_MODELS = {
        "Qwen3_5MoeForConditionalGeneration": fake_model_cls
    }
    fake_register.register_ops_to_sglang = MagicMock()
    fake_register.init_aiter_dist = MagicMock()
    fake_register.set_attn_cls = MagicMock()

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
        result = plugin_prepare.prepare_model(
            config=_Obj(model_config=_Obj()), engine="rtpllm"
        )

    fake_config_mod.generate_atom_config_for_plugin_mode.assert_called_once()
    fake_register.register_ops_to_sglang.assert_not_called()
    fake_register.set_attn_cls.assert_called_once()
    fake_register.init_aiter_dist.assert_called_once_with(config=fake_atom_config)
    fake_quant_config.remap_layer_name.assert_called_once()
    fake_model_cls.assert_called_once_with(atom_config=fake_atom_config)
    assert result is fake_model
