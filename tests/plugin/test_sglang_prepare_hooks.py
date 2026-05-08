# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for SGLang prepare/register gating behavior."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from atom.plugin import prepare as plugin_prepare


class _Obj:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_fake_register_module(model_arch: str):
    fake_model = MagicMock()
    fake_model.atom_config = None
    fake_model_cls = MagicMock(return_value=fake_model)
    fake_register = MagicMock()
    fake_register._ATOM_SUPPORTED_MODELS = {model_arch: fake_model_cls}
    fake_register._SGLANG_NATIVE_ATTN_MODEL_ARCHS = {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
    fake_register.register_ops_to_sglang = MagicMock()
    fake_register.init_aiter_dist = MagicMock()
    fake_register.set_attn_cls = MagicMock()
    return fake_register, fake_model, fake_model_cls


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


@pytest.mark.parametrize(
    "model_arch,expect_register_ops",
    (
        ("Qwen3_5ForConditionalGeneration", False),
        ("Qwen3_5MoeForConditionalGeneration", False),
        ("Qwen3NextForCausalLM", False),
        ("DeepseekV3ForCausalLM", True),
        ("Qwen3MoeForCausalLM", True),
    ),
)
def test_prepare_model_register_ops_gate(
    model_arch: str,
    expect_register_ops: bool,
):
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_register, _fake_model, fake_model_cls = _make_fake_register_module(model_arch)
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=fake_atom_config
    )
    fake_qwen35_mod = _module(
        "atom.plugin.sglang.models.qwen3_5",
        apply_prepare_model_adaptations=MagicMock(),
    )

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
            "atom.plugin.sglang.models.qwen3_5": fake_qwen35_mod,
            "atom.plugin.sglang.graph_capture_patch": MagicMock(
                apply_graph_capture_patch=MagicMock()
            ),
        },
    ):
        plugin_prepare.prepare_model(
            config=_Obj(architectures=[model_arch]),
            engine="sglang",
        )

    if expect_register_ops:
        fake_register.register_ops_to_sglang.assert_called_once_with(
            atom_config=fake_atom_config
        )
    else:
        fake_register.register_ops_to_sglang.assert_not_called()
    if model_arch in {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        fake_qwen35_mod.apply_prepare_model_adaptations.assert_called_once_with(
            fake_atom_config, model_arch
        )
    else:
        fake_qwen35_mod.apply_prepare_model_adaptations.assert_not_called()
    fake_model_cls.assert_called_once()
