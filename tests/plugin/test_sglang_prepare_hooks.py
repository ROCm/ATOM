# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for SGLang prepare/register gating behavior."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from atom.plugin import prepare as plugin_runtime
from atom.plugin.sglang import prepare as sglang_prepare


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


def _make_fake_runtime_module(model_arch: str, prepare_config):
    module = ModuleType("atom.plugin.sglang.runtime")
    model_spec = _Obj(
        prepare_config=prepare_config,
        prepare_draft_model_config=None,
        construction_context=None,
    )
    module.resolve_model_arch_spec = MagicMock(return_value=(model_arch, model_spec))
    return module


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_runtime._set_framework_backbone("atom")
    yield
    plugin_runtime._set_framework_backbone("atom")


@pytest.mark.parametrize(
    ("model_arch", "register_atom_attention"),
    (
        ("Qwen3_5ForConditionalGeneration", False),
        ("Qwen3_5MoeForConditionalGeneration", False),
        ("Qwen3NextForCausalLM", False),
        ("Qwen3_5ForCausalLM", True),
        ("Qwen3_5MoeForCausalLM", True),
        ("DeepseekV3ForCausalLM", True),
        ("GlmMoeDsaForCausalLM", True),
        ("Qwen3MoeForCausalLM", True),
    ),
)
def test_prepare_model_register_ops_gate(
    model_arch: str, register_atom_attention: bool
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
    prepare_config = (
        fake_qwen35_mod.apply_prepare_model_adaptations
        if model_arch
        in {
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        }
        else None
    )
    fake_runtime_mod = _make_fake_runtime_module(model_arch, prepare_config)

    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
            "atom.plugin.sglang.runtime": fake_runtime_mod,
            "atom.plugin.sglang.models.qwen3_5": fake_qwen35_mod,
            "atom.plugin.sglang.dflash_lm_head_bridge": _module(
                "atom.plugin.sglang.dflash_lm_head_bridge",
                install_dflash_lm_head_patch=MagicMock(),
            ),
            "atom.plugin.sglang.patches.graph_capture_patch": MagicMock(
                apply_graph_capture_patch=MagicMock()
            ),
        },
    ):
        sglang_prepare.prepare_model(config=_Obj(architectures=[model_arch]))

    if register_atom_attention:
        fake_register.register_ops_to_sglang.assert_called_once_with(
            atom_config=fake_atom_config
        )
    else:
        fake_register.register_ops_to_sglang.assert_not_called()
    if model_arch in {
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        fake_qwen35_mod.apply_prepare_model_adaptations.assert_called_once_with(
            fake_atom_config, model_arch
        )
    else:
        fake_qwen35_mod.apply_prepare_model_adaptations.assert_not_called()
    fake_model_cls.assert_called_once()


def test_prepare_model_uses_canonical_family_architecture():
    source_arch = "FutureQwen35MoeForConditionalGeneration"
    canonical_arch = "Qwen3_5MoeForConditionalGeneration"
    fake_atom_config = _Obj(plugin_config=_Obj(is_plugin_mode=True))
    fake_register, _fake_model, fake_model_cls = _make_fake_register_module(
        canonical_arch
    )
    fake_config_mod = MagicMock()
    fake_config_mod.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=fake_atom_config
    )
    prepare_config = MagicMock()
    model_spec = _Obj(
        prepare_config=prepare_config,
        prepare_draft_model_config=None,
        construction_context=None,
    )
    fake_runtime_mod = ModuleType("atom.plugin.sglang.runtime")
    fake_runtime_mod.resolve_model_arch_spec = MagicMock(
        return_value=(canonical_arch, model_spec)
    )

    config = _Obj(architectures=[source_arch], model_type="qwen3_5_moe")
    with patch.dict(
        sys.modules,
        {
            "atom.plugin.register": fake_register,
            "atom.plugin.config": fake_config_mod,
            "atom.plugin.sglang.runtime": fake_runtime_mod,
            "atom.plugin.sglang.dflash_lm_head_bridge": _module(
                "atom.plugin.sglang.dflash_lm_head_bridge",
                install_dflash_lm_head_patch=MagicMock(),
            ),
            "atom.plugin.sglang.patches.graph_capture_patch": MagicMock(
                apply_graph_capture_patch=MagicMock()
            ),
        },
    ):
        sglang_prepare.prepare_model(config=config)

    fake_runtime_mod.resolve_model_arch_spec.assert_called_once_with(config)
    prepare_config.assert_called_once_with(fake_atom_config, canonical_arch)
    fake_register.register_ops_to_sglang.assert_not_called()
    fake_model_cls.assert_called_once()
