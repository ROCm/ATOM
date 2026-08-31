"""Tests for Qwen3.5 plugin patch scope helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from atom.plugin.vllm.qwen35_plugin_scope import (
    is_qwen35_35b_a3b_vllm_plugin_model,
    is_qwen35_vllm_plugin_model,
)


def _atom_config(*, hidden: int, layers: int, experts: int):
    text_config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_experts=experts,
    )
    hf_config = SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        model_type="qwen3_5_moe",
        text_config=text_config,
    )
    plugin_config = SimpleNamespace(is_vllm=True)
    return SimpleNamespace(hf_config=hf_config, plugin_config=plugin_config)


def test_scope_matches_35b_a3b():
    cfg = _atom_config(hidden=2048, layers=40, experts=256)
    with patch(
        "atom.plugin.vllm.qwen35_plugin_scope.get_current_atom_config", return_value=cfg
    ):
        assert is_qwen35_vllm_plugin_model()
        assert is_qwen35_35b_a3b_vllm_plugin_model()


def test_scope_excludes_397b():
    cfg = _atom_config(hidden=4096, layers=60, experts=512)
    with patch(
        "atom.plugin.vllm.qwen35_plugin_scope.get_current_atom_config", return_value=cfg
    ):
        assert is_qwen35_vllm_plugin_model()
        assert not is_qwen35_35b_a3b_vllm_plugin_model()
