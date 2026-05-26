"""Tests for rtp-llm plugin registration."""

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def test_rtpllm_wrapper_registers_qwen35_moe_override():
    register_model_mock = MagicMock()

    fake_register_mod = ModuleType("rtp_llm.model_factory_register")
    fake_register_mod.register_model = register_model_mock
    fake_register_mod._model_factory = {}
    fake_register_mod._hf_architecture_2_ft = {}

    fake_atom_qwen_mod = ModuleType("atom.plugin.rtpllm.models.qwen3_5")

    class _FakeATOMQwen35Moe:
        pass

    fake_atom_qwen_mod.ATOMQwen35Moe = _FakeATOMQwen35Moe

    fake_modules = {
        "rtp_llm": _package("rtp_llm"),
        "rtp_llm.models": _package("rtp_llm.models"),
        "rtp_llm.model_factory_register": fake_register_mod,
        "atom.plugin.rtpllm.models.qwen3_5": fake_atom_qwen_mod,
    }

    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("atom.plugin.rtpllm.models.base_model_wrapper", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.base_model_wrapper")
        module = importlib.reload(module)

        assert fake_register_mod._model_factory["qwen35_moe"] is _FakeATOMQwen35Moe
        assert (
            fake_register_mod._hf_architecture_2_ft[
                "Qwen3_5MoeForConditionalGeneration"
            ]
            == "qwen35_moe"
        )
        register_model_mock.assert_called_with(
            "atom_qwen35_moe", _FakeATOMQwen35Moe, []
        )
