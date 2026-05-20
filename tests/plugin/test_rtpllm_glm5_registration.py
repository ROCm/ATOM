"""Tests for GLM5 rtp-llm plugin registration."""

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def test_rtpllm_wrapper_registers_glm5_override_and_alias():
    register_model_mock = MagicMock()

    fake_rtp_register_mod = ModuleType("rtp_llm.model_factory_register")
    fake_rtp_register_mod.register_model = register_model_mock
    fake_rtp_register_mod._model_factory = {}
    fake_rtp_register_mod._hf_architecture_2_ft = {}

    fake_atom_register_mod = ModuleType("atom.plugin.register")
    fake_atom_register_mod._ATOM_SUPPORTED_MODELS = {}

    fake_atom_deepseek_mod = ModuleType("atom.models.deepseek_v2")

    class _FakeGlmMoeDsaForCausalLM:
        pass

    fake_atom_deepseek_mod.GlmMoeDsaForCausalLM = _FakeGlmMoeDsaForCausalLM

    fake_atom_qwen_mod = ModuleType("atom.plugin.rtpllm.models.qwen3_5")

    class _FakeATOMQwen35Moe:
        pass

    fake_atom_qwen_mod.ATOMQwen35Moe = _FakeATOMQwen35Moe

    fake_atom_glm_mod = ModuleType("atom.plugin.rtpllm.models.glm5")

    class _FakeATOMGlm5Moe:
        pass

    fake_atom_glm_mod.ATOMGlm5Moe = _FakeATOMGlm5Moe

    fake_modules = {
        "rtp_llm": _package("rtp_llm"),
        "rtp_llm.models": _package("rtp_llm.models"),
        "rtp_llm.model_factory_register": fake_rtp_register_mod,
        "atom.models.deepseek_v2": fake_atom_deepseek_mod,
        "atom.plugin.register": fake_atom_register_mod,
        "atom.plugin.rtpllm.models.qwen3_5": fake_atom_qwen_mod,
        "atom.plugin.rtpllm.models.glm5": fake_atom_glm_mod,
    }

    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("atom.plugin.rtpllm.models", None)
        sys.modules.pop("atom.plugin.rtpllm.models.base_model_wrapper", None)
        importlib.import_module("atom.plugin.rtpllm.models")

        assert fake_rtp_register_mod._model_factory["glm_5"] is _FakeATOMGlm5Moe
        assert (
            fake_rtp_register_mod._hf_architecture_2_ft["GlmMoeDsaForCausalLM"]
            == "glm_5"
        )
        assert (
            fake_atom_register_mod._ATOM_SUPPORTED_MODELS["GlmMoeDsaForCausalLM"]
            is _FakeGlmMoeDsaForCausalLM
        )
        register_model_mock.assert_has_calls(
            [
                call("atom_qwen35_moe", _FakeATOMQwen35Moe, []),
                call("atom_glm5_moe", _FakeATOMGlm5Moe, []),
            ],
            any_order=False,
        )

