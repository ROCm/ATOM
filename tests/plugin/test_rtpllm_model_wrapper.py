"""Tests for rtp-llm plugin registration."""

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch


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
    fake_atom_glm_mod = ModuleType("atom.plugin.rtpllm.models.glm5")

    class _FakeATOMGlm5Moe:
        pass

    fake_atom_glm_mod.ATOMGlm5Moe = _FakeATOMGlm5Moe
    fake_atom_qwen4_mod = ModuleType("atom.plugin.rtpllm.models.qwen4_exp")

    class _FakeATOMQwen4Exp:
        pass

    fake_atom_qwen4_mod.ATOMQwen4Exp = _FakeATOMQwen4Exp
    fake_dsv2_mod = ModuleType("atom.models.deepseek_v2")
    fake_dsv2_mod.GlmMoeDsaForCausalLM = type("GlmMoeDsaForCausalLM", (), {})
    fake_native_qwen4_mod = ModuleType("atom.models.qwen4_exp")
    fake_native_qwen4_mod.Qwen4ExpForConditionalGeneration = type(
        "Qwen4ExpForConditionalGeneration", (), {}
    )
    fake_plugin_register_mod = ModuleType("atom.plugin.register")
    fake_plugin_register_mod._ATOM_SUPPORTED_MODELS = {}

    fake_modules = {
        "rtp_llm": _package("rtp_llm"),
        "rtp_llm.models": _package("rtp_llm.models"),
        "rtp_llm.model_factory_register": fake_register_mod,
        "atom.plugin.rtpllm.models.qwen3_5": fake_atom_qwen_mod,
        "atom.plugin.rtpllm.models.glm5": fake_atom_glm_mod,
        "atom.plugin.rtpllm.models.qwen4_exp": fake_atom_qwen4_mod,
        "atom.models.deepseek_v2": fake_dsv2_mod,
        "atom.models.qwen4_exp": fake_native_qwen4_mod,
        "atom.plugin.register": fake_plugin_register_mod,
    }

    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("atom.plugin.rtpllm.models", None)
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
        assert fake_register_mod._model_factory["qwen4_exp"] is _FakeATOMQwen4Exp
        assert (
            fake_register_mod._hf_architecture_2_ft[
                "Qwen4ExpForConditionalGeneration"
            ]
            == "qwen4_exp"
        )
        register_model_mock.assert_has_calls(
            [
                call("atom_qwen35_moe", _FakeATOMQwen35Moe, []),
                call("atom_glm5_moe", _FakeATOMGlm5Moe, []),
                call("atom_qwen4_exp", _FakeATOMQwen4Exp, []),
            ],
            any_order=False,
        )
