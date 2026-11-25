# SPDX-License-Identifier: MIT
"""The Triton ``next_power_of_2`` restore: DeepSeek-V4.1 only.

SGLang replaces Triton's constexpr ``next_power_of_2`` with a plain host
function at import. V4.1's mHC kernel needs the constexpr one; every other
model keeps SGLang's, whose host call is about ten times cheaper.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

triton = pytest.importorskip("triton")
pytest.importorskip("sglang")

import sglang.srt.utils.common  # noqa: F401 - SGLang's setattr runs here
from triton.runtime.jit import ConstexprFunction

from atom.plugin.sglang import prepare as sglang_prepare
from atom.plugin.sglang.patches import (
    triton_constexpr_helpers_patch as constexpr_patch,
)


@pytest.fixture
def sglang_next_power_of_2(monkeypatch):
    """SGLang's host function, restored after the test whatever it installs."""
    current = triton.next_power_of_2
    assert not isinstance(current, ConstexprFunction)
    monkeypatch.setattr(triton, "next_power_of_2", current)
    return current


def test_patch_installs_the_constexpr_function_once(sglang_next_power_of_2):
    assert constexpr_patch.apply_triton_constexpr_helpers_patch()
    restored = triton.next_power_of_2
    assert isinstance(restored, ConstexprFunction)
    # SGLang's host semantics: 1 for n <= 0; Triton's for n > 0.
    assert [restored(n) for n in (-3, 0, 1, 2, 3, 1000, 1024)] == [
        sglang_next_power_of_2(n) for n in (-3, 0, 1, 2, 3, 1000, 1024)
    ]
    assert not constexpr_patch.apply_triton_constexpr_helpers_patch()
    assert triton.next_power_of_2 is restored


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _prepare(model_arch: str, extra_modules=None):
    """`prepare_model` for ``model_arch`` with everything past the
    architecture branch faked out."""
    fake_model_cls = MagicMock(return_value=MagicMock(atom_config=None))
    fake_register = MagicMock(_ATOM_SUPPORTED_MODELS={model_arch: fake_model_cls})
    fake_config = MagicMock()
    fake_config.generate_atom_config_for_plugin_mode = MagicMock(
        return_value=_Obj(plugin_config=_Obj(is_plugin_mode=True))
    )
    spec = _Obj(
        prepare_config=MagicMock(),
        prepare_draft_model_config=None,
        construction_context=None,
    )
    fake_runtime = ModuleType("atom.plugin.sglang.runtime")
    fake_runtime.resolve_model_arch_spec = MagicMock(return_value=(model_arch, spec))
    modules = {
        "atom.plugin.register": fake_register,
        "atom.plugin.config": fake_config,
        "atom.plugin.sglang.runtime": fake_runtime,
        "atom.plugin.sglang.patches.graph_capture_patch": MagicMock(),
        **(extra_modules or {}),
    }
    with patch.dict(sys.modules, modules):
        sglang_prepare.prepare_model(config=_Obj(architectures=[model_arch]))


@pytest.mark.parametrize(
    "model_arch",
    ["Qwen3MoeForCausalLM", "GlmMoeDsaForCausalLM", "DeepseekV3ForCausalLM"],
)
def test_other_models_keep_sglangs_function(sglang_next_power_of_2, model_arch):
    _prepare(model_arch)
    assert triton.next_power_of_2 is sglang_next_power_of_2


def test_deepseek_v41_installs_the_constexpr_function(sglang_next_power_of_2):
    bridge = ModuleType("atom.plugin.sglang.deepseek_v41_bridge")
    bridge.install_deepseek_v41_proxy_pool_patch = MagicMock()
    _prepare(
        "DeepseekV41ForCausalLM",
        {"atom.plugin.sglang.deepseek_v41_bridge": bridge},
    )
    assert isinstance(triton.next_power_of_2, ConstexprFunction)
    bridge.install_deepseek_v41_proxy_pool_patch.assert_called_once_with()
