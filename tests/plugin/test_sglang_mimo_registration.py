"""MiMo plugin registration must not advertise an unadapted vLLM path."""

import subprocess
import sys

import pytest

pytest.importorskip("sglang")
pytest.importorskip("aiter")


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_mimo_registration_is_scoped_to_sglang(framework):
    # Model registration is process-global; import it in a fresh process so
    # another test's framework selection cannot hide an import-order failure.
    code = f"""
from atom.plugin.prepare import _set_framework_backbone
_set_framework_backbone({framework!r})
from atom.plugin.register import _ATOM_SUPPORTED_MODELS
expected = {framework == 'sglang'!r}
assert ('MiMoV2ForCausalLM' in _ATOM_SUPPORTED_MODELS) == expected
assert ('MiMoV2MTP' in _ATOM_SUPPORTED_MODELS) == expected
assert 'Qwen3ForCausalLM' in _ATOM_SUPPORTED_MODELS
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mimo_pool_padding_is_scoped_to_plugin_models():
    code = """
from types import SimpleNamespace
from sglang.srt.model_executor.model_runner import ModelRunner
from atom.plugin.sglang import register

def allocate(runner):
    config = runner.model_config
    return config.v_head_dim, config.swa_v_head_dim

ModelRunner.alloc_memory_pool = allocate
ModelRunner._atom_mimo_v2_pool_symmetry_patch = False
register._is_atom_external_model_enabled = lambda: enabled
register._install_mimo_v2_pool_symmetry_patch()
wrapped = ModelRunner.alloc_memory_pool
register._install_mimo_v2_pool_symmetry_patch()
assert ModelRunner.alloc_memory_pool is wrapped

for enabled in (False, True):
    for arch in ('MiMoV2ForCausalLM', 'MiMoV2MTP', 'Qwen3ForCausalLM'):
        hf = SimpleNamespace(architectures=[arch], v_head_dim=128, swa_v_head_dim=128)
        config = SimpleNamespace(
            hf_config=hf, head_dim=192, v_head_dim=128,
            swa_head_dim=192, swa_v_head_dim=128,
        )
        dims = ModelRunner.alloc_memory_pool(SimpleNamespace(model_config=config))
        expected = (192, 192) if enabled and arch.startswith('MiMo') else (128, 128)
        assert dims == expected, (enabled, arch, dims)
        assert (hf.v_head_dim, hf.swa_v_head_dim) == (128, 128)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
