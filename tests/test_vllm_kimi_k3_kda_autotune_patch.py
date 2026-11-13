# SPDX-License-Identifier: MIT
"""The Kimi-K3 KDA autotune patch drops non-argument keys, and only for K3.

vLLM's Kimi-K3 KDA kernel keys its autotune on `NC`, which is not an argument,
and Triton 3.8 rejects that when the decorator runs (vllm-project/vllm#60446).
The patch wraps `triton.autotune`; these check what the wrapper hands Triton,
with real Triton and no vLLM or GPU. The patch module is loaded from its file
so the `atom.plugin.vllm` package, which imports vLLM, is not.
"""

import importlib.util
import pathlib

import pytest

triton = pytest.importorskip("triton")

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "atom/plugin/vllm/kimi_k3_kda_autotune_patch.py"

KERNELS = """
import triton
import triton.language as tl


@triton.jit
def kernel(x, BK: tl.constexpr):
    pass
"""


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patched(monkeypatch):
    # Restores triton.autotune at teardown, whatever the patch installed.
    monkeypatch.setattr(triton, "autotune", triton.autotune)
    module = _load(PATCH, "atom_kimi_k3_kda_autotune_patch_under_test")
    assert module.apply_vllm_kimi_k3_kda_autotune_patch()
    assert not module.apply_vllm_kimi_k3_kda_autotune_patch()
    return module


def _kernel(tmp_path, module_name):
    # Triton reads the kernel's source, so it has to live in a file.
    source = tmp_path / "kernels.py"
    source.write_text(KERNELS)
    return _load(source, module_name).kernel


def test_drops_the_non_argument_key_of_a_kimi_k3_kernel(patched, tmp_path):
    kernel = _kernel(tmp_path, "vllm.models.kimi_k3.amd.ops.third_party.kda.probe")
    tuned = triton.autotune(configs=[triton.Config({})], key=["BK", "NC"])(kernel)
    assert list(tuned.keys) == ["BK"]


def test_passes_every_other_kernel_to_triton_unchanged(patched, tmp_path):
    kernel = _kernel(tmp_path, "vllm.models.other.probe")
    decorate = triton.autotune(configs=[triton.Config({})], key=["BK", "NC"])
    try:
        tuned = decorate(kernel)
    except ValueError as exc:  # Triton >= 3.8 checks keys itself
        assert "NC" in str(exc)
    else:
        assert list(tuned.keys) == ["BK", "NC"]
