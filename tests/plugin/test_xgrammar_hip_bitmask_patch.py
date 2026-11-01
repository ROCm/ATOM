"""Unit tests for the HIP xgrammar vocab-mask stopgap.

Plugin tests stay under ``tests/plugin/``. Pre Checkin ignores that tree.
"""

from __future__ import annotations

import sys
from types import ModuleType

import torch

from atom.plugin.sglang.patches import xgrammar_hip_bitmask_patch as patch
from atom.plugin.sglang.patches.xgrammar_hip_bitmask_patch import (
    _apply_bitmask_torch_gpu,
    _embedded_gfx_archs_from_blob,
    _should_replace_native_bitmask,
    apply_xgrammar_hip_bitmask_patch,
)


def test_apply_bitmask_torch_gpu_masks_disallowed_tokens():
    logits = torch.zeros(2, 64, dtype=torch.float32)
    bitmask = torch.zeros(2, 2, dtype=torch.int32)
    # Allow token 0 (bit 0 of word 0) and token 33 (bit 1 of word 1).
    bitmask[:, 0] = 1 << 0
    bitmask[:, 1] = 1 << 1

    _apply_bitmask_torch_gpu(logits, bitmask)

    assert logits[0, 0] == 0
    assert logits[0, 33] == 0
    assert torch.isneginf(logits[0, 1])
    assert torch.isneginf(logits[0, 32])
    assert torch.isneginf(logits[1, 2])


def test_apply_bitmask_torch_gpu_with_indices():
    logits = torch.zeros(3, 32, dtype=torch.float32)
    bitmask = torch.zeros(3, 1, dtype=torch.int32)
    bitmask[1, 0] = 1 << 3  # only token 3 allowed on row 1

    _apply_bitmask_torch_gpu(logits, bitmask, indices=[1])

    assert logits[1, 3] == 0
    assert torch.isneginf(logits[1, 0])
    # Untouched rows stay zero.
    assert torch.all(logits[0] == 0)
    assert torch.all(logits[2] == 0)


def test_apply_bitmask_torch_gpu_masks_logits_beyond_bitmask():
    # logits padded past bitmask coverage must be -inf (not left finite).
    logits = torch.zeros(1, 40, dtype=torch.float32)
    bitmask = torch.zeros(1, 1, dtype=torch.int32)
    bitmask[0, 0] = 1 << 0  # only token 0 allowed in the first 32

    _apply_bitmask_torch_gpu(logits, bitmask)

    assert logits[0, 0] == 0
    assert torch.isneginf(logits[0, 1])
    assert torch.isneginf(logits[0, 32])
    assert torch.isneginf(logits[0, 39])


def test_embedded_arch_scan_and_replace_gate():
    blob = b"prefix amdgcn-amd-amdhsa--gfx950 suffix amdgcn-amd-amdhsa--gfx950"
    assert _embedded_gfx_archs_from_blob(blob) == {"gfx950"}
    assert _should_replace_native_bitmask("gfx942:sramecc+:xnack-", {"gfx950"})
    assert _should_replace_native_bitmask("gfx950", {"gfx950"}) is False
    assert _should_replace_native_bitmask("gfx942", set()) is False
    assert _should_replace_native_bitmask("", {"gfx950"}) is False


def _fake_xgrammar():
    module = ModuleType("sglang.srt.constrained.xgrammar_backend")

    def native_bitmask(*_args, **_kwargs):
        raise AssertionError("native bitmask must stay bound when not replaced")

    module.apply_token_bitmask_inplace_cuda = native_bitmask
    module._atom_hip_bitmask_patched = False
    return module


def _install_sglang_stubs(monkeypatch, xgrammar):
    def package(name):
        mod = ModuleType(name)
        mod.__path__ = []
        mod.__package__ = name
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    package("sglang")
    package("sglang.srt")
    utils = package("sglang.srt.utils")
    utils.is_hip = lambda: True
    constrained = package("sglang.srt.constrained")
    constrained.xgrammar_backend = xgrammar
    monkeypatch.setitem(
        sys.modules, "sglang.srt.constrained.xgrammar_backend", xgrammar
    )


def test_patch_installs_only_when_device_arch_is_missing(monkeypatch):
    monkeypatch.delenv("ATOM_SGLANG_XGRAMMAR_HIP_TORCH_BITMASK", raising=False)
    xgrammar = _fake_xgrammar()
    native = xgrammar.apply_token_bitmask_inplace_cuda
    _install_sglang_stubs(monkeypatch, xgrammar)
    monkeypatch.setattr(patch, "_device_lacks_sgl_kernel_code_object", lambda: True)

    apply_xgrammar_hip_bitmask_patch()

    assert xgrammar._atom_hip_bitmask_patched is True
    assert xgrammar.apply_token_bitmask_inplace_cuda is patch._apply_bitmask_hip_safe
    assert xgrammar.apply_token_bitmask_inplace_cuda is not native


def test_patch_keeps_native_kernel_when_code_object_exists(monkeypatch):
    monkeypatch.setenv("ATOM_SGLANG_XGRAMMAR_HIP_TORCH_BITMASK", "1")
    xgrammar = _fake_xgrammar()
    native = xgrammar.apply_token_bitmask_inplace_cuda
    _install_sglang_stubs(monkeypatch, xgrammar)
    monkeypatch.setattr(patch, "_device_lacks_sgl_kernel_code_object", lambda: False)

    apply_xgrammar_hip_bitmask_patch()

    assert xgrammar.apply_token_bitmask_inplace_cuda is native
    assert xgrammar._atom_hip_bitmask_patched is False


def test_patch_skips_when_cuda_symbol_is_absent(monkeypatch):
    monkeypatch.setenv("ATOM_SGLANG_XGRAMMAR_HIP_TORCH_BITMASK", "1")
    xgrammar = _fake_xgrammar()
    del xgrammar.apply_token_bitmask_inplace_cuda
    _install_sglang_stubs(monkeypatch, xgrammar)
    monkeypatch.setattr(
        patch,
        "_device_lacks_sgl_kernel_code_object",
        lambda: (_ for _ in ()).throw(AssertionError("arch check must not run")),
    )

    apply_xgrammar_hip_bitmask_patch()

    assert xgrammar._atom_hip_bitmask_patched is False


def test_patch_skips_when_env_disables_it(monkeypatch):
    monkeypatch.setenv("ATOM_SGLANG_XGRAMMAR_HIP_TORCH_BITMASK", "0")
    monkeypatch.setattr(
        patch,
        "_device_lacks_sgl_kernel_code_object",
        lambda: (_ for _ in ()).throw(AssertionError("arch check must not run")),
    )

    apply_xgrammar_hip_bitmask_patch()
