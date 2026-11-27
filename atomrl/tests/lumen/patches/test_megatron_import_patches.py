# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for Megatron IMPORT patch registration and installers."""

import sys
import types
from pathlib import Path

import pytest

if "atomrl.lumen" not in sys.modules:
    _lumen = types.ModuleType("atomrl.lumen")
    _lumen.__path__ = [
        str(Path(__file__).resolve().parents[2].parent / "atomrl" / "lumen")
    ]
    sys.modules["atomrl.lumen"] = _lumen

from atomrl.lumen.patches import PatchPhase, PatchRegistry


def _load_moe_fused_router():
    for key in (
        "atomrl.lumen.patches.runtime.moe_fused_router",
        "atomrl.lumen.patches.runtime",
    ):
        sys.modules.pop(key, None)
    from atomrl.lumen.patches.runtime import moe_fused_router

    return moe_fused_router


class TestMegatronImportPatches:
    def setup_method(self):
        PatchRegistry.clear()
        self.moe_fused_router = _load_moe_fused_router()

    def test_moe_fused_router_registered(self):
        spec = PatchRegistry.get("moe_fused_router")
        assert spec.phase is PatchPhase.IMPORT
        assert spec.tags == frozenset({"core", "moe", "megatron"})
        assert spec.default is True
        assert spec.fn is self.moe_fused_router.install_moe_fused_router

    def test_install_moe_fused_router_patches_moe_utils(self):
        from atomrl.lumen.ops.moe.fused_router import (
            fused_compute_score_for_moe_aux_loss,
            fused_moe_aux_loss,
            fused_topk_with_score_function,
        )

        moe_utils = types.ModuleType("megatron.core.transformer.moe.moe_utils")
        te_ext = types.ModuleType("megatron.core.extensions.transformer_engine")

        for name in (
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.extensions",
        ):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["megatron.core.transformer.moe.moe_utils"] = moe_utils
        sys.modules["megatron.core.transformer.moe"].moe_utils = moe_utils
        sys.modules["megatron.core.extensions.transformer_engine"] = te_ext
        sys.modules["megatron.core.extensions"].transformer_engine = te_ext

        self.moe_fused_router.install_moe_fused_router()

        assert (
            moe_utils.fused_topk_with_score_function is fused_topk_with_score_function
        )
        assert (
            moe_utils.fused_compute_score_for_moe_aux_loss
            is fused_compute_score_for_moe_aux_loss
        )
        assert moe_utils.fused_moe_aux_loss is fused_moe_aux_loss
        assert te_ext.fused_topk_with_score_function is fused_topk_with_score_function
        assert moe_utils._lumen_fused_router_patched is True

        self.moe_fused_router.install_moe_fused_router()

    def test_install_moe_fused_router_skips_without_megatron(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _block_moe_utils_import(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "megatron.core.transformer.moe.moe_utils":
                raise ImportError("megatron not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _block_moe_utils_import)
        self.moe_fused_router.install_moe_fused_router()

    def _stub_moe_utils(self):
        moe_utils = types.ModuleType("megatron.core.transformer.moe.moe_utils")
        for name in (
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
        ):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["megatron.core.transformer.moe.moe_utils"] = moe_utils
        sys.modules["megatron.core.transformer.moe"].moe_utils = moe_utils
        return moe_utils

    def _raise_on_router_import(self, monkeypatch, exc):
        import builtins

        real_import = builtins.__import__

        def _fake(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "atomrl.lumen.ops.moe.fused_router":
                raise exc
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake)

    def test_skips_when_its_own_ops_are_missing(self, monkeypatch):
        """The expected failure: the module is there, the three names are not.

        That is a plain ImportError rather than ModuleNotFoundError, and it has
        to be tolerated -- this is an IMPORT-phase patch with default=True, so a
        dense run would otherwise die on its way to GPTModel.
        """
        moe_utils = self._stub_moe_utils()
        self._raise_on_router_import(
            monkeypatch,
            ImportError(
                "cannot import name 'fused_moe_aux_loss'",
                name="atomrl.lumen.ops.moe.fused_router",
            ),
        )

        self.moe_fused_router.install_moe_fused_router()

        # Marked handled, so a later pass neither retries nor warns again.
        assert moe_utils._lumen_fused_router_patched is True

    def test_reraises_when_another_module_is_broken(self, monkeypatch):
        """A broken AITER must not read as "no fused router available"."""
        moe_utils = self._stub_moe_utils()
        self._raise_on_router_import(
            monkeypatch,
            ModuleNotFoundError(
                "No module named 'aiter.ops.triton.moe'",
                name="aiter.ops.triton.moe",
            ),
        )

        with pytest.raises(ModuleNotFoundError):
            self.moe_fused_router.install_moe_fused_router()

        assert not getattr(moe_utils, "_lumen_fused_router_patched", False)


def _load_megatron_import():
    for key in (
        "atomrl.lumen.patches.runtime.megatron_import",
        "atomrl.lumen.patches.runtime",
        "atomrl.lumen.models.megatron_patches",
    ):
        sys.modules.pop(key, None)
    from atomrl.lumen.patches.runtime import megatron_import

    return megatron_import


class TestMegatronImportModuleLocation:
    def setup_method(self):
        PatchRegistry.clear()

    def test_megatron_import_registers_core_patches(self):
        _load_megatron_import()
        names = {
            spec.name for spec in PatchRegistry.all() if spec.phase is PatchPhase.IMPORT
        }
        assert "fused_layer_norm" in names
        assert "mmap_checkpoint" in names

    def test_mmap_checkpoint_does_not_replace_global_torch_load(self, monkeypatch):
        import io

        import torch

        megatron_import = _load_megatron_import()

        checkpointing = types.ModuleType("megatron.training.checkpointing")
        common = types.ModuleType("megatron.core.dist_checkpointing.strategies.common")
        checkpointing.torch = torch
        common.torch = torch
        for name in (
            "megatron",
            "megatron.training",
            "megatron.core",
            "megatron.core.dist_checkpointing",
            "megatron.core.dist_checkpointing.strategies",
        ):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(
            sys.modules, "megatron.training.checkpointing", checkpointing
        )
        monkeypatch.setitem(
            sys.modules,
            "megatron.core.dist_checkpointing.strategies.common",
            common,
        )

        calls = []

        def fake_load(*args, **kwargs):
            calls.append((args, kwargs))
            return "loaded"

        monkeypatch.setattr(torch, "load", fake_load)
        megatron_import.install_mmap_checkpoint()

        assert torch.load is fake_load
        assert checkpointing.torch is not torch
        assert common.torch is not torch
        assert checkpointing.torch.load("checkpoint.pt") == "loaded"
        assert calls[-1][1]["mmap"] is True
        assert common.torch.load(io.BytesIO(b"checkpoint")) == "loaded"
        assert "mmap" not in calls[-1][1]
