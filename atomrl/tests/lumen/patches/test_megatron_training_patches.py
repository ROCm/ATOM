# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for Megatron TRAINING-phase patch registration and apply order."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if "atomrl.lumen" not in sys.modules:
    _lumen = types.ModuleType("atomrl.lumen")
    _lumen.__path__ = [
        str(Path(__file__).resolve().parents[2].parent / "atomrl" / "lumen")
    ]
    sys.modules["atomrl.lumen"] = _lumen

from atomrl.lumen.patches import PatchPhase, PatchRegistry, apply_training_patches
from atomrl.lumen.patches.registry import _topological_sort
from atomrl.lumen.patches.types import PatchSpec


def _load_training_hooks():
    for key in (
        "atomrl.lumen.patches.training.megatron_hooks",
        "atomrl.lumen.patches.training",
    ):
        sys.modules.pop(key, None)
    from atomrl.lumen.patches.training import megatron_hooks

    return megatron_hooks


class TestMegatronTrainingPatches:
    def setup_method(self):
        PatchRegistry.clear()
        self.hooks = _load_training_hooks()

    def test_all_training_hooks_registered(self):
        expected = {
            "fp8_param_gather_hook",
            "mxfp4_weight_cache_hook",
            "fp8_param_storage_hook",
            "gc_freeze_hook",
            "hip_graphs_hook",
            "val_loss_early_stop_hook",
        }
        registered = {
            spec.name
            for spec in PatchRegistry.all()
            if spec.phase is PatchPhase.TRAINING
        }
        assert registered == expected
        for name in expected:
            spec = PatchRegistry.get(name)
            assert spec.default is False
            assert "megatron" in spec.tags
            assert "training" in spec.tags

    def test_dependency_order_for_llama2_bundle(self):
        specs = [
            PatchRegistry.get(name)
            for name in (
                "fp8_param_gather_hook",
                "fp8_param_storage_hook",
                "hip_graphs_hook",
                "val_loss_early_stop_hook",
            )
        ]
        ordered = [spec.name for spec in _topological_sort(specs)]
        assert ordered.index("fp8_param_gather_hook") < ordered.index(
            "fp8_param_storage_hook"
        )
        assert ordered.index("fp8_param_storage_hook") < ordered.index(
            "hip_graphs_hook"
        )

    def test_storage_depends_on_gather(self):
        spec = PatchRegistry.get("fp8_param_storage_hook")
        assert spec.depends_on == ("fp8_param_gather_hook",)

    def test_hip_graphs_depends_on_fp8_hooks(self):
        spec = PatchRegistry.get("hip_graphs_hook")
        assert spec.depends_on == ("fp8_param_gather_hook", "fp8_param_storage_hook")

    def test_apply_training_patches_invokes_installers(self):
        calls = []

        def _stub(patch_name, label):
            spec = PatchRegistry.get(patch_name)

            def _installer():
                calls.append(label)

            PatchRegistry._patches[patch_name] = PatchSpec(
                name=spec.name,
                phase=spec.phase,
                fn=_installer,
                description=spec.description,
                enabled=spec.enabled,
                depends_on=spec.depends_on,
                tags=spec.tags,
                config_fields=spec.config_fields,
                default=spec.default,
            )

        _stub("fp8_param_gather_hook", "install_fp8_param_gather_hook")
        _stub("fp8_param_storage_hook", "install_fp8_param_storage_hook")
        _stub("hip_graphs_hook", "install_hip_graphs_hook")
        _stub("val_loss_early_stop_hook", "install_val_loss_early_stop_hook")

        apply_training_patches(
            names={
                "fp8_param_gather_hook",
                "fp8_param_storage_hook",
                "hip_graphs_hook",
                "val_loss_early_stop_hook",
            }
        )

        assert calls.index("install_fp8_param_gather_hook") < calls.index(
            "install_fp8_param_storage_hook"
        )
        assert calls.index("install_fp8_param_storage_hook") < calls.index(
            "install_hip_graphs_hook"
        )
        assert "install_val_loss_early_stop_hook" in calls

    def test_val_loss_hook_skips_without_megatron(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _block_training_import(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "megatron.training.training":
                raise ImportError("megatron not installed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _block_training_import)
        self.hooks.install_val_loss_early_stop_hook()

    def _stub_megatron_training(self, monkeypatch, *, setup=None, get_args=None):
        """Point TRAINING installers at in-process stubs, not a real Megatron."""
        training_mod = types.ModuleType("megatron.training.training")
        training_mod.setup_model_and_optimizer = setup or (
            lambda *a, **k: ("model", "optimizer", "scheduler")
        )
        training_mod.train_step = lambda *a, **k: None

        training_pkg = types.ModuleType("megatron.training")
        training_pkg.get_args = get_args or (lambda: SimpleNamespace())
        training_pkg.print_rank_0 = lambda *a, **k: None
        training_pkg.training = training_mod

        megatron_pkg = types.ModuleType("megatron")
        megatron_pkg.training = training_pkg

        monkeypatch.setitem(sys.modules, "megatron", megatron_pkg)
        monkeypatch.setitem(sys.modules, "megatron.training", training_pkg)
        monkeypatch.setitem(sys.modules, "megatron.training.training", training_mod)
        return training_mod

    def _stub_mxfp4_hook_deps(self, monkeypatch, register):
        """Avoid importing real atomrl.lumen.quantize / megatron.py (AITER, ROCm)."""
        megatron_compat = types.ModuleType("atomrl.lumen.models.megatron")
        megatron_compat.resolve_quant_format = lambda args: (
            "mxfp4"
            if getattr(args, "linear_fp4", False)
            else getattr(args, "linear_fp8_format", None)
        )
        quant = types.ModuleType("atomrl.lumen.quantize")
        quant.register_mxfp4_weight_optimizer_hooks = register
        monkeypatch.setitem(
            sys.modules, "atomrl.lumen.models.megatron", megatron_compat
        )
        monkeypatch.setitem(sys.modules, "atomrl.lumen.quantize", quant)

    def test_mxfp4_weight_cache_hook_registers_on_mxfp4_run(self, monkeypatch):
        model, optimizer = object(), object()
        training_mod = self._stub_megatron_training(
            monkeypatch,
            setup=lambda *a, **k: (model, optimizer, None),
            get_args=lambda: SimpleNamespace(linear_fp4=True),
        )
        register = mock.MagicMock()
        self._stub_mxfp4_hook_deps(monkeypatch, register)
        monkeypatch.delenv("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE", raising=False)

        self.hooks.install_mxfp4_weight_cache_hook()
        training_mod.setup_model_and_optimizer()

        register.assert_called_once_with(model, optimizer)

    def test_mxfp4_weight_cache_hook_skips_when_disabled(self, monkeypatch):
        training_mod = self._stub_megatron_training(
            monkeypatch,
            get_args=lambda: SimpleNamespace(linear_fp4=True),
        )
        register = mock.MagicMock()
        self._stub_mxfp4_hook_deps(monkeypatch, register)
        monkeypatch.setenv("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE", "1")

        self.hooks.install_mxfp4_weight_cache_hook()
        training_mod.setup_model_and_optimizer()

        register.assert_not_called()

    def test_mxfp4_weight_cache_hook_skips_non_mxfp4_format(self, monkeypatch):
        training_mod = self._stub_megatron_training(
            monkeypatch,
            get_args=lambda: SimpleNamespace(
                linear_fp8=True, linear_fp4=False, linear_fp8_format="fp8_e4m3"
            ),
        )
        register = mock.MagicMock()
        self._stub_mxfp4_hook_deps(monkeypatch, register)

        self.hooks.install_mxfp4_weight_cache_hook()
        training_mod.setup_model_and_optimizer()

        register.assert_not_called()

    def test_gc_freeze_hook_skips_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LUMEN_GC_FREEZE", "0")
        # Returns before importing megatron, so this passes without it installed.
        self.hooks.install_gc_freeze_hook()

    def test_gc_freeze_hook_only_unwraps_itself(self, monkeypatch):
        """A wrapper installed after the hook must survive the hook's unwrap."""
        mt = types.ModuleType("megatron.training.training")
        calls = {"base": 0, "outer": 0}

        def _base_train_step(*args, **kwargs):
            calls["base"] += 1
            return "out"

        mt.train_step = _base_train_step

        # ``import megatron.training.training`` walks the whole chain, so every
        # parent has to be importable and carry the child as an attribute.
        mt_pkg = types.ModuleType("megatron.training")
        mt_pkg.print_rank_0 = lambda *a, **k: None
        mt_pkg.training = mt
        meg_pkg = types.ModuleType("megatron")
        meg_pkg.training = mt_pkg
        monkeypatch.setitem(sys.modules, "megatron", meg_pkg)
        monkeypatch.setitem(sys.modules, "megatron.training", mt_pkg)
        monkeypatch.setitem(sys.modules, "megatron.training.training", mt)

        self.hooks.install_gc_freeze_hook(warmup_steps=1)
        gc_wrapped = mt.train_step

        def _outer(*args, **kwargs):
            calls["outer"] += 1
            return gc_wrapped(*args, **kwargs)

        mt.train_step = _outer

        _outer()
        assert calls == {"base": 1, "outer": 1}
        # The freeze fired on that call. Restoring the install-time snapshot
        # would have dropped _outer; it must still be in place.
        assert mt.train_step is _outer
