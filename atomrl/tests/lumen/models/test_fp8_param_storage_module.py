# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lightweight tests for fp8_param_storage module structure."""

import importlib
import sys
import types
from pathlib import Path

if "atomrl.lumen" not in sys.modules:
    _lumen = types.ModuleType("atomrl.lumen")
    _lumen.__path__ = [
        str(Path(__file__).resolve().parents[2].parent / "atomrl" / "lumen")
    ]
    sys.modules["atomrl.lumen"] = _lumen


class TestFp8ParamStorageModule:
    def test_public_helpers_exported(self):
        mod = importlib.import_module("atomrl.lumen.models.fp8_param_storage")
        for name in (
            "shrink_frozen_weights_to_fp8",
            "patch_meta_materializer",
            "patch_float16_module",
            "patch_load_checkpoint_for_fp8",
            "install_embedding_output_fp8_hooks",
            "register_fp8_param_optimizer_hook",
            "prepare_hipblaslt_for_fp8_storage",
        ):
            assert hasattr(mod, name), name

    def test_legacy_private_aliases(self):
        mod = importlib.import_module("atomrl.lumen.models.fp8_param_storage")
        assert mod._shrink_frozen_weights_to_fp8 is mod.shrink_frozen_weights_to_fp8
        assert mod._patch_meta_materializer is mod.patch_meta_materializer

    def test_megatron_reexports_fp8_storage(self):
        megatron = importlib.import_module("atomrl.lumen.models.megatron")
        assert megatron.register_fp8_param_optimizer_hook is not None
        assert megatron._patch_meta_materializer is not None

    def test_prepare_hipblaslt_noop_without_hybrid(self):
        from types import SimpleNamespace

        from atomrl.lumen.models.fp8_param_storage import (
            prepare_hipblaslt_for_fp8_storage,
        )

        args = SimpleNamespace(lumen_fp8_format="", fp8="", linear_fp8_format="delayed")
        prepare_hipblaslt_for_fp8_storage(args)
