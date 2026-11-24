# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for LLaMA SOURCE patch registration."""

import importlib
import sys
import types
import typing
from pathlib import Path

if "atomrl.lumen" not in sys.modules:
    _lumen = types.ModuleType("atomrl.lumen")
    _lumen.__path__ = [
        str(Path(__file__).resolve().parents[2].parent / "atomrl" / "lumen")
    ]
    sys.modules["atomrl.lumen"] = _lumen

from atomrl.lumen.patches import PatchPhase, PatchRegistry, list_patches
from atomrl.lumen.patches.source import llama


class TestLlamaSourcePatches:
    def setup_method(self):
        PatchRegistry.clear()
        importlib.reload(llama)

    EXPECTED: typing.ClassVar = {
        "llama_megatron_fused_rmsnorm",
        "llama_gpt_layer_specs_rmsnorm",
        "llama_transformer_block_rmsnorm",
    }

    def test_all_llama_source_patches_registered(self):
        names = {spec.name for spec in list_patches(PatchPhase.SOURCE, tags={"llama"})}
        assert self.EXPECTED <= names

    def test_gpt_layer_specs_depends_on_wrapper(self):
        spec = PatchRegistry.get("llama_gpt_layer_specs_rmsnorm")
        assert "llama_megatron_fused_rmsnorm" in spec.depends_on

    def test_transformer_block_depends_on_wrapper(self):
        spec = PatchRegistry.get("llama_transformer_block_rmsnorm")
        assert "llama_megatron_fused_rmsnorm" in spec.depends_on

    def test_llama_patches_not_dsv4_only(self):
        dsv4_only = {
            spec.name
            for spec in list_patches(PatchPhase.SOURCE, tags={"dsv4"})
            if "llama" not in spec.tags
        }
        assert not self.EXPECTED & dsv4_only
