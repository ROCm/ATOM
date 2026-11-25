# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for LLaMA LoRA SOURCE patch registration."""

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
from atomrl.lumen.patches.source import llama_lora


class TestLlamaLoraSourcePatches:
    def setup_method(self):
        PatchRegistry.clear()
        importlib.reload(llama_lora)

    EXPECTED: typing.ClassVar = {
        "lora_requires_grad",
        "lora_checkpoint_load",
        "lora_adapter_scaling",
        "lora_sft_loss_default",
    }

    def test_all_lora_source_patches_registered(self):
        names = {spec.name for spec in list_patches(PatchPhase.SOURCE, tags={"lora"})}
        assert self.EXPECTED <= names

    def test_lora_patches_opt_in(self):
        for name in self.EXPECTED:
            assert PatchRegistry.get(name).default is False

    def test_lora_patches_not_matched_by_llama_tag(self):
        llama_names = {
            spec.name for spec in list_patches(PatchPhase.SOURCE, tags={"llama"})
        }
        assert not self.EXPECTED & llama_names

    def test_explicit_tag_includes_opt_in_patches(self):
        from atomrl.lumen.patches.registry import _filter_specs

        opt_in = {
            spec.name
            for spec in PatchRegistry.all()
            if spec.phase is PatchPhase.SOURCE
            and {"lora"}.issubset(spec.tags)
            and not spec.default
        }
        applied_names = {
            spec.name
            for spec in _filter_specs(
                PatchPhase.SOURCE, tags={"lora"}, default_only=False
            )
        }
        assert opt_in <= applied_names
        assert len(applied_names) == len(self.EXPECTED)

    def test_lora_tags(self):
        spec = PatchRegistry.get("lora_checkpoint_load")
        assert "lora" in spec.tags
        assert "finetune" in spec.tags
