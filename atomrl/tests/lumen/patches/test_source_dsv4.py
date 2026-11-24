# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for DSV4 SOURCE patch registration."""

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
from atomrl.lumen.patches.source import dsv4


class TestDsv4SourcePatches:
    def setup_method(self):
        PatchRegistry.clear()
        importlib.reload(dsv4)

    EXPECTED: typing.ClassVar = {
        "dsv4_transformer_config",
        "moe_sqrtsoftplus",
        "dsv4_training_config",
        "moe_router_freeze",
        "dsv4_hash_routing",
        "skip_none_router_expert_bias",
        "dist_ckpt_skip_dsv4_norms",
        "shared_expert_clamp",
        "dsv4_transformer_block",
        "dsv4_transformer_layer",
        "dsv4_eav_specs",
        "tp_layers_condition_init",
    }

    def test_all_dsv4_source_patches_registered(self):
        names = {spec.name for spec in list_patches(PatchPhase.SOURCE, tags={"dsv4"})}
        assert self.EXPECTED <= names

    def test_hash_routing_depends_on_sqrtsoftplus(self):
        spec = PatchRegistry.get("dsv4_hash_routing")
        assert "moe_sqrtsoftplus" in spec.depends_on

    def test_transformer_config_fields_declared(self):
        spec = PatchRegistry.get("dsv4_transformer_config")
        assert "dsv4_mode" in spec.config_fields
        assert "dsv4_n_hash_layers" in spec.config_fields
