# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for ROCm platform SOURCE patch registration."""

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
from atomrl.lumen.patches.source import rocm


class TestRocmSourcePatches:
    def setup_method(self):
        PatchRegistry.clear()
        importlib.reload(rocm)

    EXPECTED: typing.ClassVar = {
        "disable_batch_p2p_comm",
        "cpu_offload_torch_gpu_adam",
    }

    def test_all_rocm_source_patches_registered(self):
        names = {spec.name for spec in list_patches(PatchPhase.SOURCE, tags={"rocm"})}
        assert self.EXPECTED <= names

    def test_rocm_patches_not_dsv4_tagged(self):
        dsv4_names = {
            spec.name for spec in list_patches(PatchPhase.SOURCE, tags={"dsv4"})
        }
        assert not self.EXPECTED & dsv4_names

    def test_disable_batch_p2p_comm_tags(self):
        spec = PatchRegistry.get("disable_batch_p2p_comm")
        assert "rocm" in spec.tags
        assert "pipeline" in spec.tags

    def test_cpu_offload_tags(self):
        spec = PatchRegistry.get("cpu_offload_torch_gpu_adam")
        assert "rocm" in spec.tags
        assert "optimizer" in spec.tags
