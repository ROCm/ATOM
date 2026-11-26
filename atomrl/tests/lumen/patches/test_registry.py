# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.patches registry."""

import pytest
from atomrl.lumen.patches import (
    PatchPhase,
    PatchRegistry,
    apply_patches,
    list_patches,
    register_patch,
)


class TestPatchRegistry:
    def setup_method(self):
        PatchRegistry.clear()

    def teardown_method(self):
        PatchRegistry.clear()

    def test_register_and_list(self):
        @register_patch("alpha", PatchPhase.IMPORT, tags=frozenset({"core"}))
        def _alpha():
            return 1

        specs = list_patches(PatchPhase.IMPORT)
        assert len(specs) == 1
        assert specs[0].name == "alpha"
        assert specs[0].tags == frozenset({"core"})

    def test_duplicate_registration_raises(self):
        @register_patch("dup", PatchPhase.IMPORT)
        def _first():
            pass

        with pytest.raises(ValueError, match="already registered"):
            register_patch("dup", PatchPhase.IMPORT)(lambda: None)

    def test_apply_respects_dependency_order(self):
        order: list[str] = []

        @register_patch("child", PatchPhase.IMPORT, depends_on=("parent",))
        def _child():
            order.append("child")

        @register_patch("parent", PatchPhase.IMPORT)
        def _parent():
            order.append("parent")

        apply_patches(PatchPhase.IMPORT)
        assert order == ["parent", "child"]

    def test_apply_skips_disabled_by_default(self):
        calls: list[str] = []

        @register_patch(
            "disabled",
            PatchPhase.IMPORT,
            enabled=lambda: False,
        )
        def _disabled():
            calls.append("disabled")

        @register_patch("enabled", PatchPhase.IMPORT)
        def _enabled():
            calls.append("enabled")

        results = apply_patches(PatchPhase.IMPORT, default_only=False)
        assert calls == ["enabled"]
        assert results["disabled"].skipped
        assert results["enabled"].applied

    def test_default_only_excludes_opt_in_false(self):
        calls: list[str] = []

        @register_patch("default", PatchPhase.IMPORT, default=True)
        def _default():
            calls.append("default")

        @register_patch("optional", PatchPhase.IMPORT, default=False)
        def _optional():
            calls.append("optional")

        apply_patches(PatchPhase.IMPORT, default_only=True)
        assert calls == ["default"]

    def test_filter_by_tags(self):
        calls: list[str] = []

        @register_patch("a", PatchPhase.IMPORT, tags=frozenset({"dsv4"}))
        def _a():
            calls.append("a")

        @register_patch("b", PatchPhase.IMPORT, tags=frozenset({"core"}))
        def _b():
            calls.append("b")

        apply_patches(PatchPhase.IMPORT, tags={"dsv4"})
        assert calls == ["a"]

    def test_filter_by_tags_any_mode(self):
        calls: list[str] = []

        @register_patch("a", PatchPhase.IMPORT, tags=frozenset({"dsv4", "rocm"}))
        def _a():
            calls.append("a")

        @register_patch("b", PatchPhase.IMPORT, tags=frozenset({"rocm"}))
        def _b():
            calls.append("b")

        @register_patch("c", PatchPhase.IMPORT, tags=frozenset({"llama"}))
        def _c():
            calls.append("c")

        apply_patches(PatchPhase.IMPORT, tags={"dsv4", "rocm"}, tag_mode="any")
        assert calls == ["a", "b"]

    def test_dry_run_does_not_invoke_patch(self):
        calls: list[str] = []

        @register_patch("dry", PatchPhase.IMPORT)
        def _dry():
            calls.append("dry")

        results = apply_patches(PatchPhase.IMPORT, dry_run=True)
        assert calls == []
        assert results["dry"].reason == "dry-run"


class TestPrintPatchReport:
    def test_false_return_value_reported_as_skipped(self, capsys):
        from atomrl.lumen.patches.cli import print_patch_report
        from atomrl.lumen.patches.types import PatchResult

        print_patch_report(
            "/tmp/megatron",
            {
                "already_patched": PatchResult(
                    name="already_patched", applied=True, return_value=False
                )
            },
        )
        out = capsys.readouterr().out
        assert "skipped: already_patched" in out

    def test_dry_run_reported_as_would_apply(self, capsys):
        from atomrl.lumen.patches.cli import print_patch_report
        from atomrl.lumen.patches.types import PatchResult

        print_patch_report(
            "/tmp/megatron",
            {
                "llama_norm": PatchResult(
                    name="llama_norm", skipped=True, reason="dry-run"
                )
            },
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "Would patch" in out
        assert "would apply: llama_norm" in out

    def test_none_return_value_reported_as_patched(self, capsys):
        from atomrl.lumen.patches.cli import print_patch_report
        from atomrl.lumen.patches.types import PatchResult

        print_patch_report(
            "/tmp/megatron",
            {
                "runtime_patch": PatchResult(
                    name="runtime_patch", applied=True, return_value=None
                )
            },
        )
        out = capsys.readouterr().out
        assert "PATCHED: runtime_patch" in out
