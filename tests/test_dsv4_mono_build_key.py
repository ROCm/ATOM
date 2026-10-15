# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""V4 device dependencies cannot silently reuse a stale FlyDSL binary."""

import shutil
from pathlib import Path

import pytest

from atom.models.deepseek_v4.mono.sources import SOURCE_PATHS
from atom.mono.plan import build_key


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    atom = Path(__file__).resolve().parents[1] / "atom"
    for rel in SOURCE_PATHS:
        source, target = atom / rel, tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)
    monkeypatch.setattr(build_key, "_ATOM", tmp_path)
    return tmp_path, build_key.source_digest.__wrapped__


@pytest.mark.parametrize(
    "dependency",
    [
        "models/deepseek_v4/mono/kernels/plan.py",
        "models/deepseek_v4/mono/kernels/attention.py",
        "models/deepseek_v4/mono/config.py",
        "mono/device/flydsl_helpers.py",
        "mono/plan/formats.py",
    ],
)
def test_device_dependency_edits_change_the_digest(source_tree, dependency):
    tmp_path, digest = source_tree
    before = digest(*SOURCE_PATHS)
    target = tmp_path / dependency
    target.write_text(target.read_text() + "\n# changed device dependency\n")

    assert digest(*SOURCE_PATHS) != before


@pytest.mark.parametrize("module", ["session", "cache", "runner", "contract"])
def test_serving_only_edits_do_not_recompile_device_sources(source_tree, module):
    tmp_path, digest = source_tree
    before = digest(*SOURCE_PATHS)
    target = tmp_path / "models/deepseek_v4/mono" / f"{module}.py"
    target.write_text("# changed serving-only module\n")
    assert before == digest(*SOURCE_PATHS)


@pytest.mark.parametrize("builder", ["layer", "advance"])
def test_source_digest_reaches_the_compiled_key(tmp_path, monkeypatch, builder):
    pytest.importorskip("flydsl")
    from atom.models.deepseek_v4.mono.kernels import layer
    from atom.mono.runtime.compile import compile_only

    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))

    def compile_step():
        import inspect

        launch = (
            layer.build_dsv4_kernel(phase="attention_hc", hc_mult=4)
            if builder == "layer"
            else layer.build_advance_step(512, 512)
        )
        nargs = sum(
            name != "stream"
            for name in inspect.signature(launch._original_func).parameters
        )
        with compile_only():
            launch(*([0] * nargs))
        return set(tmp_path.rglob("*.pkl"))

    monkeypatch.setattr(layer, "SOURCES", "before-device-edit")
    initial = compile_step()
    assert len(initial) == 1
    assert compile_step() == initial

    monkeypatch.setattr(layer, "SOURCES", "after-device-edit")
    assert len(compile_step()) == 2
