# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The launch grid of the V4.1 candidate block maxima, per arch.

`_maxima_grid` and `_maxima_lanes` are taken out of ``indexer.py`` by name, so
the test runs without the Triton / AITER stack the module's kernels need; the
arch comes from a stand-in for ``get_gfx_runtime``.
"""

import ast
import functools
import sys
import types
from pathlib import Path

import pytest

ROWS = (0, 1, 5, 6, 36, 64, 72, 128, 132, 192, 288, 384, 480, 1170, 4096, 16384)
SOURCE = (
    Path(__file__).resolve().parents[2]
    / "atom"
    / "model_ops"
    / "deepseek_v41"
    / "indexer.py"
)


def _lanes_fn(monkeypatch, arch):
    """`_maxima_lanes` as the file defines it, on a machine of `arch`."""
    chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx_runtime = lambda: arch
    for name in ("aiter", "aiter.jit", "aiter.jit.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "aiter.jit.utils.chip_info", chip_info)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_maxima_grid", "_maxima_lanes")
    ]
    assert [node.name for node in wanted] == ["_maxima_grid", "_maxima_lanes"]
    namespace = {"functools": functools}
    module = ast.Module(body=wanted, type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace["_maxima_lanes"]


def _lanes(grid, rows):
    return min(256, max(1, grid // max(rows, 1)))


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
def test_gfx950_and_gfx1250_take_the_wide_grid(monkeypatch, arch):
    lanes = _lanes_fn(monkeypatch, arch)
    assert [lanes(rows) for rows in ROWS] == [_lanes(16384, rows) for rows in ROWS]


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a", "gfx1201"])
def test_other_archs_keep_the_grid_they_were_tuned_with(monkeypatch, arch):
    lanes = _lanes_fn(monkeypatch, arch)
    assert [lanes(rows) for rows in ROWS] == [_lanes(2048, rows) for rows in ROWS]


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
def test_lanes_stay_in_bounds(monkeypatch, arch):
    lanes = _lanes_fn(monkeypatch, arch)
    assert all(1 <= lanes(rows) <= 256 for rows in range(0, 20000, 7))
