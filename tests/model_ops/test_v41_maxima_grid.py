# SPDX-License-Identifier: MIT
"""`ATOM_DSV41_MAXIMA_GRID`: the launch grid of the V4.1 candidate block maxima.

The two functions under test are taken out of ``indexer.py`` by name, so the
test runs without the Triton / AITER stack the module's kernels need; the env
value comes from the real ``atom.utils.envs``.
"""

import ast
from pathlib import Path

import pytest

from atom.utils import envs

ENV = "ATOM_DSV41_MAXIMA_GRID"
ROWS = (1, 5, 6, 36, 72, 126, 128, 132, 192, 288, 384, 480, 1170, 4096, 8191, 16384)
SOURCE = (
    Path(__file__).resolve().parents[2]
    / "atom"
    / "model_ops"
    / "deepseek_v41"
    / "indexer.py"
)


def _lanes_fn(gfx1250=False):
    """`_maxima_lanes` as the file defines it, on a machine that is (not) gfx1250."""
    tree = ast.parse(SOURCE.read_text())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_maxima_lanes", "_maxima_grid")
    ]
    assert [node.name for node in wanted] == ["_maxima_lanes", "_maxima_grid"]
    namespace = {"_gfx1250": lambda: gfx1250}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_maxima_lanes"]


def _before_the_switch(rows):
    return min(256, max(1, 2048 // max(rows, 1)))


def test_unset_is_the_grid_of_before(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert envs.ATOM_DSV41_MAXIMA_GRID == 2048
    lanes = _lanes_fn()
    assert [lanes(rows) for rows in (0, *ROWS)] == [
        _before_the_switch(rows) for rows in (0, *ROWS)
    ]


@pytest.mark.parametrize("grid", [2048, 4096, 16384, 32768, 65536])
def test_env_sets_the_grid(monkeypatch, grid):
    monkeypatch.setenv(ENV, str(grid))
    assert envs.ATOM_DSV41_MAXIMA_GRID == grid
    lanes = _lanes_fn()
    assert [lanes(rows) for rows in ROWS] == [
        min(256, max(1, grid // rows)) for rows in ROWS
    ]


def test_decode_buckets_at_16384(monkeypatch):
    """The lane counts the bench measured: 132 / 192 / 288 / 384 rows."""
    monkeypatch.setenv(ENV, "16384")
    lanes = _lanes_fn()
    assert [lanes(rows) for rows in (132, 192, 288, 384)] == [124, 85, 56, 42]


@pytest.mark.parametrize("raw, grid", [("abc", 2048), ("", 2048), ("0", 1), ("-5", 1)])
def test_bad_values(monkeypatch, raw, grid):
    """Not a number: the default. Below one: one program, never zero lanes."""
    monkeypatch.setenv(ENV, raw)
    assert envs.ATOM_DSV41_MAXIMA_GRID == grid
    lanes = _lanes_fn()
    assert all(1 <= lanes(rows) <= 256 for rows in (0, *ROWS))


def test_lanes_stay_in_bounds_for_any_grid(monkeypatch):
    lanes = _lanes_fn()
    for grid in (1, 7, 2048, 16384, 1 << 20, 1 << 40):
        monkeypatch.setenv(ENV, str(grid))
        assert all(1 <= lanes(rows) <= 256 for rows in (0, *ROWS))


def test_read_at_each_launch(monkeypatch):
    """No cached copy: a server sets it before its graphs are captured."""
    lanes = _lanes_fn()
    monkeypatch.delenv(ENV, raising=False)
    assert lanes(384) == 5
    monkeypatch.setenv(ENV, "16384")
    assert lanes(384) == 42
    monkeypatch.delenv(ENV)
    assert lanes(384) == 5


def test_gfx1250_keeps_its_own_grid(monkeypatch):
    lanes = _lanes_fn(gfx1250=True)
    for raw in (None, "2048", "65536"):
        if raw is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, raw)
        assert [lanes(rows) for rows in ROWS] == [
            min(256, max(1, 16384 // rows)) for rows in ROWS
        ]


def test_no_line_above_the_kernels_moved():
    """Triton keys a compiled kernel by its source and the line it starts on:
    the helper sits below both kernels, so neither is compiled again."""
    lines = SOURCE.read_text().splitlines()
    starts = {
        name: next(i for i, line in enumerate(lines) if line.startswith(f"def {name}("))
        for name in ("_restrict_kernel", "_block_maxima_kernel", "_maxima_grid")
    }
    assert starts["_maxima_grid"] > starts["_block_maxima_kernel"] > starts["_restrict_kernel"]
    assert starts["_restrict_kernel"] == 28 and starts["_block_maxima_kernel"] == 144
