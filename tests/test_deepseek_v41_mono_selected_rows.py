# SPDX-License-Identifier: MIT
"""Check the real selection helper's loads, including values FULL discards.

Execute its source with NumPy scalars and bounds-checked buffer loads so this
regression runs without FlyDSL or a GPU. Checking only the returned row misses
the bug: FULL discarded an out-of-bounds candidate load's result.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from atom.models.deepseek_v41.mono import index_plan as ip

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "atom/models/deepseek_v41/mono/kernels/index_score.py"
)
TOKENS = 36
KMETA = 4
ROWS_PER_PAGE = 16
PAGE_ROWS = 32
MAIN_OFF = 7


class _Scalar(np.ndarray):
    def select(self, yes, no):
        return np.where(self, yes, no).view(_Scalar)


def _i32(value):
    return np.asarray(value, dtype=np.int32).view(_Scalar)


@pytest.fixture(scope="module")
def selected_rows_code():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_selected_rows"
    )
    return compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec")


def _run(code, t, ident, lift):
    # Distinct candidates per token and a non-identity block table expose a
    # wrong token stride, an accidental always-FULL path or bad pool mapping.
    candidates = (
        np.arange(TOKENS * ip.TOPK_BLOCKS, dtype=np.int32).reshape(TOKENS, -1) * 13 + 17
    ) % 65536
    metadata = np.zeros((TOKENS, KMETA), dtype=np.int32)
    metadata[:, 2] = np.arange(TOKENS) * 3
    table = np.arange((1 << 20) // ROWS_PER_PAGE + TOKENS * 3, dtype=np.int32)
    table = (table * 7 + 11) % len(table)
    buffers = {"icand": candidates.ravel(), "kmeta": metadata.ravel(), "table": table}

    def load(buffer, offset, **kwargs):
        offset = int(offset)
        if buffer == "icand":
            assert (
                t * ip.TOPK_BLOCKS <= offset < (t + 1) * ip.TOPK_BLOCKS
            ), f"candidate load {offset} escapes token {t}'s row"
        assert 0 <= offset < len(buffers[buffer])
        return _i32(buffers[buffer][offset])

    scope = {
        "fx": SimpleNamespace(Int32=_i32, max=np.maximum),
        "bo": SimpleNamespace(buffer_load=load),
        "rsrc": lambda buffer: buffer,
        "T": SimpleNamespace(i32=np.int32),
        "ip": ip,
        "TILE": ip.BLOCK_ROWS,
        "KMETA": KMETA,
    }
    exec(code, scope)  # noqa: S102 - the function under test, from this repository.
    args = {
        **{name: name for name in buffers},
        "ilift": _i32(lift),
        "rows_per_page": ROWS_PER_PAGE,
        "page_rows": PAGE_ROWS,
        "main_off": MAIN_OFF,
    }
    actual = scope["_selected_rows"]({"args": args}, t, _i32(ident))
    expected_row = ident
    if lift and ident >= 0:
        block, within = divmod(ident, ip.BLOCK_ROWS)
        expected_row = int(candidates[t, block]) * ip.BLOCK_ROWS + within
    page, within = divmod(max(expected_row, 0), ROWS_PER_PAGE)
    expected_pool_row = int(table[int(metadata[t, 2]) + page]) * PAGE_ROWS
    assert tuple(map(int, actual)) == (
        expected_row,
        expected_pool_row + MAIN_OFF + within,
    )


@pytest.mark.parametrize("t", [0, 1, TOKENS - 1])
@pytest.mark.parametrize("ident", [-1, 0, 16383, 16384, 316436, (1 << 20) - 1])
def test_full_history_rows_keep_candidate_loads_in_bounds(selected_rows_code, t, ident):
    _run(selected_rows_code, t, ident, lift=0)


@pytest.mark.parametrize("t", [0, 1, TOKENS - 1])
@pytest.mark.parametrize("ident", [-1, 0, 1, 7, 8, ip.TOPK_BLOCKS * ip.BLOCK_ROWS - 1])
@pytest.mark.parametrize("lift", [1, 3])
def test_reindex_lifts_candidates_and_preserves_padding(
    selected_rows_code, t, ident, lift
):
    _run(selected_rows_code, t, ident, lift)
