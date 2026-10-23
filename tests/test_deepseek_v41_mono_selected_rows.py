# SPDX-License-Identifier: MIT
"""Check the real selection helper's loads, including values FULL discards.

Execute its source with NumPy scalars and bounds-checked buffer loads so this
regression runs without FlyDSL or a GPU. Checking only the returned row misses
the bug: FULL discarded an out-of-bounds candidate load's result.

This is a bounds regression, not a FlyDSL emulator: new global dependencies
require an explicit harness update, and division/remainder must stay in the
nonnegative domain where NumPy and GPU integer arithmetic agree.
"""

import ast
import builtins
import dis
import operator
from pathlib import Path
from types import CodeType, SimpleNamespace

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
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        values = [np.asarray(value) for value in inputs]
        if ufunc in (np.floor_divide, np.remainder):
            assert np.all(values[0] >= 0) and np.all(values[1] > 0), (
                "selected_rows CPU harness requires a nonnegative dividend and "
                "positive divisor; NumPy and GPU signed division/remainder differ"
            )
        return np.asarray(getattr(ufunc, method)(*values, **kwargs)).view(_Scalar)

    def select(self, yes, no):
        return np.where(self, yes, no).view(_Scalar)


def _i32(value):
    return np.asarray(value, dtype=np.int32).view(_Scalar)


def _global_names(code):
    for instruction in dis.get_instructions(code):
        if instruction.opname == "LOAD_GLOBAL":
            yield instruction.argval
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            yield from _global_names(constant)


def _check_globals(code, scope):
    missing = set(_global_names(code)) - scope.keys() - vars(builtins).keys()
    assert not missing, (
        "_selected_rows acquired global dependencies missing from its CPU "
        f"harness: {sorted(missing)}; update the harness before running this regression"
    )


@pytest.fixture(scope="module")
def selected_rows_code():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_selected_rows"
    )
    # Run the original Python control flow on CPU; @traced only lowers it to
    # device IR. GPU tests exercise that lowering separately. Any additional
    # decorator needs an explicit decision about its CPU test semantics.
    assert [ast.unparse(node) for node in function.decorator_list] == ["traced"]
    function.decorator_list = []
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
    candidate_loads = []

    def load(buffer, offset, **kwargs):
        offset = int(offset)
        if buffer == "icand":
            assert lift and ident >= 0, "FULL and padding must not read candidates"
            assert (
                t * ip.TOPK_BLOCKS <= offset < (t + 1) * ip.TOPK_BLOCKS
            ), f"candidate load {offset} escapes token {t}'s row"
            candidate_loads.append(offset)
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
    _check_globals(code, scope)
    exec(code, scope)  # noqa: S102 - the function under test, from this repository.
    args = {
        **{name: name for name in buffers},
        "ilift": _i32(lift),
        "rows_per_page": ROWS_PER_PAGE,
        "page_rows": PAGE_ROWS,
        "main_off": MAIN_OFF,
    }
    actual = scope["_selected_rows"]({"args": args}, t, _i32(ident))
    assert len(candidate_loads) == int(bool(lift) and ident >= 0)
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
def test_full_history_rows_do_not_read_candidates(selected_rows_code, t, ident):
    _run(selected_rows_code, t, ident, lift=0)


@pytest.mark.parametrize("t", [0, 1, TOKENS - 1])
@pytest.mark.parametrize("ident", [-1, 0, 1, 7, 8, ip.TOPK_BLOCKS * ip.BLOCK_ROWS - 1])
@pytest.mark.parametrize("lift", [1, 3])
def test_reindex_lifts_candidates_and_preserves_padding(
    selected_rows_code, t, ident, lift
):
    _run(selected_rows_code, t, ident, lift)


@pytest.mark.parametrize("operation", [operator.floordiv, operator.mod])
@pytest.mark.parametrize("left,right", [(-1, 8), (1, -8), (1, 0)])
def test_scalar_rejects_unsupported_division_domain(operation, left, right):
    with pytest.raises(AssertionError, match="NumPy and GPU signed"):
        operation(_i32(left), _i32(right))


def test_harness_reports_new_global_dependencies():
    code = compile("def probe(): return added_dependency", "<probe>", "exec")
    with pytest.raises(AssertionError, match="missing.*CPU harness:.*added_dependency"):
        _check_globals(code, {})
