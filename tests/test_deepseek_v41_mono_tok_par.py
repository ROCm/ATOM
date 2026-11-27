# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Which V4.1 mono target widths run token parallel and which shard: at TP4
the 12-row width runs token parallel, as 24-48 rows do. Sharded, its K1 alone
spills scratch, which no lean rebuild leaves out, so the width was refused and
two-request steps took the original path."""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.deepseek_v41.mono import runner as mono_runner
from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner

# target verify widths: 1-8 requests of 6 rows (5 draft tokens + 1)
TARGET_WIDTHS = [6 * n for n in range(1, 9)]


def _runner(tp: int, layer_capable: bool = True) -> SimpleNamespace:
    # tok_par_from as V41MonoDecodeRunner.__init__ sets it
    from_rows = (
        mono_runner.LAYER_TOK_PAR_FROM if layer_capable else mono_runner.TOK_PAR_FROM
    )
    return SimpleNamespace(
        tp=tp,
        dp=1,
        check=False,
        split_k2=False,
        shards=True,
        max_rows=max(TARGET_WIDTHS),
        tok_par_from=from_rows.get(tp, max(TARGET_WIDTHS) + 1),
    )


def _tok_par(r, s: int) -> bool:
    return V41MonoDecodeRunner._tok_par(r, s)


def test_tp4_layer_kernel_12_rows_run_token_parallel():
    r = _runner(4)
    assert _tok_par(r, 12)
    # and so do not shard: the sharded 12-row K1 is the build that spills
    r._tok_par = lambda s: _tok_par(r, s)
    assert not V41MonoDecodeRunner._shard(r, 12)


@pytest.mark.parametrize("s", TARGET_WIDTHS)
def test_tp4_layer_kernel_widths(s):
    # one request (6 rows) stays sharded; every other width a multiple of
    # the TP size runs token parallel (18, 30, 42 rows are not)
    assert _tok_par(_runner(4), s) == (s >= 12 and s % 4 == 0)


@pytest.mark.parametrize("tp", [2, 8])
def test_other_tp_sizes_unchanged(tp):
    # token parallel never paid at TP2; TP8 has no layer-kernel threshold
    assert tp not in mono_runner.LAYER_TOK_PAR_FROM
    assert not any(_tok_par(_runner(tp), s) for s in TARGET_WIDTHS)
