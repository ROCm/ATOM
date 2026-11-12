# SPDX-License-Identifier: MIT
"""`_emit_sparse_block_table_row` for a row with no causal key (a pad row).

The helper is the one block-table emission every index cache shares -- the bf16 /
fp8 Triton selectors, the MXFP4 one, the candidate exchange -- so a pad row's
handling is tested here once, on the helper, with a selection that is *not* empty:
the case the callers' selectors never produce today and a future caller might.

Two ways to run it, same cases:

- on the CPU, always: the kernel runs under Triton's interpreter
  (``TRITON_INTERPRET=1``), which has to be set before triton is imported, so it
  runs in a child process;
- on a gfx950 device: the compiled kernel (GPU-PENDING: written to run there, not
  yet run on one).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton", reason="the helper is a @triton.jit function")
pytest.importorskip("aiter", reason="index_topk imports aiter's chip_info")

import triton
import triton.language as tl

from atom.model_ops.minimax_m3.index_topk import _emit_sparse_block_table_row

BLOCK = 128
T = 4  # selection slots a row holds
ROW_PAGES = 64  # block-table width of the probe


@triton.jit
def _probe(
    idx_ptr,
    table_ptr,
    sbt_ptr,
    sctx_ptr,
    causal_ptr,
    T: tl.constexpr,
    ROW: tl.constexpr,
):
    r = tl.program_id(0)
    idx = tl.load(idx_ptr + r * T + tl.arange(0, T))
    _emit_sparse_block_table_row(
        idx, table_ptr + r * ROW, sbt_ptr + r * T, sctx_ptr + r,
        tl.load(causal_ptr + r), T, 0, BLOCK, 1, 1, 1, T,
    )  # fmt: skip


# (causal_len, selection). A pad row of a Q=4 decode step is -3 .. 0; every selection
# is non-empty so that nothing but the helper keeps the row empty.
PAD = [
    (-3, [0, 1, -1, -1]),
    (-1, [0, 2, 1, 3]),
    (0, [0, 1, 2, 3]),
    (-127, [5, 6, 7, 8]),
]
# (causal_len, selection, block table entries it must emit, ctx): full blocks in
# selection order, then the partial tail last.
LIVE = [
    (1, [0, -1, -1, -1], [100, 0, 0, 0], 1),
    (128, [0, -1, -1, -1], [100, 0, 0, 0], 128),
    (129, [1, 0, -1, -1], [100, 101, 0, 0], 129),
    (300, [0, 2, 1, -1], [100, 101, 102, 0], 300),
]


def run_emit(rows, device):
    """Emit each (causal_len, selection) row; return (block-table rows, contexts)."""
    n = len(rows)
    i32 = {"dtype": torch.int32, "device": device}
    idx = torch.tensor([sel for _, sel in rows], **i32)
    causal = torch.tensor([c for c, _ in rows], **i32)
    table = (torch.arange(ROW_PAGES, **i32) + 100).repeat(n, 1).contiguous()
    sbt = torch.full((n, T), -7, **i32)  # a sentinel: every slot must be written
    sctx = torch.full((n,), -9, **i32)
    _probe[(n,)](idx, table, sbt, sctx, causal, T, ROW_PAGES)
    return sbt.tolist(), sctx.tolist()


def _check(sbt, sctx):
    for i, _ in enumerate(PAD):
        assert sbt[i] == [0] * T, f"pad row {PAD[i]} names a page: {sbt[i]}"
        assert sctx[i] == 0, f"pad row {PAD[i]} has context {sctx[i]}"
    for j, (c, _, want_bt, want_ctx) in enumerate(LIVE, start=len(PAD)):
        assert sbt[j] == want_bt and sctx[j] == want_ctx, f"live row {c}"


def test_a_pad_row_emits_nothing_whatever_the_selection_holds_on_the_cpu():
    env = {**os.environ, "TRITON_INTERPRET": "1", "HIP_VISIBLE_DEVICES": ""}
    code = (
        "import importlib.util, json, sys;"
        f"s = importlib.util.spec_from_file_location('probe', {__file__!r});"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
        "print('RESULT' + json.dumps(m.run_emit(m.PAD + [r[:2] for r in m.LIVE], 'cpu')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT"))
    _check(*json.loads(line[len("RESULT") :]))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="GPU-pending: the compiled kernel, run on gfx950",
)
def test_a_pad_row_emits_nothing_whatever_the_selection_holds_on_gfx950():
    _check(*run_emit(PAD + [r[:2] for r in LIVE], "cuda"))
