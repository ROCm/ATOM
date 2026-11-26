# SPDX-License-Identifier: MIT
"""`_emit_sparse_block_table_row` (shared by every index cache) on a pad row with a
non-empty selection. CPU: Triton's interpreter, in a child process (the flag must
precede the triton import). The gfx950 variant needs the GPU (run on MI355X: passes)."""

from __future__ import annotations

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

T = 4  # selection slots a row holds


@triton.jit
def _probe(idx_ptr, table_ptr, sbt_ptr, sctx_ptr, causal_ptr, T: tl.constexpr):
    r = tl.program_id(0)
    idx = tl.load(idx_ptr + r * T + tl.arange(0, T))
    _emit_sparse_block_table_row(
        idx, table_ptr + r * 64, sbt_ptr + r * T, sctx_ptr + r,
        tl.load(causal_ptr + r), T, 0, 128, 1, 1, 1, T,
    )  # fmt: skip


# (causal_len, selection). A pad row of a Q=4 decode step is -3 .. 0.
PAD = [(-3, [0, 1, -1, -1])]
# (causal_len, selection, block-table row, ctx): full blocks first, the tail last.
LIVE = [(300, [0, 2, 1, -1], [100, 101, 102, 0], 300)]


def run_emit(rows, device):
    n, i32 = len(rows), {"dtype": torch.int32, "device": device}
    idx = torch.tensor([sel for _, sel in rows], **i32)
    causal = torch.tensor([c for c, _ in rows], **i32)
    table = (torch.arange(64, **i32) + 100).repeat(n, 1).contiguous()
    sbt, sctx = torch.full((n, T), -7, **i32), torch.full((n,), -9, **i32)
    _probe[(n,)](idx, table, sbt, sctx, causal, T)
    return sbt.tolist(), sctx.tolist()


def _check(sbt, sctx):
    for i, row in enumerate(PAD):
        assert (sbt[i], sctx[i]) == ([0] * T, 0), f"pad row {row} emitted something"
    for j, (c, _, bt, ctx) in enumerate(LIVE, start=len(PAD)):
        assert (sbt[j], sctx[j]) == (bt, ctx), f"live row {c}"


def test_a_pad_row_emits_nothing_whatever_the_selection_holds_on_the_cpu():
    code = (
        "import importlib.util;"
        f"s = importlib.util.spec_from_file_location('probe', {__file__!r});"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
        "m._check(*m.run_emit(m.PAD + [r[:2] for r in m.LIVE], 'cpu'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={**os.environ, "TRITON_INTERPRET": "1", "HIP_VISIBLE_DEVICES": ""},
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="GPU-pending: the compiled kernel, run on gfx950",
)
def test_a_pad_row_emits_nothing_whatever_the_selection_holds_on_gfx950():
    _check(*run_emit(PAD + [r[:2] for r in LIVE], "cuda"))
