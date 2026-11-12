# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL is imported where an MXFP4 index step or a wide-row selection reaches for it,
not by the modules every deployment loads.

`aiter.ops.flydsl` imports `flydsl` itself, so a module-level import of its names makes
FlyDSL mandatory for every MHA model (`attention_mha` -> `aiter_attention` ->
`sparse_attn` -> `index_topk`) and every bf16 / fp8 cache. Run in a subprocess with
`flydsl` made unimportable, so that what an earlier test already imported cannot hide it.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("aiter", reason="the kernel modules import aiter at the top")
pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parent.parent

# Absent the way a host without the package is: the import raises and `find_spec`
# (which aiter probes with) answers None.
_BLOCK_FLYDSL = textwrap.dedent("""
    import importlib.util, sys
    class _B:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] == "flydsl":
                raise ModuleNotFoundError("No module named 'flydsl'", name="flydsl")
    sys.meta_path.insert(0, _B())
    _find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda n, p=None: (
        None if n.split(".")[0] == "flydsl" else _find_spec(n, p)
    )
    """)


def _run_without_flydsl(body: str):
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_FLYDSL + textwrap.dedent(body)],
        check=False,
        cwd=ROOT,
        env={
            "PYTHONPATH": str(ROOT),
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]


def test_the_shared_backend_and_the_m3_modules_import_without_flydsl():
    _run_without_flydsl("""
        import atom.model_ops.attention_mha
        import atom.model_ops.attentions.aiter_attention
        import atom.model_ops.minimax_m3.index_topk
        import atom.model_ops.minimax_m3.sparse_attn
        assert "flydsl" not in sys.modules
        """)


def test_a_bf16_or_fp8_step_needs_no_flydsl():
    """No scorer record: the work map is not asked for, the Triton path runs."""
    _run_without_flydsl("""
        import torch
        from atom.model_ops.minimax_m3 import sparse_attn
        from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata
        sparse_attn.build_n_valid_column_per_row = lambda *a, **k: None  # a Triton launch
        md = make_sparse_decode_metadata(
            seq_lens=torch.tensor([300], dtype=torch.int32),
            block_table=torch.zeros(1, 4, dtype=torch.int32),
            slot_mapping=torch.zeros(1, dtype=torch.int64),
            max_seq_len=300,
            num_idx_heads=1,
        )
        assert md.index_scorer is None
        """)


def test_the_flydsl_paths_fail_where_they_are_selected():
    """The MXFP4 index and the wide-row selector are the two things that need FlyDSL;
    each raises when reached, rather than at import or by falling back."""
    _run_without_flydsl("""
        import torch
        from atom.model_ops.minimax_m3 import index_topk
        from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers
        from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

        def refused(call):
            try:
                call()
            except ImportError:
                return
            raise AssertionError("no ImportError")

        refused(index_topk.flydsl_ops)
        refused(index_topk._small_k_selector)
        record = IndexScoreBuffers(scores=None, work_map=torch.empty(4, 2), max_block=3)
        refused(lambda: make_sparse_decode_metadata(
            seq_lens=torch.tensor([300], dtype=torch.int32),
            block_table=torch.zeros(1, 4, dtype=torch.int32),
            slot_mapping=torch.zeros(1, dtype=torch.int64),
            max_seq_len=300,
            num_idx_heads=1,
            index_scorer=record,
        ))
        """)
