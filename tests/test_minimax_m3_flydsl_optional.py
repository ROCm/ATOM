# SPDX-License-Identifier: MIT
"""FlyDSL is imported where an MXFP4 index step or a wide-row selection reaches for it,
not by the modules every deployment loads (`aiter.ops.flydsl` imports `flydsl`, so a
module-level import would make it mandatory for every MHA model and bf16 / fp8 cache).
Run in a subprocess with `flydsl` made unimportable."""

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
_SCRIPT = textwrap.dedent("""
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

    import torch
    import atom.model_ops.attention_mha
    import atom.model_ops.attentions.aiter_attention
    from atom.model_ops.minimax_m3 import index_topk, sparse_attn
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata
    assert "flydsl" not in sys.modules

    # a bf16 / fp8 step: no scorer record, the work map is not asked for
    sparse_attn.build_n_valid_column_per_row = lambda *a, **k: None  # a Triton launch
    step = dict(
        seq_lens=torch.tensor([300], dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        max_seq_len=300,
        num_idx_heads=1,
    )
    assert make_sparse_decode_metadata(**step).index_scorer is None

    # the MXFP4 index and the wide-row selector raise where reached, not at import
    def refused(call):
        try:
            call()
        except ImportError:
            return
        raise AssertionError("no ImportError")

    refused(index_topk.flydsl_ops)
    refused(index_topk._small_k_selector)
    record = IndexScoreBuffers(scores=None, work_map=torch.empty(4, 2), max_block=3)
    refused(lambda: make_sparse_decode_metadata(**step, index_scorer=record))
    """)


def test_bf16_and_fp8_need_no_flydsl_and_the_flydsl_paths_fail_where_selected():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
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
