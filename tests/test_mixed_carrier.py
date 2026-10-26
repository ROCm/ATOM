# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The mixed decode view's row checks, and the one mixed carrier both builders use."""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

# The V4 attention module imports aiter at module scope.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions.deepseek_v4_attn import (
    MixedViewUnslicedField,
    _mixed_carrier,
    _MixedDecodeView,
)

_V4_ATTN = (
    pathlib.Path(__file__).resolve().parents[1]
    / "atom/model_ops/attentions/deepseek_v4_attn.py"
)


def _batch(n_p=2, n_d=3, block_rows=None):
    n = n_p + n_d
    tables = [[10 + i] for i in range(n)]
    return SimpleNamespace(
        total_seqs_num=n,
        total_seqs_num_decode=n_d,
        total_tokens_num_prefill=8 * n_p,
        total_tokens_num_decode=n_d,
        num_spec_step=0,
        is_dummy_run=False,
        context_lens=list(range(n)),
        block_tables=tables if block_rows is None else tables[:block_rows],
        swa_block_tables=None,
        state_slots_committed=list(range(n)),
        state_fork_srcs=None,
    )


def test_view_slices_the_decode_rows():
    view = _MixedDecodeView(_batch(), 2)
    assert view.block_tables == [[12], [13], [14]]


def test_view_refuses_block_tables_that_are_not_one_per_row():
    """A row with an empty table (filtered out of `block_tables`) would shift
    every decode row onto its neighbour's pages if sliced by position."""
    with pytest.raises(MixedViewUnslicedField, match="block_tables has 4 entries"):
        _MixedDecodeView(_batch(block_rows=4), 2)


def _meta(cu, slots, max_q):
    return SimpleNamespace(
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        state_slot_out=torch.tensor(slots, dtype=torch.int32),
        max_seqlen_q=max_q,
        max_seqlen_k=max_q,
        has_cached=False,
        total_kv=0,
    )


def test_carrier_has_whole_batch_spans_and_state_slots():
    prefill = _meta([0, 5, 9], [7, 8], 5)
    decode = _meta([0, 1, 2, 3], [4, 5, 6], 1)
    merged = _mixed_carrier(prefill, decode, 2, 3, 9, 3, 6)
    assert merged.is_mixed is True
    assert merged.cu_seqlens_q.tolist() == [0, 5, 9, 10, 11, 12]
    # Padded to running_bs with 0, as prepare_decode pads.
    assert merged.state_slot_out.tolist() == [7, 8, 4, 5, 6, 0]
    assert (merged.num_prefill_seqs, merged.num_decode_seqs) == (2, 3)
    assert (merged.num_prefill_tokens, merged.num_decode_tokens) == (9, 3)
    assert merged.prefill_attn_metadata is prefill
    assert merged.decode_attn_metadata is decode


@pytest.mark.parametrize("fn", ["prepare_mixed", "_build_ubatch_mixed_metadata"])
def test_both_builders_assemble_the_carrier_through_the_helper(fn):
    """A second hand-built copy is how the two drifted apart before."""
    tree = ast.parse(_V4_ATTN.read_text(encoding="utf-8"))
    body = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn
    )
    src = ast.unparse(body)
    assert "_mixed_carrier(" in src
    assert "AttentionMetaData_DSV4(" not in src
