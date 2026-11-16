# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The embeddings' pre-mix comes from a shared (1, 0, ..., 0) table: the same
values as the zero-and-fill it replaces, and no table a recording may have
baked is ever freed."""

import pytest
import torch

mhc = pytest.importorskip(
    "atom.model_ops.deepseek_v41.mhc",
    reason="mhc imports aiter at module load",
    exc_type=ImportError,
)

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")


@pytest.fixture(autouse=True)
def fresh_tables(monkeypatch):
    monkeypatch.setattr(mhc, "_PRE_MIX_TABLES", {})


def reference(shape):
    pre_mix = torch.zeros(shape, dtype=torch.float32)
    pre_mix[..., 0] = 1
    return pre_mix


@pytest.mark.parametrize("shape", [(6, 4), (1, 6, 4), (2, 3, 4), (5, 2)])
def test_values_match_the_zero_and_fill(shape):
    for _ in range(2):  # the allocating call, then the table's slice
        pre_mix = mhc._embedding_pre_mix(shape, "cpu")
        assert pre_mix.shape == shape and pre_mix.dtype == torch.float32
        assert pre_mix.is_contiguous() and torch.equal(pre_mix, reference(shape))


def test_rows_come_from_one_table_and_outgrown_tables_live_on():
    big = mhc._embedding_pre_mix((8, 4), "cpu")
    small = mhc._embedding_pre_mix((1, 3, 4), "cpu")
    assert small.data_ptr() == big.data_ptr()
    bigger = mhc._embedding_pre_mix((16, 4), "cpu")
    tables = mhc._PRE_MIX_TABLES[(torch.device("cpu"), 4)]
    assert [t.shape[0] for t in tables] == [8, 16]
    assert tables[0].data_ptr() == big.data_ptr()
    assert mhc._embedding_pre_mix((12, 4), "cpu").data_ptr() == bigger.data_ptr()


def test_from_embeddings_state():
    hidden = torch.randn(1, 5, 32)
    state = mhc.SinglePassHCState.from_embeddings(hidden, 4)
    assert torch.equal(state.pre_mix, reference((1, 5, 4)))
    assert torch.equal(state.residual, hidden.unsqueeze(-2).expand(1, 5, 4, 32))


def test_compiled_trace_allocates(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    pre_mix = mhc._embedding_pre_mix((3, 4), "cpu")
    assert torch.equal(pre_mix, reference((3, 4)))
    assert mhc._PRE_MIX_TABLES == {}


@gpu
def test_recording_reads_the_table_and_never_grows_it():
    dev = torch.device("cuda", torch.cuda.current_device())
    warm = mhc._embedding_pre_mix((4, 4), dev)
    graph, outs = torch.cuda.CUDAGraph(), {}
    with torch.cuda.graph(graph):
        outs["fits"] = mhc._embedding_pre_mix((2, 4), dev)
        outs["outgrows"] = mhc._embedding_pre_mix((9, 4), dev)
    graph.replay()
    torch.cuda.synchronize()
    assert outs["fits"].data_ptr() == warm.data_ptr()
    tables = mhc._PRE_MIX_TABLES[(dev, 4)]
    assert [t.shape[0] for t in tables] == [4]
    assert torch.equal(outs["outgrows"].cpu(), reference((9, 4)))
