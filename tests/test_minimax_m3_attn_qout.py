# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_M3_ATTN_QOUT: ``SparseMHAPagedAttentionImpl._qout_buffers`` (when the NVFP4 decode attention writes
o_proj's MXFP8 input itself) and ``attn_qout.aiter_supports`` (feature detection of aiter's ``quant_out``).
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl
    from atom.model_ops.minimax_m3 import attn_qout
except ImportError as exc:  # no aiter on a CPU-only runner
    skip_if_dependency_missing(exc, "atom.model_ops.minimax_m3.attn_qout")

_MOD = "aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode"


def _call(impl, rows, q_rows=None):
    q = torch.zeros((q_rows or rows, 16, 128), dtype=torch.bfloat16)
    return SparseMHAPagedAttentionImpl._qout_buffers(impl, q, rows)


def test_buffers_shapes_and_parking():
    impl = SimpleNamespace(_qout_enabled=True, _qout_result="stale")
    codes, scales = _call(impl, 56)
    assert codes.shape == (56, 2048) and codes.dtype == torch.uint8
    assert scales.shape == (56, 64) and scales.dtype == torch.uint8
    pc, ps, rows = impl._qout_result
    assert rows == 56 and pc.dtype == attn_qout.FP8_DTYPE
    assert (
        ps.dtype == torch.float8_e8m0fnu
        and pc.shape == (56, 2048)
        and ps.shape == (56, 64)
    )
    # the parked tensors are views of the buffers the attention writes
    assert pc.data_ptr() == codes.data_ptr() and ps.data_ptr() == scales.data_ptr()


@pytest.mark.parametrize(
    "enabled, rows, q_rows",
    [
        (False, 56, 56),  # flag / eligibility off
        (True, 4, 4),  # below the measured window
        (True, 192, 192),  # above it
        (True, 56, 64),  # pad rows: the bf16 output keeps its path
    ],
)
def test_refused_clears_parked_result(enabled, rows, q_rows):
    impl = SimpleNamespace(_qout_enabled=enabled, _qout_result="stale")
    assert _call(impl, rows, q_rows) is None
    assert impl._qout_result is None


def test_aiter_supports(monkeypatch):
    mod = types.ModuleType(_MOD)

    def new(q, kv_cache, topk_idx, quant_out=None):
        return None

    def old(q, kv_cache, topk_idx):
        return None

    for fn, want in ((new, True), (old, False)):
        mod.sparse_pa_nvfp4_decode = fn
        monkeypatch.setitem(sys.modules, _MOD, mod)
        attn_qout.aiter_supports.cache_clear()
        assert attn_qout.aiter_supports() is want
    monkeypatch.setitem(
        sys.modules, _MOD, None
    )  # ImportError: an aiter without the kernel
    attn_qout.aiter_supports.cache_clear()
    assert attn_qout.aiter_supports() is False
    attn_qout.aiter_supports.cache_clear()


def _qout_layer(monkeypatch, *, runs_bf16=False, mode=None, sp_columns=None):
    """A MiniMaxM3SparseAttention stand-in that reaches ``_attn_qout`` with every other gate open."""
    pytest.importorskip("aiter")
    from atom.models.minimax_m3 import model as m3

    monkeypatch.setattr(m3.envs, "ATOM_M3_ATTN_QOUT", True, raising=False)
    monkeypatch.setattr(attn_qout, "aiter_supports", lambda: True)
    cc = SimpleNamespace(cudagraph_mode=mode)
    monkeypatch.setattr(
        m3, "get_current_atom_config", lambda: SimpleNamespace(compilation_config=cc)
    )
    impl = SimpleNamespace(
        kv_cache_dtype="nvfp4", indexer_cp_group=None, _qout_enabled=False
    )
    layer = SimpleNamespace(
        o_proj=SimpleNamespace(native_a8_group_rows=4, runs_bf16=runs_bf16),
        attn=SimpleNamespace(impl=impl, sp_columns=sp_columns),
    )
    return m3.MiniMaxM3SparseAttention._attn_qout(layer), impl


def test_attn_qout_gate_open(monkeypatch):
    ok, impl = _qout_layer(monkeypatch)
    assert ok and impl._qout_enabled


def test_attn_qout_refused_on_bf16_oproj(monkeypatch):
    ok, impl = _qout_layer(monkeypatch, runs_bf16=True)
    assert not ok and not impl._qout_enabled


@pytest.mark.parametrize("name", ["TR_HYBRID", "FULL_AND_TR_HYBRID"])
def test_attn_qout_refused_under_tr_hybrid(monkeypatch, name):
    from atom.config import CUDAGraphMode

    ok, impl = _qout_layer(monkeypatch, mode=CUDAGraphMode[name])
    assert not ok and not impl._qout_enabled


def test_attn_qout_allowed_under_full_graph(monkeypatch):
    from atom.config import CUDAGraphMode

    ok, _ = _qout_layer(monkeypatch, mode=CUDAGraphMode.FULL_DECODE_ONLY)
    assert ok


def test_attn_qout_refused_under_ulysses_sp(monkeypatch):
    """The head exchange (gather_heads) only carries the BF16 output, so the codes cannot follow it."""
    ok, impl = _qout_layer(monkeypatch, sp_columns=object())
    assert not ok and not impl._qout_enabled
