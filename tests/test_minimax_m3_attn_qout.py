# SPDX-License-Identifier: Apache-2.0
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
