# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 fp8 KV cache + gluon PA port — subclass framework integration.

These tests cover the port of the fp8 KV-cache + gluon paged-attention sparse
attention from ``origin/ganyi/shuffle_kv_cache_fp8_eagle`` onto main's attention
framework, rebuilt as a first-class ``PagedAttentionImpl`` subclass
(``SparseMHAPagedAttentionImpl``) selected through ``Attention(impl_cls=...)``.

Source line ranges grafted by later tasks (origin/ganyi/shuffle_kv_cache_fp8_eagle):
  sparse_attn.py:
    971-973  ASM_PAGE_SIZE / PAGES_PER_SPARSE_BLOCK constants
    231-358  _fused_qknorm_rope_kv_insert_shuffle_kernel
    360-419  minimax_m3_fused_qknorm_rope_kv_insert_shuffle (host wrapper)
    976-1085 _build_sparse_block_table_kernel + minimax_m3_build_sparse_block_table
    1087-... minimax_m3_sparse_attn_decode_asm (topk->block table->gluon decode)
    1223-1330 _build_sparse_block_table_prefill_kernel + ..._prefill wrapper
    1332-... minimax_m3_sparse_attn_prefill_asm
    1412-... _run_prefill_fp8_gluon / gluon decode runner (run_pa_decode_gluon)
  index_topk.py:
    EMIT_SPARSE_BT constexpr branch + emit_sparse_block_table arg
    (exclude MAX_Q / causal_len eagle3 spec-decode machinery)

Run (most GPU-free; gluon/insert tests require GPU):
    source /opt/venv/bin/activate
    python -m pytest tests/test_minimax_m3_fp8_gluon_pa.py -v
"""

from __future__ import annotations

import inspect

import pytest  # noqa: F401


# The repo conftest stubs atom.config for CPU scheduler tests; evict so the real
# atom.model_ops modules import (mirrors tests/test_minimax_m3_sparse_attn_asm.py).
def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()


# ── Task 0: subclass selection ────────────────────────────────────────────────


def test_sparse_impl_is_paged_attention_subclass():
    """SparseMHAPagedAttentionImpl extends the standard MHA impl so it reuses
    forward / forward_impl and only overrides rope_cache + dispatch_backend."""
    from atom.model_ops.attention_mha import (
        PagedAttentionImpl,
        SparseMHAPagedAttentionImpl,
    )

    assert issubclass(SparseMHAPagedAttentionImpl, PagedAttentionImpl)


def test_sparse_impl_overrides_only_rope_cache_and_dispatch():
    """The subclass must override exactly rope_cache and dispatch_backend (the two
    framework hooks forward_impl calls) and inherit everything else."""
    from atom.model_ops.attention_mha import (
        PagedAttentionImpl,
        SparseMHAPagedAttentionImpl,
    )

    for name in ("rope_cache", "dispatch_backend"):
        assert (
            SparseMHAPagedAttentionImpl.__dict__.get(name) is not None
        ), f"{name} must be overridden on the subclass"
    # forward_impl/forward stay inherited (confirmed design: standard forward)
    assert "forward_impl" not in SparseMHAPagedAttentionImpl.__dict__
    assert "forward" not in SparseMHAPagedAttentionImpl.__dict__
    assert SparseMHAPagedAttentionImpl.forward_impl is PagedAttentionImpl.forward_impl


def test_attention_accepts_impl_cls_kwarg():
    """Attention.__init__ exposes an impl_cls override so a model can plug in the
    sparse impl while keeping the backend's metadata builder."""
    from atom.model_ops.paged_attention import Attention

    params = inspect.signature(Attention.__init__).parameters
    assert "impl_cls" in params, "Attention.__init__ must accept impl_cls"
    # Default None so existing models fall back to attn_backend.get_impl_cls().
    assert params["impl_cls"].default is None
