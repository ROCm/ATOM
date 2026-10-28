# SPDX-License-Identifier: MIT
"""Start-up warm-up of the prefix-hit prefill unified_attention path.

Host-side only: the token table, the warm-up shape list and the gating. They
live beside the kernel wrappers in `atom.model_ops.attention_mha`, which imports
triton and aiter, so this file self-skips on the CPU CI runner like the other
attention tests. No device is touched: the impls are built without __init__.
"""

from __future__ import annotations

import itertools

import pytest

pytest.importorskip("triton", reason="attention_mha imports triton kernels")
pytest.importorskip("aiter", reason="attention_mha imports the AITER runtime")

import torch
from torch import nn

from atom.model_ops import attention_mha as mha


class TestTokenBlockTable:
    def test_columns_are_padded_to_a_multiple_of_16(self):
        """The row stride is a Triton int specialization: one value class only."""
        for max_k in (1, 15, 16, 17, 407, 170585):
            cu_k = torch.tensor([0], dtype=torch.int32)
            table = mha.token_block_table(cu_k, max_k, max_k)
            assert table.shape == (1, -(-max_k // 16) * 16)
            assert table.stride(0) % 16 == 0
            assert table.dtype == torch.int32

    def test_rows_point_at_each_requests_packed_rows(self):
        ctx = [5, 21, 3]
        cu_k = torch.tensor([0, 5, 26], dtype=torch.int32)
        total = sum(ctx)
        table = mha.token_block_table(cu_k, max(ctx), total)
        for i, n in enumerate(ctx):
            start = int(cu_k[i])
            assert table[i, :n].tolist() == list(range(start, start + n))
        # padding and the tails of shorter rows stay inside the gathered tensor
        assert int(table.max()) == total - 1
        assert int(table.min()) == 0


class TestWarmupShapes:
    @pytest.mark.parametrize("max_q", [1, 32, 63, 64, 100, 4096, 6144])
    def test_every_launch_is_a_prefix_hit_within_max_q(self, max_q):
        shapes = mha.unified_prefill_warmup_shapes(max_q)
        assert shapes
        for q_lens, ctx_lens in shapes:
            assert len(q_lens) == len(ctx_lens) >= 1
            assert max(q_lens) <= max_q
            # bottom-right causal over a cached prefix: context covers the rows
            assert all(c > q > 0 for q, c in zip(q_lens, ctx_lens))

    def test_largest_step_and_both_batch_kinds_are_covered(self):
        shapes = mha.unified_prefill_warmup_shapes(4096)
        assert any(max(q) == 4096 for q, _ in shapes)
        assert any(len(q) == 1 for q, _ in shapes)
        assert any(len(q) > 1 for q, _ in shapes)

    def test_rows_step_no_wider_than_the_narrowest_split_bucket(self):
        """Adjacent KV-split counts sit >= 1.33x apart in program count."""
        rows = sorted(
            {q[0] for q, _ in mha.unified_prefill_warmup_shapes(4096) if q[0] >= 64}
        )
        assert rows[0] == 64
        assert all(b / a <= 1.26 for a, b in itertools.pairwise(rows))

    def test_no_sixteen_request_batch(self):
        """num_seqs % 16 == 0 is its own specialization; c<=12 never reaches it."""
        assert all(len(q) % 16 for q, _ in mha.unified_prefill_warmup_shapes(4096))


def _impl(heads, kv_heads, head_dim):
    impl = object.__new__(mha.PagedAttentionImpl)
    nn.Module.__init__(impl)
    impl.num_heads, impl.num_kv_heads, impl.head_dim = heads, kv_heads, head_dim
    return impl


class TestWarmupGate:
    def _model(self, monkeypatch, supported=True):
        calls = []
        monkeypatch.setattr(
            mha.PagedAttentionImpl,
            "_unified_fp8_prefill_supported",
            lambda self: supported,
        )
        monkeypatch.setattr(
            mha.PagedAttentionImpl,
            "warmup_unified_fp8_prefill_attention",
            lambda self, dtype: calls.append((self.num_heads, dtype)) or 1,
        )
        model = nn.ModuleList([_impl(8, 1, 256), _impl(8, 1, 256), _impl(16, 2, 256)])
        return model, calls

    def test_one_warmup_per_head_geometry(self, monkeypatch):
        monkeypatch.setattr(mha.envs, "ATOM_PREFILL_UNIFIED_ATTN_MAX_Q", 4096)
        monkeypatch.setattr(mha.envs, "ATOM_PREFILL_UNIFIED_ATTN_WARMUP", True)
        model, calls = self._model(monkeypatch)
        mha.warmup_unified_prefill_attention(model, torch.bfloat16)
        assert calls == [(8, torch.bfloat16), (16, torch.bfloat16)]

    @pytest.mark.parametrize("max_q, warmup", [(0, True), (4096, False)])
    def test_off_switches(self, monkeypatch, max_q, warmup):
        monkeypatch.setattr(mha.envs, "ATOM_PREFILL_UNIFIED_ATTN_MAX_Q", max_q)
        monkeypatch.setattr(mha.envs, "ATOM_PREFILL_UNIFIED_ATTN_WARMUP", warmup)
        model, calls = self._model(monkeypatch)
        mha.warmup_unified_prefill_attention(model, torch.bfloat16)
        assert calls == []

    def test_unsupported_layers_are_skipped(self, monkeypatch):
        monkeypatch.setattr(mha.envs, "ATOM_PREFILL_UNIFIED_ATTN_MAX_Q", 4096)
        monkeypatch.setattr(mha.envs, "ATOM_PREFILL_UNIFIED_ATTN_WARMUP", True)
        model, calls = self._model(monkeypatch, supported=False)
        mha.warmup_unified_prefill_attention(model, torch.bfloat16)
        assert calls == []
