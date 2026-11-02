# SPDX-License-Identifier: MIT
"""Start-up warm-up of the prefix-hit prefill unified_attention path.

Host-side only: the token table, the warm-up shape list and the gating, all in
`atom.model_ops.unified_prefill_warmup`, which needs only torch, so this file
runs in the CPU unit-test job. No device is touched: the impls are stand-in
modules carrying the two methods the gate calls.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from torch import nn

from atom.model_ops import unified_prefill_warmup as upw


class TestTokenBlockTable:
    def test_columns_are_padded_to_a_multiple_of_16(self):
        """The row stride is a Triton int specialization: one value class only."""
        for max_k in (1, 15, 16, 17, 407, 170585):
            cu_k = torch.tensor([0], dtype=torch.int32)
            table = upw.token_block_table(cu_k, max_k, max_k)
            assert table.shape == (1, -(-max_k // 16) * 16)
            assert table.stride(0) % 16 == 0
            assert table.dtype == torch.int32

    def test_rows_point_at_each_requests_packed_rows(self):
        ctx = [5, 21, 3]
        cu_k = torch.tensor([0, 5, 26], dtype=torch.int32)
        total = sum(ctx)
        table = upw.token_block_table(cu_k, max(ctx), total)
        for i, n in enumerate(ctx):
            start = int(cu_k[i])
            assert table[i, :n].tolist() == list(range(start, start + n))
        # padding and the tails of shorter rows stay inside the gathered tensor
        assert int(table.max()) == total - 1
        assert int(table.min()) == 0


class TestWarmupShapes:
    @pytest.mark.parametrize("max_q", [1, 32, 63, 64, 100, 4096, 6144])
    def test_every_launch_is_a_prefix_hit_within_max_q(self, max_q):
        shapes = upw.unified_prefill_warmup_shapes(max_q)
        assert shapes
        for q_lens, ctx_lens in shapes:
            assert len(q_lens) == len(ctx_lens) >= 1
            assert max(q_lens) <= max_q
            # bottom-right causal over a cached prefix: context covers the rows
            assert all(c > q > 0 for q, c in zip(q_lens, ctx_lens))

    def test_largest_step_and_both_batch_kinds_are_covered(self):
        shapes = upw.unified_prefill_warmup_shapes(4096)
        assert any(max(q) == 4096 for q, _ in shapes)
        assert any(len(q) == 1 for q, _ in shapes)
        assert any(len(q) > 1 for q, _ in shapes)

    def test_rows_step_no_wider_than_the_narrowest_split_bucket(self):
        """Adjacent KV-split counts sit >= 1.33x apart in program count."""
        rows = sorted(
            {q[0] for q, _ in upw.unified_prefill_warmup_shapes(4096) if q[0] >= 64}
        )
        assert rows[0] == 64
        assert all(b / a <= 1.26 for a, b in itertools.pairwise(rows))

    @pytest.mark.parametrize("max_q", [32, 100, 4096])
    def test_every_row_count_covers_each_request_count_specialization(self, max_q):
        """Triton specializes num_seqs on == 1 and % 16 == 0, so every longest
        row count (which sets the query block) is warmed with 1 request, a few,
        and a multiple of 16 -- a step with 16 or 32 requests compiles nothing."""
        kinds = {}
        for q, _ in upw.unified_prefill_warmup_shapes(max_q):
            n = len(q)
            kinds.setdefault(max(q), set()).add(
                "one" if n == 1 else "x16" if n % 16 == 0 else "few"
            )
        assert kinds
        assert all(k == {"one", "few", "x16"} for k in kinds.values()), kinds


class _Impl(nn.Module):
    """Stands in for PagedAttentionImpl: the gate reads only these members."""

    def __init__(self, heads, kv_heads, head_dim, supported, calls):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = heads, kv_heads, head_dim
        self._supported, self._calls = supported, calls

    def _unified_fp8_prefill_supported(self):
        return self._supported

    def warmup_unified_fp8_prefill_attention(self, dtype):
        self._calls.append((self.num_heads, dtype))
        return 1


class TestWarmupGate:
    def _model(self, supported=True):
        calls = []
        geometries = [(8, 1, 256), (8, 1, 256), (16, 2, 256)]
        impls = [_Impl(*g, supported, calls) for g in geometries]
        # a module without the warm-up method (e.g. a linear layer) is ignored
        model = nn.ModuleList([nn.Linear(2, 2), *impls])
        return model, calls

    def test_one_warmup_per_head_geometry(self, monkeypatch):
        monkeypatch.setattr(upw.envs, "ATOM_PREFILL_UNIFIED_ATTN_MAX_Q", 4096)
        monkeypatch.setattr(upw.envs, "ATOM_PREFILL_UNIFIED_ATTN_WARMUP", True)
        model, calls = self._model()
        upw.warmup_unified_prefill_attention(model, torch.bfloat16)
        assert calls == [(8, torch.bfloat16), (16, torch.bfloat16)]

    @pytest.mark.parametrize("max_q, warmup", [(0, True), (4096, False)])
    def test_off_switches(self, monkeypatch, max_q, warmup):
        monkeypatch.setattr(upw.envs, "ATOM_PREFILL_UNIFIED_ATTN_MAX_Q", max_q)
        monkeypatch.setattr(upw.envs, "ATOM_PREFILL_UNIFIED_ATTN_WARMUP", warmup)
        model, calls = self._model()
        upw.warmup_unified_prefill_attention(model, torch.bfloat16)
        assert calls == []

    def test_unsupported_layers_are_skipped(self, monkeypatch):
        monkeypatch.setattr(upw.envs, "ATOM_PREFILL_UNIFIED_ATTN_MAX_Q", 4096)
        monkeypatch.setattr(upw.envs, "ATOM_PREFILL_UNIFIED_ATTN_WARMUP", True)
        model, calls = self._model(supported=False)
        upw.warmup_unified_prefill_attention(model, torch.bfloat16)
        assert calls == []
