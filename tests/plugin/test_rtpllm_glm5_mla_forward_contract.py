"""Contract-executable tests for GLM5 RTP MLA native forward."""

import builtins
import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention


_FORBIDDEN_CUDA_SPARSE_MODULES = (
    "flashmla_sparse",
    "flash_mla",
    "sparse_mla",
    "attention_mla_sparse",
)


class _FakeDenseBackend:
    def __init__(self, v_head_dim: int):
        self.v_head_dim = v_head_dim
        self.calls = []

    def forward(
        self,
        q,
        compressed_kv,
        k_pe,
        kv_cache,
        layer_id,
        topk_indices=None,
        positions=None,
    ):
        self.calls.append(
            {
                "q": q,
                "compressed_kv": compressed_kv,
                "k_pe": k_pe,
                "kv_cache": kv_cache,
                "layer_id": layer_id,
                "topk_indices": topk_indices,
                "positions": positions,
            }
        )
        return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))


def test_rtp_mla_attention_keeps_legacy_dense_boundary_during_migration():
    backend = _FakeDenseBackend(v_head_dim=16)
    attention = RTPMLAAttention(dense_backend=backend, layer_id=7, kv_cache="cache")
    q = torch.empty(3, 2, 12, dtype=torch.bfloat16)
    compressed_kv = torch.empty(3, 8, dtype=torch.bfloat16)
    k_pe = torch.empty(3, 4, dtype=torch.bfloat16)
    positions = torch.arange(3, dtype=torch.int32)

    output = attention.forward(
        q,
        compressed_kv,
        k_pe,
        positions=positions,
        topk_indices=None,
    )

    assert output.shape == (3, 2, 16)
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["q"] is q
    assert call["compressed_kv"] is compressed_kv
    assert call["k_pe"] is k_pe
    assert call["kv_cache"] == "cache"
    assert call["layer_id"] == 7
    assert call["topk_indices"] is None
    assert call["positions"] is positions


def _guard_sparse_kernel_imports(monkeypatch):
    original_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if any(part in _FORBIDDEN_CUDA_SPARSE_MODULES for part in name.split(".")):
            raise AssertionError(f"M1 dense contract must not import sparse kernel module: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def test_rtp_mla_attention_accepts_explicit_topk_and_passes_it_to_dense_backend(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    attention = RTPMLAAttention(dense_backend=_FakeDenseBackend(v_head_dim=16))
    q = torch.empty(1, 2, 12)
    compressed_kv = torch.empty(1, 8)
    k_pe = torch.empty(1, 4)
    positions = torch.arange(1, dtype=torch.int32)
    topk = torch.tensor([[3, 1, 0, 2]], dtype=torch.int32)

    output = attention.forward(
        q,
        compressed_kv,
        k_pe,
        positions=positions,
        topk_indices=topk,
    )

    assert output.shape == (1, 2, 16)
    assert len(attention.dense_backend.calls) == 1
    assert attention.dense_backend.calls[0]["topk_indices"] is topk


def test_dense_backend_output_does_not_depend_on_topk_values(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    backend = _FakeDenseBackend(v_head_dim=16)
    attention = RTPMLAAttention(dense_backend=backend)
    q = torch.ones(2, 2, 12)
    compressed_kv = torch.empty(2, 8)
    k_pe = torch.empty(2, 4)
    positions = torch.arange(2, dtype=torch.int32)
    topk_a = torch.tensor([[3, 1, 0, 2], [2, 0, 1, 3]], dtype=torch.int32)
    topk_b = torch.tensor([[0, 2, 1, 3], [3, 1, 2, 0]], dtype=torch.int32)

    out_a = attention.forward(
        q,
        compressed_kv,
        k_pe,
        positions=positions,
        topk_indices=topk_a,
    )
    out_b = attention.forward(
        q,
        compressed_kv,
        k_pe,
        positions=positions,
        topk_indices=topk_b,
    )

    assert torch.equal(out_a, out_b)
    assert backend.calls[0]["topk_indices"] is topk_a
    assert backend.calls[1]["topk_indices"] is topk_b


def test_native_forward_signature_exposes_q_scale_argument():
    signature = inspect.signature(RTPMLAAttention.forward)

    assert "q_scale" in signature.parameters


@pytest.mark.parametrize("attr", ["q_proj", "o_proj", "kv_b_proj", "v_head_dim"])
def test_constructor_injects_native_mla_module_attributes(attr):
    modules = SimpleNamespace(
        q_proj=object(),
        o_proj=object(),
        kv_b_proj=object(),
        v_head_dim=16,
    )
    attention = RTPMLAAttention(mla_modules=modules)

    assert getattr(attention, attr) == getattr(modules, attr)


class _FakeQProj:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, query, q_scale=None):
        self.calls.append((query, q_scale))
        return self.output


class _FakeOProj:
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.calls = []

    def __call__(self, tensor):
        self.calls.append(tensor)
        return tensor.new_empty((tensor.shape[0], self.hidden_dim))


def test_native_five_tuple_projects_latent_query_and_applies_o_proj(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    token_count = 3
    num_heads = 2
    qk_head_dim = 4
    v_head_dim = 5
    hidden_dim = 7
    query = torch.arange(token_count * 6, dtype=torch.float32).reshape(token_count, 6)
    q_scale = torch.ones(token_count, 1)
    projected_q = torch.arange(
        token_count * num_heads * qk_head_dim, dtype=torch.float32
    ).reshape(token_count, num_heads * qk_head_dim)
    compressed_kv = torch.empty(token_count, 8)
    k_rope = torch.empty(token_count, 3)
    positions = torch.arange(token_count, dtype=torch.int32)
    backend = _FakeDenseBackend(v_head_dim=v_head_dim)
    modules = SimpleNamespace(
        q_proj=_FakeQProj(projected_q),
        o_proj=_FakeOProj(hidden_dim=hidden_dim),
        kv_b_proj=object(),
        v_head_dim=v_head_dim,
        qk_head_dim=qk_head_dim,
        num_heads=num_heads,
        num_local_heads=num_heads,
    )
    attention = RTPMLAAttention(
        mla_modules=modules,
        dense_backend=backend,
        layer_num=5,
        kv_cache="kv-cache",
    )

    output = attention.forward(
        query,
        compressed_kv,
        k_rope,
        positions=positions,
        q_scale=q_scale,
    )

    assert modules.q_proj.calls == [(query, q_scale)]
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["q"].shape == (token_count, num_heads, qk_head_dim)
    assert torch.equal(call["q"].reshape(token_count, -1), projected_q)
    assert call["compressed_kv"] is compressed_kv
    assert call["k_pe"] is k_rope
    assert call["kv_cache"] == "kv-cache"
    assert call["layer_id"] == 5
    assert len(modules.o_proj.calls) == 1
    assert modules.o_proj.calls[0].shape == (token_count, num_heads * v_head_dim)
    assert output.shape == (token_count, hidden_dim)


def test_rtp_mla_attention_builds_m0_backend_from_mla_modules():
    modules = SimpleNamespace(v_head_dim=16)
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)
    q = torch.empty(2, 4, 12)
    compressed_kv = torch.empty(2, 8)
    k_pe = torch.empty(2, 4)

    output = attention(q, compressed_kv, k_pe, positions=torch.arange(2))

    assert output.shape == (2, 4, 16)


def test_rtp_mla_attention_defaults_to_sparse_backend_from_mla_modules(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    from atom.plugin.rtpllm.attention_backend.rtp_dense_mla_backend import (
        RTPDenseMlaBackend,
    )
    from atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend import (
        RTPSparseMlaBackend,
    )

    modules = SimpleNamespace(v_head_dim=16)
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    assert isinstance(attention.dense_backend, RTPSparseMlaBackend)
    assert isinstance(attention.dense_backend.dense_backend, RTPDenseMlaBackend)


class _FakeKVProj:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.calls = []

    def __call__(self, compressed_kv):
        self.calls.append(compressed_kv)
        output = self.output.to(device=compressed_kv.device, dtype=compressed_kv.dtype)
        if output.shape[0] == 1 and compressed_kv.shape[0] != 1:
            output = output.expand(compressed_kv.shape[0], -1).contiguous()
        return output


class _DeterministicKVProj:
    def __init__(self, output_dim: int):
        self.output_dim = output_dim
        self.calls = []

    def __call__(self, compressed_kv):
        self.calls.append(compressed_kv.detach().clone())
        token_signal = compressed_kv.float().mean(dim=-1, keepdim=True)
        basis = torch.linspace(
            0.0,
            1.0,
            self.output_dim,
            device=compressed_kv.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        return (token_signal + basis).to(dtype=compressed_kv.dtype)


class _FakeRotaryEmbedding:
    is_neox_style = True

    def __init__(self):
        self.calls = []

    def __call__(self, positions, query, key):
        self.calls.append(
            {
                "positions": positions.detach().clone(),
                "query": query.detach().clone(),
                "key": key.detach().clone(),
            }
        )
        offset = positions.to(device=query.device, dtype=query.dtype)
        while offset.ndim < query.ndim:
            offset = offset.unsqueeze(-1)
        query = query + offset
        key_offset = positions.to(device=key.device, dtype=key.dtype)
        while key_offset.ndim < key.ndim:
            key_offset = key_offset.unsqueeze(-1)
        key = key + key_offset
        return query, key


def _patch_forward_context(
    monkeypatch,
    *,
    is_prefill,
    query_start_loc,
    seq_lens=None,
    block_table=None,
    slot_mapping=None,
    kv_cache_data=None,
):
    plugin_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        rtp_cu_seqlens_q=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
    )
    fake_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(
            plugin_metadata=plugin_metadata,
            rtp_kernel_seq_size_per_block=4,
        ),
        context=SimpleNamespace(is_prefill=is_prefill),
        kv_cache_data=kv_cache_data,
    )
    forward_context_module = importlib.import_module("atom.utils.forward_context")
    monkeypatch.setattr(
        forward_context_module,
        "get_forward_context",
        lambda: fake_context,
    )


def _patch_forward_context_with_top_level_attn_metadata(
    monkeypatch,
    *,
    is_prefill,
    seq_lens,
    block_table,
    slot_mapping,
    kv_cache_data=None,
):
    fake_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(
            plugin_metadata=None,
            context_lens=seq_lens,
            block_tables=block_table,
            slot_mapping=slot_mapping,
            cu_seqlens_q=None,
            rtp_kernel_seq_size_per_block=4,
        ),
        context=SimpleNamespace(is_prefill=is_prefill),
        kv_cache_data=kv_cache_data,
    )
    forward_context_module = importlib.import_module("atom.utils.forward_context")
    monkeypatch.setattr(
        forward_context_module,
        "get_forward_context",
        lambda: fake_context,
    )


def _patch_forward_context_without_is_prefill(monkeypatch, *, query_start_loc):
    plugin_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        rtp_cu_seqlens_q=query_start_loc,
        seq_lens=None,
        block_table=None,
        slot_mapping=None,
    )
    fake_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(
            plugin_metadata=plugin_metadata,
            rtp_kernel_seq_size_per_block=4,
        ),
        context=SimpleNamespace(),
    )
    forward_context_module = importlib.import_module("atom.utils.forward_context")
    monkeypatch.setattr(
        forward_context_module,
        "get_forward_context",
        lambda: fake_context,
    )


def test_default_dense_mla_backend_computes_nonzero_attention(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    from atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend import (
        RTPSparseMlaBackend,
    )

    q = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
    compressed_kv = torch.ones(2, 4, dtype=torch.float32)
    # Per token: [k_nope_dim=2, v_head_dim=1].
    kv_projection = torch.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]])
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(kv_projection),
    )
    _patch_forward_context(
        monkeypatch,
        is_prefill=True,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    output = attention(q, compressed_kv, q.new_empty((2, 0)), positions=torch.arange(2))

    assert isinstance(attention.dense_backend, RTPSparseMlaBackend)
    assert output.shape == (2, 1, 1)
    assert not torch.equal(output, torch.zeros_like(output))
    assert len(modules.kv_b_proj.calls) == 1
    assert modules.kv_b_proj.calls[0] is compressed_kv


def test_default_dense_mla_backend_rejects_missing_multi_token_metadata(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.randn(2, 1, 2)
    compressed_kv = torch.ones(2, 4)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.empty(2, 3)),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    with pytest.raises(ValueError, match="query_start_loc metadata"):
        attention(q, compressed_kv, q.new_empty((2, 0)), positions=torch.arange(2))


def test_default_dense_mla_backend_decode_reads_history_from_raw_cache(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    compressed_kv = torch.tensor([[9.0, 9.0, 9.0, 9.0]], dtype=torch.float32)
    # The backend projects each latent token into [k_nope0, k_nope1, v].
    kv_projection = torch.tensor([[0.0, 0.0, 1.0]])
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(kv_projection),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 4))
    # Three historical latent tokens are already in cache.
    layer_cache.kv_cache_base[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    layer_cache.kv_cache_base[0, 1] = torch.tensor([2.0, 0.0, 0.0, 0.0])
    layer_cache.kv_cache_base[0, 2] = torch.tensor([3.0, 0.0, 0.0, 0.0])
    _patch_forward_context(
        monkeypatch,
        is_prefill=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([3], dtype=torch.int32),
    )
    attention = RTPMLAAttention(
        mla_modules=modules,
        layer_num=3,
        kv_cache=layer_cache,
    )

    output = attention(q, compressed_kv, q.new_empty((1, 0)), positions=torch.arange(1))

    assert output.shape == (1, 1, 1)
    assert layer_cache.kv_cache_base[0, 3].tolist() == [9.0, 9.0, 9.0, 9.0]


def test_default_dense_mla_backend_decode_uses_top_level_rtp_metadata(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    compressed_kv = torch.tensor([[9.0, 9.0, 9.0, 9.0]], dtype=torch.float32)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.tensor([[0.0, 0.0, 1.0]])),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 4))
    layer_cache.kv_cache_base[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    _patch_forward_context_with_top_level_attn_metadata(
        monkeypatch,
        is_prefill=False,
        seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([1], dtype=torch.int32),
    )
    attention = RTPMLAAttention(
        mla_modules=modules,
        layer_num=3,
        kv_cache=layer_cache,
    )

    output = attention(q, compressed_kv, q.new_empty((1, 0)), positions=torch.arange(1))

    assert output.shape == (1, 1, 1)
    assert layer_cache.kv_cache_base[0, 1].tolist() == [9.0, 9.0, 9.0, 9.0]


def test_default_dense_mla_backend_decode_rebuilds_stale_query_start_loc(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    compressed_kv = torch.tensor([[9.0, 9.0, 9.0, 9.0]], dtype=torch.float32)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.tensor([[0.0, 0.0, 1.0]])),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 4))
    layer_cache.kv_cache_base[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    _patch_forward_context(
        monkeypatch,
        is_prefill=False,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([1], dtype=torch.int32),
    )
    attention = RTPMLAAttention(
        mla_modules=modules,
        layer_num=3,
        kv_cache=layer_cache,
    )

    output = attention(q, compressed_kv, q.new_empty((1, 0)), positions=torch.arange(1))

    assert output.shape == (1, 1, 1)
    assert layer_cache.kv_cache_base[0, 1].tolist() == [9.0, 9.0, 9.0, 9.0]


def test_default_sparse_wrapper_validates_topk_but_falls_back_to_dense(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    from atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend import (
        RTPSparseMlaBackend,
    )

    dense_backend = _FakeDenseBackend(v_head_dim=4)
    sparse_impl = SimpleNamespace(calls=[])
    backend = RTPSparseMlaBackend(
        dense_backend=dense_backend,
        sparse_impl=sparse_impl,
        v_head_dim=4,
    )
    q = torch.ones(2, 1, 3)
    compressed_kv = torch.ones(2, 5)
    k_pe = torch.ones(2, 2)
    positions = torch.arange(2)
    topk = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)

    output = backend.forward(
        q,
        compressed_kv,
        k_pe,
        kv_cache="cache",
        layer_id=9,
        topk_indices=topk,
        positions=positions,
    )

    assert output.shape == (2, 1, 4)
    assert len(dense_backend.calls) == 1
    assert dense_backend.calls[0]["topk_indices"] is topk
    assert dense_backend.calls[0]["positions"] is positions
    assert sparse_impl.calls == []


def test_default_dense_mla_backend_resolves_kv_cache_from_forward_context(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    compressed_kv = torch.tensor([[9.0, 9.0, 9.0, 9.0]], dtype=torch.float32)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.tensor([[0.0, 0.0, 1.0]])),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 4))
    layer_cache.kv_cache_base[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    _patch_forward_context(
        monkeypatch,
        is_prefill=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([1], dtype=torch.int32),
        kv_cache_data={"layer_3": SimpleNamespace(k_cache=layer_cache)},
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    output = attention(q, compressed_kv, q.new_empty((1, 0)), positions=torch.arange(1))

    assert output.shape == (1, 1, 1)
    assert layer_cache.kv_cache_base[0, 1].tolist() == [9.0, 9.0, 9.0, 9.0]


def test_default_dense_mla_backend_accepts_noncontiguous_compressed_kv(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
    storage = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    compressed_kv = storage[:, ::2]
    assert not compressed_kv.is_contiguous()
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])),
    )
    _patch_forward_context(
        monkeypatch,
        is_prefill=True,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    output = attention(q, compressed_kv, q.new_empty((2, 0)), positions=torch.arange(2))

    assert output.shape == (2, 1, 1)
    assert modules.kv_b_proj.calls[0].is_contiguous()


def test_default_dense_mla_backend_skips_negative_slot_mapping(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
    compressed_kv = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32
    )
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 4))
    _patch_forward_context(
        monkeypatch,
        is_prefill=True,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([-1, 1], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3, kv_cache=layer_cache)

    attention(q, compressed_kv, q.new_empty((2, 0)), positions=torch.arange(2))

    assert torch.equal(layer_cache.kv_cache_base[0, -1], torch.zeros(4))
    assert torch.equal(layer_cache.kv_cache_base[0, 1], compressed_kv[1])


def test_default_dense_mla_backend_rejects_oob_slot_mapping(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    compressed_kv = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.tensor([[1.0, 0.0, 1.0]])),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 4))
    _patch_forward_context(
        monkeypatch,
        is_prefill=True,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        slot_mapping=torch.tensor([4], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3, kv_cache=layer_cache)

    with pytest.raises(RuntimeError, match="out-of-bounds slot_mapping"):
        attention(q, compressed_kv, q.new_empty((1, 0)), positions=torch.arange(1))

    assert torch.equal(layer_cache.kv_cache_base, torch.zeros_like(layer_cache.kv_cache_base))


def test_default_dense_mla_backend_writes_post_rope_kpe_to_cache(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    rotary_emb = _FakeRotaryEmbedding()
    q = torch.tensor([[[1.0, 2.0, 10.0, 20.0]], [[3.0, 4.0, 30.0, 40.0]]])
    compressed_kv = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    k_pe = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
    positions = torch.tensor([5, 7], dtype=torch.long)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        rotary_emb=rotary_emb,
        kv_b_proj=_FakeKVProj(torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 5))
    _patch_forward_context(
        monkeypatch,
        is_prefill=True,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3, kv_cache=layer_cache)

    attention(q, compressed_kv, k_pe, positions=positions)

    expected_k_pe = k_pe + positions.to(k_pe.dtype).unsqueeze(-1)
    expected_cache = torch.cat((compressed_kv, expected_k_pe), dim=-1)
    assert torch.equal(layer_cache.kv_cache_base[0, :2], expected_cache)
    assert torch.equal(rotary_emb.calls[0]["positions"], positions)


def test_default_dense_mla_backend_uses_post_rope_q_for_attention(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    from atom.plugin.rtpllm.attention_backend.rtp_dense_mla_backend import (
        RTPDenseMlaBackend,
    )

    captured = {}

    def _fake_causal_attention(q, key, value, query_start_loc, scale):
        del key, query_start_loc, scale
        captured["q"] = q.detach().clone()
        return value.new_zeros((q.shape[0], q.shape[1], value.shape[-1]))

    monkeypatch.setattr(
        RTPDenseMlaBackend,
        "_causal_attention",
        staticmethod(_fake_causal_attention),
    )
    rotary_emb = _FakeRotaryEmbedding()
    q = torch.tensor([[[1.0, 2.0, 10.0, 20.0]], [[3.0, 4.0, 30.0, 40.0]]])
    compressed_kv = torch.ones(2, 3)
    k_pe = torch.ones(2, 2)
    positions = torch.tensor([5, 7], dtype=torch.long)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        rotary_emb=rotary_emb,
        kv_b_proj=_FakeKVProj(torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])),
    )
    _patch_forward_context(
        monkeypatch,
        is_prefill=True,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    attention(q, compressed_kv, k_pe, positions=positions)

    expected_q = q.clone()
    expected_q[..., -2:] += positions.to(q.dtype).view(2, 1, 1)
    assert torch.equal(captured["q"], expected_q)


def test_default_dense_mla_backend_decode_history_kpe_not_double_roped(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    from atom.plugin.rtpllm.attention_backend.rtp_dense_mla_backend import (
        RTPDenseMlaBackend,
    )

    captured_k_pe = []
    original_project_kv = RTPDenseMlaBackend._project_kv

    def _capture_project_kv(self, q, compressed_kv, k_pe):
        captured_k_pe.append(k_pe.detach().clone())
        return original_project_kv(self, q, compressed_kv, k_pe)

    monkeypatch.setattr(RTPDenseMlaBackend, "_project_kv", _capture_project_kv)
    rotary_emb = _FakeRotaryEmbedding()
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        rotary_emb=rotary_emb,
        kv_b_proj=_FakeKVProj(torch.tensor([[1.0, 0.0, 1.0]])),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 5))
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3, kv_cache=layer_cache)
    prefill_k_pe = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    prefill_positions = torch.tensor([4, 5], dtype=torch.long)
    _patch_forward_context(
        monkeypatch,
        is_prefill=True,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
    )
    attention(
        torch.ones(2, 1, 4),
        torch.ones(2, 3),
        prefill_k_pe,
        positions=prefill_positions,
    )

    decode_k_pe = torch.tensor([[50.0, 60.0]])
    decode_positions = torch.tensor([6], dtype=torch.long)
    _patch_forward_context(
        monkeypatch,
        is_prefill=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([2], dtype=torch.int32),
    )
    attention(
        torch.ones(1, 1, 4),
        torch.ones(1, 3),
        decode_k_pe,
        positions=decode_positions,
    )

    expected_history_k_pe = torch.cat(
        (
            prefill_k_pe + prefill_positions.to(prefill_k_pe.dtype).unsqueeze(-1),
            decode_k_pe + decode_positions.to(decode_k_pe.dtype).unsqueeze(-1),
        ),
        dim=0,
    )
    assert torch.equal(captured_k_pe[-1], expected_history_k_pe)


def test_default_dense_mla_backend_rejects_missing_is_prefill_metadata(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.randn(1, 1, 2)
    compressed_kv = torch.ones(1, 4)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.empty(1, 3)),
    )
    _patch_forward_context_without_is_prefill(
        monkeypatch,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    with pytest.raises(ValueError, match="context.is_prefill"):
        attention(q, compressed_kv, q.new_empty((1, 0)), positions=torch.arange(1))


@pytest.mark.parametrize(
    ("field_name", "seq_lens", "block_table", "slot_mapping"),
    [
        ("seq_lens", None, torch.tensor([[0]], dtype=torch.int32), torch.tensor([0])),
        ("block_table", torch.tensor([1], dtype=torch.int32), None, torch.tensor([0])),
        ("slot_mapping", torch.tensor([1], dtype=torch.int32), torch.tensor([[0]], dtype=torch.int32), None),
    ],
)
def test_default_dense_mla_backend_decode_requires_rtp_metadata(
    monkeypatch, field_name, seq_lens, block_table, slot_mapping
):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.randn(1, 1, 2)
    compressed_kv = torch.ones(1, 4)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.empty(1, 3)),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 4))
    _patch_forward_context(
        monkeypatch,
        is_prefill=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3, kv_cache=layer_cache)

    with pytest.raises(ValueError, match=field_name):
        attention(q, compressed_kv, q.new_empty((1, 0)), positions=torch.arange(1))


def test_default_dense_mla_backend_decode_requires_readable_cache(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.randn(1, 1, 2)
    compressed_kv = torch.ones(1, 4)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.empty(1, 3)),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.empty(0))
    _patch_forward_context(
        monkeypatch,
        is_prefill=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3, kv_cache=layer_cache)

    with pytest.raises(ValueError, match="kv_cache_base"):
        attention(q, compressed_kv, q.new_empty((1, 0)), positions=torch.arange(1))


def test_default_dense_mla_backend_rejects_fp8_kv_cache(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.randn(1, 1, 2)
    compressed_kv = torch.ones(1, 4)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.empty(1, 3)),
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, 4, dtype=torch.uint8))
    _patch_forward_context(
        monkeypatch,
        is_prefill=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3, kv_cache=layer_cache)

    with pytest.raises(NotImplementedError, match="FP8 KV cache"):
        attention(q, compressed_kv, q.new_empty((1, 0)), positions=torch.arange(1))


def test_default_dense_mla_backend_glm5_shape_bf16_cache_roundtrip(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    num_heads = 32
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    kv_lora_rank = 512
    kv_dim = kv_lora_rank + qk_rope_head_dim
    output_dim = num_heads * (qk_nope_head_dim + v_head_dim)
    kv_proj = _DeterministicKVProj(output_dim)
    rotary_emb = _FakeRotaryEmbedding()
    modules = SimpleNamespace(
        v_head_dim=v_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        rotary_emb=rotary_emb,
        kv_b_proj=kv_proj,
    )
    layer_cache = SimpleNamespace(kv_cache_base=torch.zeros(1, 4, kv_dim, dtype=torch.bfloat16))
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3, kv_cache=layer_cache)
    q_prefill = torch.randn(
        3,
        num_heads,
        qk_nope_head_dim + qk_rope_head_dim,
        dtype=torch.bfloat16,
    )
    compressed_prefill = torch.randn(3, kv_lora_rank, dtype=torch.bfloat16)
    k_pe_prefill = torch.randn(3, qk_rope_head_dim, dtype=torch.bfloat16)
    _patch_forward_context(
        monkeypatch,
        is_prefill=True,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    prefill_output = attention(
        q_prefill,
        compressed_prefill,
        k_pe_prefill,
        positions=torch.arange(3),
    )

    assert prefill_output.shape == (3, num_heads, v_head_dim)
    expected_prefill_k_pe = k_pe_prefill + torch.arange(3).to(
        dtype=k_pe_prefill.dtype
    ).unsqueeze(-1)
    expected_prefill_cache = torch.cat((compressed_prefill, expected_prefill_k_pe), dim=-1)
    assert torch.equal(layer_cache.kv_cache_base[0, :3], expected_prefill_cache)

    q_decode = torch.randn(
        1,
        num_heads,
        qk_nope_head_dim + qk_rope_head_dim,
        dtype=torch.bfloat16,
    )
    compressed_decode = torch.randn(1, kv_lora_rank, dtype=torch.bfloat16)
    k_pe_decode = torch.randn(1, qk_rope_head_dim, dtype=torch.bfloat16)
    _patch_forward_context(
        monkeypatch,
        is_prefill=False,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([3], dtype=torch.int32),
    )

    decode_output = attention(
        q_decode,
        compressed_decode,
        k_pe_decode,
        positions=torch.arange(1),
    )

    assert decode_output.shape == (1, num_heads, v_head_dim)
    expected_decode_k_pe = k_pe_decode + torch.arange(1).to(
        dtype=k_pe_decode.dtype
    ).unsqueeze(-1)
    expected_decode_cache = torch.cat((compressed_decode, expected_decode_k_pe), dim=-1)
    assert torch.equal(layer_cache.kv_cache_base[0, 3:4], expected_decode_cache)
    expected_history = torch.cat((compressed_prefill, compressed_decode), dim=0)
    assert torch.equal(kv_proj.calls[-1], expected_history)


def test_default_dense_mla_backend_rejects_bad_kv_projection_shape(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    q = torch.randn(2, 1, 2)
    compressed_kv = torch.ones(2, 4)
    modules = SimpleNamespace(
        v_head_dim=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=0,
        kv_b_proj=_FakeKVProj(torch.empty(2, 2)),
    )
    _patch_forward_context(
        monkeypatch,
        is_prefill=True,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )
    attention = RTPMLAAttention(mla_modules=modules, layer_num=3)

    with pytest.raises(ValueError, match="kv_b_proj output shape mismatch"):
        attention(q, compressed_kv, q.new_empty((2, 0)), positions=torch.arange(2))


def test_rtp_mla_attention_explicit_dense_backend_overrides_sparse_default(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    dense_backend = _FakeDenseBackend(v_head_dim=16)
    modules = SimpleNamespace(v_head_dim=16)

    attention = RTPMLAAttention(mla_modules=modules, dense_backend=dense_backend)

    assert attention.dense_backend is dense_backend

